//! GPU streaming pipeline orchestration

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;
use tokio::sync::mpsc;

use super::{GpuBufferPool, GpuStreamConfig, GpuStreamKernel};

/// GPU streaming pipeline
pub struct GpuStreamPipeline {
    device: Arc<CudaContext>,
    config: GpuStreamConfig,
    /// Pipeline stages
    stages: Vec<Box<dyn GpuStreamKernel>>,
    /// CUDA streams for pipelining
    streams: Vec<Arc<CudaStream>>,
    /// Buffer pools for each stage
    buffer_pools: Vec<GpuBufferPool>,
    /// Statistics
    stats: PipelineStatistics,
}

impl GpuStreamPipeline {
    /// Create new GPU streaming pipeline
    pub fn new(device: Arc<CudaContext>, config: GpuStreamConfig) -> Result<Self> {
        // Create CUDA streams
        let mut streams = Vec::with_capacity(config.num_streams);
        let default_stream = device.default_stream();
        for _ in 0..config.num_streams {
            streams.push(default_stream.fork()?);
        }

        Ok(Self {
            device,
            config,
            stages: Vec::new(),
            streams,
            buffer_pools: Vec::new(),
            stats: PipelineStatistics::default(),
        })
    }

    /// Add stage to pipeline
    pub fn add_stage(&mut self, stage: Box<dyn GpuStreamKernel>) -> Result<()> {
        // Create buffer pool for this stage
        let pool = GpuBufferPool::new(
            self.device.clone(),
            self.config.num_streams * 2,
            self.config.buffer_size,
        )?;

        self.buffer_pools.push(pool);
        self.stages.push(stage);

        Ok(())
    }

    /// Process data through pipeline
    pub async fn process(&mut self, input_data: Vec<u8>) -> Result<Vec<u8>> {
        let start_time = std::time::Instant::now();

        // Upload input to GPU
        let stream = self.device.default_stream();
        let mut current_buffer = stream.clone_htod(&input_data)?;

        // Process through each stage
        for (stage_idx, stage) in self.stages.iter().enumerate() {
            let stream_idx = stage_idx % self.streams.len();
            let stream = &self.streams[stream_idx];

            // Get output buffer from pool
            let _output_size = stage.output_size(current_buffer.len());
            let buffer_idx = self.buffer_pools[stage_idx]
                .acquire()
                .ok_or_else(|| anyhow::anyhow!("No available buffers for stage {}", stage_idx))?;

            // Get output buffer reference and process stage
            let output_buffer = self.buffer_pools[stage_idx]
                .get_buffer_mut(buffer_idx)
                .ok_or_else(|| anyhow::anyhow!("Invalid buffer index"))?;
            stage.process(&current_buffer, output_buffer, stream)?;

            // Wait for completion
            self.device.synchronize()?;

            // Release previous buffer if not the input
            if stage_idx > 0 {
                // Store buffer index to release later to avoid borrowing conflicts
                // In a real implementation, we'd use a proper release queue
                // self.buffer_pools[stage_idx - 1].release(buffer_idx);
            }

            // Update current buffer
            current_buffer = output_buffer.clone();
        }

        // Download result
        let result = stream.clone_dtoh(&current_buffer)?;

        // Update statistics
        self.stats.chunks_processed += 1;
        self.stats.bytes_processed += input_data.len() as u64;
        self.stats.total_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(result)
    }

    /// Process batch of data
    pub async fn process_batch(&mut self, batch: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>> {
        let mut results = Vec::with_capacity(batch.len());

        for data in batch {
            results.push(self.process(data).await?);
        }

        Ok(results)
    }

    /// Get pipeline statistics
    pub fn statistics(&self) -> &PipelineStatistics {
        &self.stats
    }

    /// Run pipeline in streaming mode
    pub async fn run_streaming(
        mut self,
        mut input_rx: mpsc::Receiver<Vec<u8>>,
        output_tx: mpsc::Sender<Vec<u8>>,
    ) -> Result<()> {
        while let Some(input) = input_rx.recv().await {
            let output = self.process(input).await?;
            if output_tx.send(output).await.is_err() {
                break;
            }
        }

        Ok(())
    }
}

/// Pipeline statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStatistics {
    pub chunks_processed: u64,
    pub bytes_processed: u64,
    pub total_time_ms: u64,
    pub errors: u64,
}

impl PipelineStatistics {
    /// Calculate throughput in GB/s
    pub fn throughput_gbps(&self) -> f64 {
        if self.total_time_ms > 0 {
            let bytes_per_ms = self.bytes_processed as f64 / self.total_time_ms as f64;
            bytes_per_ms * 1000.0 / (1024.0 * 1024.0 * 1024.0)
        } else {
            0.0
        }
    }

    /// Calculate average latency per chunk
    pub fn avg_latency_ms(&self) -> f64 {
        if self.chunks_processed > 0 {
            self.total_time_ms as f64 / self.chunks_processed as f64
        } else {
            0.0
        }
    }
}

/// Pipeline builder for fluent API
pub struct PipelineBuilder {
    device: Arc<CudaContext>,
    config: GpuStreamConfig,
    stages: Vec<Box<dyn GpuStreamKernel>>,
}

impl PipelineBuilder {
    /// Create new pipeline builder
    pub fn new(device: Arc<CudaContext>) -> Self {
        Self {
            device,
            config: GpuStreamConfig::default(),
            stages: Vec::new(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: GpuStreamConfig) -> Self {
        self.config = config;
        self
    }

    /// Add stage
    pub fn add_stage(mut self, stage: Box<dyn GpuStreamKernel>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Build pipeline
    pub fn build(self) -> Result<GpuStreamPipeline> {
        let mut pipeline = GpuStreamPipeline::new(self.device, self.config)?;

        for stage in self.stages {
            pipeline.add_stage(stage)?;
        }

        Ok(pipeline)
    }
}

/// Multi-stream pipeline for maximum throughput
pub struct MultiStreamPipeline {
    pipelines: Vec<GpuStreamPipeline>,
    round_robin: usize,
}

impl MultiStreamPipeline {
    /// Create multi-stream pipeline
    pub fn new(
        num_pipelines: usize,
        device: Arc<CudaContext>,
        config: GpuStreamConfig,
    ) -> Result<Self> {
        let mut pipelines = Vec::with_capacity(num_pipelines);

        for _ in 0..num_pipelines {
            pipelines.push(GpuStreamPipeline::new(device.clone(), config.clone())?);
        }

        Ok(Self {
            pipelines,
            round_robin: 0,
        })
    }

    /// Process data using round-robin scheduling
    pub async fn process(&mut self, input: Vec<u8>) -> Result<Vec<u8>> {
        let pipeline_idx = self.round_robin % self.pipelines.len();
        self.round_robin += 1;

        self.pipelines[pipeline_idx].process(input).await
    }

    /// Get aggregated statistics
    pub fn statistics(&self) -> PipelineStatistics {
        let mut total_stats = PipelineStatistics::default();

        for pipeline in &self.pipelines {
            let stats = pipeline.statistics();
            total_stats.chunks_processed += stats.chunks_processed;
            total_stats.bytes_processed += stats.bytes_processed;
            total_stats.total_time_ms += stats.total_time_ms;
            total_stats.errors += stats.errors;
        }

        total_stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock kernel for testing
    struct MockKernel {
        name: String,
        output_multiplier: f32,
    }

    impl GpuStreamKernel for MockKernel {
        fn name(&self) -> &str {
            &self.name
        }

        fn process(
            &self,
            _input: &CudaSlice<u8>,
            _output: &mut CudaSlice<u8>,
            _stream: &CudaStream,
        ) -> Result<()> {
            // Mock processing
            Ok(())
        }

        fn output_size(&self, input_size: usize) -> usize {
            (input_size as f32 * self.output_multiplier) as usize
        }
    }

    #[test]
    fn test_pipeline_builder() {
        if let Ok(ctx) = CudaContext::new(0) {
            let pipeline = PipelineBuilder::new(ctx)
                .with_config(GpuStreamConfig {
                    chunk_size: 1024,
                    num_streams: 2,
                    ..Default::default()
                })
                .add_stage(Box::new(MockKernel {
                    name: "stage1".to_string(),
                    output_multiplier: 1.0,
                }))
                .add_stage(Box::new(MockKernel {
                    name: "stage2".to_string(),
                    output_multiplier: 0.5,
                }))
                .build()
                .unwrap();

            assert_eq!(pipeline.stages.len(), 2);
        }
    }

    #[test]
    fn test_pipeline_statistics() {
        let mut stats = PipelineStatistics::default();
        stats.bytes_processed = 1_000_000_000; // 1GB
        stats.total_time_ms = 1000; // 1 second
        stats.chunks_processed = 1000;

        assert_eq!(stats.throughput_gbps(), 1.0 / 1.073741824); // ~0.93 GB/s
        assert_eq!(stats.avg_latency_ms(), 1.0);
    }
}
