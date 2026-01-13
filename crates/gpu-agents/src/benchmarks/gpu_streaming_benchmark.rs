//! GPU Streaming pipeline benchmarks

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;

use crate::streaming::{
    CompressionAlgorithm, GpuCompressor, GpuStreamConfig, GpuStreamPipeline, GpuTransformer,
    PipelineBuilder, TransformType,
};

/// Streaming benchmark configuration
#[derive(Debug, Clone)]
pub struct StreamingBenchmarkConfig {
    /// Data sizes to test (in MB)
    pub data_sizes_mb: Vec<usize>,
    /// Chunk sizes to test
    pub chunk_sizes: Vec<usize>,
    /// Number of streams for pipelining
    pub num_streams: Vec<usize>,
    /// Compression algorithms to test
    pub compression_algorithms: Vec<CompressionAlgorithm>,
    /// Transform types to test
    pub transform_types: Vec<TransformType>,
    /// Number of iterations per test
    pub iterations: usize,
}

impl Default for StreamingBenchmarkConfig {
    fn default() -> Self {
        Self {
            data_sizes_mb: vec![1, 10, 100, 1000],
            chunk_sizes: vec![64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024],
            num_streams: vec![1, 2, 4, 8],
            compression_algorithms: vec![
                CompressionAlgorithm::Lz4,
                CompressionAlgorithm::Rle,
                CompressionAlgorithm::Delta,
            ],
            transform_types: vec![
                TransformType::JsonParse,
                TransformType::CsvParse,
                TransformType::Normalize,
            ],
            iterations: 10,
        }
    }
}

/// Streaming benchmark results
#[derive(Debug, Clone)]
pub struct StreamingBenchmarkResults {
    pub data_size_mb: usize,
    pub chunk_size: usize,
    pub num_streams: usize,
    pub total_time_ms: f64,
    pub throughput_gbps: f64,
    pub latency_ms: f64,
    pub compression_ratio: f64,
    pub memory_usage_mb: f64,
}

/// GPU Streaming benchmark suite
pub struct GpuStreamingBenchmark {
    ctx: Arc<cudarc::driver::CudaContext>,
    config: StreamingBenchmarkConfig,
}

impl GpuStreamingBenchmark {
    /// Create new benchmark suite
    pub fn new(device_id: i32, config: StreamingBenchmarkConfig) -> Result<Self> {
        let ctx = cudarc::driver::CudaContext::new(device_id as usize)?;
        Ok(Self { ctx, config })
    }

    /// Run all benchmarks
    pub async fn run_all(&mut self) -> Result<Vec<StreamingBenchmarkResults>> {
        let mut results = Vec::new();

        for &data_size_mb in &self.config.data_sizes_mb {
            for &chunk_size in &self.config.chunk_sizes {
                for &num_streams in &self.config.num_streams {
                    println!(
                        "\nBenchmarking: data={}MB, chunk={}KB, streams={}",
                        data_size_mb,
                        chunk_size / 1024,
                        num_streams
                    );

                    match self
                        .benchmark_pipeline(data_size_mb, chunk_size, num_streams)
                        .await
                    {
                        Ok(result) => {
                            println!("  Time: {:.2}ms", result.total_time_ms);
                            println!("  Throughput: {:.2} GB/s", result.throughput_gbps);
                            println!("  Latency: {:.2}ms", result.latency_ms);
                            results.push(result);
                        }
                        Err(e) => {
                            eprintln!("  Benchmark failed: {}", e);
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Benchmark pipeline with specific parameters
    async fn benchmark_pipeline(
        &self,
        data_size_mb: usize,
        chunk_size: usize,
        num_streams: usize,
    ) -> Result<StreamingBenchmarkResults> {
        let config = GpuStreamConfig {
            chunk_size,
            num_streams,
            buffer_size: chunk_size * 2,
            use_pinned_memory: true,
            batch_size: 8,
        };

        // Create pipeline
        let mut pipeline = GpuStreamPipeline::new(self.ctx.clone(), config.clone())?;

        // Add compression stage
        let compressor = Box::new(GpuCompressor::new(
            self.ctx.clone(),
            CompressionAlgorithm::Lz4,
            chunk_size,
        )?);
        pipeline.add_stage(compressor)?;

        // Generate test data
        let total_bytes = data_size_mb * 1024 * 1024;
        let test_data = self.generate_test_data(total_bytes);

        let start = Instant::now();
        let mut total_processed = 0;

        // Process in chunks
        for chunk_data in test_data.chunks(chunk_size) {
            let result = pipeline.process(chunk_data.to_vec()).await?;
            total_processed += result.len();
        }

        let elapsed = start.elapsed();

        // Calculate metrics
        let throughput_gbps =
            (total_bytes as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        let num_chunks = (total_bytes + chunk_size - 1) / chunk_size;
        let latency_ms = (elapsed.as_secs_f64() * 1000.0) / num_chunks as f64;

        let _stats = pipeline.statistics();

        Ok(StreamingBenchmarkResults {
            data_size_mb,
            chunk_size,
            num_streams,
            total_time_ms: elapsed.as_secs_f64() * 1000.0,
            throughput_gbps,
            latency_ms,
            compression_ratio: total_bytes as f64 / total_processed as f64,
            memory_usage_mb: (num_streams * chunk_size * 2) as f64 / (1024.0 * 1024.0),
        })
    }

    /// Generate test data
    fn generate_test_data(&self, size: usize) -> Vec<u8> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Generate semi-compressible data
        let mut data = vec![0u8; size];
        let patterns = vec![
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            b"The quick brown fox jumps over the lazy dog.            ",
            b"0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123ABCD",
            b"{ \"key\": \"value\", \"number\": 42 } { \"key\": \"value\" }     ",
        ];

        let mut offset = 0;
        while offset < size {
            let pattern = patterns[rng.gen_range(0..patterns.len())];
            let copy_len = pattern.len().min(size - offset);
            data[offset..offset + copy_len].copy_from_slice(&pattern[..copy_len]);
            offset += copy_len;

            // Add some random data
            if offset < size && rng.gen_bool(0.3) {
                let random_len = rng.gen_range(1..100).min(size - offset);
                for i in 0..random_len {
                    data[offset + i] = rng.r#gen();
                }
                offset += random_len;
            }
        }

        data
    }

    /// Benchmark compression algorithms
    pub async fn benchmark_compression(&self) -> Result<()> {
        println!("\n=== Compression Algorithm Benchmarks ===");

        let _data_size = 10 * 1024 * 1024; // 10MB
        let chunk_size = 1024 * 1024; // 1MB
        let test_data = self.generate_test_data(chunk_size);

        for &algorithm in &self.config.compression_algorithms {
            println!("\nAlgorithm: {:?}", algorithm);

            let config = GpuStreamConfig {
                chunk_size,
                num_streams: 1,
                ..Default::default()
            };

            let mut pipeline = PipelineBuilder::new(self.ctx.clone())
                .with_config(config)
                .add_stage(Box::new(GpuCompressor::new(
                    self.ctx.clone(),
                    algorithm,
                    chunk_size,
                )?))
                .build()?;

            let start = Instant::now();
            let mut total_compressed = 0;

            for _ in 0..self.config.iterations {
                let result = pipeline.process(test_data.clone()).await?;
                total_compressed += result.len();
            }

            let elapsed = start.elapsed();
            let throughput = (chunk_size * self.config.iterations) as f64
                / elapsed.as_secs_f64()
                / (1024.0 * 1024.0 * 1024.0);
            let compression_ratio =
                (chunk_size * self.config.iterations) as f64 / total_compressed as f64;

            println!(
                "  Throughput: {:.2} GB/s, Compression ratio: {:.2}:1",
                throughput, compression_ratio
            );
        }

        Ok(())
    }

    /// Benchmark transform operations
    pub async fn benchmark_transforms(&self) -> Result<()> {
        println!("\n=== Transform Operation Benchmarks ===");

        let chunk_size = 1024 * 1024; // 1MB

        for &transform_type in &self.config.transform_types {
            println!("\nTransform: {:?}", transform_type);

            let test_data = match transform_type {
                TransformType::JsonParse => {
                    r#"{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}"#
                        .repeat(chunk_size / 60)
                        .into_bytes()
                }
                TransformType::CsvParse => "id,name,value\n1,Alice,100\n2,Bob,200\n"
                    .repeat(chunk_size / 30)
                    .into_bytes(),
                TransformType::Normalize => {
                    // Float data for normalization
                    let floats: Vec<f32> = (0..chunk_size / 4).map(|i| i as f32 * 0.1).collect();
                    bytemuck::cast_slice(&floats).to_vec()
                }
                _ => vec![0u8; chunk_size],
            };

            let config = GpuStreamConfig {
                chunk_size: test_data.len(),
                num_streams: 1,
                ..Default::default()
            };

            let mut pipeline = PipelineBuilder::new(self.ctx.clone())
                .with_config(config)
                .add_stage(Box::new(GpuTransformer::new(
                    self.ctx.clone(),
                    transform_type,
                    test_data.len() * 2,
                )?))
                .build()?;

            let start = Instant::now();

            for _ in 0..self.config.iterations {
                let _ = pipeline.process(test_data.clone()).await?;
            }

            let elapsed = start.elapsed();
            let throughput = (test_data.len() * self.config.iterations) as f64
                / elapsed.as_secs_f64()
                / (1024.0 * 1024.0 * 1024.0);

            println!("  Throughput: {:.2} GB/s", throughput);
        }

        Ok(())
    }

    /// Benchmark pipeline scaling
    pub async fn benchmark_scaling(&self) -> Result<()> {
        println!("\n=== Pipeline Scaling Benchmarks ===");

        let data_size = 100 * 1024 * 1024; // 100MB
        let chunk_size = 1024 * 1024; // 1MB

        for &num_streams in &self.config.num_streams {
            println!("\nStreams: {}", num_streams);

            let config = GpuStreamConfig {
                chunk_size,
                num_streams,
                ..Default::default()
            };

            // Create multi-stage pipeline
            let mut pipeline = PipelineBuilder::new(self.ctx.clone())
                .with_config(config)
                .add_stage(Box::new(GpuTransformer::new(
                    self.ctx.clone(),
                    TransformType::Normalize,
                    chunk_size,
                )?))
                .add_stage(Box::new(GpuCompressor::new(
                    self.ctx.clone(),
                    CompressionAlgorithm::Lz4,
                    chunk_size,
                )?))
                .build()?;

            let test_data = self.generate_test_data(data_size);
            let start = Instant::now();

            // Process all data
            for chunk in test_data.chunks(chunk_size) {
                let _ = pipeline.process(chunk.to_vec()).await?;
            }

            let elapsed = start.elapsed();
            let throughput = data_size as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);

            println!(
                "  Time: {:.2}s, Throughput: {:.2} GB/s",
                elapsed.as_secs_f64(),
                throughput
            );
        }

        Ok(())
    }
}

/// Run streaming benchmarks
pub async fn run_streaming_benchmarks() -> Result<()> {
    println!("=== GPU Streaming Pipeline Benchmarks ===");

    let config = StreamingBenchmarkConfig::default();
    let mut benchmark = GpuStreamingBenchmark::new(0, config)?;

    // Run comprehensive benchmarks
    let results = benchmark.run_all().await?;

    // Print summary
    println!("\n=== Summary ===");
    for result in &results {
        println!(
            "Data: {}MB, Chunk: {}KB, Streams: {}, Throughput: {:.2} GB/s",
            result.data_size_mb,
            result.chunk_size / 1024,
            result.num_streams,
            result.throughput_gbps
        );
    }

    // Run specialized benchmarks
    benchmark.benchmark_compression().await?;
    benchmark.benchmark_transforms().await?;
    benchmark.benchmark_scaling().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_streaming_benchmark_small() {
        let config = StreamingBenchmarkConfig {
            data_sizes_mb: vec![1],
            chunk_sizes: vec![64 * 1024],
            num_streams: vec![1],
            iterations: 2,
            ..Default::default()
        };

        if let Ok(mut benchmark) = GpuStreamingBenchmark::new(0, config) {
            let results = benchmark.run_all().await?;
            assert!(!results.is_empty());
            assert!(results[0].throughput_gbps > 0.0);
        }
    }
}
