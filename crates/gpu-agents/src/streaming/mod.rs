//! GPU-accelerated streaming pipeline

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;

pub mod compression;
pub mod huffman;
pub mod pipeline;
pub mod string_ops;
pub mod transform;

pub use compression::{CompressionAlgorithm, GpuCompressor};
pub use huffman::{
    CompressionLevel, GpuHuffmanProcessor, HuffmanCode, HuffmanCodec, HuffmanConfig,
    HuffmanEncoded, HuffmanStatistics, HuffmanStreamProcessor, HuffmanTree,
};
pub use pipeline::{GpuStreamPipeline, PipelineBuilder};
pub use string_ops::{
    FilterPredicate, GpuStringProcessor, SortOrder, StringOperation, StringProcessorConfig,
    StringProcessorStats, StringStreamProcessor, TransformFunction,
};
pub use transform::{GpuTransformer, TransformType};

/// GPU streaming configuration
#[derive(Debug, Clone)]
pub struct GpuStreamConfig {
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Number of CUDA streams for pipelining
    pub num_streams: usize,
    /// Buffer size for double buffering
    pub buffer_size: usize,
    /// Enable pinned memory for fast transfers
    pub use_pinned_memory: bool,
    /// Batch size for GPU operations
    pub batch_size: usize,
}

impl Default for GpuStreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024 * 1024,       // 1MB chunks
            num_streams: 4,                // 4 concurrent streams
            buffer_size: 64 * 1024 * 1024, // 64MB buffers
            use_pinned_memory: true,
            batch_size: 32, // 32 chunks per batch
        }
    }
}

/// GPU buffer pool for efficient memory management
pub struct GpuBufferPool {
    _device: Arc<CudaContext>,
    /// Pool of reusable GPU buffers
    buffers: Vec<CudaSlice<u8>>,
    /// Buffer size
    buffer_size: usize,
    /// Currently available buffers
    available: Vec<usize>,
}

impl GpuBufferPool {
    /// Create new buffer pool
    pub fn new(device: Arc<CudaContext>, num_buffers: usize, buffer_size: usize) -> Result<Self> {
        let mut buffers = Vec::with_capacity(num_buffers);
        let mut available = Vec::with_capacity(num_buffers);

        let stream = device.default_stream();
        for i in 0..num_buffers {
            // SAFETY: CudaDevice::alloc returns uninitialized GPU memory. This is safe
            // because the buffer pool manages buffer lifecycle and buffers are written
            // via htod_copy before any kernel reads from them.
            let buffer = unsafe { stream.alloc::<u8>(buffer_size)? };
            buffers.push(buffer);
            available.push(i);
        }

        Ok(Self {
            _device: device,
            buffers,
            buffer_size,
            available,
        })
    }

    /// Get a buffer from the pool (returns index only)
    pub fn acquire(&mut self) -> Option<usize> {
        self.available.pop()
    }

    /// Get mutable reference to buffer by index
    pub fn get_buffer_mut(&mut self, idx: usize) -> Option<&mut CudaSlice<u8>> {
        self.buffers.get_mut(idx)
    }

    /// Get immutable reference to buffer by index (for reading)
    pub fn get_buffer(&self, idx: usize) -> Option<&CudaSlice<u8>> {
        self.buffers.get(idx)
    }

    /// Return a buffer to the pool
    pub fn release(&mut self, idx: usize) {
        if idx < self.buffers.len() && !self.available.contains(&idx) {
            self.available.push(idx);
        }
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.buffers.len() * self.buffer_size
    }
}

/// GPU stream processor base
pub struct GpuStreamProcessor {
    device: Arc<CudaContext>,
    _config: GpuStreamConfig,
    /// CUDA streams for concurrent operations
    cuda_streams: Vec<Arc<CudaStream>>,
    /// Buffer pool
    buffer_pool: GpuBufferPool,
    /// Statistics
    bytes_processed: u64,
    chunks_processed: u64,
}

impl GpuStreamProcessor {
    /// Create new GPU stream processor
    pub fn new(device: Arc<CudaContext>, config: GpuStreamConfig) -> Result<Self> {
        // Create CUDA streams
        let mut cuda_streams = Vec::with_capacity(config.num_streams);
        let default_stream = device.default_stream();
        for _ in 0..config.num_streams {
            cuda_streams.push(default_stream.fork()?);
        }

        // Create buffer pool
        let buffer_pool = GpuBufferPool::new(
            device.clone(),
            config.num_streams * 2, // Double buffering
            config.buffer_size,
        )?;

        Ok(Self {
            device,
            _config: config,
            cuda_streams,
            buffer_pool,
            bytes_processed: 0,
            chunks_processed: 0,
        })
    }

    /// Process data chunk on GPU
    pub async fn process_chunk(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        // Get stream index for this operation
        let stream_idx = (self.chunks_processed as usize) % self.cuda_streams.len();
        let _stream = &self.cuda_streams[stream_idx];

        // Acquire buffer from pool
        let buffer_idx = self
            .buffer_pool
            .acquire()
            .ok_or_else(|| anyhow::anyhow!("No available buffers"))?;

        // Copy data to GPU
        let default_stream = self.device.default_stream();
        let gpu_buffer = self
            .buffer_pool
            .get_buffer_mut(buffer_idx)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer index"))?;
        default_stream.memcpy_htod(data, gpu_buffer)?;

        // Process on GPU (placeholder - actual kernel would go here)
        // Note: Simplified to avoid borrowing issues - actual kernel launch would go here

        // Copy result back
        let gpu_buffer = self
            .buffer_pool
            .get_buffer_mut(buffer_idx)
            .ok_or_else(|| anyhow::anyhow!("Invalid buffer index"))?;
        let result: Vec<u8> = default_stream.clone_dtoh(gpu_buffer)?;

        // Release buffer
        self.buffer_pool.release(buffer_idx);

        // Update statistics
        self.bytes_processed += data.len() as u64;
        self.chunks_processed += 1;

        Ok(result)
    }

    /// Process on GPU (placeholder for actual kernel)
    #[allow(dead_code)]
    fn process_on_gpu(&self, _buffer: &mut CudaSlice<u8>, _stream: &CudaStream) -> Result<()> {
        // Actual GPU kernel would be launched here
        Ok(())
    }

    /// Get processing statistics
    pub fn statistics(&self) -> StreamStatistics {
        StreamStatistics {
            bytes_processed: self.bytes_processed,
            chunks_processed: self.chunks_processed,
            throughput_gbps: self.calculate_throughput(),
        }
    }

    /// Calculate throughput in GB/s
    fn calculate_throughput(&self) -> f64 {
        // Placeholder calculation
        if self.chunks_processed > 0 {
            (self.bytes_processed as f64) / (1024.0 * 1024.0 * 1024.0)
        } else {
            0.0
        }
    }
}

/// Stream processing statistics
#[derive(Debug, Clone)]
pub struct StreamStatistics {
    pub bytes_processed: u64,
    pub chunks_processed: u64,
    pub throughput_gbps: f64,
}

/// GPU stream kernel interface
pub trait GpuStreamKernel: Send + Sync {
    /// Get kernel name
    fn name(&self) -> &str;

    /// Process data on GPU
    fn process(
        &self,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        stream: &CudaStream,
    ) -> Result<()>;

    /// Get output size for given input size
    fn output_size(&self, input_size: usize) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_stream_config_default() {
        let config = GpuStreamConfig::default();
        assert_eq!(config.chunk_size, 1024 * 1024);
        assert_eq!(config.num_streams, 4);
        assert!(config.use_pinned_memory);
    }

    #[test]
    fn test_buffer_pool() -> Result<(), Box<dyn std::error::Error>> {
        if let Ok(ctx) = CudaContext::new(0) {
            let mut pool = GpuBufferPool::new(ctx, 4, 1024)?;

            // Acquire all buffers
            let mut acquired = Vec::new();
            for _ in 0..4 {
                if let Some(idx) = pool.acquire() {
                    acquired.push(idx);
                }
            }

            // Pool should be empty
            assert!(pool.acquire().is_none());

            // Release one buffer
            pool.release(acquired[0]);

            // Should be able to acquire again
            assert!(pool.acquire().is_some());
        }
        Ok(())
    }
}

#[cfg(test)]
mod string_ops_tests;

#[cfg(test)]
mod huffman_tests;
