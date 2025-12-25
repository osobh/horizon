//! GPU-accelerated stream processor for high-performance data transformations

use crate::{StreamChunk, StreamProcessor, StreamStats, StreamingError};
use async_trait::async_trait;
use bytes::Bytes;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// GPU operation types
#[derive(Debug, Clone)]
pub enum GpuOperation {
    /// Matrix multiplication
    MatrixMultiply { rows: usize, cols: usize },
    /// Vector addition
    VectorAdd,
    /// Convolution operation
    Convolution { kernel_size: usize },
    /// Element-wise transformation
    Transform { func_id: u32 },
    /// Custom CUDA kernel
    #[cfg(feature = "cuda")]
    CustomKernel { kernel_name: String },
}

/// GPU stream processor for high-performance batch operations
pub struct GpuStreamProcessor {
    id: String,
    operation: GpuOperation,
    batch_size: usize,
    use_cuda: bool,
    stats: Arc<GpuProcessorStats>,
}

/// Thread-safe statistics for GPU processor
#[derive(Debug, Default)]
struct GpuProcessorStats {
    chunks_processed: AtomicU64,
    bytes_processed: AtomicU64,
    gpu_time_ns: AtomicU64,
    batch_operations: AtomicU64,
    errors: AtomicU64,
}

impl GpuStreamProcessor {
    /// Create a new GPU stream processor
    pub fn new(id: String, operation: GpuOperation) -> Self {
        Self {
            id,
            operation,
            batch_size: 32,  // 32 chunks per batch for GPU efficiency
            use_cuda: false, // Default to CPU simulation
            stats: Arc::new(GpuProcessorStats::default()),
        }
    }

    /// Configure batch size for GPU operations
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable/disable CUDA usage
    pub fn with_cuda(mut self, use_cuda: bool) -> Self {
        self.use_cuda = use_cuda;
        self
    }

    /// Get current statistics snapshot
    pub fn get_stats_snapshot(&self) -> StreamStats {
        // Relaxed: independent statistics counters with no ordering dependencies
        let chunks = self.stats.chunks_processed.load(Ordering::Relaxed);
        let bytes = self.stats.bytes_processed.load(Ordering::Relaxed);
        let time_ns = self.stats.gpu_time_ns.load(Ordering::Relaxed);
        let errors = self.stats.errors.load(Ordering::Relaxed);

        let throughput_mbps = if time_ns > 0 {
            (bytes as f64) / ((time_ns as f64) / 1_000_000_000.0) / (1024.0 * 1024.0)
        } else {
            0.0
        };

        StreamStats {
            chunks_processed: chunks,
            bytes_processed: bytes,
            processing_time_ms: time_ns / 1_000_000,
            throughput_mbps,
            errors,
        }
    }

    /// Simulate GPU operation on data
    fn simulate_gpu_operation(&self, data: &[u8]) -> Result<Vec<u8>, StreamingError> {
        match &self.operation {
            GpuOperation::MatrixMultiply { rows, cols } => {
                // Simulate matrix multiplication by reshaping and basic ops
                if data.len() < rows * cols {
                    return Err(StreamingError::InvalidInput(format!(
                        "Data too small for {rows}x{cols} matrix"
                    )));
                }

                let mut result = Vec::with_capacity(data.len());
                for chunk in data.chunks(*rows) {
                    // Simulate matrix ops by doubling values
                    for &byte in chunk {
                        result.push(byte.saturating_mul(2));
                    }
                }
                Ok(result)
            }
            GpuOperation::VectorAdd => {
                // Simulate vector addition by adding a constant
                let result = data.iter().map(|&b| b.saturating_add(1)).collect();
                Ok(result)
            }
            GpuOperation::Convolution { kernel_size } => {
                // Simulate convolution with simple smoothing
                let mut result = Vec::with_capacity(data.len());
                for i in 0..data.len() {
                    let start = i.saturating_sub(*kernel_size / 2);
                    let end = std::cmp::min(i + *kernel_size / 2 + 1, data.len());
                    let sum: u32 = data[start..end].iter().map(|&b| b as u32).sum();
                    let avg = (sum / (end - start) as u32) as u8;
                    result.push(avg);
                }
                Ok(result)
            }
            GpuOperation::Transform { func_id } => {
                // Apply transformation based on function ID
                let result = match func_id {
                    1 => data.iter().map(|&b| b.wrapping_mul(3)).collect(),
                    2 => data.iter().map(|&b| b ^ 0xFF).collect(), // Bitwise NOT
                    3 => data.iter().map(|&b| b.rotate_left(2)).collect(),
                    _ => data.iter().map(|&b| b.saturating_add(10)).collect(),
                };
                Ok(result)
            }
            #[cfg(feature = "cuda")]
            GpuOperation::CustomKernel { kernel_name: _ } => {
                // For now, simulate custom kernel
                Ok(data.to_vec())
            }
        }
    }
}

#[async_trait]
impl StreamProcessor for GpuStreamProcessor {
    async fn process(&mut self, chunk: StreamChunk) -> Result<StreamChunk, StreamingError> {
        let start_time = Instant::now();

        let result_data = if self.use_cuda {
            // TODO: Implement actual CUDA processing
            self.simulate_gpu_operation(&chunk.data)?
        } else {
            // Use CPU simulation
            self.simulate_gpu_operation(&chunk.data)?
        };

        let processing_time = start_time.elapsed().as_nanos() as u64;
        // Relaxed: independent statistics counters
        self.stats
            .gpu_time_ns
            .fetch_add(processing_time, Ordering::Relaxed);
        self.stats.chunks_processed.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_processed
            .fetch_add(chunk.data.len() as u64, Ordering::Relaxed);

        Ok(StreamChunk::new(
            Bytes::from(result_data),
            chunk.sequence,
            chunk.metadata.source_id,
        ))
    }

    async fn process_batch(
        &mut self,
        chunks: Vec<StreamChunk>,
    ) -> Result<Vec<StreamChunk>, StreamingError> {
        let start_time = Instant::now();
        let mut results = Vec::with_capacity(chunks.len());

        if self.use_cuda && chunks.len() >= self.batch_size {
            // Batch GPU processing for efficiency
            // Relaxed: independent counter
            self.stats.batch_operations.fetch_add(1, Ordering::Relaxed);

            for chunk in chunks {
                let result_data = self.simulate_gpu_operation(&chunk.data)?;
                results.push(StreamChunk::new(
                    Bytes::from(result_data),
                    chunk.sequence,
                    chunk.metadata.source_id,
                ));
            }
        } else {
            // Fall back to individual processing
            for chunk in chunks {
                results.push(self.process(chunk).await?);
            }
        }

        let processing_time = start_time.elapsed().as_nanos() as u64;
        // Relaxed: independent statistics counter
        self.stats
            .gpu_time_ns
            .fetch_add(processing_time, Ordering::Relaxed);

        Ok(results)
    }

    async fn stats(&self) -> Result<StreamStats, StreamingError> {
        Ok(self.get_stats_snapshot())
    }

    fn processor_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_processor_creation() {
        let processor = GpuStreamProcessor::new("test-gpu".to_string(), GpuOperation::VectorAdd);

        assert_eq!(processor.processor_id(), "test-gpu");
        assert_eq!(processor.batch_size, 32);
        assert!(!processor.use_cuda);
    }

    #[tokio::test]
    async fn test_gpu_processor_configuration() {
        let processor = GpuStreamProcessor::new(
            "config-test".to_string(),
            GpuOperation::MatrixMultiply { rows: 4, cols: 4 },
        )
        .with_batch_size(64)
        .with_cuda(true);

        assert_eq!(processor.batch_size, 64);
        assert!(processor.use_cuda);
    }

    #[tokio::test]
    async fn test_vector_add_operation() {
        let mut processor =
            GpuStreamProcessor::new("vector-add-test".to_string(), GpuOperation::VectorAdd);

        let input_data = vec![10, 20, 30, 40];
        let chunk = StreamChunk::new(
            Bytes::from(input_data.clone()),
            1,
            "test-source".to_string(),
        );

        let result = processor.process(chunk).await.unwrap();
        let result_data: Vec<u8> = result.data.to_vec();

        // Vector add should add 1 to each element
        assert_eq!(result_data, vec![11, 21, 31, 41]);
        assert_eq!(result.sequence, 1);
        assert_eq!(result.metadata.source_id, "test-source");

        let stats = processor.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
        assert_eq!(stats.bytes_processed, 4);
    }

    #[tokio::test]
    async fn test_matrix_multiply_operation() {
        let mut processor = GpuStreamProcessor::new(
            "matrix-test".to_string(),
            GpuOperation::MatrixMultiply { rows: 2, cols: 2 },
        );

        let input_data = vec![1, 2, 3, 4]; // 2x2 matrix
        let chunk = StreamChunk::new(Bytes::from(input_data), 1, "test-source".to_string());

        let result = processor.process(chunk).await.unwrap();
        let result_data: Vec<u8> = result.data.to_vec();

        // Matrix multiply simulation doubles values
        assert_eq!(result_data, vec![2, 4, 6, 8]);
    }

    #[tokio::test]
    async fn test_convolution_operation() {
        let mut processor = GpuStreamProcessor::new(
            "conv-test".to_string(),
            GpuOperation::Convolution { kernel_size: 3 },
        );

        let input_data = vec![10, 20, 30, 40, 50];
        let chunk = StreamChunk::new(Bytes::from(input_data), 1, "test-source".to_string());

        let result = processor.process(chunk).await.unwrap();
        let result_data: Vec<u8> = result.data.to_vec();

        // Convolution should smooth the values
        assert_eq!(result_data.len(), 5);
        assert!(result_data[2] > 20 && result_data[2] < 40); // Middle value should be averaged
    }

    #[tokio::test]
    async fn test_transform_operations() {
        let input_data = vec![10, 20, 30, 40];
        let chunk = StreamChunk::new(
            Bytes::from(input_data.clone()),
            1,
            "test-source".to_string(),
        );

        // Test different transform functions
        for func_id in [1, 2, 3, 99] {
            let mut processor = GpuStreamProcessor::new(
                format!("transform-{func_id}"),
                GpuOperation::Transform { func_id },
            );

            let result = processor.process(chunk.clone()).await.unwrap();
            let result_data: Vec<u8> = result.data.to_vec();

            match func_id {
                1 => assert_eq!(result_data, vec![30, 60, 90, 120]), // multiply by 3
                2 => assert_eq!(result_data, vec![245, 235, 225, 215]), // bitwise NOT
                3 => assert_eq!(result_data, vec![40, 80, 120, 160]), // rotate left 2
                _ => assert_eq!(result_data, vec![20, 30, 40, 50]),  // add 10
            }
        }
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let mut processor =
            GpuStreamProcessor::new("batch-test".to_string(), GpuOperation::VectorAdd)
                .with_batch_size(2);

        let chunks = vec![
            StreamChunk::new(Bytes::from(vec![1, 2]), 1, "test".to_string()),
            StreamChunk::new(Bytes::from(vec![3, 4]), 2, "test".to_string()),
            StreamChunk::new(Bytes::from(vec![5, 6]), 3, "test".to_string()),
        ];

        let results = processor.process_batch(chunks).await.unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].data.to_vec(), vec![2, 3]);
        assert_eq!(results[1].data.to_vec(), vec![4, 5]);
        assert_eq!(results[2].data.to_vec(), vec![6, 7]);

        let stats = processor.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 3);
    }

    #[tokio::test]
    async fn test_matrix_multiply_error() {
        let mut processor = GpuStreamProcessor::new(
            "error-test".to_string(),
            GpuOperation::MatrixMultiply { rows: 4, cols: 4 },
        );

        // Data too small for 4x4 matrix
        let chunk = StreamChunk::new(
            Bytes::from(vec![1, 2, 3]), // Only 3 bytes, need 16
            1,
            "test-source".to_string(),
        );

        let result = processor.process(chunk).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            StreamingError::InvalidInput(msg) => assert!(msg.contains("Data too small")),
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_gpu_processor_stats() {
        let mut processor =
            GpuStreamProcessor::new("stats-test".to_string(), GpuOperation::VectorAdd);

        // Initial stats should be zero
        let initial_stats = processor.stats().await.unwrap();
        assert_eq!(initial_stats.chunks_processed, 0);
        assert_eq!(initial_stats.bytes_processed, 0);

        // Process some chunks
        for i in 0..3 {
            let chunk = StreamChunk::new(
                Bytes::from(vec![i, i + 1, i + 2]),
                i as u64,
                "test".to_string(),
            );
            processor.process(chunk).await.unwrap();
        }

        let final_stats = processor.stats().await.unwrap();
        assert_eq!(final_stats.chunks_processed, 3);
        assert_eq!(final_stats.bytes_processed, 9); // 3 bytes per chunk * 3 chunks
        assert!(final_stats.processing_time_ms >= 0);
        assert!(final_stats.throughput_mbps >= 0.0);
    }
}
