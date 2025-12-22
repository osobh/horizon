//! CUDA stream management for asynchronous operations

use crate::error::{CudaError, CudaResult};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

/// Stream creation flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamFlags {
    /// Non-blocking stream
    pub non_blocking: bool,
    /// Disable implicit synchronization
    pub disable_timing: bool,
    /// Raw bit flags (for compatibility)
    pub bits: u32,
}

impl Default for StreamFlags {
    fn default() -> Self {
        Self {
            non_blocking: true,
            disable_timing: false,
            bits: 0,
        }
    }
}

/// CUDA stream for asynchronous operations
#[derive(Debug)]
pub struct Stream {
    /// Unique stream ID
    pub id: Uuid,
    /// Stream name
    pub name: String,
    /// Stream handle (would be cudaStream_t in real implementation)
    handle: u64,
    /// Creation flags
    pub flags: StreamFlags,
    /// Whether this is a mock stream
    is_mock: bool,
    /// Operation counter for testing
    operation_count: AtomicU64,
}

impl Stream {
    /// Create a new stream
    pub fn new(_name: String, _flags: StreamFlags) -> CudaResult<Self> {
        #[cfg(cuda_mock)]
        {
            Ok(Self {
                id: Uuid::new_v4(),
                name: _name,
                handle: rand::random::<u64>(),
                flags: _flags,
                is_mock: true,
                operation_count: AtomicU64::new(0),
            })
        }

        #[cfg(not(cuda_mock))]
        {
            // In real implementation, would use cudaStreamCreateWithFlags
            Err(CudaError::MockModeError)
        }
    }

    /// Create new stream with default configuration
    pub fn new_default() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "default".to_string(),
            handle: 0, // Default stream handle is 0
            flags: StreamFlags::default(),
            is_mock: cfg!(cuda_mock),
            operation_count: AtomicU64::new(0),
        }
    }

    /// Synchronize stream (wait for all operations to complete)
    pub async fn synchronize(&self) -> CudaResult<()> {
        #[cfg(cuda_mock)]
        {
            // Simulate async wait
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
            Ok(())
        }

        #[cfg(not(cuda_mock))]
        {
            // In real implementation, would use cudaStreamSynchronize
            Err(CudaError::MockModeError)
        }
    }

    /// Check if stream operations are complete
    pub fn is_complete(&self) -> CudaResult<bool> {
        #[cfg(cuda_mock)]
        {
            // Mock: always complete after first check
            Ok(true)
        }

        #[cfg(not(cuda_mock))]
        {
            // In real implementation, would use cudaStreamQuery
            Err(CudaError::MockModeError)
        }
    }

    /// Queue a kernel launch on this stream (mock)
    pub async fn launch_kernel(
        &self,
        _kernel_name: &str,
        _grid_dim: (u32, u32, u32),
        _block_dim: (u32, u32, u32),
    ) -> CudaResult<()> {
        #[cfg(cuda_mock)]
        {
            self.operation_count.fetch_add(1, Ordering::SeqCst);

            // Validate dimensions
            if _block_dim.0 * _block_dim.1 * _block_dim.2 > 1024 {
                return Err(CudaError::InvalidParameter {
                    message: "Block size exceeds maximum threads per block".to_string(),
                });
            }

            // Simulate kernel launch
            tokio::time::sleep(tokio::time::Duration::from_micros(10)).await;

            Ok(())
        }

        #[cfg(not(cuda_mock))]
        {
            Err(CudaError::MockModeError)
        }
    }

    /// Queue a memory copy on this stream
    pub async fn copy_async(&self, _dst: u64, _src: u64, size: usize) -> CudaResult<()> {
        #[cfg(cuda_mock)]
        {
            self.operation_count.fetch_add(1, Ordering::SeqCst);

            // Basic validation
            if size == 0 {
                return Err(CudaError::InvalidParameter {
                    message: "Copy size cannot be zero".to_string(),
                });
            }

            // Simulate async copy
            tokio::time::sleep(tokio::time::Duration::from_micros(5)).await;

            Ok(())
        }

        #[cfg(not(cuda_mock))]
        {
            let _ = size;
            Err(CudaError::MockModeError)
        }
    }

    /// Get handle
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Get operation count (for testing)
    pub fn operation_count(&self) -> u64 {
        self.operation_count.load(Ordering::SeqCst)
    }

    /// Check if mock stream
    pub fn is_mock(&self) -> bool {
        self.is_mock
    }

    /// Wait for an event on this stream
    pub async fn wait_event(&self, _event_id: Uuid) -> CudaResult<()> {
        #[cfg(cuda_mock)]
        {
            // Simulate waiting for event
            tokio::time::sleep(tokio::time::Duration::from_micros(50)).await;
            Ok(())
        }

        #[cfg(not(cuda_mock))]
        {
            Err(CudaError::MockModeError)
        }
    }

    /// Add a callback to be executed on stream completion
    pub async fn add_callback<F>(&self, _callback: F) -> CudaResult<()>
    where
        F: FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = CudaResult<()>> + Send>>
            + Send
            + 'static,
    {
        #[cfg(cuda_mock)]
        {
            // In mock mode, we just simulate the callback registration
            self.operation_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        #[cfg(not(cuda_mock))]
        {
            Err(CudaError::MockModeError)
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // In real implementation, would call cudaStreamDestroy
        #[cfg(not(cuda_mock))]
        {
            // cudaStreamDestroy(self.handle);
        }
    }
}

/// Stream priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamPriority {
    /// Lowest priority
    Low = -1,
    /// Normal priority (default)
    Normal = 0,
    /// High priority
    High = 1,
}

impl Default for Stream {
    fn default() -> Self {
        Self::new_default()
    }
}

/// Stream pool for managing multiple streams
pub struct StreamPool {
    /// Pool name
    pub name: String,
    /// Available streams
    streams: Vec<Arc<Stream>>,
    /// Next stream index for round-robin
    next_index: AtomicU64,
}

impl StreamPool {
    /// Create a new stream pool
    pub fn new(name: String, size: usize) -> CudaResult<Self> {
        let mut streams = Vec::with_capacity(size);

        for i in 0..size {
            let stream_name = format!("{name}_stream_{i}");
            let stream = Stream::new(stream_name, StreamFlags::default())?;
            streams.push(Arc::new(stream));
        }

        Ok(Self {
            name,
            streams,
            next_index: AtomicU64::new(0),
        })
    }

    /// Get next available stream (round-robin)
    pub fn get_stream(&self) -> Arc<Stream> {
        let index = self.next_index.fetch_add(1, Ordering::Relaxed) as usize;
        let stream_index = index % self.streams.len();
        self.streams[stream_index].clone()
    }

    /// Synchronize all streams
    pub async fn synchronize_all(&self) -> CudaResult<()> {
        for stream in &self.streams {
            stream.synchronize().await?;
        }
        Ok(())
    }

    /// Get pool size
    pub fn size(&self) -> usize {
        self.streams.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_flags_default() {
        let flags = StreamFlags::default();
        assert!(flags.non_blocking);
        assert!(!flags.disable_timing);
    }

    #[tokio::test]
    async fn test_stream_creation() {
        #[cfg(cuda_mock)]
        {
            let stream = Stream::new("test_stream".to_string(), StreamFlags::default()).unwrap();

            assert_eq!(stream.name, "test_stream");
            assert!(stream.is_mock());
            assert_eq!(stream.operation_count(), 0);
        }
    }

    #[test]
    fn test_default_stream() {
        let stream = Stream::default();
        assert_eq!(stream.name, "default");
        assert_eq!(stream.handle(), 0);
    }

    #[tokio::test]
    async fn test_stream_synchronize() {
        #[cfg(cuda_mock)]
        {
            let stream = Stream::new("test_stream".to_string(), StreamFlags::default()).unwrap();

            assert!(stream.synchronize().await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_stream_is_complete() {
        #[cfg(cuda_mock)]
        {
            let stream = Stream::new("test_stream".to_string(), StreamFlags::default()).unwrap();

            assert!(stream.is_complete()?);
        }
    }

    #[tokio::test]
    async fn test_launch_kernel() {
        #[cfg(cuda_mock)]
        {
            let stream = Stream::new("test_stream".to_string(), StreamFlags::default()).unwrap();

            // Valid launch
            assert!(stream
                .launch_kernel(
                    "test_kernel",
                    (32, 1, 1),  // grid
                    (256, 1, 1), // block
                )
                .await
                .is_ok());

            assert_eq!(stream.operation_count(), 1);

            // Invalid launch (too many threads)
            let result = stream
                .launch_kernel(
                    "test_kernel",
                    (32, 1, 1),
                    (32, 32, 2), // 2048 threads
                )
                .await;

            assert!(matches!(result, Err(CudaError::InvalidParameter { .. })));
        }
    }

    #[tokio::test]
    async fn test_copy_async() {
        #[cfg(cuda_mock)]
        {
            let stream = Stream::new("test_stream".to_string(), StreamFlags::default()).unwrap();

            // Valid copy
            assert!(stream.copy_async(0x1000, 0x2000, 1024,).await.is_ok());

            assert_eq!(stream.operation_count(), 1);

            // Invalid copy (size 0)
            let result = stream.copy_async(0x1000, 0x2000, 0).await;

            assert!(matches!(result, Err(CudaError::InvalidParameter { .. })));
        }
    }

    #[tokio::test]
    async fn test_stream_pool() {
        #[cfg(cuda_mock)]
        {
            let pool = StreamPool::new("test_pool".to_string(), 4).unwrap();
            assert_eq!(pool.size(), 4);

            // Get streams
            let stream1 = pool.get_stream();
            let stream2 = pool.get_stream();

            // Should be different streams
            assert_ne!(stream1.id, stream2.id);

            // Test round-robin
            let mut stream_names = Vec::new();
            for _ in 0..8 {
                let stream = pool.get_stream();
                stream_names.push(stream.name.clone());
            }

            // Should cycle through all 4 streams twice
            assert_eq!(stream_names[0], stream_names[4]);
            assert_eq!(stream_names[1], stream_names[5]);
        }
    }

    #[tokio::test]
    async fn test_stream_pool_synchronize_all() {
        #[cfg(cuda_mock)]
        {
            let pool = StreamPool::new("test_pool".to_string(), 4).unwrap();

            // Launch operations on all streams
            for _ in 0..4 {
                let stream = pool.get_stream();
                stream
                    .launch_kernel("test_kernel", (1, 1, 1), (1, 1, 1))
                    .await
                    .unwrap();
            }

            // Synchronize all
            assert!(pool.synchronize_all().await.is_ok());
        }
    }
}
