//! CUDA context management for device isolation

use crate::error::{CudaError, CudaResult};
use bitflags::bitflags;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

bitflags! {
    /// Context creation flags
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ContextFlags: u32 {
        /// Enable automatic scheduling
        const SCHED_AUTO = 0x00;
        /// Spin CPU while waiting for GPU
        const SCHED_SPIN = 0x01;
        /// Yield CPU while waiting
        const SCHED_YIELD = 0x02;
        /// Block CPU while waiting
        const SCHED_BLOCKING_SYNC = 0x04;
        /// Support mapped pinned allocations
        const MAP_HOST = 0x08;
        /// Keep local memory allocation after launch
        const LMEM_RESIZE_TO_MAX = 0x10;
    }
}

impl Default for ContextFlags {
    fn default() -> Self {
        Self::SCHED_AUTO
    }
}

/// GPU context properties
#[derive(Debug, Clone, PartialEq)]
pub struct ContextProperties {
    /// Total GPU memory in bytes
    pub total_memory: usize,
    /// Compute capability (major, minor)
    pub compute_capability: (u32, u32),
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Maximum block dimensions [x, y, z]
    pub max_block_dims: [u32; 3],
    /// Maximum grid dimensions [x, y, z]
    pub max_grid_dims: [u32; 3],
    /// Warp size
    pub warp_size: u32,
    /// Maximum shared memory per block
    pub max_shared_memory_per_block: u32,
    /// GPU clock rate in kHz
    pub clock_rate: u32,
}

/// Cache configuration options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheConfig {
    /// Prefer shared memory over L1 cache
    PreferShared,
    /// Prefer L1 cache over shared memory
    PreferCache,
    /// Equal split between shared memory and L1 cache
    PreferEqual,
    /// No preference
    PreferNone,
}

/// Shared memory bank size configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedMemConfig {
    /// Default bank size
    DefaultBankSize,
    /// 4-byte bank size
    FourByteBankSize,
    /// 8-byte bank size
    EightByteBankSize,
}

/// Resource limit types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceLimit {
    /// Stack size limit
    StackSize,
    /// Printf FIFO size
    PrintfFifoSize,
    /// Malloc heap size
    MallocHeapSize,
}

/// L2 cache configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2CacheConfig {
    /// Reserved size in bytes
    pub reserved_size: usize,
}

/// CUDA context for device execution
pub struct Context {
    /// Unique context ID
    pub id: Uuid,
    /// Device ID this context is bound to
    pub device_id: i32,
    /// Context flags
    pub flags: ContextFlags,
    /// Context handle (would be CUcontext in real implementation)
    handle: u64,
    /// Whether this is a mock context
    is_mock: bool,
    /// Whether context is current
    is_current: Arc<RwLock<bool>>,
}

impl Context {
    /// Create a new context for a device
    pub fn new(device_id: i32, _flags: ContextFlags) -> CudaResult<Self> {
        // Validate device ID
        if device_id < 0 {
            return Err(CudaError::InvalidDevice { device: device_id });
        }

        #[cfg(cuda_mock)]
        {
            // In mock mode, accept any device ID up to 999 for error testing
            if device_id > 999 {
                return Err(CudaError::InvalidDevice { device: device_id });
            }

            // But only device IDs 0-7 are valid for normal operation
            if device_id > 7 && device_id != 999 {
                return Err(CudaError::InvalidDevice { device: device_id });
            }

            Ok(Self {
                id: Uuid::new_v4(),
                device_id,
                flags: _flags,
                handle: (device_id as u64) << 32 | rand::random::<u32>() as u64,
                is_mock: true,
                is_current: Arc::new(RwLock::new(false)),
            })
        }

        #[cfg(not(cuda_mock))]
        {
            // In real implementation, would use cuCtxCreate
            Err(CudaError::MockModeError)
        }
    }

    /// Make this context current on the calling thread
    pub async fn set_current(&self) -> CudaResult<()> {
        #[cfg(cuda_mock)]
        {
            let mut is_current = self.is_current.write().await;
            *is_current = true;
            Ok(())
        }

        #[cfg(not(cuda_mock))]
        {
            // In real implementation, would use cuCtxSetCurrent
            Err(CudaError::MockModeError)
        }
    }

    /// Push context on current thread stack
    pub async fn push(&self) -> CudaResult<()> {
        self.set_current().await
    }

    /// Pop context from current thread stack
    pub async fn pop(&self) -> CudaResult<()> {
        #[cfg(cuda_mock)]
        {
            let mut is_current = self.is_current.write().await;
            *is_current = false;
            Ok(())
        }

        #[cfg(not(cuda_mock))]
        {
            // In real implementation, would use cuCtxPopCurrent
            Err(CudaError::MockModeError)
        }
    }

    /// Synchronize context (wait for all operations) - async version
    #[cfg(not(cuda_mock))]
    pub async fn synchronize_async(&self) -> CudaResult<()> {
        // In real implementation, would use cuCtxSynchronize
        Err(CudaError::MockModeError)
    }

    /// Get free and total memory for this context - async version  
    #[cfg(not(cuda_mock))]
    pub async fn get_memory_info_async(&self) -> CudaResult<(usize, usize)> {
        // In real implementation, would use cuMemGetInfo
        Err(CudaError::MockModeError)
    }

    /// Get context handle
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Check if context is current - async version
    #[cfg(not(cuda_mock))]
    pub async fn is_current_async(&self) -> bool {
        *self.is_current.read().await
    }

    /// Check if mock context
    pub fn is_mock(&self) -> bool {
        self.is_mock
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Synchronous make current (for tests)
    #[cfg(cuda_mock)]
    pub fn make_current(&self) -> CudaResult<()> {
        // Mock implementation - would use cuCtxSetCurrent in real code
        Ok(())
    }

    /// Synchronize context (wait for all operations)
    #[cfg(cuda_mock)]
    pub fn synchronize(&self) -> CudaResult<()> {
        // Mock implementation - would use cuCtxSynchronize in real code
        Ok(())
    }

    /// Check if context is current
    #[cfg(cuda_mock)]
    pub fn is_current(&self) -> bool {
        // Mock implementation - in reality would check thread-local context
        true
    }

    /// Get context properties
    #[cfg(cuda_mock)]
    pub fn get_properties(&self) -> CudaResult<ContextProperties> {
        Ok(ContextProperties {
            total_memory: match self.device_id {
                0 => 24 * 1024 * 1024 * 1024, // 24GB
                1 => 16 * 1024 * 1024 * 1024, // 16GB
                2 => 12 * 1024 * 1024 * 1024, // 12GB
                3 => 8 * 1024 * 1024 * 1024,  // 8GB
                _ => 4 * 1024 * 1024 * 1024,  // 4GB
            },
            compute_capability: (8, 6),
            max_threads_per_block: 1024,
            max_block_dims: [1024, 1024, 64],
            max_grid_dims: [2147483647, 65535, 65535],
            warp_size: 32,
            max_shared_memory_per_block: 49152,
            clock_rate: 1770000,
        })
    }

    /// Set cache configuration
    #[cfg(cuda_mock)]
    pub fn set_cache_config(&self, _config: CacheConfig) -> CudaResult<()> {
        Ok(())
    }

    /// Set shared memory configuration
    #[cfg(cuda_mock)]
    pub fn set_shared_mem_config(&self, _config: SharedMemConfig) -> CudaResult<()> {
        Ok(())
    }

    /// Get stream priority range
    #[cfg(cuda_mock)]
    pub fn get_stream_priority_range(&self) -> CudaResult<(i32, i32)> {
        Ok((-1, 0)) // Lower values = higher priority
    }

    /// Get API version
    #[cfg(cuda_mock)]
    pub fn get_api_version(&self) -> CudaResult<u32> {
        Ok(12000) // CUDA 12.0
    }

    /// Check if can map host memory
    #[cfg(cuda_mock)]
    pub fn can_map_host_memory(&self) -> CudaResult<bool> {
        Ok(true)
    }

    /// Check if supports managed memory
    #[cfg(cuda_mock)]
    pub fn supports_managed_memory(&self) -> CudaResult<bool> {
        Ok(true)
    }

    /// Check if supports cooperative launch
    #[cfg(cuda_mock)]
    pub fn supports_cooperative_launch(&self) -> CudaResult<bool> {
        Ok(true)
    }

    /// Get memory info
    #[cfg(cuda_mock)]
    pub fn get_memory_info(&self) -> CudaResult<(usize, usize)> {
        let total = match self.device_id {
            0 => 24 * 1024 * 1024 * 1024, // 24GB
            1 => 16 * 1024 * 1024 * 1024, // 16GB
            2 => 12 * 1024 * 1024 * 1024, // 12GB
            3 => 8 * 1024 * 1024 * 1024,  // 8GB
            _ => 4 * 1024 * 1024 * 1024,  // 4GB
        };
        let free = total * 9 / 10; // 90% free
        Ok((free, total))
    }

    /// Get resource limit
    #[cfg(cuda_mock)]
    pub fn get_limit(&self, limit: ResourceLimit) -> CudaResult<usize> {
        match limit {
            ResourceLimit::StackSize => Ok(1024),
            ResourceLimit::PrintfFifoSize => Ok(1048576),
            ResourceLimit::MallocHeapSize => Ok(8 * 1024 * 1024),
        }
    }

    /// Set resource limit
    #[cfg(cuda_mock)]
    pub fn set_limit(&self, _limit: ResourceLimit, _value: usize) -> CudaResult<()> {
        Ok(())
    }

    /// Check if can access peer device
    #[cfg(cuda_mock)]
    pub fn can_access_peer(&self, _peer_device: i32) -> CudaResult<bool> {
        Ok(true)
    }

    /// Enable peer access
    #[cfg(cuda_mock)]
    pub fn enable_peer_access(&self, _peer_device: i32) -> CudaResult<()> {
        Ok(())
    }

    /// Disable peer access
    #[cfg(cuda_mock)]
    pub fn disable_peer_access(&self, _peer_device: i32) -> CudaResult<()> {
        Ok(())
    }

    /// Reset context
    #[cfg(cuda_mock)]
    pub fn reset(&self) -> CudaResult<()> {
        Ok(())
    }

    /// Start profiler
    #[cfg(cuda_mock)]
    pub fn profiler_start(&self) -> CudaResult<()> {
        Ok(())
    }

    /// Stop profiler
    #[cfg(cuda_mock)]
    pub fn profiler_stop(&self) -> CudaResult<()> {
        Ok(())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // In real implementation, would call cuCtxDestroy
        #[cfg(not(cuda_mock))]
        {
            // cuCtxDestroy(self.handle);
        }
    }
}

/// Context guard for automatic push/pop
pub struct ContextGuard {
    context: Arc<Context>,
}

impl ContextGuard {
    /// Create a new context guard
    pub async fn new(context: Arc<Context>) -> CudaResult<Self> {
        context.push().await?;
        Ok(Self { context })
    }
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        // Note: Can't await in drop, so we spawn a task
        let context = self.context.clone();
        tokio::spawn(async move {
            let _ = context.pop().await;
        });
    }
}

/// Primary context management
pub struct PrimaryContext {
    device_id: i32,
    context: Arc<RwLock<Option<Arc<Context>>>>,
}

impl PrimaryContext {
    /// Create primary context manager for device
    pub fn new(device_id: i32) -> Self {
        Self {
            device_id,
            context: Arc::new(RwLock::new(None)),
        }
    }

    /// Get or create primary context
    pub async fn get(&self) -> CudaResult<Arc<Context>> {
        let mut ctx = self.context.write().await;

        if let Some(context) = ctx.as_ref() {
            Ok(context.clone())
        } else {
            let context = Arc::new(Context::new(self.device_id, ContextFlags::default())?);
            *ctx = Some(context.clone());
            Ok(context)
        }
    }

    /// Reset primary context
    pub async fn reset(&self) -> CudaResult<()> {
        let mut ctx = self.context.write().await;
        *ctx = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_flags_default() {
        let flags = ContextFlags::default();
        assert!(flags.contains(ContextFlags::SCHED_AUTO));
        assert!(!flags.contains(ContextFlags::SCHED_SPIN));
        assert!(!flags.contains(ContextFlags::MAP_HOST));
    }

    #[tokio::test]
    async fn test_context_creation() {
        #[cfg(cuda_mock)]
        {
            let context = Context::new(0, ContextFlags::default()).unwrap();
            assert_eq!(context.device_id, 0);
            assert!(context.is_mock());
            assert!(!context.is_current().await);
        }
    }

    #[tokio::test]
    async fn test_invalid_device() {
        let result = Context::new(-1, ContextFlags::default());
        assert!(matches!(result, Err(CudaError::InvalidDevice { .. })));

        #[cfg(cuda_mock)]
        {
            let result = Context::new(8, ContextFlags::default());
            assert!(matches!(result, Err(CudaError::InvalidDevice { .. })));
        }
    }

    #[tokio::test]
    async fn test_context_current() {
        #[cfg(cuda_mock)]
        {
            let context = Context::new(0, ContextFlags::default()).unwrap();

            assert!(!context.is_current().await);

            context.set_current().await?;
            assert!(context.is_current().await);

            context.pop().await?;
            assert!(!context.is_current().await);
        }
    }

    #[tokio::test]
    async fn test_context_synchronize() {
        #[cfg(cuda_mock)]
        {
            let context = Context::new(0, ContextFlags::default()).unwrap();
            assert!(context.synchronize().await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_memory_info() {
        #[cfg(cuda_mock)]
        {
            let context = Context::new(0, ContextFlags::default()).unwrap();
            let (free, total) = context.get_memory_info().await.unwrap();

            assert_eq!(total, 24 * 1024 * 1024 * 1024);
            assert!(free < total);
            assert!(free > total / 2);
        }
    }

    #[tokio::test]
    async fn test_context_guard() {
        #[cfg(cuda_mock)]
        {
            let context = Arc::new(Context::new(0, ContextFlags::default()).unwrap());

            assert!(!context.is_current().await);

            {
                let _guard = ContextGuard::new(context.clone()).await?;
                assert!(context.is_current().await);
            }

            // Give time for the drop task to execute
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            assert!(!context.is_current().await);
        }
    }

    #[tokio::test]
    async fn test_primary_context() {
        #[cfg(cuda_mock)]
        {
            let primary = PrimaryContext::new(0);

            // First get creates context
            let ctx1 = primary.get().await?;
            assert_eq!(ctx1.device_id, 0);

            // Second get returns same context
            let ctx2 = primary.get().await?;
            assert_eq!(ctx1.id, ctx2.id);

            // Reset clears context
            primary.reset().await.unwrap();

            // Next get creates new context
            let ctx3 = primary.get().await.unwrap();
            assert_ne!(ctx1.id, ctx3.id);
        }
    }
}
