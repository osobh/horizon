//! Memory pool for efficient allocation patterns

use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::{GpuMemoryHandle, MemoryError, MemoryManager};

/// Memory pool for reusing allocations of the same size
///
/// Cache-line aligned (64 bytes) with mutex-guarded fields grouped
/// to reduce contention during concurrent acquire/release operations.
#[repr(C, align(64))]
pub struct MemoryPool {
    // Immutable configuration fields (rarely accessed after init)
    allocator: Arc<dyn MemoryManager>,
    max_blocks: usize,
    block_size: usize,
    // Padding to separate config from hot mutexes
    _config_padding: [u8; 40],
    // Hot mutex fields - second cache line
    pool: Arc<Mutex<VecDeque<GpuMemoryHandle>>>,
    total_allocated: Arc<Mutex<usize>>, // Track total blocks ever allocated
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(block_size: usize, max_blocks: usize, allocator: Arc<dyn MemoryManager>) -> Self {
        Self {
            // Immutable config fields
            allocator,
            max_blocks,
            block_size,
            _config_padding: [0; 40],
            // Hot mutex fields
            pool: Arc::new(Mutex::new(VecDeque::new())),
            total_allocated: Arc::new(Mutex::new(0)),
        }
    }

    /// Get a block from the pool or allocate new one
    pub async fn acquire(&self) -> Result<GpuMemoryHandle, MemoryError> {
        // Check if we have a handle in the pool first
        let existing_handle = {
            let mut pool = self.pool.lock().map_err(|e| MemoryError::PoolInitFailed {
                reason: format!("Failed to acquire pool lock: {e}"),
            })?;
            pool.pop_front()
        };

        if let Some(handle) = existing_handle {
            Ok(handle)
        } else {
            // Pool is empty, allocate new block
            let handle = self.allocator.allocate(self.block_size).await?;

            // Track that we allocated a new block
            let mut total =
                self.total_allocated
                    .lock()
                    .map_err(|e| MemoryError::PoolInitFailed {
                        reason: format!("Failed to acquire total_allocated lock: {e}"),
                    })?;
            *total += 1;

            Ok(handle)
        }
    }

    /// Return a block to the pool
    pub fn release(&self, handle: GpuMemoryHandle) -> Result<(), MemoryError> {
        if handle.size != self.block_size {
            return Err(MemoryError::InvalidSize { size: handle.size });
        }

        let mut pool = self.pool.lock().map_err(|e| MemoryError::PoolInitFailed {
            reason: format!("Failed to acquire pool lock: {e}"),
        })?;

        if pool.len() < self.max_blocks {
            pool.push_back(handle);
        }
        // If pool is full, just drop the handle (it will be deallocated)

        Ok(())
    }

    /// Get pool statistics
    pub fn stats(&self) -> Result<PoolStats, MemoryError> {
        let pool = self.pool.lock().map_err(|e| MemoryError::PoolInitFailed {
            reason: format!("Failed to acquire pool lock: {e}"),
        })?;

        let total_allocated =
            self.total_allocated
                .lock()
                .map_err(|e| MemoryError::PoolInitFailed {
                    reason: format!("Failed to acquire total_allocated lock: {e}"),
                })?;

        // Utilization = (blocks currently in use / total allocated) * 100
        // In use = total allocated - available in pool
        let blocks_in_use = *total_allocated - pool.len();
        let utilization_percent = if *total_allocated == 0 {
            0.0
        } else {
            (blocks_in_use as f32 / *total_allocated as f32) * 100.0
        };

        Ok(PoolStats {
            block_size: self.block_size,
            available_blocks: pool.len(),
            max_blocks: self.max_blocks,
            utilization_percent,
        })
    }
}

/// Memory pool statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PoolStats {
    pub block_size: usize,
    pub available_blocks: usize,
    pub max_blocks: usize,
    pub utilization_percent: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuMemoryAllocator;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_pool_creation() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));
        let pool = MemoryPool::new(1024, 10, allocator);

        let stats = pool.stats().expect("Failed to get stats");
        assert_eq!(stats.block_size, 1024);
        assert_eq!(stats.max_blocks, 10);
        assert_eq!(stats.available_blocks, 0);
        assert_eq!(stats.utilization_percent, 0.0);
    }

    #[tokio::test]
    async fn test_pool_acquire_new_block() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));
        let pool = MemoryPool::new(1024, 10, allocator);

        let handle = pool.acquire().await.expect("Failed to acquire block");
        assert_eq!(handle.size, 1024);
    }

    #[tokio::test]
    async fn test_pool_release_and_reuse() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));
        let pool = MemoryPool::new(1024, 10, allocator);

        // Acquire a block
        let handle = pool.acquire().await.expect("Failed to acquire block");
        let original_id = handle.id;

        // Release it back to pool
        pool.release(handle).expect("Failed to release block");

        // Check pool stats
        let stats = pool.stats().expect("Failed to get stats");
        assert_eq!(stats.available_blocks, 1);
        assert_eq!(stats.utilization_percent, 0.0); // 0 blocks in use out of 1 allocated

        // Acquire again - should get the same block
        let reused_handle = pool.acquire().await.expect("Failed to reacquire block");
        assert_eq!(reused_handle.id, original_id);
        assert_eq!(reused_handle.size, 1024);

        // Pool should now be empty again
        let stats = pool.stats().expect("Failed to get stats");
        assert_eq!(stats.available_blocks, 0);
    }

    #[tokio::test]
    async fn test_pool_release_wrong_size() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));
        let pool = MemoryPool::new(1024, 10, allocator.clone());

        // Get a block with different size
        let wrong_handle = allocator.allocate(2048).await.expect("Failed to allocate");

        // Try to release wrong size block
        let result = pool.release(wrong_handle);
        assert!(matches!(
            result,
            Err(MemoryError::InvalidSize { size: 2048 })
        ));
    }

    #[tokio::test]
    async fn test_pool_max_capacity() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));
        let pool = MemoryPool::new(1024, 2, allocator); // Max 2 blocks

        // Acquire and release 3 blocks
        let handle1 = pool.acquire().await.expect("Failed to acquire block 1");
        let handle2 = pool.acquire().await.expect("Failed to acquire block 2");
        let handle3 = pool.acquire().await.expect("Failed to acquire block 3");

        pool.release(handle1).expect("Failed to release block 1");
        pool.release(handle2).expect("Failed to release block 2");
        pool.release(handle3).expect("Failed to release block 3"); // Should be dropped

        // Pool should only have 2 blocks (max capacity)
        let stats = pool.stats().expect("Failed to get stats");
        assert_eq!(stats.available_blocks, 2);
        // 3 blocks allocated, 2 available, so 1 in use: 1/3 = 33.33%
        assert!((stats.utilization_percent - 33.333336).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_pool_stats_calculation() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));
        let pool = MemoryPool::new(512, 5, allocator);

        let stats = pool.stats().expect("Failed to get stats");
        assert_eq!(stats.block_size, 512);
        assert_eq!(stats.max_blocks, 5);
        assert_eq!(stats.available_blocks, 0);
        assert_eq!(stats.utilization_percent, 0.0);
    }

    #[tokio::test]
    async fn test_mutex_poisoning_acquire() {
        use crate::test_helpers::tests::PoisonedMemoryPool;

        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));

        let poisoned_pool = PoisonedMemoryPool::new(1024, 10);
        let pool = MemoryPool {
            block_size: poisoned_pool.block_size,
            pool: poisoned_pool.pool,
            max_blocks: poisoned_pool.max_blocks,
            allocator,
            total_allocated: Arc::new(Mutex::new(0)),
        };

        let result = pool.acquire().await;
        assert!(result.is_err());

        match result {
            Err(MemoryError::PoolInitFailed { reason }) => {
                assert!(
                    reason.contains("Failed to acquire pool lock")
                        || reason.contains("Failed to acquire total_allocated lock")
                );
            }
            _ => panic!("Expected PoolInitFailed error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_release() {
        use crate::test_helpers::tests::PoisonedMemoryPool;

        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));

        let poisoned_pool = PoisonedMemoryPool::new(1024, 10);
        let pool = MemoryPool {
            block_size: poisoned_pool.block_size,
            pool: poisoned_pool.pool,
            max_blocks: poisoned_pool.max_blocks,
            allocator,
            total_allocated: Arc::new(Mutex::new(0)),
        };

        let handle = GpuMemoryHandle {
            #[cfg(feature = "cuda")]
            ptr: unsafe { cust::memory::DevicePointer::from_raw(0u64) },
            #[cfg(not(feature = "cuda"))]
            ptr: 0usize,
            size: 1024,
            id: Uuid::new_v4(),
        };

        let result = pool.release(handle);
        assert!(result.is_err());

        match result {
            Err(MemoryError::PoolInitFailed { reason }) => {
                assert!(
                    reason.contains("Failed to acquire pool lock")
                        || reason.contains("Failed to acquire total_allocated lock")
                );
            }
            _ => panic!("Expected PoolInitFailed error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_stats() {
        use crate::test_helpers::tests::PoisonedMemoryPool;

        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));

        let poisoned_pool = PoisonedMemoryPool::new(1024, 10);
        let pool = MemoryPool {
            block_size: poisoned_pool.block_size,
            pool: poisoned_pool.pool,
            max_blocks: poisoned_pool.max_blocks,
            allocator,
            total_allocated: Arc::new(Mutex::new(0)),
        };

        let result = pool.stats();
        assert!(result.is_err());

        match result {
            Err(MemoryError::PoolInitFailed { reason }) => {
                assert!(
                    reason.contains("Failed to acquire pool lock")
                        || reason.contains("Failed to acquire total_allocated lock")
                );
            }
            _ => panic!("Expected PoolInitFailed error with lock failure"),
        }
    }

    #[test]
    fn test_mutex_poisoning_stats_total_allocated() {
        use crate::test_helpers::tests::create_poisoned_mutex;

        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));

        let pool = MemoryPool {
            block_size: 1024,
            pool: Arc::new(Mutex::new(VecDeque::new())),
            max_blocks: 10,
            allocator,
            total_allocated: create_poisoned_mutex(),
        };

        let result = pool.stats();
        assert!(result.is_err());

        match result {
            Err(MemoryError::PoolInitFailed { reason }) => {
                assert!(reason.contains("Failed to acquire total_allocated lock"));
            }
            _ => panic!("Expected PoolInitFailed error with total_allocated lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_total_allocated() {
        use crate::test_helpers::tests::create_poisoned_mutex;

        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));

        let pool = MemoryPool {
            block_size: 1024,
            pool: Arc::new(Mutex::new(VecDeque::new())),
            max_blocks: 10,
            allocator,
            total_allocated: create_poisoned_mutex(),
        };

        let result = pool.acquire().await;
        assert!(result.is_err());

        match result {
            Err(MemoryError::PoolInitFailed { reason }) => {
                assert!(
                    reason.contains("Failed to acquire pool lock")
                        || reason.contains("Failed to acquire total_allocated lock")
                );
            }
            _ => panic!("Expected PoolInitFailed error with lock failure"),
        }
    }
}
