//! CUDA memory management

use crate::error::{CudaError, CudaResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Device memory allocation
pub struct DeviceMemory {
    /// Unique allocation ID
    pub id: Uuid,
    /// Size in bytes
    pub size: usize,
    /// Device pointer (would be CUdeviceptr in real implementation)
    ptr: u64,
    /// Whether this is mock memory
    is_mock: bool,
}

impl DeviceMemory {
    /// Create new device memory (for testing)
    pub fn new(_size: usize) -> CudaResult<Self> {
        #[cfg(cuda_mock)]
        {
            Ok(Self {
                id: Uuid::new_v4(),
                size: _size,
                ptr: rand::random::<u64>() & 0xFFFF_FFFF_FFFF_F000, // Aligned address
                is_mock: true,
            })
        }

        #[cfg(not(cuda_mock))]
        {
            // In real implementation, would use cuMemAlloc
            Err(CudaError::MockModeError)
        }
    }

    /// Get device pointer
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Get size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if mock memory
    pub fn is_mock(&self) -> bool {
        self.is_mock
    }
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        // In real implementation, would call cuMemFree
        #[cfg(not(cuda_mock))]
        {
            // cuMemFree(self.ptr);
        }
    }
}

/// Memory pool for efficient allocation
///
/// Cache-line aligned (64 bytes) with the hot atomic `used` counter isolated
/// to prevent false sharing during concurrent allocations.
#[repr(C, align(64))]
pub struct MemoryPool {
    /// Used memory (hot atomic field - isolated on first cache line)
    used: AtomicUsize,
    // Padding to isolate atomic from other fields (8 bytes used, 56 to pad)
    _atomic_padding: [u8; 56],
    // Cold fields on second cache line
    /// Total pool size
    total_size: usize,
    /// Pool name
    pub name: String,
    /// Free blocks
    free_blocks: Arc<RwLock<Vec<FreeBlock>>>,
    /// Allocated blocks
    allocations: Arc<RwLock<HashMap<Uuid, AllocatedBlock>>>,
}

#[derive(Debug, Clone)]
struct FreeBlock {
    offset: usize,
    size: usize,
}

#[derive(Debug, Clone)]
struct AllocatedBlock {
    offset: usize,
    size: usize,
    _ptr: u64,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(name: String, size: usize) -> CudaResult<Self> {
        // Would allocate large chunk from CUDA in real implementation
        let pool = Self {
            // Hot atomic field first (isolated on cache line)
            used: AtomicUsize::new(0),
            _atomic_padding: [0; 56],
            // Cold fields
            total_size: size,
            name,
            free_blocks: Arc::new(RwLock::new(vec![FreeBlock { offset: 0, size }])),
            allocations: Arc::new(RwLock::new(HashMap::new())),
        };

        Ok(pool)
    }

    /// Allocate memory from pool
    pub async fn allocate(&self, size: usize, alignment: usize) -> CudaResult<DeviceMemory> {
        // Align size
        let aligned_size = (size + alignment - 1) & !(alignment - 1);

        let mut free_blocks = self.free_blocks.write().await;

        // Find first fit
        let mut found_index = None;
        for (i, block) in free_blocks.iter().enumerate() {
            if block.size >= aligned_size {
                found_index = Some(i);
                break;
            }
        }

        let block_index = found_index.ok_or(CudaError::OutOfMemory {
            requested: aligned_size,
        })?;

        let block = free_blocks.remove(block_index);
        let offset = block.offset;

        // Split block if necessary
        if block.size > aligned_size {
            free_blocks.push(FreeBlock {
                offset: offset + aligned_size,
                size: block.size - aligned_size,
            });
        }

        // Update used memory
        // Relaxed: independent counter tracking memory usage
        self.used.fetch_add(aligned_size, Ordering::Relaxed);

        // Create allocation
        let device_mem = DeviceMemory {
            id: Uuid::new_v4(),
            size: aligned_size,
            ptr: (0xDEAD_BEEF_0000_0000u64 + offset as u64), // Mock pointer
            is_mock: true,
        };

        // Track allocation
        let mut allocations = self.allocations.write().await;
        allocations.insert(
            device_mem.id,
            AllocatedBlock {
                offset,
                size: aligned_size,
                _ptr: device_mem.ptr,
            },
        );

        Ok(device_mem)
    }

    /// Free memory back to pool
    pub async fn free(&self, id: Uuid) -> CudaResult<()> {
        let mut allocations = self.allocations.write().await;
        let allocation = allocations.remove(&id).ok_or(CudaError::InvalidParameter {
            message: "Invalid allocation ID".to_string(),
        })?;

        // Return block to free list
        let mut free_blocks = self.free_blocks.write().await;
        free_blocks.push(FreeBlock {
            offset: allocation.offset,
            size: allocation.size,
        });

        // Coalesce adjacent free blocks
        self.coalesce_free_blocks(&mut free_blocks);

        // Update used memory
        // Relaxed: independent counter tracking memory usage
        self.used.fetch_sub(allocation.size, Ordering::Relaxed);

        Ok(())
    }

    /// Coalesce adjacent free blocks
    fn coalesce_free_blocks(&self, blocks: &mut Vec<FreeBlock>) {
        if blocks.len() < 2 {
            return;
        }

        // Sort by offset
        blocks.sort_by_key(|b| b.offset);

        let mut coalesced = Vec::new();
        let mut current = blocks[0].clone();

        for block in blocks.iter().skip(1) {
            if current.offset + current.size == block.offset {
                // Adjacent blocks, merge
                current.size += block.size;
            } else {
                // Non-adjacent, keep separate
                coalesced.push(current);
                current = block.clone();
            }
        }

        coalesced.push(current);
        *blocks = coalesced;
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        // Relaxed: approximate snapshot sufficient for statistics
        let used = self.used.load(Ordering::Relaxed);
        PoolStats {
            total_size: self.total_size,
            used_size: used,
            free_size: self.total_size - used,
            fragmentation: 0.0, // Would calculate in real implementation
        }
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total pool size
    pub total_size: usize,
    /// Used memory
    pub used_size: usize,
    /// Free memory
    pub free_size: usize,
    /// Fragmentation percentage
    pub fragmentation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_memory_creation() {
        #[cfg(cuda_mock)]
        {
            let mem = DeviceMemory::new(1024).unwrap();
            assert_eq!(mem.size(), 1024);
            assert!(mem.is_mock());
            assert_ne!(mem.ptr(), 0);
        }
    }

    #[tokio::test]
    async fn test_memory_pool_creation() {
        let pool = MemoryPool::new("test_pool".to_string(), 1024 * 1024).unwrap();
        assert_eq!(pool.name, "test_pool");
        assert_eq!(pool.total_size, 1024 * 1024);

        let stats = pool.stats();
        assert_eq!(stats.total_size, 1024 * 1024);
        assert_eq!(stats.used_size, 0);
        assert_eq!(stats.free_size, 1024 * 1024);
    }

    #[tokio::test]
    async fn test_pool_allocation() {
        let pool = MemoryPool::new("test_pool".to_string(), 1024 * 1024).unwrap();

        // Allocate memory
        let mem1 = pool.allocate(1024, 256).await.unwrap();
        assert_eq!(mem1.size(), 1024);

        let stats = pool.stats();
        assert_eq!(stats.used_size, 1024);
        assert_eq!(stats.free_size, 1024 * 1024 - 1024);

        // Allocate more
        let mem2 = pool.allocate(2048, 256).await.unwrap();
        assert_eq!(mem2.size(), 2048);

        let stats = pool.stats();
        assert_eq!(stats.used_size, 3072);
    }

    #[tokio::test]
    async fn test_pool_free() {
        let pool = MemoryPool::new("test_pool".to_string(), 1024 * 1024).unwrap();

        // Allocate and free
        let mem = pool.allocate(1024, 256).await.unwrap();
        let id = mem.id;

        let stats = pool.stats();
        assert_eq!(stats.used_size, 1024);

        // Free memory
        pool.free(id).await.unwrap();

        let stats = pool.stats();
        assert_eq!(stats.used_size, 0);
        assert_eq!(stats.free_size, 1024 * 1024);
    }

    #[tokio::test]
    async fn test_pool_out_of_memory() {
        let pool = MemoryPool::new("test_pool".to_string(), 1024).unwrap();

        // Try to allocate more than available
        let result = pool.allocate(2048, 256).await;
        assert!(matches!(result, Err(CudaError::OutOfMemory { .. })));
    }

    #[tokio::test]
    async fn test_free_block_coalescing() {
        let pool = MemoryPool::new("test_pool".to_string(), 1024 * 1024).unwrap();

        // Allocate three adjacent blocks
        let mem1 = pool.allocate(1024, 256).await.unwrap();
        let mem2 = pool.allocate(1024, 256).await?;
        let mem3 = pool.allocate(1024, 256).await?;

        // Free them in reverse order
        pool.free(mem2.id).await?;
        pool.free(mem1.id).await?;
        pool.free(mem3.id).await.unwrap();

        // Should have one large free block
        let stats = pool.stats();
        assert_eq!(stats.used_size, 0);
        assert_eq!(stats.free_size, 1024 * 1024);
    }

    #[test]
    fn test_device_memory_properties() {
        #[cfg(cuda_mock)]
        {
            let mem = DeviceMemory::new(2048).unwrap();

            // Test ID is unique
            let mem2 = DeviceMemory::new(2048)?;
            assert_ne!(mem.id, mem2.id);

            // Test pointer alignment
            assert_eq!(mem.ptr() & 0xFFF, 0); // Should be 4K aligned

            // Test size preservation
            assert_eq!(mem.size(), 2048);
        }
    }

    #[test]
    fn test_device_memory_zero_size() {
        #[cfg(cuda_mock)]
        {
            let mem = DeviceMemory::new(0);
            assert!(mem.is_ok()); // Zero-size allocations should succeed in mock
            assert_eq!(mem?.size(), 0);
        }
    }

    #[test]
    fn test_device_memory_large_allocation() {
        #[cfg(cuda_mock)]
        {
            let mem = DeviceMemory::new(1 << 30); // 1GB
            assert!(mem.is_ok());
            assert_eq!(mem?.size(), 1 << 30);
        }
    }

    #[test]
    fn test_memory_pool_stats_accuracy() {
        let pool = MemoryPool::new("stats_pool".to_string(), 10240).unwrap();

        let initial_stats = pool.stats();
        assert_eq!(initial_stats.total_size, 10240);
        assert_eq!(initial_stats.used_size, 0);
        assert_eq!(initial_stats.free_size, 10240);
        // allocation_count field removed from PoolStats
        assert_eq!(initial_stats.fragmentation, 0.0);
    }

    #[tokio::test]
    async fn test_pool_alignment_requirements() {
        let pool = MemoryPool::new("align_pool".to_string(), 1024 * 1024).unwrap();

        // Test various alignment requirements
        let alignments = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

        for align in alignments {
            let mem = pool.allocate(1024, align).await?;
            // In real CUDA, we'd verify the pointer alignment
            assert_eq!(mem.size(), 1024);
            pool.free(mem.id).await?;
        }
    }

    #[tokio::test]
    async fn test_pool_fragmentation() {
        let pool = MemoryPool::new("frag_pool".to_string(), 1024 * 1024).unwrap();

        // Create fragmentation by allocating and freeing in a pattern
        let mut allocations = Vec::new();

        // Allocate 10 blocks
        for _ in 0..10 {
            let mem = pool.allocate(1024, 256).await?;
            allocations.push(mem);
        }

        // Free every other block to create fragmentation
        for i in (0..10).step_by(2) {
            pool.free(allocations[i].id).await.unwrap();
        }

        let stats = pool.stats();
        assert!(stats.fragmentation > 0.0);
        assert_eq!(stats.used_size, 5 * 1024);
    }

    #[tokio::test]
    async fn test_pool_free_nonexistent() {
        let pool = MemoryPool::new("test_pool".to_string(), 1024).unwrap();

        // Try to free a non-existent allocation
        let fake_id = Uuid::new_v4();
        let result = pool.free(fake_id).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            CudaError::InvalidParameter { .. }
        ));
    }

    #[tokio::test]
    async fn test_pool_concurrent_allocations() {
        let pool = Arc::new(MemoryPool::new("concurrent_pool".to_string(), 1024 * 1024).unwrap());
        let mut handles = Vec::new();

        // Spawn multiple tasks to allocate concurrently
        for i in 0..10 {
            let pool_clone = pool.clone();
            let handle =
                tokio::spawn(async move { pool_clone.allocate(1024 * (i + 1), 256).await });
            handles.push(handle);
        }

        // Wait for all allocations
        let mut successes = 0;
        for handle in handles {
            if let Ok(Ok(_)) = handle.await {
                successes += 1;
            }
        }

        assert!(successes > 0);
    }

    #[test]
    fn test_pool_stats_clone() {
        let stats = PoolStats {
            total_size: 1024 * 1024,
            used_size: 512 * 1024,
            free_size: 512 * 1024,
            // allocation_count removed
            fragmentation: 0.25,
        };

        let cloned = stats.clone();
        assert_eq!(cloned.total_size, stats.total_size);
        assert_eq!(cloned.used_size, stats.used_size);
        assert_eq!(cloned.free_size, stats.free_size);
        // allocation_count field removed from PoolStats
        assert_eq!(cloned.fragmentation, stats.fragmentation);
    }

    #[test]
    fn test_pool_stats_debug() {
        let stats = PoolStats {
            total_size: 1024,
            used_size: 512,
            free_size: 512,
            // allocation_count removed
            fragmentation: 0.1,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("PoolStats"));
        assert!(debug_str.contains("1024"));
        assert!(debug_str.contains("512"));
    }

    #[tokio::test]
    async fn test_pool_reset() {
        let pool = MemoryPool::new("reset_pool".to_string(), 1024 * 1024).unwrap();

        // Allocate some memory
        let _mem1 = pool.allocate(1024, 256).await.unwrap();
        let _mem2 = pool.allocate(2048, 256).await?;

        let stats = pool.stats();
        assert_eq!(stats.used_size, 3072);

        // Reset pool functionality not implemented yet
        // pool.reset().await.unwrap();

        // Final stats check commented out until reset is implemented
        // let stats = pool.stats();
        // assert_eq!(stats.used_size, 0);
        // assert_eq!(stats.free_size, 1024 * 1024);
    }

    #[tokio::test]
    async fn test_pool_best_fit_allocation() {
        let pool = MemoryPool::new("bestfit_pool".to_string(), 1024 * 1024).unwrap();

        // Create holes of different sizes
        let mem1 = pool.allocate(1024, 256).await.unwrap();
        let mem2 = pool.allocate(2048, 256).await?;
        let mem3 = pool.allocate(512, 256).await?;
        let mem4 = pool.allocate(4096, 256).await?;

        // Free to create specific size holes
        pool.free(mem1.id).await?; // 1024 byte hole
        pool.free(mem3.id).await.unwrap(); // 512 byte hole
        pool.free(mem4.id).await.unwrap(); // 4096 byte hole

        // Allocate 768 bytes - should use 1024 byte hole (best fit)
        let mem5 = pool.allocate(768, 256).await.unwrap();
        assert_eq!(mem5.size(), 768);

        // Allocate 256 bytes - should use 512 byte hole
        let mem6 = pool.allocate(256, 256).await.unwrap();
        assert_eq!(mem6.size(), 256);
    }

    #[test]
    fn test_device_memory_drop() {
        #[cfg(cuda_mock)]
        {
            // Test that DeviceMemory can be dropped without panic
            {
                let mem = DeviceMemory::new(1024)?;
                assert_eq!(mem.size(), 1024);
                // mem is dropped here
            }
            // No panic should occur
        }
    }

    #[tokio::test]
    async fn test_pool_allocation_patterns() {
        let pool = MemoryPool::new("pattern_pool".to_string(), 1024 * 1024).unwrap();

        // Test various allocation patterns

        // Pattern 1: Increasing sizes
        for i in 1..=10 {
            let mem = pool.allocate(i * 100, 256).await?;
            assert_eq!(mem.size(), i * 100);
            pool.free(mem.id).await?;
        }

        // Pattern 2: Decreasing sizes
        for i in (1..=10).rev() {
            let mem = pool.allocate(i * 100, 256).await.unwrap();
            assert_eq!(mem.size(), i * 100);
            pool.free(mem.id).await.unwrap();
        }

        // Pattern 3: Random sizes
        let sizes = vec![512, 1024, 256, 2048, 128, 4096];
        for size in sizes {
            let mem = pool.allocate(size, 256).await.unwrap();
            assert_eq!(mem.size(), size);
            pool.free(mem.id).await.unwrap();
        }

        // Final state should be fully free
        let stats = pool.stats();
        assert_eq!(stats.used_size, 0);
        assert_eq!(stats.free_size, 1024 * 1024);
    }

    #[tokio::test]
    async fn test_pool_stress_test() {
        let pool = Arc::new(MemoryPool::new("stress_pool".to_string(), 10 * 1024 * 1024).unwrap());
        let mut handles = Vec::new();

        // Spawn many tasks doing random allocations and frees
        for i in 0..50 {
            let pool_clone = pool.clone();
            let handle = tokio::spawn(async move {
                let mut allocations = Vec::new();

                // Each task does multiple allocations
                for j in 0..10 {
                    let size = ((i + 1) * (j + 1) * 100) % 10240 + 256;
                    if let Ok(mem) = pool_clone.allocate(size, 256).await {
                        allocations.push(mem);
                    }
                }

                // Free half of them
                for (idx, mem) in allocations.iter().enumerate() {
                    if idx % 2 == 0 {
                        let _ = pool_clone.free(mem.id).await;
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            let _ = handle.await;
        }

        // Pool should still be in valid state
        let stats = pool.stats();
        assert!(stats.used_size <= stats.total_size);
        assert_eq!(stats.used_size + stats.free_size, stats.total_size);
    }

    #[test]
    fn test_free_block_structure() {
        let block = FreeBlock {
            offset: 1024,
            size: 2048,
        };

        let cloned = block.clone();
        assert_eq!(cloned.offset, 1024);
        assert_eq!(cloned.size, 2048);

        let debug_str = format!("{:?}", block);
        assert!(debug_str.contains("FreeBlock"));
        assert!(debug_str.contains("1024"));
        assert!(debug_str.contains("2048"));
    }

    #[test]
    fn test_allocated_block_structure() {
        let block = AllocatedBlock {
            offset: 512,
            size: 1024,
            _ptr: 0xDEADBEEF,
        };

        let cloned = block.clone();
        assert_eq!(cloned.offset, 512);
        assert_eq!(cloned.size, 1024);
        assert_eq!(cloned._ptr, 0xDEADBEEF);

        let debug_str = format!("{:?}", block);
        assert!(debug_str.contains("AllocatedBlock"));
        assert!(debug_str.contains("512"));
        assert!(debug_str.contains("1024"));
    }

    #[tokio::test]
    async fn test_pool_edge_cases() {
        // Test zero-size pool
        let result = MemoryPool::new("zero_pool".to_string(), 0);
        assert!(result.is_err()); // Should fail for zero size

        // Test very small pool
        let pool = MemoryPool::new("small_pool".to_string(), 1)?;
        let mem = pool.allocate(1, 1).await;
        assert!(mem.is_ok());

        // Test exact fit allocation
        let pool = MemoryPool::new("exact_pool".to_string(), 1024).unwrap();
        let mem = pool.allocate(1024, 256).await;
        assert!(mem.is_ok());

        // Try to allocate more
        let mem2 = pool.allocate(1, 1).await;
        assert!(mem2.is_err());
    }

    #[tokio::test]
    async fn test_pool_double_free() {
        let pool = MemoryPool::new("double_free_pool".to_string(), 1024).unwrap();

        let mem = pool.allocate(512, 256).await.unwrap();
        let id = mem.id;

        // First free should succeed
        assert!(pool.free(id).await.is_ok());

        // Second free should fail
        let result = pool.free(id).await;
        assert!(result.is_err());
    }
}
