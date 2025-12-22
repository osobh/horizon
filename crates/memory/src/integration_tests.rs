//! Integration tests for memory crate components

#[cfg(test)]
mod tests {
    use crate::*;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::timeout;

    /// Test complete allocation workflow
    #[tokio::test]
    async fn test_complete_allocation_workflow() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));

        // Test allocation
        let handle = allocator.allocate(1024).await.expect("Failed to allocate");
        assert_eq!(handle.size, 1024);

        // Check stats after allocation
        let stats = allocator.stats().await.expect("Failed to get stats");
        assert_eq!(stats.used_bytes, 1024);
        assert_eq!(stats.available_bytes, 1024 * 1024 - 1024);
        assert_eq!(stats.allocated_blocks, 1);

        // Test deallocation
        allocator
            .deallocate(handle)
            .await
            .expect("Failed to deallocate");

        // Check stats after deallocation
        let stats = allocator
            .stats()
            .await
            .expect("Failed to get stats after deallocation");
        assert_eq!(stats.used_bytes, 0);
        assert_eq!(stats.available_bytes, 1024 * 1024);
        assert_eq!(stats.allocated_blocks, 0);
    }

    /// Test memory pool with allocator integration
    #[tokio::test]
    async fn test_pool_allocator_integration() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));
        let pool = MemoryPool::new(512, 5, allocator.clone());

        // Acquire multiple blocks
        let handle1 = pool.acquire().await.expect("Failed to acquire block 1");
        let handle2 = pool.acquire().await.expect("Failed to acquire block 2");

        assert_eq!(handle1.size, 512);
        assert_eq!(handle2.size, 512);

        // Check allocator stats
        let allocator_stats = allocator
            .stats()
            .await
            .expect("Failed to get allocator stats");
        assert_eq!(allocator_stats.used_bytes, 1024); // 2 * 512
        assert_eq!(allocator_stats.allocated_blocks, 2);

        // Release one block back to pool
        pool.release(handle1).expect("Failed to release block 1");

        // Pool should have 1 available block
        let pool_stats = pool.stats().expect("Failed to get pool stats");
        assert_eq!(pool_stats.available_blocks, 1);

        // Allocator should still show 2 blocks allocated (not deallocated)
        let allocator_stats = allocator
            .stats()
            .await
            .expect("Failed to get allocator stats");
        assert_eq!(allocator_stats.allocated_blocks, 2);

        // Acquire again - should reuse the pooled block
        let reused_handle = pool.acquire().await.expect("Failed to reacquire block");
        assert_eq!(reused_handle.size, 512);

        // Pool should be empty again
        let pool_stats = pool.stats().expect("Failed to get pool stats");
        assert_eq!(pool_stats.available_blocks, 0);
    }

    /// Test memory fragmentation scenarios
    #[tokio::test]
    async fn test_memory_fragmentation() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(4096).expect("Failed to create allocator"));

        // Allocate many small blocks to create fragmentation
        let mut handles = Vec::new();
        for _ in 0..8 {
            let handle = allocator.allocate(256).await.expect("Failed to allocate");
            handles.push(handle);
        }

        // Check stats
        let stats = allocator.stats().await.expect("Failed to get stats");
        assert_eq!(stats.used_bytes, 8 * 256);
        assert_eq!(stats.allocated_blocks, 8);
        assert!(stats.fragmentation_percent > 0.0);

        // Deallocate every other block to create holes
        for i in (0..handles.len()).step_by(2) {
            allocator
                .deallocate(handles.remove(i))
                .await
                .expect("Failed to deallocate");
        }

        // Check fragmentation after partial deallocation
        let stats = allocator
            .stats()
            .await
            .expect("Failed to get fragmented stats");
        assert_eq!(stats.allocated_blocks, 4);
        assert!(stats.used_bytes < 8 * 256);
    }

    /// Test concurrent allocation and deallocation
    #[tokio::test]
    async fn test_concurrent_memory_operations() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));

        let mut handles = Vec::new();
        let mut tasks = Vec::new();

        // Spawn multiple tasks that allocate concurrently
        for i in 0..10 {
            let allocator_clone = allocator.clone();
            let task = tokio::spawn(async move {
                let size = (i + 1) * 1024; // Variable sizes
                timeout(Duration::from_secs(5), allocator_clone.allocate(size))
                    .await
                    .expect("Allocation timed out")
                    .expect("Failed to allocate")
            });
            tasks.push(task);
        }

        // Collect all handles
        for task in tasks {
            let handle = task.await.expect("Task failed");
            handles.push(handle);
        }

        assert_eq!(handles.len(), 10);

        // Check total allocation
        let stats = allocator.stats().await.expect("Failed to get stats");
        let expected_total: usize = (1..=10).map(|i| i * 1024).sum();
        assert_eq!(stats.used_bytes, expected_total);
        assert_eq!(stats.allocated_blocks, 10);

        // Deallocate concurrently
        let mut dealloc_tasks = Vec::new();
        for handle in handles {
            let allocator_clone = allocator.clone();
            let task = tokio::spawn(async move {
                timeout(Duration::from_secs(5), allocator_clone.deallocate(handle))
                    .await
                    .expect("Deallocation timed out")
                    .expect("Failed to deallocate")
            });
            dealloc_tasks.push(task);
        }

        // Wait for all deallocations
        for task in dealloc_tasks {
            task.await.expect("Deallocation task failed");
        }

        // Verify all memory is freed
        let stats = allocator.stats().await.expect("Failed to get final stats");
        assert_eq!(stats.used_bytes, 0);
        assert_eq!(stats.allocated_blocks, 0);
    }

    /// Test pool under sequential load (avoiding Send issues)
    #[tokio::test]
    async fn test_pool_sequential_stress() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));
        let pool = MemoryPool::new(1024, 20, allocator);

        // Sequential acquire and release operations
        for _ in 0..100 {
            let handle = pool.acquire().await.expect("Failed to acquire");
            assert_eq!(handle.size, 1024);

            pool.release(handle).expect("Failed to release");
        }

        // Pool should be functional after stress test
        let stats = pool
            .stats()
            .expect("Failed to get pool stats after stress test");
        assert!(stats.available_blocks <= 20); // Should not exceed max
    }

    /// Test error propagation through the stack
    #[tokio::test]
    async fn test_error_propagation() {
        let small_allocator =
            Arc::new(GpuMemoryAllocator::new(1024).expect("Failed to create small allocator"));
        let pool = MemoryPool::new(2048, 5, small_allocator); // Block size > total memory

        // Pool should fail to acquire because allocator can't satisfy the request
        let result = pool.acquire().await;
        assert!(result.is_err());

        match result {
            Err(MemoryError::OutOfMemory {
                requested,
                available,
            }) => {
                assert_eq!(requested, 2048);
                assert_eq!(available, 1024);
            }
            _ => panic!("Expected OutOfMemory error"),
        }
    }

    /// Test memory manager trait implementation
    #[tokio::test]
    async fn test_memory_manager_trait() {
        let manager: Arc<dyn MemoryManager> =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));

        // Test trait methods
        let handle = manager.allocate(1024).await.expect("Failed to allocate");
        assert_eq!(handle.size, 1024);

        let stats = manager.stats().await.expect("Failed to get stats");
        assert_eq!(stats.used_bytes, 1024);

        manager
            .deallocate(handle)
            .await
            .expect("Failed to deallocate");

        let stats = manager.stats().await.expect("Failed to get final stats");
        assert_eq!(stats.used_bytes, 0);
    }

    /// Test memory handle uniqueness
    #[tokio::test]
    async fn test_handle_uniqueness() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));

        let mut handles = Vec::new();
        for _ in 0..100 {
            let handle = allocator.allocate(1024).await.expect("Failed to allocate");
            handles.push(handle);
        }

        // Check all handles have unique IDs
        let mut ids = std::collections::HashSet::new();
        for handle in &handles {
            assert!(ids.insert(handle.id), "Found duplicate handle ID");
        }

        // Clean up
        for handle in handles {
            allocator
                .deallocate(handle)
                .await
                .expect("Failed to deallocate");
        }
    }

    /// Test large allocation scenarios
    #[tokio::test]
    async fn test_large_allocations() {
        let large_memory = 1024 * 1024 * 1024; // 1GB
        let allocator = Arc::new(
            GpuMemoryAllocator::new(large_memory).expect("Failed to create large allocator"),
        );

        // Test allocating half the memory
        let large_handle = allocator
            .allocate(large_memory / 2)
            .await
            .expect("Failed to allocate large block");

        let stats = allocator.stats().await.expect("Failed to get stats");
        assert_eq!(stats.used_bytes, large_memory / 2);
        assert_eq!(stats.utilization_percent(), 50.0);

        // Should not be able to allocate another half (plus overhead)
        let result = allocator.allocate(large_memory / 2 + 1).await;
        assert!(result.is_err());

        // Clean up
        allocator
            .deallocate(large_handle)
            .await
            .expect("Failed to deallocate large block");
    }

    /// Test pool statistics accuracy
    #[tokio::test]
    async fn test_pool_statistics_accuracy() {
        let allocator =
            Arc::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));
        let pool = MemoryPool::new(512, 10, allocator);

        // Initial state
        let stats = pool.stats().expect("Failed to get initial stats");
        assert_eq!(stats.utilization_percent, 0.0);

        // Acquire 3 blocks
        let handle1 = pool.acquire().await.expect("Failed to acquire block 1");
        let handle2 = pool.acquire().await.expect("Failed to acquire block 2");
        let handle3 = pool.acquire().await.expect("Failed to acquire block 3");

        let stats = pool.stats().expect("Failed to get stats after acquire");
        assert_eq!(stats.utilization_percent, 100.0); // 3/3 = 100%

        // Release 1 block
        pool.release(handle1).expect("Failed to release block 1");

        let stats = pool.stats().expect("Failed to get stats after release");
        assert_eq!(stats.available_blocks, 1);
        assert!((stats.utilization_percent - 66.666664).abs() < 0.001); // 2/3 = 66.67%

        // Release remaining blocks
        pool.release(handle2).expect("Failed to release block 2");
        pool.release(handle3).expect("Failed to release block 3");

        let stats = pool.stats().expect("Failed to get final stats");
        assert_eq!(stats.available_blocks, 3);
        assert_eq!(stats.utilization_percent, 0.0); // 0/3 = 0%
    }
}
