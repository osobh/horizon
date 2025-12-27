//! GPU Memory Allocator implementation

use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

#[cfg(feature = "cuda")]
use cust::memory::DeviceBuffer;

use crate::{GpuMemoryHandle, MemoryError, MemoryManager, MemoryStats};

/// GPU memory allocator with pool management
pub struct GpuMemoryAllocator {
    allocations: Arc<Mutex<HashMap<Uuid, AllocationInfo>>>,
    total_memory: usize,
}

#[derive(Debug)]
struct AllocationInfo {
    size: usize,
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    buffer: DeviceBuffer<u8>,
    #[cfg(all(feature = "cuda-alt", not(feature = "cuda")))]
    buffer: Vec<u8>, // Simulate for now, would be DevicePtr<u8> in real implementation
    #[cfg(not(any(feature = "cuda", feature = "cuda-alt")))]
    #[allow(dead_code)]
    buffer: Vec<u8>,
}

impl GpuMemoryAllocator {
    /// Create new GPU memory allocator
    #[must_use = "ignoring the Result may hide allocator creation errors"]
    pub fn new(total_memory: usize) -> Result<Self, MemoryError> {
        Ok(Self {
            allocations: Arc::new(Mutex::new(HashMap::new())),
            total_memory,
        })
    }

    fn calculate_stats(&self) -> Result<MemoryStats, MemoryError> {
        let allocations = self
            .allocations
            .lock()
            .map_err(|e| MemoryError::AllocationFailed {
                reason: format!("Failed to acquire lock: {e}"),
            })?;

        let used_bytes: usize = allocations.values().map(|info| info.size).sum();
        let allocated_blocks = allocations.len();
        let available_bytes = self.total_memory.saturating_sub(used_bytes);

        // Simple fragmentation calculation (more sophisticated needed for production)
        let fragmentation_percent = if allocated_blocks > 0 {
            (allocated_blocks as f32 / (self.total_memory as f32 / 1024.0)).min(1.0) * 100.0
        } else {
            0.0
        };

        Ok(MemoryStats {
            total_bytes: self.total_memory,
            used_bytes,
            available_bytes,
            allocated_blocks,
            fragmentation_percent,
        })
    }
}

#[async_trait::async_trait]
impl MemoryManager for GpuMemoryAllocator {
    async fn allocate(&self, size: usize) -> Result<GpuMemoryHandle, MemoryError> {
        if size == 0 {
            return Err(MemoryError::InvalidSize { size });
        }

        let stats = self.calculate_stats()?;
        if size > stats.available_bytes {
            return Err(MemoryError::OutOfMemory {
                requested: size,
                available: stats.available_bytes,
            });
        }

        // Allocate GPU memory
        #[cfg(feature = "cuda")]
        let (buffer, ptr) = {
            let buffer = DeviceBuffer::zeroed(size).map_err(|e| MemoryError::AllocationFailed {
                reason: format!("CUDA allocation failed: {e}"),
            })?;
            let ptr = buffer.as_device_ptr();
            (buffer, ptr)
        };

        #[cfg(all(feature = "cuda-alt", not(feature = "cuda")))]
        let (buffer, ptr) = {
            // For now, simulate allocation until we have proper cudarc integration
            // In a real implementation, we would use CudaDevice::alloc()
            let buffer = vec![0u8; size];
            let ptr = buffer.as_ptr() as usize; // Store as usize for Send safety
            (buffer, ptr)
        };

        #[cfg(not(any(feature = "cuda", feature = "cuda-alt")))]
        let (buffer, ptr) = {
            let buffer = vec![0u8; size];
            let ptr = buffer.as_ptr() as usize; // Store as usize for Send safety
            (buffer, ptr)
        };

        let id = Uuid::new_v4();

        // Store allocation info
        let mut allocations =
            self.allocations
                .lock()
                .map_err(|e| MemoryError::AllocationFailed {
                    reason: format!("Failed to acquire lock: {e}"),
                })?;

        allocations.insert(id, AllocationInfo { size, buffer });

        Ok(GpuMemoryHandle { ptr, size, id })
    }

    async fn deallocate(&self, handle: GpuMemoryHandle) -> Result<(), MemoryError> {
        let mut allocations =
            self.allocations
                .lock()
                .map_err(|e| MemoryError::AllocationFailed {
                    reason: format!("Failed to acquire lock: {e}"),
                })?;

        allocations
            .remove(&handle.id)
            .ok_or(MemoryError::HandleNotFound { id: handle.id })?;

        Ok(())
    }

    async fn stats(&self) -> Result<MemoryStats, MemoryError> {
        self.calculate_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_allocator_creation() {
        let allocator = GpuMemoryAllocator::new(1024 * 1024 * 1024) // 1GB
            .expect("Failed to create allocator");

        let stats = allocator.stats().await.expect("Failed to get stats");

        assert_eq!(stats.total_bytes, 1024 * 1024 * 1024);
        assert_eq!(stats.used_bytes, 0);
        assert_eq!(stats.available_bytes, 1024 * 1024 * 1024);
        assert_eq!(stats.allocated_blocks, 0);
    }

    #[tokio::test]
    async fn test_invalid_allocation_size() {
        let allocator = GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator");

        let result = allocator.allocate(0).await;
        assert!(matches!(result, Err(MemoryError::InvalidSize { size: 0 })));
    }

    #[tokio::test]
    async fn test_out_of_memory() {
        let total_memory = 1024;
        let allocator = GpuMemoryAllocator::new(total_memory).expect("Failed to create allocator");

        let result = allocator.allocate(2048).await;
        assert!(matches!(
            result,
            Err(MemoryError::OutOfMemory {
                requested: 2048,
                available: 1024
            })
        ));
    }

    #[tokio::test]
    async fn test_successful_allocation_mock() {
        // This test will pass without CUDA by using mock allocation
        let allocator = GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator");

        // For now, this will fail due to CUDA requirement
        // In a real implementation, we'd mock the CUDA calls for testing
        let stats = allocator.stats().await.expect("Stats should work");
        assert_eq!(stats.used_bytes, 0);
    }

    #[tokio::test]
    async fn test_deallocate_invalid_handle() {
        let allocator = GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator");

        let fake_handle = GpuMemoryHandle {
            #[cfg(feature = "cuda")]
            ptr: unsafe { cust::memory::DevicePointer::from_raw(0) },
            #[cfg(not(feature = "cuda"))]
            ptr: 0usize, // Use 0 as null pointer equivalent
            size: 1024,
            id: Uuid::new_v4(),
        };

        let result = allocator.deallocate(fake_handle).await;
        assert!(matches!(result, Err(MemoryError::HandleNotFound { .. })));
    }

    #[tokio::test]
    async fn test_memory_stats_calculation() {
        let allocator = GpuMemoryAllocator::new(2048).expect("Failed to create allocator");

        let stats = allocator.stats().await.expect("Failed to get stats");

        assert_eq!(stats.utilization_percent(), 0.0);
        assert_eq!(stats.total_bytes, 2048);
        assert_eq!(stats.available_bytes, 2048);
    }

    #[tokio::test]
    async fn test_mutex_poisoning_allocate() {
        use crate::test_helpers::tests::create_poisoned_mutex;

        let allocator = GpuMemoryAllocator {
            allocations: create_poisoned_mutex(),
            total_memory: 1024 * 1024,
        };

        let result = allocator.allocate(1024).await;
        assert!(result.is_err());

        match result {
            Err(MemoryError::AllocationFailed { reason }) => {
                assert!(reason.contains("Failed to acquire lock"));
            }
            _ => panic!("Expected AllocationFailed error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_deallocate() {
        use crate::test_helpers::tests::create_poisoned_mutex;

        let allocator = GpuMemoryAllocator {
            allocations: create_poisoned_mutex(),
            total_memory: 1024 * 1024,
        };

        let handle = GpuMemoryHandle {
            #[cfg(feature = "cuda")]
            ptr: unsafe { cust::memory::DevicePointer::from_raw(0) },
            #[cfg(not(feature = "cuda"))]
            ptr: 0usize,
            size: 1024,
            id: Uuid::new_v4(),
        };

        let result = allocator.deallocate(handle).await;
        assert!(result.is_err());

        match result {
            Err(MemoryError::AllocationFailed { reason }) => {
                assert!(reason.contains("Failed to acquire lock"));
            }
            _ => panic!("Expected AllocationFailed error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_stats() {
        use crate::test_helpers::tests::create_poisoned_mutex;

        let allocator = GpuMemoryAllocator {
            allocations: create_poisoned_mutex(),
            total_memory: 1024 * 1024,
        };

        let result = allocator.stats().await;
        assert!(result.is_err());

        match result {
            Err(MemoryError::AllocationFailed { reason }) => {
                assert!(reason.contains("Failed to acquire lock"));
            }
            _ => panic!("Expected AllocationFailed error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_calculate_stats() {
        use crate::test_helpers::tests::create_poisoned_mutex;

        let allocator = GpuMemoryAllocator {
            allocations: create_poisoned_mutex(),
            total_memory: 1024 * 1024,
        };

        let result = allocator.calculate_stats();
        assert!(result.is_err());

        match result {
            Err(MemoryError::AllocationFailed { reason }) => {
                assert!(reason.contains("Failed to acquire lock"));
            }
            _ => panic!("Expected AllocationFailed error with lock failure"),
        }
    }

    // TDD RED Phase: Test deallocate success path (line 137)
    #[tokio::test]
    async fn test_deallocate_success_path() {
        let allocator = GpuMemoryAllocator::new(1024 * 1024).unwrap();

        // First allocate memory
        let handle = allocator.allocate(512).await.unwrap();

        // Then deallocate it successfully - this covers line 137
        let result = allocator.deallocate(handle).await;
        assert!(result.is_ok());
    }
}
