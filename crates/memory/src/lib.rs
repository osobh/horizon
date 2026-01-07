//! Stratoswarm GPU Memory Management
//!
//! This crate provides GPU memory allocation and management for the Stratoswarm
//! agent-first operating system.

use anyhow::Result;

#[cfg(feature = "cuda")]
use cust::memory::DevicePointer;

pub mod allocator;
pub mod error;
pub mod pool;

// GPU Memory Tier modules
pub mod gpu_memory_tier;
pub mod memory_pool;
pub mod multi_gpu;
pub mod unified_memory;

#[cfg(test)]
mod test_helpers;

#[cfg(test)]
mod integration_tests;

pub use allocator::GpuMemoryAllocator;
pub use error::MemoryError;
pub use pool::MemoryPool;

// Export GPU Memory Tier types
pub use gpu_memory_tier::{AllocationStrategy, GpuMemoryTier, MemoryConfig, MemoryMetrics};

/// GPU memory allocation interface for containers
#[async_trait::async_trait]
pub trait MemoryManager: Send + Sync {
    /// Allocate GPU memory for a container
    async fn allocate(&self, size: usize) -> Result<GpuMemoryHandle, MemoryError>;

    /// Deallocate GPU memory
    async fn deallocate(&self, handle: GpuMemoryHandle) -> Result<(), MemoryError>;

    /// Get memory statistics
    async fn stats(&self) -> Result<MemoryStats, MemoryError>;
}

/// Handle to allocated GPU memory
///
/// This type encapsulates a GPU memory allocation. The raw pointer is kept private
/// to ensure memory safety - access to the underlying pointer requires unsafe code.
#[derive(Debug, Clone)]
pub struct GpuMemoryHandle {
    #[cfg(feature = "cuda")]
    ptr: DevicePointer<u8>,
    #[cfg(not(feature = "cuda"))]
    ptr: usize, // Store as usize for Send safety in development mode
    size: usize,
    id: uuid::Uuid,
}

impl GpuMemoryHandle {
    /// Create a new GPU memory handle (crate-internal use only)
    ///
    /// This constructor is `pub(crate)` to prevent external code from creating
    /// handles with arbitrary pointers. Only the memory allocator should create handles.
    #[cfg(feature = "cuda")]
    pub(crate) fn new(ptr: DevicePointer<u8>, size: usize, id: uuid::Uuid) -> Self {
        Self { ptr, size, id }
    }

    /// Create a new GPU memory handle (crate-internal use only)
    #[cfg(not(feature = "cuda"))]
    pub(crate) fn new(ptr: usize, size: usize, id: uuid::Uuid) -> Self {
        Self { ptr, size, id }
    }

    /// Create a handle with an arbitrary pointer for testing purposes.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` points to valid GPU memory (or is a null/test pointer that won't be dereferenced)
    /// - The memory region is at least `size` bytes
    /// - The handle will not outlive the underlying allocation
    #[cfg(feature = "cuda")]
    pub unsafe fn new_unchecked(ptr: DevicePointer<u8>, size: usize, id: uuid::Uuid) -> Self {
        Self { ptr, size, id }
    }

    /// Create a handle with an arbitrary pointer for testing purposes.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` represents valid GPU memory (or is a test value that won't be dereferenced)
    /// - The memory region is at least `size` bytes
    /// - The handle will not outlive the underlying allocation
    #[cfg(not(feature = "cuda"))]
    pub unsafe fn new_unchecked(ptr: usize, size: usize, id: uuid::Uuid) -> Self {
        Self { ptr, size, id }
    }

    /// Get the size of the allocated memory in bytes
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the unique identifier for this allocation
    #[inline]
    #[must_use]
    pub fn id(&self) -> uuid::Uuid {
        self.id
    }

    /// Get the raw device pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The pointer is not used after the handle is deallocated
    /// - Proper synchronization is maintained for concurrent access
    /// - The pointer is used according to GPU memory access rules
    #[cfg(feature = "cuda")]
    #[inline]
    pub unsafe fn as_raw_ptr(&self) -> DevicePointer<u8> {
        self.ptr
    }

    /// Get the raw pointer value.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The pointer is not used after the handle is deallocated
    /// - Proper synchronization is maintained for concurrent access
    #[cfg(not(feature = "cuda"))]
    #[inline]
    pub unsafe fn as_raw_ptr(&self) -> usize {
        self.ptr
    }
}

// SAFETY: GpuMemoryHandle is Send because:
// 1. When cuda feature is enabled: `DevicePointer<u8>` represents a GPU memory
//    address that can be safely transferred between threads (CUDA is thread-safe)
// 2. When cuda feature is disabled: `ptr: usize` is just a numeric value
// 3. `size: usize` is a trivially Send primitive type
// 4. `id: uuid::Uuid` is Send (just bytes, no references)
// 5. The handle represents ownership of a GPU allocation that can be moved
unsafe impl Send for GpuMemoryHandle {}

// SAFETY: GpuMemoryHandle is Sync because:
// 1. All fields are either primitive types or thread-safe wrappers
// 2. DevicePointer (when cuda enabled) is just an address value
// 3. Reading the handle's fields from multiple threads is safe
// 4. Actual GPU memory operations require going through CUDA APIs
//    which provide their own synchronization
// 5. No interior mutability - the handle is immutable after creation
unsafe impl Sync for GpuMemoryHandle {}

/// Memory usage statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryStats {
    pub total_bytes: usize,
    pub used_bytes: usize,
    pub available_bytes: usize,
    pub allocated_blocks: usize,
    pub fragmentation_percent: f32,
}

impl MemoryStats {
    pub fn utilization_percent(&self) -> f32 {
        (self.used_bytes as f32 / self.total_bytes as f32) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_memory_stats_utilization_calculation() {
        let stats = MemoryStats {
            total_bytes: 1024,
            used_bytes: 512,
            available_bytes: 512,
            allocated_blocks: 2,
            fragmentation_percent: 10.0,
        };

        assert_eq!(stats.utilization_percent(), 50.0);
    }

    #[test]
    fn test_memory_stats_zero_total() {
        let stats = MemoryStats {
            total_bytes: 0,
            used_bytes: 0,
            available_bytes: 0,
            allocated_blocks: 0,
            fragmentation_percent: 0.0,
        };

        // Should not panic with zero division
        assert!(stats.utilization_percent().is_nan() || stats.utilization_percent() == 0.0);
    }

    #[test]
    fn test_memory_stats_full_utilization() {
        let stats = MemoryStats {
            total_bytes: 2048,
            used_bytes: 2048,
            available_bytes: 0,
            allocated_blocks: 4,
            fragmentation_percent: 25.0,
        };

        assert_eq!(stats.utilization_percent(), 100.0);
    }

    #[test]
    fn test_memory_stats_fields() {
        let stats = MemoryStats {
            total_bytes: 1024 * 1024,
            used_bytes: 512 * 1024,
            available_bytes: 512 * 1024,
            allocated_blocks: 10,
            fragmentation_percent: 15.5,
        };

        assert_eq!(stats.total_bytes, 1024 * 1024);
        assert_eq!(stats.used_bytes, 512 * 1024);
        assert_eq!(stats.available_bytes, 512 * 1024);
        assert_eq!(stats.allocated_blocks, 10);
        assert_eq!(stats.fragmentation_percent, 15.5);
    }

    #[test]
    fn test_gpu_memory_handle_creation() {
        // SAFETY: Test pointer that won't be dereferenced - used for API testing only
        #[cfg(feature = "cuda")]
        let handle = unsafe {
            GpuMemoryHandle::new_unchecked(
                cust::memory::DevicePointer::from_raw(0x1000),
                1024,
                uuid::Uuid::new_v4(),
            )
        };
        #[cfg(not(feature = "cuda"))]
        let handle = unsafe { GpuMemoryHandle::new_unchecked(0x1000, 1024, uuid::Uuid::new_v4()) };

        assert_eq!(handle.size(), 1024);
        #[cfg(not(feature = "cuda"))]
        assert_eq!(unsafe { handle.as_raw_ptr() }, 0x1000);
    }

    #[test]
    fn test_gpu_memory_handle_clone() {
        // SAFETY: Test pointer that won't be dereferenced - used for clone testing only
        #[cfg(feature = "cuda")]
        let original = unsafe {
            GpuMemoryHandle::new_unchecked(
                cust::memory::DevicePointer::from_raw(0x2000),
                2048,
                uuid::Uuid::new_v4(),
            )
        };
        #[cfg(not(feature = "cuda"))]
        let original = unsafe { GpuMemoryHandle::new_unchecked(0x2000, 2048, uuid::Uuid::new_v4()) };

        let cloned = original.clone();
        assert_eq!(cloned.size(), original.size());
        assert_eq!(cloned.id(), original.id());
        #[cfg(not(feature = "cuda"))]
        assert_eq!(unsafe { cloned.as_raw_ptr() }, unsafe {
            original.as_raw_ptr()
        });
    }

    #[test]
    fn test_gpu_memory_handle_debug() {
        // SAFETY: Test pointer that won't be dereferenced - used for debug output testing only
        #[cfg(feature = "cuda")]
        let handle = unsafe {
            GpuMemoryHandle::new_unchecked(
                cust::memory::DevicePointer::from_raw(0x3000),
                4096,
                uuid::Uuid::new_v4(),
            )
        };
        #[cfg(not(feature = "cuda"))]
        let handle = unsafe { GpuMemoryHandle::new_unchecked(0x3000, 4096, uuid::Uuid::new_v4()) };

        let debug_str = format!("{:?}", handle);
        assert!(debug_str.contains("GpuMemoryHandle"));
        assert!(debug_str.contains("4096"));
    }

    #[test]
    fn test_memory_stats_edge_cases() {
        // Test with maximum values
        let max_stats = MemoryStats {
            total_bytes: usize::MAX,
            used_bytes: usize::MAX,
            available_bytes: 0,
            allocated_blocks: usize::MAX,
            fragmentation_percent: 100.0,
        };

        assert_eq!(max_stats.utilization_percent(), 100.0);

        // Test with precision
        let precise_stats = MemoryStats {
            total_bytes: 3,
            used_bytes: 1,
            available_bytes: 2,
            allocated_blocks: 1,
            fragmentation_percent: 33.333336,
        };

        assert!((precise_stats.utilization_percent() - 33.333336).abs() < 0.001);
    }

    #[test]
    fn test_memory_stats_default_values() {
        let stats = MemoryStats {
            total_bytes: 0,
            used_bytes: 0,
            available_bytes: 0,
            allocated_blocks: 0,
            fragmentation_percent: 0.0,
        };

        assert_eq!(stats.allocated_blocks, 0);
        assert_eq!(stats.fragmentation_percent, 0.0);
    }

    #[tokio::test]
    async fn test_memory_manager_trait_object() {
        let allocator: Box<dyn MemoryManager> =
            Box::new(GpuMemoryAllocator::new(1024 * 1024).expect("Failed to create allocator"));

        // Test that trait object works correctly
        let handle = allocator.allocate(1024).await.expect("Failed to allocate");
        assert_eq!(handle.size(), 1024);

        let stats = allocator.stats().await.expect("Failed to get stats");
        assert_eq!(stats.used_bytes, 1024);

        allocator
            .deallocate(handle)
            .await
            .expect("Failed to deallocate");
    }

    #[test]
    fn test_memory_handle_send_sync() {
        // Test that GpuMemoryHandle is Send and Sync
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<GpuMemoryHandle>();
        assert_sync::<GpuMemoryHandle>();
    }

    #[test]
    fn test_memory_manager_trait_bounds() {
        // Test that MemoryManager is Send and Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn MemoryManager>>();
    }

    #[test]
    fn test_module_exports() {
        // Test that all public types are accessible
        let _allocator = GpuMemoryAllocator::new(1024).expect("Failed to create allocator");
        let _error = MemoryError::InvalidSize { size: 0 };
        let _pool = MemoryPool::new(512, 10, Arc::new(_allocator));
    }

    #[test]
    fn test_memory_stats_clone_debug() {
        let stats = MemoryStats {
            total_bytes: 8192,
            used_bytes: 4096,
            available_bytes: 4096,
            allocated_blocks: 8,
            fragmentation_percent: 12.5,
        };

        // Test Clone
        let cloned = stats.clone();
        assert_eq!(cloned.total_bytes, stats.total_bytes);
        assert_eq!(cloned.used_bytes, stats.used_bytes);

        // Test Debug
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("MemoryStats"));
        assert!(debug_str.contains("8192"));
    }

    #[test]
    fn test_memory_stats_floating_point_precision() {
        let stats = MemoryStats {
            total_bytes: 7,
            used_bytes: 5,
            available_bytes: 2,
            allocated_blocks: 3,
            fragmentation_percent: 71.42857,
        };

        let utilization = stats.utilization_percent();
        assert!((utilization - 71.42857).abs() < 0.001);
    }

    #[test]
    fn test_uuid_in_handle() {
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();

        // SAFETY: Test pointers that won't be dereferenced - used for UUID comparison only
        #[cfg(feature = "cuda")]
        let handle1 = unsafe {
            GpuMemoryHandle::new_unchecked(
                cust::memory::DevicePointer::from_raw(0x1000),
                1024,
                id1,
            )
        };
        #[cfg(not(feature = "cuda"))]
        let handle1 = unsafe { GpuMemoryHandle::new_unchecked(0x1000, 1024, id1) };

        #[cfg(feature = "cuda")]
        let handle2 = unsafe {
            GpuMemoryHandle::new_unchecked(
                cust::memory::DevicePointer::from_raw(0x2000),
                1024,
                id2,
            )
        };
        #[cfg(not(feature = "cuda"))]
        let handle2 = unsafe { GpuMemoryHandle::new_unchecked(0x2000, 1024, id2) };

        assert_ne!(handle1.id(), handle2.id());
    }
}
