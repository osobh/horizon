//! ExoRust GPU Memory Management
//!
//! This crate provides GPU memory allocation and management for the ExoRust
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
pub mod unified_memory;
pub mod multi_gpu;

#[cfg(test)]
mod test_helpers;

#[cfg(test)]
mod integration_tests;

pub use allocator::GpuMemoryAllocator;
pub use error::MemoryError;
pub use pool::MemoryPool;

// Export GPU Memory Tier types
pub use gpu_memory_tier::{GpuMemoryTier, MemoryConfig, AllocationStrategy, MemoryMetrics};

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
#[derive(Debug, Clone)]
pub struct GpuMemoryHandle {
    #[cfg(feature = "cuda")]
    pub ptr: DevicePointer<u8>,
    #[cfg(not(feature = "cuda"))]
    pub ptr: usize, // Store as usize for Send safety in development mode
    pub size: usize,
    pub id: uuid::Uuid,
}

// Make GpuMemoryHandle Send and Sync for development mode
unsafe impl Send for GpuMemoryHandle {}
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
        let handle = GpuMemoryHandle {
            #[cfg(feature = "cuda")]
            ptr: unsafe { cust::memory::DevicePointer::from_raw(0x1000) },
            #[cfg(not(feature = "cuda"))]
            ptr: 0x1000,
            size: 1024,
            id: uuid::Uuid::new_v4(),
        };

        assert_eq!(handle.size, 1024);
        #[cfg(not(feature = "cuda"))]
        assert_eq!(handle.ptr, 0x1000);
    }

    #[test]
    fn test_gpu_memory_handle_clone() {
        let original = GpuMemoryHandle {
            #[cfg(feature = "cuda")]
            ptr: unsafe { cust::memory::DevicePointer::from_raw(0x2000) },
            #[cfg(not(feature = "cuda"))]
            ptr: 0x2000,
            size: 2048,
            id: uuid::Uuid::new_v4(),
        };

        let cloned = original.clone();
        assert_eq!(cloned.size, original.size);
        assert_eq!(cloned.id, original.id);
        #[cfg(not(feature = "cuda"))]
        assert_eq!(cloned.ptr, original.ptr);
    }

    #[test]
    fn test_gpu_memory_handle_debug() {
        let handle = GpuMemoryHandle {
            #[cfg(feature = "cuda")]
            ptr: unsafe { cust::memory::DevicePointer::from_raw(0x3000) },
            #[cfg(not(feature = "cuda"))]
            ptr: 0x3000,
            size: 4096,
            id: uuid::Uuid::new_v4(),
        };

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
        assert_eq!(handle.size, 1024);

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

        let handle1 = GpuMemoryHandle {
            #[cfg(feature = "cuda")]
            ptr: unsafe { cust::memory::DevicePointer::from_raw(0x1000) },
            #[cfg(not(feature = "cuda"))]
            ptr: 0x1000,
            size: 1024,
            id: id1,
        };

        let handle2 = GpuMemoryHandle {
            #[cfg(feature = "cuda")]
            ptr: unsafe { cust::memory::DevicePointer::from_raw(0x2000) },
            #[cfg(not(feature = "cuda"))]
            ptr: 0x2000,
            size: 1024,
            id: id2,
        };

        assert_ne!(handle1.id, handle2.id);
    }
}
