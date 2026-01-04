// GPU Memory Tier TDD Tests - RED Phase
// These tests MUST fail initially as we implement TDD correctly

use std::sync::Arc;
use stratoswarm_memory::gpu_memory_tier::{AllocationStrategy, GpuMemoryTier, MemoryConfig};

#[cfg(test)]
mod gpu_memory_tier_tests {
    use super::*;
    use std::ptr::NonNull;

    #[test]
    fn test_gpu_allocation_as_primary() {
        // RED: This test should fail initially
        let config = MemoryConfig {
            gpu_memory_gb: 24, // RTX 5090 spec
            cpu_memory_gb: 64,
            prefer_gpu: true,
            enable_unified_memory: true,
            overflow_threshold: 0.85,
        };

        let tier = GpuMemoryTier::new(config).expect("Failed to create GPU memory tier");

        // Allocate 1GB on GPU
        let allocation = tier
            .allocate(1024 * 1024 * 1024, AllocationStrategy::GpuPrimary)
            .expect("Failed to allocate on GPU");

        assert!(allocation.is_gpu_resident());
        assert_eq!(allocation.size(), 1024 * 1024 * 1024);
        assert!(tier.gpu_utilization() > 0.0);
        assert!(tier.gpu_utilization() < 0.1); // Should be ~4% of 24GB
    }

    #[test]
    fn test_automatic_cpu_overflow() {
        // RED: This test should fail initially
        let config = MemoryConfig {
            gpu_memory_gb: 8, // Smaller GPU for testing overflow
            cpu_memory_gb: 64,
            prefer_gpu: true,
            enable_unified_memory: true,
            overflow_threshold: 0.85,
        };

        let tier = GpuMemoryTier::new(config).expect("Failed to create GPU memory tier");

        // Allocate 7GB on GPU (under threshold)
        let alloc1 = tier
            .allocate(7 * 1024 * 1024 * 1024, AllocationStrategy::GpuPrimary)
            .expect("First allocation should succeed on GPU");
        assert!(alloc1.is_gpu_resident());

        // Try to allocate 2GB more (should overflow to CPU)
        let alloc2 = tier
            .allocate(2 * 1024 * 1024 * 1024, AllocationStrategy::GpuPrimary)
            .expect("Second allocation should overflow to CPU");
        assert!(alloc2.is_cpu_resident());
        assert!(tier.gpu_utilization() > 0.85);
        assert!(tier.cpu_utilization() > 0.0);
    }

    #[test]
    fn test_unified_addressing() {
        // RED: This test should fail initially
        let config = MemoryConfig {
            gpu_memory_gb: 24,
            cpu_memory_gb: 64,
            prefer_gpu: true,
            enable_unified_memory: true,
            overflow_threshold: 0.85,
        };

        let tier = GpuMemoryTier::new(config).expect("Failed to create GPU memory tier");

        // Allocate memory with unified addressing
        let allocation = tier
            .allocate_unified(1024 * 1024, AllocationStrategy::Auto)
            .expect("Failed to allocate unified memory");

        // Should be accessible from both CPU and GPU
        assert!(allocation.cpu_accessible());
        assert!(allocation.gpu_accessible());

        // Get unified pointer
        let ptr = allocation.unified_ptr();
        assert!(!ptr.is_null());

        // Test migration hint
        allocation.hint_gpu_usage();
        assert!(allocation.is_gpu_resident());

        allocation.hint_cpu_usage();
        assert!(allocation.is_cpu_resident());
    }

    #[test]
    fn test_zero_copy_operations() {
        // RED: This test should fail initially
        let config = MemoryConfig {
            gpu_memory_gb: 24,
            cpu_memory_gb: 64,
            prefer_gpu: true,
            enable_unified_memory: true,
            overflow_threshold: 0.85,
        };

        let tier = GpuMemoryTier::new(config).expect("Failed to create GPU memory tier");

        // Create zero-copy buffer
        let buffer = tier
            .create_zero_copy_buffer(4096)
            .expect("Failed to create zero-copy buffer");

        // Write from CPU
        let data = vec![42u8; 4096];
        buffer.write_cpu(&data).expect("Failed to write from CPU");

        // Read from GPU without copy
        let gpu_view = buffer.gpu_view();
        assert_eq!(gpu_view.len(), 4096);

        // Verify data integrity
        let read_data = buffer.read_cpu();
        assert_eq!(read_data[0], 42);
        assert_eq!(read_data[4095], 42);
    }

    #[test]
    fn test_memory_pressure_handling() {
        // RED: This test should fail initially
        let config = MemoryConfig {
            gpu_memory_gb: 4, // Small GPU to test pressure
            cpu_memory_gb: 32,
            prefer_gpu: true,
            enable_unified_memory: true,
            overflow_threshold: 0.85,
        };

        let tier = GpuMemoryTier::new(config).expect("Failed to create GPU memory tier");

        // Fill GPU memory to create pressure
        let mut allocations = Vec::new();
        for _ in 0..10 {
            let alloc = tier.allocate(512 * 1024 * 1024, AllocationStrategy::GpuPrimary);
            if let Ok(a) = alloc {
                allocations.push(a);
            }
        }

        // Check that overflow happened correctly
        let gpu_allocs = allocations.iter().filter(|a| a.is_gpu_resident()).count();
        let cpu_allocs = allocations.iter().filter(|a| a.is_cpu_resident()).count();

        assert!(gpu_allocs > 0);
        assert!(cpu_allocs > 0);
        assert!(tier.gpu_utilization() >= 0.85);
    }

    #[test]
    fn test_pinned_memory_allocation() {
        // RED: This test should fail initially
        let config = MemoryConfig {
            gpu_memory_gb: 24,
            cpu_memory_gb: 64,
            prefer_gpu: true,
            enable_unified_memory: true,
            overflow_threshold: 0.85,
        };

        let tier = GpuMemoryTier::new(config).expect("Failed to create GPU memory tier");

        // Allocate pinned memory for fast DMA
        let pinned = tier
            .allocate_pinned(1024 * 1024)
            .expect("Failed to allocate pinned memory");

        assert!(pinned.is_pinned());
        assert!(pinned.is_dma_capable());
        assert_eq!(pinned.size(), 1024 * 1024);

        // Test DMA transfer
        let gpu_dest = tier
            .allocate(1024 * 1024, AllocationStrategy::GpuOnly)
            .expect("Failed to allocate GPU destination");

        let transfer_time = pinned.dma_copy_to(&gpu_dest).expect("DMA transfer failed");

        assert!(transfer_time.as_micros() < 1000); // Should be fast
    }

    #[test]
    fn test_memory_pool_management() {
        // RED: This test should fail initially
        let config = MemoryConfig {
            gpu_memory_gb: 24,
            cpu_memory_gb: 64,
            prefer_gpu: true,
            enable_unified_memory: true,
            overflow_threshold: 0.85,
        };

        let tier = GpuMemoryTier::new(config).expect("Failed to create GPU memory tier");

        // Create memory pool
        let pool = tier
            .create_pool("inference", 2 * 1024 * 1024 * 1024)
            .expect("Failed to create memory pool");

        // Allocate from pool
        let alloc1 = pool
            .allocate(512 * 1024 * 1024)
            .expect("Failed to allocate from pool");
        let alloc2 = pool
            .allocate(512 * 1024 * 1024)
            .expect("Failed to allocate from pool");

        assert_eq!(pool.used_bytes(), 1024 * 1024 * 1024);
        assert_eq!(pool.available_bytes(), 1024 * 1024 * 1024);

        // Return to pool
        pool.deallocate(alloc1);
        assert_eq!(pool.used_bytes(), 512 * 1024 * 1024);

        // Reuse pooled memory
        let alloc3 = pool
            .allocate(256 * 1024 * 1024)
            .expect("Should reuse pooled memory");
        assert!(alloc3.from_pool());
    }

    #[test]
    fn test_multi_gpu_memory_tier() {
        // RED: This test should fail initially
        let config = MemoryConfig {
            gpu_memory_gb: 24,
            cpu_memory_gb: 64,
            prefer_gpu: true,
            enable_unified_memory: true,
            overflow_threshold: 0.85,
        };

        let tier =
            GpuMemoryTier::new_multi_gpu(config, 4).expect("Failed to create multi-GPU tier");

        // Allocate across GPUs
        let alloc1 = tier
            .allocate_on_gpu(1024 * 1024 * 1024, 0)
            .expect("Failed to allocate on GPU 0");
        let alloc2 = tier
            .allocate_on_gpu(1024 * 1024 * 1024, 1)
            .expect("Failed to allocate on GPU 1");

        assert_eq!(alloc1.gpu_id(), 0);
        assert_eq!(alloc2.gpu_id(), 1);

        // Test P2P transfer
        let transfer_time = tier.p2p_copy(&alloc1, &alloc2).expect("P2P copy failed");
        assert!(transfer_time.as_micros() < 5000); // P2P should be fast
    }

    #[test]
    fn test_memory_migration() {
        // RED: This test should fail initially
        let config = MemoryConfig {
            gpu_memory_gb: 24,
            cpu_memory_gb: 64,
            prefer_gpu: true,
            enable_unified_memory: true,
            overflow_threshold: 0.85,
        };

        let tier = GpuMemoryTier::new(config).expect("Failed to create GPU memory tier");

        // Allocate on CPU initially
        let mut allocation = tier
            .allocate(1024 * 1024, AllocationStrategy::CpuOnly)
            .expect("Failed to allocate on CPU");

        assert!(allocation.is_cpu_resident());

        // Migrate to GPU
        allocation
            .migrate_to_gpu()
            .expect("Migration to GPU failed");
        assert!(allocation.is_gpu_resident());

        // Migrate back to CPU
        allocation
            .migrate_to_cpu()
            .expect("Migration to CPU failed");
        assert!(allocation.is_cpu_resident());
    }

    #[test]
    fn test_memory_metrics() {
        // RED: This test should fail initially
        let config = MemoryConfig {
            gpu_memory_gb: 24,
            cpu_memory_gb: 64,
            prefer_gpu: true,
            enable_unified_memory: true,
            overflow_threshold: 0.85,
        };

        let tier = GpuMemoryTier::new(config).expect("Failed to create GPU memory tier");

        // Get initial metrics
        let metrics = tier.get_metrics();
        assert_eq!(metrics.gpu_allocated_bytes, 0);
        assert_eq!(metrics.cpu_allocated_bytes, 0);
        assert_eq!(metrics.total_allocations, 0);

        // Make allocations
        let _alloc1 = tier
            .allocate(1024 * 1024 * 1024, AllocationStrategy::GpuPrimary)
            .expect("Allocation failed");
        let _alloc2 = tier
            .allocate(512 * 1024 * 1024, AllocationStrategy::CpuOnly)
            .expect("Allocation failed");

        // Check updated metrics
        let metrics = tier.get_metrics();
        assert_eq!(metrics.gpu_allocated_bytes, 1024 * 1024 * 1024);
        assert_eq!(metrics.cpu_allocated_bytes, 512 * 1024 * 1024);
        assert_eq!(metrics.total_allocations, 2);
        assert!(metrics.overflow_events == 0);
    }
}
