//! Unit tests for gpu_dma_lock kernel module
//!
//! Tests GPU memory protection, DMA access control, and quota enforcement

#![cfg(test)]

use gpu_dma_lock::*;

mod gpu_allocation_tests {
    use super::*;

    #[test]
    fn test_gpu_device_registration() {
        let device = GpuDevice::new(0, "NVIDIA RTX 4090", 24 << 30); // 24GB
        assert_eq!(device.id(), 0);
        assert_eq!(device.name(), "NVIDIA RTX 4090");
        assert_eq!(device.total_memory(), 24 << 30);
        assert_eq!(device.available_memory(), 24 << 30);
        assert_eq!(device.allocated_memory(), 0);
    }

    #[test]
    fn test_agent_quota_creation() {
        let quota = AgentGpuQuota::new(1000, 4 << 30); // 4GB quota
        assert_eq!(quota.agent_id(), 1000);
        assert_eq!(quota.memory_limit(), 4 << 30);
        assert_eq!(quota.current_usage(), 0);
        assert!(quota.can_allocate(1 << 30)); // Can allocate 1GB
        assert!(!quota.can_allocate(5 << 30)); // Cannot allocate 5GB
    }

    #[test]
    fn test_memory_allocation_tracking() {
        let mut tracker = AllocationTracker::new();
        let agent_id = 100;
        let allocation_size = 512 << 20; // 512MB

        // Track allocation
        let alloc_id = tracker.track_allocation(agent_id, allocation_size, 0);
        assert!(alloc_id > 0);

        // Verify allocation exists
        let allocation = tracker.get_allocation(alloc_id).unwrap();
        assert_eq!(allocation.agent_id, agent_id);
        assert_eq!(allocation.size, allocation_size);
        assert_eq!(allocation.device_id, 0);

        // Track deallocation
        tracker.track_deallocation(alloc_id);
        assert!(tracker.get_allocation(alloc_id).is_none());
    }

    #[test]
    fn test_quota_enforcement() {
        let mut enforcer = QuotaEnforcer::new();
        let agent_id = 200;
        let quota_limit = 2 << 30; // 2GB

        // Set quota
        enforcer.set_agent_quota(agent_id, quota_limit);

        // Test allocations within quota
        assert!(enforcer.check_allocation(agent_id, 1 << 30)); // 1GB - OK
        enforcer.record_allocation(agent_id, 1 << 30);

        assert!(enforcer.check_allocation(agent_id, 512 << 20)); // 512MB - OK
        enforcer.record_allocation(agent_id, 512 << 20);

        // Test allocation exceeding quota
        assert!(!enforcer.check_allocation(agent_id, 1 << 30)); // 1GB - Would exceed

        // Test deallocation
        enforcer.record_deallocation(agent_id, 512 << 20);
        assert!(enforcer.check_allocation(agent_id, 512 << 20)); // Now OK again
    }

    #[test]
    fn test_multiple_gpu_devices() {
        let mut manager = GpuManager::new();

        // Register multiple devices
        manager.register_device(0, "GPU0", 16 << 30);
        manager.register_device(1, "GPU1", 24 << 30);
        manager.register_device(2, "GPU2", 48 << 30);

        // Verify devices
        assert_eq!(manager.device_count(), 3);
        assert_eq!(manager.total_memory(), 88 << 30);

        // Test device-specific allocation
        let agent_id = 300;
        assert!(manager.allocate(agent_id, 8 << 30, Some(1)).is_ok()); // 8GB on GPU1
        assert!(manager.allocate(agent_id, 20 << 30, Some(1)).is_err()); // Too much for GPU1
        assert!(manager.allocate(agent_id, 20 << 30, Some(2)).is_ok()); // OK on GPU2
    }
}

mod dma_access_tests {
    use super::*;

    #[test]
    fn test_dma_permission_creation() {
        let permission = DmaPermission::new(
            400,         // agent_id
            0x1000_0000, // start_addr
            0x2000_0000, // end_addr
            DmaAccessMode::ReadWrite,
        );

        assert_eq!(permission.agent_id(), 400);
        assert!(permission.can_access(0x1500_0000, DmaAccessMode::Read));
        assert!(permission.can_access(0x1500_0000, DmaAccessMode::Write));
        assert!(!permission.can_access(0x0500_0000, DmaAccessMode::Read)); // Out of range
        assert!(!permission.can_access(0x2500_0000, DmaAccessMode::Read)); // Out of range
    }

    #[test]
    fn test_dma_access_control_list() {
        let mut acl = DmaAccessControlList::new();

        // Add permissions
        acl.grant_access(500, 0x1000_0000, 0x2000_0000, DmaAccessMode::ReadOnly);
        acl.grant_access(501, 0x2000_0000, 0x3000_0000, DmaAccessMode::ReadWrite);

        // Test access checks
        assert!(acl.check_access(500, 0x1500_0000, DmaAccessMode::Read));
        assert!(!acl.check_access(500, 0x1500_0000, DmaAccessMode::Write)); // Read-only
        assert!(acl.check_access(501, 0x2500_0000, DmaAccessMode::Write));
        assert!(!acl.check_access(502, 0x1500_0000, DmaAccessMode::Read)); // No permission

        // Test revocation
        acl.revoke_access(500);
        assert!(!acl.check_access(500, 0x1500_0000, DmaAccessMode::Read));
    }

    #[test]
    fn test_pcie_bar_protection() {
        let mut bar_protection = PcieBarProtection::new();

        // Register BAR regions
        bar_protection.register_bar(0, 0, 0xF000_0000, 0x100_0000); // 16MB BAR
        bar_protection.register_bar(1, 0, 0xE000_0000, 0x200_0000); // 32MB BAR

        // Test protection
        assert!(bar_protection.is_protected(0xF010_0000)); // Within BAR0
        assert!(bar_protection.is_protected(0xE100_0000)); // Within BAR1
        assert!(!bar_protection.is_protected(0xD000_0000)); // Outside BARs

        // Test authorized access
        bar_protection.authorize_agent(600, 0); // Agent 600 can access device 0
        assert!(bar_protection.check_agent_access(600, 0xF010_0000));
        assert!(!bar_protection.check_agent_access(601, 0xF010_0000)); // Not authorized
    }

    #[test]
    fn test_gpu_context_isolation() {
        let mut context_manager = GpuContextManager::new();

        // Create contexts for different agents
        let ctx1 = context_manager.create_context(700, 0); // Agent 700, GPU 0
        let ctx2 = context_manager.create_context(701, 0); // Agent 701, GPU 0

        assert_ne!(ctx1, ctx2); // Different contexts

        // Verify isolation
        assert!(context_manager.is_isolated(ctx1, ctx2));

        // Test context switching overhead
        let switch_time = context_manager.measure_switch_overhead(ctx1, ctx2);
        assert!(switch_time < 1000); // Less than 1µs
    }

    #[test]
    fn test_gpu_direct_rdma() {
        let rdma_manager = GpuDirectRdmaManager::new();

        // Check if GPUDirect is available
        if !rdma_manager.is_available() {
            return; // Skip test if not available
        }

        // Test RDMA registration
        let agent_id = 800;
        let gpu_mem_addr = 0x8000_0000;
        let size = 1 << 20; // 1MB

        let handle = rdma_manager.register_memory(agent_id, gpu_mem_addr, size);
        assert!(handle.is_ok());

        // Test RDMA transfer authorization
        assert!(rdma_manager.authorize_transfer(agent_id, handle.unwrap()));
        assert!(!rdma_manager.authorize_transfer(801, handle.unwrap())); // Different agent
    }
}

mod statistics_tests {
    use super::*;

    #[test]
    fn test_allocation_statistics() {
        let mut stats = AllocationStatistics::new();

        // Record allocations
        stats.record_allocation(900, 1 << 30, 0); // 1GB on GPU0
        stats.record_allocation(900, 512 << 20, 0); // 512MB on GPU0
        stats.record_allocation(901, 2 << 30, 1); // 2GB on GPU1

        // Check per-agent stats
        let agent_stats = stats.get_agent_stats(900);
        assert_eq!(agent_stats.total_allocated, (1 << 30) + (512 << 20));
        assert_eq!(agent_stats.allocation_count, 2);

        // Check per-device stats
        let device_stats = stats.get_device_stats(0);
        assert_eq!(device_stats.allocated_memory, (1 << 30) + (512 << 20));

        // Record deallocations
        stats.record_deallocation(900, 512 << 20, 0);
        let agent_stats = stats.get_agent_stats(900);
        assert_eq!(agent_stats.total_allocated, 1 << 30);
    }

    #[test]
    fn test_dma_transfer_tracking() {
        let mut tracker = DmaTransferTracker::new();

        // Track transfers
        tracker.track_transfer(1000, 1 << 20, 100); // 1MB in 100µs
        tracker.track_transfer(1000, 2 << 20, 150); // 2MB in 150µs
        tracker.track_transfer(1001, 4 << 20, 200); // 4MB in 200µs

        // Get statistics
        let agent_stats = tracker.get_agent_stats(1000);
        assert_eq!(agent_stats.total_bytes, 3 << 20);
        assert_eq!(agent_stats.transfer_count, 2);
        assert_eq!(agent_stats.avg_latency_us, 125);

        // Check bandwidth calculation
        let bandwidth = tracker.calculate_bandwidth(1000);
        assert!(bandwidth > 0.0);
    }

    #[test]
    fn test_memory_pressure_detection() {
        let mut pressure_monitor = MemoryPressureMonitor::new();

        // Set thresholds
        pressure_monitor.set_warning_threshold(0.8); // 80%
        pressure_monitor.set_critical_threshold(0.95); // 95%

        // Update usage
        pressure_monitor.update_usage(0, 20 << 30, 24 << 30); // 20GB/24GB = 83%

        // Check pressure level
        assert_eq!(
            pressure_monitor.get_pressure_level(0),
            PressureLevel::Warning
        );

        // Update to critical
        pressure_monitor.update_usage(0, 23 << 30, 24 << 30); // 23GB/24GB = 96%
        assert_eq!(
            pressure_monitor.get_pressure_level(0),
            PressureLevel::Critical
        );
    }
}

mod security_tests {
    use super::*;

    #[test]
    fn test_unauthorized_allocation_prevention() {
        let mut security_manager = GpuSecurityManager::new();

        // Attempt allocation without authorization
        let result = security_manager.validate_allocation(1100, 1 << 30, 0);
        assert!(result.is_err());

        // Authorize agent
        security_manager.authorize_agent(1100, vec![0, 1]);

        // Now allocation should succeed
        let result = security_manager.validate_allocation(1100, 1 << 30, 0);
        assert!(result.is_ok());

        // But not on unauthorized device
        let result = security_manager.validate_allocation(1100, 1 << 30, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_boundary_validation() {
        let validator = MemoryBoundaryValidator::new();

        // Test valid allocations
        assert!(validator.validate_allocation(0x1000_0000, 1 << 20).is_ok()); // 1MB aligned

        // Test invalid allocations
        assert!(validator.validate_allocation(0x1000_0001, 1 << 20).is_err()); // Misaligned
        assert!(validator.validate_allocation(0xFFFF_F000, 1 << 20).is_err()); // Overflow
    }

    #[test]
    fn test_isolation_between_agents() {
        let mut isolation_manager = IsolationManager::new();

        // Create isolated regions
        isolation_manager.create_region(1200, 0x2000_0000, 0x1000_0000); // 256MB
        isolation_manager.create_region(1201, 0x3000_0000, 0x1000_0000); // 256MB

        // Test access violations
        assert!(!isolation_manager.can_access(1200, 0x3100_0000)); // Agent 1200 can't access 1201's region
        assert!(!isolation_manager.can_access(1201, 0x2100_0000)); // Agent 1201 can't access 1200's region

        // Test valid access
        assert!(isolation_manager.can_access(1200, 0x2100_0000));
        assert!(isolation_manager.can_access(1201, 0x3100_0000));
    }
}

mod performance_tests {
    use super::*;

    #[test]
    fn test_allocation_performance() {
        let mut allocator = FastAllocator::new();
        let iterations = 10000;

        let start = std::time::Instant::now();
        for i in 0..iterations {
            allocator.allocate(i, 1 << 20, 0); // 1MB allocations
        }
        let elapsed = start.elapsed();

        let per_alloc_ns = elapsed.as_nanos() / iterations;
        assert!(per_alloc_ns < 1000); // Less than 1µs per allocation
    }

    #[test]
    fn test_lookup_performance() {
        let mut tracker = AllocationTracker::new();

        // Pre-populate with allocations
        for i in 0..10000 {
            tracker.track_allocation(i % 100, 1 << 20, 0);
        }

        // Test lookup performance
        let start = std::time::Instant::now();
        for i in 0..10000 {
            let _ = tracker.get_allocation(i);
        }
        let elapsed = start.elapsed();

        let per_lookup_ns = elapsed.as_nanos() / 10000;
        assert!(per_lookup_ns < 100); // Less than 100ns per lookup
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(ConcurrentGpuManager::new());
        let threads = 8;
        let ops_per_thread = 1000;

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let mgr = Arc::clone(&manager);
                thread::spawn(move || {
                    for i in 0..ops_per_thread {
                        mgr.allocate(t * 1000 + i, 1 << 20, t % 4);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all allocations succeeded without conflicts
        assert_eq!(manager.total_allocations(), threads * ops_per_thread);
    }
}

mod integration_tests {
    use super::*;

    #[test]
    fn test_multi_gpu_workload() {
        let mut system = GpuDmaLockSystem::new();

        // Initialize with 4 GPUs
        for i in 0..4 {
            system.add_gpu(i, &format!("GPU{}", i), 24 << 30);
        }

        // Create agents with different quotas
        for i in 0..10 {
            system.create_agent(i, 4 << 30); // 4GB quota per agent
        }

        // Simulate workload
        for agent in 0..10 {
            for _ in 0..5 {
                let gpu = agent % 4;
                let size = 512 << 20; // 512MB
                assert!(system.allocate(agent, size, gpu).is_ok());
            }
        }

        // Verify distribution
        for gpu in 0..4 {
            let stats = system.get_gpu_stats(gpu);
            assert!(stats.allocated > 0);
            assert!(stats.allocated <= 24 << 30);
        }
    }

    #[test]
    fn test_oversubscription_handling() {
        let mut manager = OversubscriptionManager::new();
        manager.set_oversubscription_ratio(1.5); // Allow 50% oversubscription

        let physical_memory = 24 << 30; // 24GB
        let virtual_limit = (physical_memory as f64 * 1.5) as usize; // 36GB

        // Allocate up to virtual limit
        let mut total_allocated = 0;
        for i in 0..40 {
            let size = 1 << 30; // 1GB
            if total_allocated + size <= virtual_limit {
                assert!(manager.allocate(i, size, 0).is_ok());
                total_allocated += size;
            } else {
                assert!(manager.allocate(i, size, 0).is_err());
            }
        }

        // Verify we can allocate more than physical memory
        assert!(total_allocated > physical_memory);
        assert!(total_allocated <= virtual_limit);
    }

    #[test]
    fn test_memory_migration() {
        let mut migration_manager = MemoryMigrationManager::new();

        // Allocate on GPU0
        let alloc_id = migration_manager.allocate(1500, 2 << 30, 0).unwrap(); // 2GB

        // Trigger migration to GPU1
        let result = migration_manager.migrate(alloc_id, 1);
        assert!(result.is_ok());

        // Verify allocation is now on GPU1
        let info = migration_manager.get_allocation_info(alloc_id).unwrap();
        assert_eq!(info.device_id, 1);

        // Verify memory is freed on GPU0 and allocated on GPU1
        let gpu0_stats = migration_manager.get_device_stats(0);
        let gpu1_stats = migration_manager.get_device_stats(1);
        assert_eq!(gpu0_stats.allocated, 0);
        assert_eq!(gpu1_stats.allocated, 2 << 30);
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;

    pub fn create_test_gpu_device(id: u32, memory_gb: u32) -> GpuDevice {
        GpuDevice::new(id, &format!("TestGPU{}", id), memory_gb << 30)
    }

    pub fn create_test_agent(id: u64, quota_gb: u32) -> AgentGpuQuota {
        AgentGpuQuota::new(id, quota_gb << 30)
    }
}

#[cfg(test)]
mod ffi_tests {
    use super::*;
    use gpu_dma_lock::ffi::*;
    use std::ffi::CString;

    #[test]
    fn test_gpu_dma_init_cleanup() {
        unsafe {
            assert_eq!(gpu_dma_lock_init(), 0);
            gpu_dma_lock_cleanup();
        }
    }

    #[test]
    fn test_device_registration() {
        unsafe {
            gpu_dma_lock_init();

            let name = CString::new("Test GPU").unwrap();
            assert_eq!(gpu_dma_register_device(0, name.as_ptr(), 8 << 30), 0);

            // Test device count
            assert_eq!(gpu_dma_get_device_count(), 1);

            gpu_dma_lock_cleanup();
        }
    }

    #[test]
    fn test_agent_management() {
        unsafe {
            gpu_dma_lock_init();

            // Create agent
            let result = gpu_dma_create_agent(100, 2 << 30);
            assert_eq!(result.success, 1);

            // Remove agent
            let result = gpu_dma_remove_agent(100);
            assert_eq!(result.success, 1);

            gpu_dma_lock_cleanup();
        }
    }

    #[test]
    fn test_allocation_deallocation() {
        unsafe {
            gpu_dma_lock_init();

            let name = CString::new("Test GPU").unwrap();
            gpu_dma_register_device(0, name.as_ptr(), 8 << 30);
            gpu_dma_create_agent(100, 2 << 30);

            // Allocate
            let result = gpu_dma_allocate(100, 1 << 20, 0);
            assert_eq!(result.success, 1);
            let alloc_id = result.value;

            // Deallocate
            let result = gpu_dma_deallocate(alloc_id);
            assert_eq!(result.success, 1);

            gpu_dma_lock_cleanup();
        }
    }

    #[test]
    fn test_dma_access_control() {
        unsafe {
            gpu_dma_lock_init();

            // Grant access
            assert_eq!(gpu_dma_grant_access(100, 0x1000, 0x2000, 1), 0);

            // Check access
            assert_eq!(gpu_dma_check_access(100, 0x1500, 1), 1);
            assert_eq!(gpu_dma_check_access(100, 0x1500, 2), 0);

            gpu_dma_lock_cleanup();
        }
    }

    #[test]
    fn test_context_management() {
        unsafe {
            gpu_dma_lock_init();

            // Create context
            let result = gpu_dma_create_context(100, 0);
            assert_eq!(result.success, 1);
            let ctx_id = result.value;

            // Switch context
            let result = gpu_dma_switch_context(0, ctx_id);
            assert_eq!(result.success, 1);

            gpu_dma_lock_cleanup();
        }
    }

    #[test]
    fn test_stats_retrieval() {
        unsafe {
            gpu_dma_lock_init();

            let mut stats = CGpuDmaStats {
                total_allocations: 0,
                total_deallocations: 0,
                total_bytes_allocated: 0,
                dma_checks: 0,
                dma_denials: 0,
                context_switches: 0,
            };

            assert_eq!(gpu_dma_get_stats(&mut stats), 0);
            assert!(stats.total_allocations >= 0);

            gpu_dma_lock_cleanup();
        }
    }

    #[test]
    fn test_debug_enable() {
        unsafe {
            gpu_dma_lock_init();

            gpu_dma_enable_debug(1);
            gpu_dma_enable_debug(0);

            gpu_dma_lock_cleanup();
        }
    }
}
