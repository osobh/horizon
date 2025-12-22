//! Tests for CUDA context management

use crate::context::*;
use crate::error::CudaResult;

#[test]
fn test_context_creation() {
    let flags = ContextFlags::default();
    assert_eq!(flags.bits(), 0);

    let flags = ContextFlags::SCHED_YIELD | ContextFlags::MAP_HOST;
    assert!(flags.contains(ContextFlags::SCHED_YIELD));
    assert!(flags.contains(ContextFlags::MAP_HOST));
}

#[test]
fn test_context_flags_all_variants() {
    let all_flags = vec![
        ContextFlags::SCHED_AUTO,
        ContextFlags::SCHED_SPIN,
        ContextFlags::SCHED_YIELD,
        ContextFlags::SCHED_BLOCKING_SYNC,
        ContextFlags::MAP_HOST,
        ContextFlags::LMEM_RESIZE_TO_MAX,
    ];

    for flag in all_flags {
        assert!(flag.bits() != 0);
        let combined = ContextFlags::empty() | flag;
        assert!(combined.contains(flag));
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_mock_context_creation() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();
    assert_eq!(ctx.device_id(), 0);
    assert!(ctx.is_current());
}

#[cfg(cuda_mock)]
#[test]
fn test_mock_context_make_current() {
    let ctx1 = Context::new(0, ContextFlags::default()).unwrap();
    let ctx2 = Context::new(1, ContextFlags::default()).unwrap();

    assert!(ctx2.make_current().is_ok());
    assert!(ctx2.is_current());
    assert!(!ctx1.is_current());
}

#[cfg(cuda_mock)]
#[test]
fn test_mock_context_synchronize() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();
    assert!(ctx.synchronize().is_ok());
}

#[cfg(cuda_mock)]
#[test]
fn test_mock_context_properties() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();

    let props = ctx.get_properties().unwrap();
    assert!(props.total_memory > 0);
    assert!(props.compute_capability.0 >= 7);
    assert!(props.max_threads_per_block > 0);
    assert!(props.max_block_dims[0] > 0);
    assert!(props.max_grid_dims[0] > 0);
}

#[cfg(cuda_mock)]
#[test]
fn test_multiple_contexts() {
    let contexts: Vec<_> = (0..4)
        .map(|i| Context::new(i, ContextFlags::default()).unwrap())
        .collect();

    for (i, ctx) in contexts.iter().enumerate() {
        assert_eq!(ctx.device_id(), i as i32);
    }
}

#[test]
fn test_context_properties_struct() {
    let props = ContextProperties {
        total_memory: 1024 * 1024 * 1024 * 8, // 8GB
        compute_capability: (8, 6),
        max_threads_per_block: 1024,
        max_block_dims: [1024, 1024, 64],
        max_grid_dims: [2147483647, 65535, 65535],
        warp_size: 32,
        max_shared_memory_per_block: 49152,
        clock_rate: 1770000, // 1.77 GHz
    };

    assert_eq!(props.total_memory, 8589934592);
    assert_eq!(props.compute_capability, (8, 6));
    assert_eq!(props.warp_size, 32);
}

#[cfg(cuda_mock)]
#[test]
fn test_context_cache_configuration() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();

    // Test setting different cache configurations
    assert!(ctx.set_cache_config(CacheConfig::PreferShared).is_ok());
    assert!(ctx.set_cache_config(CacheConfig::PreferCache).is_ok());
    assert!(ctx.set_cache_config(CacheConfig::PreferEqual).is_ok());
    assert!(ctx.set_cache_config(CacheConfig::PreferNone).is_ok());
}

#[cfg(cuda_mock)]
#[test]
fn test_context_shared_memory_configuration() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();

    // Test setting different shared memory bank sizes
    assert!(ctx
        .set_shared_mem_config(SharedMemConfig::DefaultBankSize)
        .is_ok());
    assert!(ctx
        .set_shared_mem_config(SharedMemConfig::FourByteBankSize)
        .is_ok());
    assert!(ctx
        .set_shared_mem_config(SharedMemConfig::EightByteBankSize)
        .is_ok());
}

#[cfg(cuda_mock)]
#[test]
fn test_context_stream_priority_range() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();

    let (least, greatest) = ctx.get_stream_priority_range().unwrap();
    assert!(least >= greatest); // Lower values = higher priority in CUDA
}

#[cfg(cuda_mock)]
#[test]
fn test_context_api_version() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();

    let version = ctx.get_api_version().unwrap();
    assert!(version >= 11000); // CUDA 11.0+
}

#[cfg(cuda_mock)]
#[test]
fn test_context_device_attributes() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();

    assert!(ctx.can_map_host_memory().unwrap());
    assert!(ctx.supports_managed_memory().unwrap());
    assert!(ctx.supports_cooperative_launch()?);
}

#[cfg(cuda_mock)]
#[test]
fn test_context_memory_info() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();

    let (free, total) = ctx.get_memory_info().unwrap();
    assert!(free > 0);
    assert!(total > 0);
    assert!(free <= total);
}

#[cfg(cuda_mock)]
#[test]
fn test_context_drop() {
    // Context should properly clean up when dropped
    {
        let _ctx = Context::new(0, ContextFlags::default()).unwrap();
        // Context dropped here
    }

    // Should be able to create new context
    let _ctx2 = Context::new(0, ContextFlags::default())?;
}

#[test]
fn test_context_flags_bitwise_operations() {
    let flag1 = ContextFlags::SCHED_YIELD;
    let flag2 = ContextFlags::MAP_HOST;

    // OR operation
    let combined = flag1 | flag2;
    assert!(combined.contains(flag1));
    assert!(combined.contains(flag2));

    // AND operation
    let intersect = combined & flag1;
    assert_eq!(intersect, flag1);

    // XOR operation
    let xor = flag1 ^ flag2;
    assert!(xor.contains(flag1));
    assert!(xor.contains(flag2));
    assert_eq!(xor, combined);

    // NOT operation
    let not_flag1 = !flag1;
    assert!(!not_flag1.contains(flag1));
}

#[cfg(cuda_mock)]
#[test]
fn test_context_with_different_devices() {
    // Test creating contexts on different devices
    for device_id in 0..4 {
        let ctx = Context::new(device_id, ContextFlags::default()).unwrap();
        assert_eq!(ctx.device_id(), device_id);

        let props = ctx.get_properties()?;
        // Different devices might have different properties
        assert!(props.total_memory > 0);
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_context_resource_limits() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();

    // Test getting various resource limits
    let stack_size = ctx.get_limit(ResourceLimit::StackSize).unwrap();
    assert!(stack_size > 0);

    let printf_fifo_size = ctx.get_limit(ResourceLimit::PrintfFifoSize)?;
    assert!(printf_fifo_size > 0);

    let malloc_heap_size = ctx.get_limit(ResourceLimit::MallocHeapSize)?;
    assert!(malloc_heap_size > 0);
}

#[cfg(cuda_mock)]
#[test]
fn test_context_set_resource_limits() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();

    // Test setting resource limits
    let new_stack_size = 1024 * 16; // 16KB
    assert!(ctx
        .set_limit(ResourceLimit::StackSize, new_stack_size)
        .is_ok());

    let actual = ctx.get_limit(ResourceLimit::StackSize)?;
    assert_eq!(actual, new_stack_size);
}

#[cfg(cuda_mock)]
#[test]
fn test_context_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let ctx = Arc::new(Context::new(0, ContextFlags::default()).unwrap());
    let mut handles = vec![];

    // Multiple threads accessing context properties
    for i in 0..10 {
        let ctx_clone = ctx.clone();
        let handle = thread::spawn(move || {
            let props = ctx_clone.get_properties().unwrap();
            assert!(props.total_memory > 0);
            assert_eq!(ctx_clone.device_id(), 0);
            i
        });
        handles.push(handle);
    }

    for handle in handles {
        let _result = handle.join().unwrap();
    }
}

#[test]
fn test_compute_capability_comparison() {
    let cc1 = (7, 5);
    let cc2 = (8, 0);
    let cc3 = (8, 6);

    // Manual comparison for compute capabilities
    assert!(cc1.0 < cc2.0 || (cc1.0 == cc2.0 && cc1.1 < cc2.1));
    assert!(cc2.0 < cc3.0 || (cc2.0 == cc3.0 && cc2.1 < cc3.1));
}

#[cfg(cuda_mock)]
#[test]
fn test_context_reset() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();

    // Should be able to reset context
    assert!(ctx.reset().is_ok());

    // Context should still be usable after reset
    assert!(ctx.synchronize().is_ok());
}

#[cfg(cuda_mock)]
#[test]
fn test_context_with_flags() {
    // Test creating contexts with different flag combinations
    let flag_combinations = vec![
        ContextFlags::SCHED_AUTO,
        ContextFlags::SCHED_SPIN,
        ContextFlags::SCHED_YIELD,
        ContextFlags::SCHED_YIELD | ContextFlags::MAP_HOST,
        ContextFlags::MAP_HOST | ContextFlags::LMEM_RESIZE_TO_MAX,
    ];

    for flags in flag_combinations {
        let ctx = Context::new(0, flags).unwrap();
        assert_eq!(ctx.device_id(), 0);
        // Flags should affect context behavior (implementation specific)
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_context_peer_access() {
    let ctx0 = Context::new(0, ContextFlags::default()).unwrap();
    let ctx1 = Context::new(1, ContextFlags::default()).unwrap();

    // Test enabling peer access between contexts
    ctx0.make_current()?;
    let can_access = ctx0.can_access_peer(1)?;

    if can_access {
        assert!(ctx0.enable_peer_access(1).is_ok());
        assert!(ctx0.disable_peer_access(1).is_ok());
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_context_l2_cache_config() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();

    // Test L2 cache configuration
    let config = L2CacheConfig {
        reserved_size: 1024 * 1024, // 1MB
                                    // other fields...
    };

    // Would test L2 cache configuration if supported
    // ctx.set_l2_cache_config(config).unwrap();
}

#[cfg(cuda_mock)]
#[test]
fn test_context_creation_error_handling() {
    // Test creating context on invalid device
    let result = Context::new(999, ContextFlags::default());
    assert!(result.is_err());
}

#[cfg(cuda_mock)]
#[test]
fn test_context_profiler_control() {
    let ctx = Context::new(0, ContextFlags::default()).unwrap();

    // Test profiler control
    assert!(ctx.profiler_start().is_ok());
    assert!(ctx.profiler_stop().is_ok());
}
