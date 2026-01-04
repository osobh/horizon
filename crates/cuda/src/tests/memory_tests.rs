//! Tests for CUDA memory management

use crate::error::CudaResult;
use crate::memory::*;

#[test]
fn test_memory_pool_creation() {
    let pool = MemoryPool::new();
    assert_eq!(pool.total_allocated(), 0);
    assert_eq!(pool.peak_usage(), 0);
    assert!(pool.available_memory() > 0);
}

#[cfg(cuda_mock)]
#[test]
fn test_device_memory_allocation() {
    let mut mem = DeviceMemory::<i32>::allocate(1024).unwrap();
    assert_eq!(mem.size(), 1024);
    assert_eq!(mem.size_bytes(), 1024 * std::mem::size_of::<i32>());
    assert!(!mem.ptr().is_null());
}

#[cfg(cuda_mock)]
#[test]
fn test_device_memory_zero() {
    let mut mem = DeviceMemory::<f32>::allocate(512).unwrap();
    mem.fill_zero().unwrap();

    // In mock mode, operation should succeed
    assert_eq!(mem.size(), 512);
}

#[cfg(cuda_mock)]
#[test]
fn test_device_memory_copy_from_host() {
    let host_data: Vec<i32> = (0..100).collect();
    let mut device_mem = DeviceMemory::<i32>::allocate(100).unwrap();

    device_mem.copy_from_host(&host_data).unwrap();

    // Verify copy operation succeeded
    assert_eq!(device_mem.size(), 100);
}

#[cfg(cuda_mock)]
#[test]
fn test_device_memory_copy_to_host() {
    let mut device_mem = DeviceMemory::<f64>::allocate(50).unwrap();
    device_mem.fill_zero().unwrap();

    let host_data = device_mem.copy_to_host().unwrap();
    assert_eq!(host_data.len(), 50);
}

#[cfg(cuda_mock)]
#[test]
fn test_device_memory_copy_device_to_device() {
    let mut src_mem = DeviceMemory::<u32>::allocate(200).unwrap();
    let mut dst_mem = DeviceMemory::<u32>::allocate(200).unwrap();

    src_mem.fill_zero().unwrap();
    dst_mem.copy_from_device(&src_mem)?;

    assert_eq!(src_mem.size(), dst_mem.size());
}

#[cfg(cuda_mock)]
#[test]
fn test_pinned_memory_allocation() {
    let pinned = PinnedMemory::<i64>::allocate(1000).unwrap();
    assert_eq!(pinned.size(), 1000);
    assert!(!pinned.ptr().is_null());
    assert!(pinned.is_pinned());
}

#[cfg(cuda_mock)]
#[test]
fn test_pinned_memory_slice_access() {
    let mut pinned = PinnedMemory::<i32>::allocate(10).unwrap();

    // Write data through slice
    {
        let slice = pinned.as_mut_slice();
        for (i, item) in slice.iter_mut().enumerate() {
            *item = i as i32;
        }
    }

    // Read data back
    {
        let slice = pinned.as_slice();
        for (i, &item) in slice.iter().enumerate() {
            assert_eq!(item, i as i32);
        }
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_managed_memory_allocation() {
    let managed = ManagedMemory::<f32>::allocate(2048).unwrap();
    assert_eq!(managed.size(), 2048);
    assert!(!managed.ptr().is_null());
    assert!(managed.is_accessible_from_host());
    assert!(managed.is_accessible_from_device());
}

#[cfg(cuda_mock)]
#[test]
fn test_managed_memory_slice_access() {
    let mut managed = ManagedMemory::<u64>::allocate(100).unwrap();

    // Access from host
    {
        let slice = managed.as_mut_slice();
        for (i, item) in slice.iter_mut().enumerate() {
            *item = (i * i) as u64;
        }
    }

    // Verify data
    {
        let slice = managed.as_slice();
        for (i, &item) in slice.iter().enumerate() {
            assert_eq!(item, (i * i) as u64);
        }
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_pool_allocate_deallocate() {
    let mut pool = MemoryPool::new();

    // Allocate multiple chunks
    let mem1 = pool.allocate::<i32>(1024).unwrap();
    let mem2 = pool.allocate::<f32>(512)?;
    let mem3 = pool.allocate::<u8>(2048)?;

    assert_eq!(pool.allocation_count(), 3);
    assert!(pool.total_allocated() > 0);

    // Memory should be automatically deallocated when dropped
    drop(mem1);
    drop(mem2);
    drop(mem3);
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_pool_statistics() {
    let mut pool = MemoryPool::new();

    let initial_allocated = pool.total_allocated();
    let initial_peak = pool.peak_usage();

    {
        let _mem = pool.allocate::<i64>(1000)?;

        assert!(pool.total_allocated() > initial_allocated);
        assert!(pool.peak_usage() >= pool.total_allocated());
    }

    // After deallocation, total should decrease but peak remains
    assert_eq!(pool.total_allocated(), initial_allocated);
    assert!(pool.peak_usage() >= initial_peak);
}

#[test]
fn test_memory_alignment() {
    use std::mem::align_of;

    // Test alignment for different types
    assert_eq!(DeviceMemory::<u8>::alignment(), align_of::<u8>());
    assert_eq!(DeviceMemory::<i32>::alignment(), align_of::<i32>());
    assert_eq!(DeviceMemory::<f64>::alignment(), align_of::<f64>());
    assert_eq!(
        DeviceMemory::<[f32; 4]>::alignment(),
        align_of::<[f32; 4]>()
    );
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_info_query() {
    let info = get_memory_info().unwrap();

    assert!(info.total_memory > 0);
    assert!(info.free_memory > 0);
    assert!(info.free_memory <= info.total_memory);
    assert!(info.used_memory >= 0);
    assert_eq!(info.total_memory, info.free_memory + info.used_memory);
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_bandwidth_test() {
    let bandwidth = measure_memory_bandwidth(1024 * 1024).unwrap(); // 1MB test

    assert!(bandwidth.host_to_device_gbps > 0.0);
    assert!(bandwidth.device_to_host_gbps > 0.0);
    assert!(bandwidth.device_to_device_gbps > 0.0);

    // Device-to-device should typically be fastest
    assert!(bandwidth.device_to_device_gbps >= bandwidth.host_to_device_gbps);
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_pattern_fill() {
    let mut mem = DeviceMemory::<u32>::allocate(1000).unwrap();

    let pattern = 0xDEADBEEF;
    mem.fill_pattern(pattern).unwrap();

    // Verify pattern was set
    let host_data = mem.copy_to_host()?;
    for &value in &host_data {
        assert_eq!(value, pattern);
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_prefetch() {
    let managed = ManagedMemory::<i32>::allocate(2048).unwrap();

    // Test prefetching to device
    managed.prefetch_to_device(0).unwrap();

    // Test prefetching to host
    managed.prefetch_to_host()?;
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_advice() {
    let managed = ManagedMemory::<f32>::allocate(1024).unwrap();

    // Test setting memory advice
    managed.set_advice(MemoryAdvice::PreferredLocation, 0)?;
    managed.set_advice(MemoryAdvice::AccessedBy, 0)?;
    managed.set_advice(MemoryAdvice::ReadMostly, 0)?;
}

#[test]
fn test_memory_type_sizes() {
    // Verify memory types have expected sizes
    assert_eq!(
        std::mem::size_of::<DevicePtr<u8>>(),
        std::mem::size_of::<*mut u8>()
    );
    assert_eq!(
        std::mem::size_of::<DevicePtr<i32>>(),
        std::mem::size_of::<*mut i32>()
    );
    assert_eq!(
        std::mem::size_of::<DevicePtr<f64>>(),
        std::mem::size_of::<*mut f64>()
    );
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_allocation_alignment() {
    // Test allocation alignment for various sizes
    let sizes = vec![1, 16, 64, 256, 1024, 4096];

    for size in sizes {
        let mem = DeviceMemory::<u8>::allocate(size)?;
        let ptr = mem.ptr() as usize;

        // Should be aligned to at least the type alignment
        assert_eq!(ptr % std::mem::align_of::<u8>(), 0);

        // For CUDA, should typically be aligned to at least 256 bytes
        if size >= 256 {
            assert_eq!(ptr % 256, 0);
        }
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_error_handling() {
    // Test allocation failure scenarios
    let result = DeviceMemory::<u8>::allocate(usize::MAX);
    assert!(result.is_err());

    // Test invalid size
    let result = DeviceMemory::<i32>::allocate(0);
    assert!(result.is_err());
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_copy_size_mismatch() {
    let mut device_mem = DeviceMemory::<i32>::allocate(100).unwrap();
    let host_data: Vec<i32> = (0..50).collect(); // Different size

    let result = device_mem.copy_from_host(&host_data);
    assert!(result.is_err()); // Should fail due to size mismatch
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_pool_fragmentation() {
    let mut pool = MemoryPool::new();

    // Allocate and deallocate in a pattern that could cause fragmentation
    let mut allocations = Vec::new();

    for i in 0..10 {
        let size = (i + 1) * 100;
        let mem = pool.allocate::<u8>(size)?;
        allocations.push(mem);
    }

    // Free every other allocation
    for i in (0..allocations.len()).step_by(2) {
        allocations[i] = pool.allocate::<u8>(1).unwrap(); // Replace with smaller allocation
    }

    // Pool should still function correctly
    let _mem = pool.allocate::<u8>(5000).unwrap();
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_stream_operations() {
    let stream = crate::stream::Stream::new(crate::stream::StreamFlags::default()).unwrap();

    let mut device_mem = DeviceMemory::<f32>::allocate(1000).unwrap();
    let host_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();

    // Asynchronous copy operations
    device_mem.copy_from_host_async(&host_data, &stream)?;
    stream.synchronize()?;

    let result = device_mem.copy_to_host_async(&stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(result.len(), 1000);
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_concurrent_access() {
    use std::sync::Arc;
    use std::thread;

    let managed = Arc::new(ManagedMemory::<i32>::allocate(1000).unwrap());
    let mut handles = Vec::new();

    // Multiple threads accessing managed memory
    for i in 0..5 {
        let managed_clone = managed.clone();
        let handle = thread::spawn(move || {
            let slice = managed_clone.as_slice();
            assert_eq!(slice.len(), 1000);
            i
        });
        handles.push(handle);
    }

    for handle in handles {
        let _result = handle.join().unwrap();
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_performance_hints() {
    let device_mem = DeviceMemory::<f64>::allocate(2048).unwrap();

    // Test performance hints
    device_mem.set_cache_preference(CachePreference::L1)?;
    device_mem.set_cache_preference(CachePreference::Shared)?;
    device_mem
        .set_cache_preference(CachePreference::None)
        .unwrap();
}

#[test]
fn test_device_pointer_arithmetic() {
    let base_ptr = 0x1000 as *mut u8;
    let device_ptr = DevicePtr::new(base_ptr);

    let offset_ptr = device_ptr.offset(100);
    assert_eq!(offset_ptr.as_ptr() as usize, 0x1000 + 100);

    let byte_ptr = device_ptr.as_bytes();
    assert_eq!(byte_ptr.as_ptr(), base_ptr);
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_pool_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let pool = Arc::new(MemoryPool::new());
    let mut handles = Vec::new();

    // Multiple threads allocating from the same pool
    for i in 0..10 {
        let pool_clone = pool.clone();
        let handle = thread::spawn(move || {
            let _mem = pool_clone.allocate::<u32>((i + 1) * 100).unwrap();
            // Memory is dropped at end of scope
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Pool should be in a consistent state
    assert_eq!(pool.total_allocated(), 0);
}
