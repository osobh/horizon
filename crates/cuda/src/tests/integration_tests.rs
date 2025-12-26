//! Integration tests for CUDA crate

use crate::*;
use std::sync::Arc;

#[cfg(cuda_mock)]
#[test]
fn test_full_cuda_workflow() {
    // Initialize CUDA
    assert!(init().is_ok());

    // Get toolkit info
    let toolkit_info = toolkit_info()?;
    assert!(toolkit_info.device_count > 0);

    // Create context
    let ctx = context::Context::new(0, context::ContextFlags::default())?;

    // Create stream
    let stream = stream::Stream::new(stream::StreamFlags::default()).unwrap();

    // Allocate memory
    let mut device_mem = memory::DeviceMemory::<f32>::allocate(1024).unwrap();
    let host_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();

    // Copy data to device
    device_mem.copy_from_host(&host_data).unwrap();

    // Create and load kernel
    let kernel_source = kernel::KernelSource::CudaC(
        r#"
        extern "C" __global__ void scale_array(float* data, float scale, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] *= scale;
            }
        }
    "#
        .to_string(),
    );

    let module = kernel::KernelModule::new(kernel_source).unwrap();
    let kernel_fn = module.get_kernel("scale_array").unwrap();

    // Launch kernel
    let config = kernel::LaunchConfig {
        grid_dim: (4, 1, 1),
        block_dim: (256, 1, 1),
        shared_memory_bytes: 0,
        stream: Some(stream.clone()),
    };

    let args = vec![
        kernel::KernelArg::Pointer(device_mem.ptr() as *mut u8),
        kernel::KernelArg::Scalar(2.0f32),
        kernel::KernelArg::Scalar(1024i32),
    ];

    kernel_fn.launch(&config, args).unwrap();

    // Synchronize and copy back
    stream.synchronize().unwrap();
    let result = device_mem.copy_to_host().unwrap();

    // Verify results
    for (i, &value) in result.iter().enumerate() {
        assert_eq!(value, (i as f32) * 2.0);
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_multi_device_workflow() {
    let toolkit_info = toolkit_info().unwrap();

    if toolkit_info.device_count < 2 {
        return; // Skip test if not enough devices
    }

    // Create contexts for different devices
    let ctx0 = context::Context::new(0, context::ContextFlags::default())?;
    let ctx1 = context::Context::new(1, context::ContextFlags::default())?;

    // Allocate memory on both devices
    ctx0.make_current().unwrap();
    let mem0 = memory::DeviceMemory::<i32>::allocate(512).unwrap();

    ctx1.make_current().unwrap();
    let mem1 = memory::DeviceMemory::<i32>::allocate(512).unwrap();

    // Both allocations should succeed
    assert_eq!(mem0.size(), 512);
    assert_eq!(mem1.size(), 512);
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_and_compute_pipeline() {
    // Create multiple streams for overlapping operations
    let copy_stream = stream::Stream::new(stream::StreamFlags::default()).unwrap();
    let compute_stream = stream::Stream::new(stream::StreamFlags::default()).unwrap();

    // Allocate memory
    let mut input_mem = memory::DeviceMemory::<f32>::allocate(2048)?;
    let mut output_mem = memory::DeviceMemory::<f32>::allocate(2048)?;
    let input_data: Vec<f32> = (0..2048).map(|i| (i as f32).sin()).collect();

    // Copy input data asynchronously
    input_mem
        .copy_from_host_async(&input_data, &copy_stream)
        .unwrap();

    // Create event to synchronize between streams
    let copy_done = stream::Event::new().unwrap();
    copy_done.record(&copy_stream).unwrap();
    compute_stream.wait_event(&copy_done).unwrap();

    // Launch computation on compute stream
    let kernel_source = kernel::KernelSource::CudaC(
        r#"
        extern "C" __global__ void square_array(float* input, float* output, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = input[idx] * input[idx];
            }
        }
    "#
        .to_string(),
    );

    let module = kernel::KernelModule::new(kernel_source).unwrap();
    let kernel_fn = module.get_kernel("square_array").unwrap();

    let config = kernel::LaunchConfig {
        grid_dim: (8, 1, 1),
        block_dim: (256, 1, 1),
        shared_memory_bytes: 0,
        stream: Some(compute_stream.clone()),
    };

    let args = vec![
        kernel::KernelArg::Pointer(input_mem.ptr() as *mut u8),
        kernel::KernelArg::Pointer(output_mem.ptr() as *mut u8),
        kernel::KernelArg::Scalar(2048i32),
    ];

    kernel_fn.launch(&config, args).unwrap();

    // Copy result back
    compute_stream.synchronize().unwrap();
    let result = output_mem.copy_to_host().unwrap();

    // Verify results
    for (i, &value) in result.iter().enumerate() {
        let expected = input_data[i] * input_data[i];
        assert!((value - expected).abs() < 1e-6);
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_error_recovery_workflow() {
    // Initialize CUDA
    assert!(init().is_ok());

    // Try to create context on invalid device
    let result = context::Context::new(999, context::ContextFlags::default());
    assert!(result.is_err());

    // Should still be able to create valid context
    let ctx = context::Context::new(0, context::ContextFlags::default())?;

    // Try invalid memory allocation
    let result = memory::DeviceMemory::<u8>::allocate(usize::MAX);
    assert!(result.is_err());

    // Should still be able to allocate valid memory
    let _mem = memory::DeviceMemory::<u8>::allocate(1024).unwrap();
}

#[cfg(cuda_mock)]
#[test]
fn test_resource_cleanup() {
    let initial_memory = memory::get_memory_info().unwrap().used_memory;

    {
        // Allocate resources in a scope
        let _ctx = context::Context::new(0, context::ContextFlags::default())?;
        let _stream = stream::Stream::new(stream::StreamFlags::default())?;
        let _mem1 = memory::DeviceMemory::<i32>::allocate(1000)?;
        let _mem2 = memory::DeviceMemory::<f32>::allocate(2000)?;

        // Resources should be allocated
        let used_memory = memory::get_memory_info().unwrap().used_memory;
        assert!(used_memory > initial_memory);
    } // Resources dropped here

    // After cleanup, memory usage should return to initial state
    // Note: In real CUDA, this might not be immediate due to caching
    let final_memory = memory::get_memory_info().unwrap().used_memory;
    assert!(final_memory <= initial_memory + 1024); // Allow for small overhead
}

#[cfg(cuda_mock)]
#[test]
fn test_concurrent_operations() {
    use std::thread;

    let mut handles = Vec::new();

    for thread_id in 0..4 {
        let handle = thread::spawn(move || {
            // Each thread creates its own context
            let ctx = context::Context::new(0, context::ContextFlags::default())?;
            let stream = stream::Stream::new(stream::StreamFlags::default())?;

            // Allocate and process data
            let mut mem = memory::DeviceMemory::<u32>::allocate(1000).unwrap();
            let data: Vec<u32> = (0..1000).map(|i| (thread_id * 1000 + i) as u32).collect();

            mem.copy_from_host(&data).unwrap();

            // Fill with pattern
            mem.fill_pattern(thread_id as u32).unwrap();

            stream.synchronize().unwrap();
            let result = mem.copy_to_host().unwrap();

            // Verify pattern
            for &value in &result {
                assert_eq!(value, thread_id as u32);
            }

            thread_id
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        let _result = handle.join().unwrap();
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_performance_measurement() {
    let start_event = stream::Event::new().unwrap();
    let end_event = stream::Event::new().unwrap();
    let stream = stream::Stream::new(stream::StreamFlags::default()).unwrap();

    // Start timing
    start_event.record(&stream)?;

    // Perform multiple operations
    for i in 0..10 {
        let mem = memory::DeviceMemory::<f32>::allocate(1000)?;
        mem.fill_zero().unwrap();

        let data: Vec<f32> = (0..1000).map(|j| (i * 1000 + j) as f32).collect();
        mem.copy_from_host(&data).unwrap();
    }

    // End timing
    end_event.record(&stream).unwrap();
    stream.synchronize().unwrap();

    let elapsed_ms = start_event.elapsed_time(&end_event).unwrap();
    assert!(elapsed_ms >= 0.0);
}

#[cfg(cuda_mock)]
#[test]
fn test_memory_bandwidth_measurement() {
    let sizes = vec![1024, 10240, 102400, 1024000]; // Various sizes

    for size in sizes {
        let bandwidth = memory::measure_memory_bandwidth(size).unwrap();

        assert!(bandwidth.host_to_device_gbps > 0.0);
        assert!(bandwidth.device_to_host_gbps > 0.0);
        assert!(bandwidth.device_to_device_gbps > 0.0);

        // Bandwidth should be reasonable (> 1 GB/s in mock mode)
        assert!(bandwidth.device_to_device_gbps >= 1.0);
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_unified_memory_workflow() {
    let managed = memory::ManagedMemory::<f64>::allocate(1000).unwrap();

    // Initialize data on host
    {
        let slice = managed.as_mut_slice();
        for (i, item) in slice.iter_mut().enumerate() {
            *item = (i as f64).sqrt();
        }
    }

    // Process on device (would be actual kernel in real scenario)
    managed.prefetch_to_device(0).unwrap();

    // Simulate kernel processing
    let kernel_source = kernel::KernelSource::CudaC(
        r#"
        extern "C" __global__ void process_managed(double* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = data[idx] * data[idx]; // Square the values
            }
        }
    "#
        .to_string(),
    );

    let module = kernel::KernelModule::new(kernel_source).unwrap();
    let kernel_fn = module.get_kernel("process_managed").unwrap();

    let config = kernel::LaunchConfig::default();
    let args = vec![
        kernel::KernelArg::Pointer(managed.ptr() as *mut u8),
        kernel::KernelArg::Scalar(1000i32),
    ];

    kernel_fn.launch(&config, args)?;

    // Access results on host
    managed.prefetch_to_host().unwrap();

    {
        let slice = managed.as_slice();
        for (i, &value) in slice.iter().enumerate() {
            let expected = (i as f64).sqrt(); // Original value
            let squared = expected * expected; // After kernel processing
            assert!((value - squared).abs() < 1e-10);
        }
    }
}

#[test]
fn test_system_integration() {
    // Test basic system integration without GPU operations
    let result = detection::initialize_cuda();

    #[cfg(cuda_mock)]
    {
        assert!(result.is_ok());

        let info = detection::get_toolkit_info()?;
        assert!(info.is_mock);
        assert_eq!(info.device_count, 4);
    }

    #[cfg(not(cuda_mock))]
    {
        // In real mode, might not have CUDA available
        // Just test that it doesn't panic
        let _ = result;
    }
}
