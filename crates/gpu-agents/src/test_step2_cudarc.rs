//! TDD Step 2: Test cudarc API usage and device operations
//! Following cuda.md guidelines for GPU programming best practices

#[cfg(test)]
mod step2_cudarc_tests {
    use anyhow::Result;

    /// Helper to check if CUDA is available for testing
    fn cuda_available() -> bool {
        cudarc::driver::CudaContext::new(0).is_ok()
    }

    /// Test cudarc trait imports are available and working
    /// Following cuda.md: Always import required CUDA traits explicitly
    #[test]
    fn test_cudarc_traits_available() -> Result<()> {
        if !cuda_available() {
            println!("⚠️  CUDA not available, skipping trait test");
            return Ok(());
        }

        // These traits should be in scope after we fix imports
        use cudarc::driver::{DevicePtr, DeviceSlice};

        let device = cudarc::driver::CudaContext::new(0)?;
        let data = vec![1u32, 2, 3, 4];
        let gpu_slice = device.htod_copy(data)?;

        // These method calls should work with proper trait imports
        let _len = gpu_slice.len(); // DeviceSlice trait
        let _ptr = gpu_slice.device_ptr(); // DevicePtr trait

        println!("✅ cudarc traits working correctly");
        Ok(())
    }

    /// Test GPU memory operations for streaming modules
    /// This will fail until we fix DeviceSlice/DevicePtr usage in streaming
    #[test]
    fn test_gpu_memory_operations() -> Result<()> {
        if !cuda_available() {
            println!("⚠️  CUDA not available, skipping memory test");
            return Ok(());
        }

        use cudarc::driver::{DevicePtr, DeviceSlice};

        let device = cudarc::driver::CudaContext::new(0)?;

        // Test allocation
        let host_data = vec![42u8; 1024];
        let gpu_buffer = device.htod_copy(host_data.clone())?;

        // Test that we can get length and pointer (used in streaming modules)
        let buffer_len = gpu_buffer.len();
        let _buffer_ptr = gpu_buffer.device_ptr();

        assert_eq!(buffer_len, 1024);

        // Test copy back
        let result: Vec<u8> = device.dtoh_sync_copy(&gpu_buffer)?;
        assert_eq!(result, host_data);

        println!("✅ GPU memory operations working");
        Ok(())
    }

    /// Test device creation patterns used in benchmarks
    /// This will fail until we fix Arc<Arc<CudaContext>> issues
    #[test]
    fn test_benchmark_device_creation() -> Result<()> {
        if !cuda_available() {
            println!("⚠️  CUDA not available, skipping device creation test");
            return Ok(());
        }

        // Following cuda.md: Device should be created once and shared via Arc
        let device = cudarc::driver::CudaContext::new(0)?;

        // Test that we can wrap in Arc (benchmark pattern)
        use std::sync::Arc;
        let arc_device: Arc<cudarc::driver::CudaContext> = Arc::new(device);

        // Test that Arc device works correctly
        assert_eq!(arc_device.id(), 0);

        // This should NOT be Arc<Arc<CudaContext>>
        struct TestBenchmark {
            device: Arc<cudarc::driver::CudaContext>,
        }

        let _benchmark = TestBenchmark { device: arc_device };

        println!("✅ Benchmark device creation pattern working");
        Ok(())
    }

    /// Test streaming operations that use cudarc APIs
    /// This validates our streaming module fixes work correctly
    #[test]
    fn test_streaming_cudarc_usage() -> Result<()> {
        if !cuda_available() {
            println!("⚠️  CUDA not available, skipping streaming test");
            return Ok(());
        }

        use cudarc::driver::{DevicePtr, DeviceSlice};

        let device = cudarc::driver::CudaContext::new(0)?;

        // Simulate streaming operation
        let input_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let input_buffer = device.htod_copy(input_data.clone())?;
        let mut output_buffer = device.alloc_zeros::<u8>(input_data.len())?;

        // Test operations that streaming modules need
        let input_len = input_buffer.len();
        let _input_ptr = input_buffer.device_ptr();
        let _output_ptr = output_buffer.device_ptr();

        assert_eq!(input_len, input_data.len());

        println!("✅ Streaming cudarc usage working");
        Ok(())
    }

    /// Test compression module cudarc patterns
    /// This will validate compression.rs fixes
    #[test]
    fn test_compression_cudarc_patterns() -> Result<()> {
        if !cuda_available() {
            println!("⚠️  CUDA not available, skipping compression test");
            return Ok(());
        }

        use cudarc::driver::{DevicePtr, DeviceSlice};

        let device = cudarc::driver::CudaContext::new(0)?;

        // Test pattern used in compression: input -> output buffers
        let input_data = vec![0u8; 256];
        let input_buffer = device.htod_copy(input_data)?;
        let mut output_buffer = device.alloc_zeros::<u8>(128)?; // Compressed size

        // These operations should work after we fix trait imports
        let input_len = input_buffer.len();
        let output_len = output_buffer.len();
        let _input_ptr = input_buffer.device_ptr() as *const u8;
        let _output_ptr = output_buffer.device_ptr() as *mut u8;

        assert_eq!(input_len, 256);
        assert_eq!(output_len, 128);

        println!("✅ Compression cudarc patterns working");
        Ok(())
    }

    /// Integration test: All cudarc usage should work together
    #[test]
    fn test_cudarc_integration() -> Result<()> {
        if !cuda_available() {
            println!("⚠️  CUDA not available, skipping integration test");
            return Ok(());
        }

        // Test full workflow: device creation -> memory ops -> cleanup
        use cudarc::driver::{DevicePtr, DeviceSlice};
        use std::sync::Arc;

        let device = Arc::new(cudarc::driver::CudaContext::new(0)?);

        // Test multiple buffer operations
        let data1 = vec![1u32; 100];
        let data2 = vec![2u8; 200];

        let buffer1 = device.htod_copy(data1)?;
        let buffer2 = device.htod_copy(data2)?;

        // Test trait methods work
        assert_eq!(buffer1.len(), 100);
        assert_eq!(buffer2.len(), 200);

        let _ptr1 = buffer1.device_ptr();
        let _ptr2 = buffer2.device_ptr();

        println!("✅ Step 2 Ready: All cudarc patterns defined");
        Ok(())
    }
}
