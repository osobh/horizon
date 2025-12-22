//! CUDA device handling tests following cuda.md guidelines
//!
//! These tests validate proper GPU device management and cudarc API usage

use anyhow::Result;
use std::sync::Arc;

/// Test CUDA device creation without double Arc wrapping
/// Following cuda.md: Validate GPU device initialization
#[test]
fn test_cuda_device_creation() -> Result<()> {
    // This test will fail until we have proper CUDA setup
    // Following cuda.md: Always test GPU availability before running tests

    if !cuda_is_available() {
        println!("⚠️  CUDA not available, skipping device test");
        return Ok(());
    }

    // Test that device creation doesn't double-wrap Arc
    let device = cudarc::driver::CudaDevice::new(0)?;

    // Verify device properties
    assert_eq!(device.id(), 0);

    // Test that we can wrap in Arc if needed
    let _arc_device: Arc<cudarc::driver::CudaDevice> = Arc::new(device);

    Ok(())
}

/// Test cudarc trait imports are available
/// These traits must be in scope for device operations
#[test]
fn test_cudarc_trait_availability() -> Result<()> {
    if !cuda_is_available() {
        println!("⚠️  CUDA not available, skipping trait test");
        return Ok(());
    }

    // Following cuda.md: Import required traits explicitly
    use cudarc::driver::{DevicePtr, DeviceSlice};

    let device = cudarc::driver::CudaDevice::new(0)?;

    // Test that we can allocate memory and use traits
    let data = vec![1u32, 2, 3, 4];
    let gpu_slice = device.htod_copy(data)?;

    // These methods should be available via traits
    let _len = gpu_slice.len(); // DeviceSlice trait
    let _ptr = gpu_slice.device_ptr(); // DevicePtr trait

    Ok(())
}

/// Test GPU memory operations for streaming
/// Following cuda.md: Test GPU memory management patterns
#[test]
fn test_gpu_memory_operations() -> Result<()> {
    if !cuda_is_available() {
        println!("⚠️  CUDA not available, skipping memory test");
        return Ok(());
    }

    use cudarc::driver::{DevicePtr, DeviceSlice};

    let device = cudarc::driver::CudaDevice::new(0)?;

    // Test allocation and copy operations
    let host_data = vec![0u8; 1024];
    let gpu_buffer = device.htod_copy(host_data.clone())?;

    // Test synchronous copy back
    let result: Vec<u8> = device.dtoh_sync_copy(&gpu_buffer)?;
    assert_eq!(result.len(), host_data.len());

    Ok(())
}

/// Test benchmark device creation patterns
/// This validates our benchmark modules can create devices correctly
#[test]
fn test_benchmark_device_patterns() -> Result<()> {
    if !cuda_is_available() {
        println!("⚠️  CUDA not available, skipping benchmark device test");
        return Ok(());
    }

    // Following our benchmark pattern - device should NOT be double-wrapped
    let device = cudarc::driver::CudaDevice::new(0)?;

    // Verify this is the correct type for our structs
    struct TestBenchmark {
        device: Arc<cudarc::driver::CudaDevice>,
    }

    let benchmark = TestBenchmark {
        device: Arc::new(device),
    };

    assert_eq!(benchmark.device.id(), 0);

    Ok(())
}

/// Helper function to check CUDA availability
/// Following cuda.md: Always validate GPU environment before tests
fn cuda_is_available() -> bool {
    cudarc::driver::CudaDevice::new(0).is_ok()
}
