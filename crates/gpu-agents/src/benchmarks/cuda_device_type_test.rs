//! Test to determine actual return type of CudaDevice::new
//! Following TDD approach from rust.md

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaDevice;
    use std::sync::Arc;
    
    #[test]
    fn test_cuda_device_new_return_type() {
        // Skip if no GPU
        match CudaDevice::new(0) {
            Ok(device) => {
                // Let's check what type device actually is
                // If this compiles, device is CudaDevice
                let _: CudaDevice = device;
                println!("✓ CudaDevice::new returns CudaDevice (not Arc<CudaDevice>)");
            }
            Err(_) => {
                println!("Skipping - no GPU available");
            }
        }
    }
    
    #[test]
    fn test_arc_wrapping_pattern() {
        // Skip if no GPU
        if let Ok(device) = CudaDevice::new(0) {
            // Correct pattern: device is CudaDevice, wrap it in Arc
            let arc_device: Arc<CudaDevice> = Arc::new(device);
            
            // Verify the type
            fn expect_arc_cuda_device(_: Arc<CudaDevice>) {
                println!("✓ Received Arc<CudaDevice> as expected");
            }
            
            expect_arc_cuda_device(arc_device);
        }
    }
}