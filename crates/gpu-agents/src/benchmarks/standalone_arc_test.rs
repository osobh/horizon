//! Standalone test to isolate Arc issue
//! Following TDD from rust.md

#[cfg(test)]
mod tests {
    #[test]
    fn test_arc_wrapping() {
        // Simplified version to understand the issue
        use std::sync::Arc;
        
        // Test 1: Basic Arc wrapping
        {
            struct Device {
                id: i32,
            }
            
            let device = Device { id: 0 };
            let arc_device: Arc<Device> = Arc::new(device);
            
            // Function that takes Arc<Device>
            fn use_device(d: Arc<Device>) {
                println!("Using device: {}", d.id);
            }
            
            use_device(arc_device);
        }
        
        // Test 2: With Result
        {
            struct Device {
                id: i32,
            }
            
            impl Device {
                fn new(id: i32) -> Result<Self, &'static str> {
                    Ok(Device { id })
                }
            }
            
            let device = Device::new(0)?;
            let arc_device = Arc::new(device);
            
            fn use_device(d: Arc<Device>) {
                println!("Using device: {}", d.id);
            }
            
            use_device(arc_device);
        }
    }
    
    #[test]
    fn test_cudarc_pattern() {
        // Skip if no GPU
        if cudarc::driver::CudaDevice::new(0).is_err() {
            return;
        }
        
        use std::sync::Arc;
        use cudarc::driver::CudaDevice;
        
        // Pattern 1: Direct
        let device1 = CudaDevice::new(0)?;
        let arc1 = Arc::new(device1);
        let _: Arc<CudaDevice> = arc1;
        
        // Pattern 2: With intermediate variable
        let device2 = CudaDevice::new(0).unwrap();
        let arc2: Arc<CudaDevice> = Arc::new(device2);
        let _: Arc<CudaDevice> = arc2;
        
        // Pattern 3: Inline
        let arc3: Arc<CudaDevice> = Arc::new(CudaDevice::new(0).unwrap());
        let _: Arc<CudaDevice> = arc3;
    }
}