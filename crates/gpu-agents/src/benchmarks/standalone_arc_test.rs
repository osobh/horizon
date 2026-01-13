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

            let device = Device::new(0).unwrap();
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
        if cudarc::driver::CudaContext::new(0).is_err() {
            return;
        }

        use std::sync::Arc;
        use cudarc::driver::CudaContext;

        // Pattern 1: In 0.18.1, CudaContext::new returns Arc<CudaContext>
        let ctx1 = CudaContext::new(0).unwrap();
        let _: Arc<CudaContext> = ctx1;

        // Pattern 2: Direct assignment
        let ctx2 = CudaContext::new(0).unwrap();
        let _: Arc<CudaContext> = ctx2;

        // Pattern 3: No double wrapping needed
        let ctx3 = CudaContext::new(0).unwrap();
        let _: Arc<CudaContext> = ctx3;
    }
}
