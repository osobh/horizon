//! Diagnostic test to understand the Arc issue
//! Following TDD from rust.md

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaDevice;
    
    #[test]
    fn diagnose_cuda_device_type() {
        // Let's check what CudaDevice::new actually returns
        match CudaDevice::new(0) {
            Ok(device) => {
                // Print the type information
                println!("Type of device: {}", std::any::type_name_of_val(&device));
                
                // Try to assign to check type
                let _: CudaDevice = device;
                println!("✓ CudaDevice::new returns CudaDevice");
            }
            Err(_) => {
                println!("No GPU available for testing");
            }
        }
    }
    
    #[test]
    fn diagnose_arc_creation() {
        use std::sync::Arc;
        
        if let Ok(device) = CudaDevice::new(0) {
            println!("Type before Arc: {}", std::any::type_name_of_val(&device));
            
            let arc_device = Arc::new(device);
            println!("Type after Arc::new: {}", std::any::type_name_of_val(&arc_device));
            
            // This should be Arc<CudaDevice>
            let _: Arc<CudaDevice> = arc_device;
            println!("✓ Successfully created Arc<CudaDevice>");
        }
    }
    
    #[test]
    fn diagnose_std_sync_arc() {
        if let Ok(device) = CudaDevice::new(0) {
            // Using fully qualified path
            let arc_device = std::sync::Arc::new(device);
            println!("Type with std::sync::Arc: {}", std::any::type_name_of_val(&arc_device));
            
            // Check the type
            let _: std::sync::Arc<CudaDevice> = arc_device;
            println!("✓ std::sync::Arc creates correct type");
        }
    }
}