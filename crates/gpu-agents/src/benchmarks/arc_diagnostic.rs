//! Diagnostic test to understand the Arc issue
//! Following TDD from rust.md

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;
    
    #[test]
    fn diagnose_cuda_device_type() {
        // Let's check what CudaContext::new actually returns
        match CudaContext::new(0) {
            Ok(ctx) => {
                // Print the type information
                println!("Type of ctx: {}", std::any::type_name_of_val(&ctx));

                // In 0.18.1, CudaContext::new returns Arc<CudaContext> directly
                let _: std::sync::Arc<CudaContext> = ctx;
                println!("✓ CudaContext::new returns Arc<CudaContext>");
            }
            Err(_) => {
                println!("No GPU available for testing");
            }
        }
    }
    
    #[test]
    fn diagnose_arc_creation() {
        use std::sync::Arc;

        if let Ok(ctx) = CudaContext::new(0) {
            println!("Type from new: {}", std::any::type_name_of_val(&ctx));

            // In 0.18.1, CudaContext::new already returns Arc<CudaContext>
            // No need to wrap again
            let _: Arc<CudaContext> = ctx;
            println!("✓ CudaContext::new returns Arc<CudaContext> directly");
        }
    }
    
    #[test]
    fn diagnose_std_sync_arc() {
        if let Ok(ctx) = CudaContext::new(0) {
            // In 0.18.1, CudaContext::new returns Arc<CudaContext> directly
            println!("Type from new: {}", std::any::type_name_of_val(&ctx));

            // Check the type - no wrapping needed
            let _: std::sync::Arc<CudaContext> = ctx;
            println!("✓ CudaContext::new returns Arc<CudaContext> directly");
        }
    }
}