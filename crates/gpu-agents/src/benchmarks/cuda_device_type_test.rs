//! Test to determine actual return type of CudaContext::new
//! Following TDD approach from rust.md

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    #[test]
    fn test_cuda_context_new_return_type() {
        // Skip if no GPU
        match CudaContext::new(0) {
            Ok(ctx) => {
                // In 0.18.1, CudaContext::new returns Arc<CudaContext>
                let _: Arc<CudaContext> = ctx;
                println!("✓ CudaContext::new returns Arc<CudaContext>");
            }
            Err(_) => {
                println!("Skipping - no GPU available");
            }
        }
    }

    #[test]
    fn test_arc_wrapping_pattern() {
        // Skip if no GPU
        if let Ok(ctx) = CudaContext::new(0) {
            // In 0.18.1, ctx is already Arc<CudaContext>
            // No need to wrap again

            // Verify the type
            fn expect_arc_cuda_context(_: Arc<CudaContext>) {
                println!("✓ Received Arc<CudaContext> as expected");
            }

            expect_arc_cuda_context(ctx);
        }
    }
}
