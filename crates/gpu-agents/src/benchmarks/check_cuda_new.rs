//! Check what CudaContext::new actually returns

#[cfg(test)]
mod tests {
    #[test]
    fn check_cuda_context_new_signature() {
        // In 0.18.1, CudaContext::new returns Arc<CudaContext>
        use cudarc::driver::CudaContext;
        use std::sync::Arc;

        if let Ok(ctx) = CudaContext::new(0) {
            // CudaContext::new returns Arc<CudaContext> directly
            let ctx_arc: Arc<CudaContext> = ctx;
            println!("Got Arc<CudaContext>: {:p}", &*ctx_arc);
        }
    }
}