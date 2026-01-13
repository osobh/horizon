//! Minimal reproduction of Arc issue
//! Following TDD from rust.md

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use cudarc::driver::CudaContext;
    use anyhow::Result;

    #[test]
    fn test_arc_creation_pattern() -> Result<()> {
        // Skip if no GPU
        if CudaContext::new(0).is_err() {
            return Ok(());
        }

        // In 0.18.1, CudaContext::new returns Arc<CudaContext>
        let ctx = CudaContext::new(0)?;

        // Test the type
        fn expect_arc_cuda_context(c: Arc<CudaContext>) -> Arc<CudaContext> {
            c
        }

        let _verified = expect_arc_cuda_context(ctx);

        Ok(())
    }

    #[test]
    fn test_upload_to_gpu_direct() -> Result<()> {
        use crate::knowledge::{KnowledgeGraph, KnowledgeNode};

        // Skip if no GPU
        if CudaContext::new(0).is_err() {
            return Ok(());
        }

        let mut graph = KnowledgeGraph::new(128);
        graph.add_node(KnowledgeNode {
            id: 0,
            content: "test".to_string(),
            node_type: "test".to_string(),
            embedding: vec![0.0; 128],
        });

        // In 0.18.1, CudaContext::new returns Arc<CudaContext>
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // This should work
        let _gpu_graph = graph.upload_to_gpu(ctx, stream)?;

        Ok(())
    }
}
