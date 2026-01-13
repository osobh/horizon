//! Test to diagnose Arc type issues
//! Following TDD approach from rust.md

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use cudarc::driver::CudaContext;
    use crate::knowledge::{KnowledgeGraph, KnowledgeNode};

    #[test]
    fn test_arc_type_inference() {
        // Test 1: In 0.18.1, CudaContext::new returns Arc<CudaContext>
        let ctx1 = CudaContext::new(0).unwrap();
        let _type_check1: Arc<CudaContext> = ctx1;

        // Test 2: With type annotation
        let ctx2: Arc<CudaContext> = CudaContext::new(0).unwrap();
        let _type_check2: Arc<CudaContext> = ctx2;

        // Test 3: Direct usage
        let ctx3 = CudaContext::new(0).unwrap();
        let _type_check3: Arc<CudaContext> = ctx3;
    }

    #[test]
    fn test_upload_to_gpu_types() {
        // Skip if no GPU
        if CudaContext::new(0).is_err() {
            return;
        }

        // Create a simple graph
        let mut graph = KnowledgeGraph::new(128);
        let node = KnowledgeNode {
            id: 0,
            content: "test".to_string(),
            node_type: "test".to_string(),
            embedding: vec![0.0; 128],
        };
        graph.add_node(node);

        // Test upload with Arc<CudaContext>
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let result = graph.upload_to_gpu(ctx, stream);
        assert!(result.is_ok());
    }
}
