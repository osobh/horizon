//! Test exact pattern from benchmark
//! Following TDD from rust.md

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use anyhow::{Context, Result};
    use cudarc::driver::CudaContext;
    use crate::knowledge::{KnowledgeGraph, KnowledgeNode};

    #[test]
    fn test_exact_benchmark_pattern() -> Result<()> {
        // Skip if no GPU
        if CudaContext::new(0).is_err() {
            println!("Skipping - no GPU");
            return Ok(());
        }

        // Create CPU knowledge graph exactly as in benchmark
        let mut cpu_graph = KnowledgeGraph::new(128);

        // Add one node exactly as in benchmark
        let node = KnowledgeNode {
            id: 0,
            content: "node_0".to_string(),
            node_type: "data".to_string(),
            embedding: vec![0.0; 128],
        };
        cpu_graph.add_node(node);

        // In 0.18.1, CudaContext::new returns Arc<CudaContext>
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        let _gpu_graph = cpu_graph.upload_to_gpu(ctx, stream)?;

        // If we get here, it worked
        println!("Successfully uploaded to GPU");
        Ok(())
    }

    #[test]
    fn test_type_annotations() {
        // Skip if no GPU
        if CudaContext::new(0).is_err() {
            return;
        }

        // Test with explicit type annotations
        let ctx: Arc<CudaContext> = CudaContext::new(0).unwrap();

        // Verify the type
        fn check_type(ctx: Arc<CudaContext>) {
            println!("Type is correct: Arc<CudaContext>");
        }

        check_type(ctx);
    }
}
