//! Minimal test to reproduce Arc issue
//! Following TDD from rust.md

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::sync::Arc;
    use anyhow::Result;
    use cudarc::driver::CudaContext;
    use crate::knowledge::{KnowledgeGraph, KnowledgeNode};

    #[test]
    fn test_minimal_upload() -> Result<()> {
        // Skip if no GPU
        if CudaContext::new(0).is_err() {
            println!("Skipping - no GPU");
            return Ok(());
        }

        // Exactly as in our benchmark
        let mut cpu_graph = KnowledgeGraph::new(128);

        // Add one node
        let node = KnowledgeNode {
            id: 0,
            content: "test".to_string(),
            node_type: "test".to_string(),
            embedding: vec![0.0; 128],
        };
        cpu_graph.add_node(node);

        // In 0.18.1, CudaContext::new returns Arc<CudaContext>
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // This line should work
        let _gpu_graph = cpu_graph.upload_to_gpu(ctx, stream)?;

        Ok(())
    }
}
