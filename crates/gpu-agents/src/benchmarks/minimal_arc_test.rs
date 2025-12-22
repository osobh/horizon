//! Minimal test to reproduce Arc issue
//! Following TDD from rust.md

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::sync::Arc;
    use anyhow::Result;
    use cudarc::driver::CudaDevice;
    use crate::knowledge::{KnowledgeGraph, KnowledgeNode};
    
    #[test]
    fn test_minimal_upload() -> Result<()> {
        // Skip if no GPU
        if CudaDevice::new(0).is_err() {
            println!("Skipping - no GPU");
            return Ok(());
        }
        
        // Exactly as in our benchmark
        let mut cpu_graph = KnowledgeGraph::new();
        
        // Add one node
        let node = KnowledgeNode {
            id: 0,
            content: "test".to_string(),
            node_type: "test".to_string(), 
            embedding: vec![0.0; 128],
        };
        cpu_graph.add_node(node);
        
        // This is exactly what we do in the benchmark
        let cuda_device = CudaDevice::new(0)?;
        let device_arc = Arc::new(cuda_device);
        
        // This line should work
        let _gpu_graph = cpu_graph.upload_to_gpu(device_arc)?;
        
        Ok(())
    }
}