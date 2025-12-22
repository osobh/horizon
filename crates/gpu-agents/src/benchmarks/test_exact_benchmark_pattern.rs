//! Test exact pattern from benchmark
//! Following TDD from rust.md

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use anyhow::{Context, Result};
    use cudarc::driver::CudaDevice;
    use crate::knowledge::{KnowledgeGraph, KnowledgeNode};
    
    #[test]
    fn test_exact_benchmark_pattern() -> Result<()> {
        // Skip if no GPU
        if CudaDevice::new(0).is_err() {
            println!("Skipping - no GPU");
            return Ok(());
        }
        
        // Create CPU knowledge graph exactly as in benchmark
        let mut cpu_graph = KnowledgeGraph::new();
        
        // Add one node exactly as in benchmark
        let node = KnowledgeNode {
            id: 0,
            content: "node_0".to_string(),
            node_type: "data".to_string(),
            embedding: vec![0.0; 128],
        };
        cpu_graph.add_node(node);
        
        // Exactly as in benchmark lines 129-132
        let cuda_device = CudaDevice::new(0)?;
        let device_arc = Arc::new(cuda_device);
        
        let gpu_graph = cpu_graph.upload_to_gpu(device_arc)?;
        
        // If we get here, it worked
        println!("Successfully uploaded to GPU");
        Ok(())
    }
    
    #[test]
    fn test_type_annotations() {
        // Skip if no GPU
        if CudaDevice::new(0).is_err() {
            return;
        }
        
        // Test with explicit type annotations
        let cuda_device: CudaDevice = CudaDevice::new(0)?;
        let device_arc: Arc<CudaDevice> = Arc::new(cuda_device);
        
        // Verify the type
        fn check_type(device: Arc<CudaDevice>) {
            println!("Type is correct: Arc<CudaDevice>");
        }
        
        check_type(device_arc);
    }
}