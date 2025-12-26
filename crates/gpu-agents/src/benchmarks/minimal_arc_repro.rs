//! Minimal reproduction of Arc issue
//! Following TDD from rust.md

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use cudarc::driver::CudaDevice;
    use anyhow::Result;
    
    #[test]
    fn test_arc_creation_pattern() -> Result<()> {
        // Skip if no GPU
        if CudaDevice::new(0).is_err() {
            return Ok(());
        }
        
        // This is exactly what we do in the benchmark
        let cuda_device = CudaDevice::new(0)?;
        let device = Arc::new(cuda_device);
        
        // Test the type
        fn expect_arc_cuda_device(d: Arc<CudaDevice>) -> Arc<CudaDevice> {
            d
        }
        
        let _verified = expect_arc_cuda_device(device);
        
        Ok(())
    }
    
    #[test]
    fn test_upload_to_gpu_direct() -> Result<()> {
        use crate::knowledge::{KnowledgeGraph, KnowledgeNode};
        
        // Skip if no GPU
        if CudaDevice::new(0).is_err() {
            return Ok(());
        }
        
        let mut graph = KnowledgeGraph::new();
        graph.add_node(KnowledgeNode {
            id: 0,
            content: "test".to_string(),
            node_type: "test".to_string(),
            embedding: vec![0.0; 128],
        });
        
        // Exact pattern from benchmark
        let cuda_device = CudaDevice::new(0)?;
        let device = Arc::new(cuda_device);
        
        // This should work
        let _gpu_graph = graph.upload_to_gpu(device)?;
        
        Ok(())
    }
}