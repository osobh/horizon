//! Test to diagnose Arc type issues
//! Following TDD approach from rust.md

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use cudarc::driver::CudaDevice;
    use crate::knowledge::{KnowledgeGraph, KnowledgeNode};
    
    #[test]
    fn test_arc_type_inference() {
        // Test 1: Direct Arc creation
        let device1 = Arc::new(CudaDevice::new(0).unwrap());
        let _type_check1: Arc<CudaDevice> = device1; // This should compile
        
        // Test 2: With type annotation
        let device2: Arc<CudaDevice> = Arc::new(CudaDevice::new(0)?);
        let _type_check2: Arc<CudaDevice> = device2; // This should compile
        
        // Test 3: What happens with our pattern
        let device3 = Arc::new(
            CudaDevice::new(0).unwrap()
        );
        let _type_check3: Arc<CudaDevice> = device3; // This should compile
    }
    
    #[test]
    fn test_upload_to_gpu_types() {
        // Skip if no GPU
        if CudaDevice::new(0).is_err() {
            return;
        }
        
        // Create a simple graph
        let mut graph = KnowledgeGraph::new();
        let node = KnowledgeNode {
            id: 0,
            content: "test".to_string(),
            node_type: "test".to_string(),
            embedding: vec![0.0; 128],
        };
        graph.add_node(node);
        
        // Test upload with explicit type
        let device: Arc<CudaDevice> = Arc::new(CudaDevice::new(0).unwrap());
        let result = graph.upload_to_gpu(device);
        assert!(result.is_ok());
    }
}