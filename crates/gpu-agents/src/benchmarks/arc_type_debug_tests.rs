//! TDD tests to debug Arc type issues
//! Following rust.md standards

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cudarc::driver::CudaDevice;
    use std::sync::Arc;

    #[test]
    fn test_arc_type_creation() {
        // Skip if no GPU
        if CudaDevice::new(0).is_err() {
            println!("Skipping - no GPU available");
            return;
        }

        // Test 1: What type does CudaDevice::new return?
        let cuda_result = CudaDevice::new(0);
        assert!(cuda_result.is_ok(), "CudaDevice creation should succeed");

        // Test 2: What happens when we unwrap and wrap in Arc?
        let cuda_device = cuda_result.unwrap();
        let arc_device = Arc::new(cuda_device);

        // This should be Arc<CudaDevice>
        fn assert_arc_cuda_device(_: Arc<CudaDevice>) {}
        assert_arc_cuda_device(arc_device.clone());

        println!("✓ Arc<CudaDevice> created successfully");
    }

    #[test]
    fn test_arc_type_inference_issue() -> Result<()> {
        // Skip if no GPU
        if CudaDevice::new(0).is_err() {
            return Ok(());
        }

        // Let's test what's happening in our benchmark

        // Pattern 1: Direct Arc::new
        let device1 = Arc::new(CudaDevice::new(0)?);
        fn check_type1(_: Arc<CudaDevice>) {}
        check_type1(device1);

        // Pattern 2: With intermediate variable
        let cuda_device = CudaDevice::new(0)?;
        let device2 = Arc::new(cuda_device);
        fn check_type2(_: Arc<CudaDevice>) {}
        check_type2(device2);

        // Pattern 3: What if Arc is already imported differently?
        {
            use std::sync::Arc as StdArc;
            let device3 = StdArc::new(CudaDevice::new(0)?);
            fn check_type3(_: Arc<CudaDevice>) {}
            check_type3(device3);
        }

        Ok(())
    }

    #[test]
    fn test_knowledge_graph_upload_signature() -> Result<()> {
        use crate::knowledge::{KnowledgeGraph, KnowledgeNode};

        // Skip if no GPU
        if CudaDevice::new(0).is_err() {
            return Ok(());
        }

        // Create a simple graph
        let mut graph = KnowledgeGraph::new();
        graph.add_node(KnowledgeNode {
            id: 0,
            content: "test".to_string(),
            node_type: "test".to_string(),
            embedding: vec![0.0; 128],
        });

        // Test what upload_to_gpu expects
        let device = Arc::new(CudaDevice::new(0)?);

        // This should work if upload_to_gpu expects Arc<CudaDevice>
        let result = graph.upload_to_gpu(device);
        assert!(result.is_ok(), "upload_to_gpu should succeed");

        Ok(())
    }

    #[test]
    fn test_llm_integration_signature() -> Result<()> {
        use crate::{LlmConfig, LlmIntegration};

        // Skip if no GPU
        if CudaDevice::new(0).is_err() {
            return Ok(());
        }

        let config = LlmConfig {
            model_path: "/tmp/test_model".to_string(),
            context_length: 512,
            batch_size: 4,
            temperature: 0.7,
        };

        // Test what LlmIntegration::new expects
        let device = Arc::new(CudaDevice::new(0)?);

        // This should work if new expects Arc<CudaDevice>
        let result = LlmIntegration::new(config, device);
        // Note: This might fail due to model path, but type should be OK

        Ok(())
    }

    #[test]
    fn test_gpu_swarm_device_field() -> Result<()> {
        use crate::{GpuSwarm, GpuSwarmConfig};

        // Skip if no GPU
        if CudaDevice::new(0).is_err() {
            return Ok(());
        }

        let config = GpuSwarmConfig {
            device_id: 0,
            max_agents: 100,
            enable_migrations: false,
            enable_llm_support: false,
            enable_knowledge_graph: false,
            optimize_kernel_launch: true,
        };

        // Test GpuSwarm creation
        let swarm = GpuSwarm::new(config)?;

        // If this compiles, the device field is correctly typed
        Ok(())
    }

    #[test]
    fn diagnose_arc_wrapping() {
        // This test helps us understand if there's double-wrapping

        // Create a type to test with
        struct TestDevice {
            id: i32,
        }

        // Normal Arc wrapping
        let device = TestDevice { id: 0 };
        let arc1: Arc<TestDevice> = Arc::new(device);

        // What happens if we accidentally wrap twice?
        // This would create Arc<Arc<TestDevice>>
        // let arc2: Arc<Arc<TestDevice>> = Arc::new(arc1.clone());

        // Function expecting Arc<TestDevice>
        fn use_device(d: Arc<TestDevice>) {
            println!("Using device {}", d.id);
        }

        // This works
        use_device(arc1.clone());

        // This would NOT work if arc1 was Arc<Arc<TestDevice>>
        // The compiler would complain about type mismatch
    }
}
