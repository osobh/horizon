//! TDD tests to debug Arc type issues
//! Following rust.md standards

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    #[test]
    fn test_arc_type_creation() {
        // Skip if no GPU
        if CudaContext::new(0).is_err() {
            println!("Skipping - no GPU available");
            return;
        }

        // Test 1: What type does CudaContext::new return?
        let cuda_result = CudaContext::new(0);
        assert!(cuda_result.is_ok(), "CudaContext creation should succeed");

        // Test 2: In 0.18.1, CudaContext::new already returns Arc<CudaContext>
        let ctx = cuda_result.unwrap();

        // This should be Arc<CudaContext> already
        fn assert_arc_cuda_context(_: Arc<CudaContext>) {}
        assert_arc_cuda_context(ctx.clone());

        println!("✓ Arc<CudaContext> created successfully");
    }

    #[test]
    fn test_arc_type_inference_issue() -> Result<()> {
        // Skip if no GPU
        if CudaContext::new(0).is_err() {
            return Ok(());
        }

        // Let's test what's happening in our benchmark

        // Pattern 1: In 0.18.1, CudaContext::new returns Arc<CudaContext>
        let ctx1 = CudaContext::new(0)?;
        fn check_type1(_: Arc<CudaContext>) {}
        check_type1(ctx1);

        // Pattern 2: Direct assignment
        let ctx2 = CudaContext::new(0)?;
        fn check_type2(_: Arc<CudaContext>) {}
        check_type2(ctx2);

        // Pattern 3: With type alias
        {
            use std::sync::Arc as StdArc;
            let ctx3 = CudaContext::new(0)?;
            fn check_type3(_: Arc<CudaContext>) {}
            check_type3(ctx3);
        }

        Ok(())
    }

    #[test]
    fn test_knowledge_graph_upload_signature() -> Result<()> {
        use crate::knowledge::{KnowledgeGraph, KnowledgeNode};

        // Skip if no GPU
        if CudaContext::new(0).is_err() {
            return Ok(());
        }

        // Create a simple graph
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

        // This should work if upload_to_gpu expects Arc<CudaContext>
        let result = graph.upload_to_gpu(ctx, stream);
        assert!(result.is_ok(), "upload_to_gpu should succeed");

        Ok(())
    }

    #[test]
    fn test_llm_integration_signature() -> Result<()> {
        use crate::{LlmConfig, LlmIntegration};

        // Skip if no GPU
        if CudaContext::new(0).is_err() {
            return Ok(());
        }

        let config = LlmConfig {
            model_type: "llama".to_string(),
            batch_size: 4,
            max_context_length: 512,
            temperature: 0.7,
            enable_embeddings: false,
            embedding_dim: 768,
            gpu_memory_mb: 1024,
        };

        // In 0.18.1, CudaContext::new returns Arc<CudaContext>
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // This should work if new expects Arc<CudaContext>
        let _result = LlmIntegration::new(config, ctx, stream);
        // Note: This might fail due to model details, but type should be OK

        Ok(())
    }

    #[test]
    fn test_gpu_swarm_device_field() -> Result<()> {
        use crate::{GpuSwarm, GpuSwarmConfig};

        // Skip if no GPU
        if CudaContext::new(0).is_err() {
            return Ok(());
        }

        let config = GpuSwarmConfig {
            device_id: 0,
            max_agents: 100,
            block_size: 256,
            shared_memory_size: 48 * 1024,
            evolution_interval: 100,
            enable_llm: false,
            enable_collective_intelligence: false,
            enable_collective_knowledge: false,
            enable_knowledge_graph: false,
        };

        // Test GpuSwarm creation
        let _swarm = GpuSwarm::new(config)?;

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
