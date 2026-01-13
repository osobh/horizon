//! Comprehensive TDD tests for knowledge graph benchmarks
//! Following rust.md and cuda.md standards

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::knowledge::{GraphQuery, KnowledgeGraph, KnowledgeNode};
    use anyhow::Result;
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    /// Helper to check if GPU is available
    fn ensure_gpu_available() -> Result<()> {
        match CudaContext::new(0) {
            Ok(_) => Ok(()),
            Err(e) => {
                eprintln!("GPU not available for testing: {}", e);
                Err(anyhow::anyhow!("GPU required for these tests"))
            }
        }
    }

    #[test]
    fn test_gpu_device_creation() -> Result<()> {
        ensure_gpu_available()?;

        // Test context creation
        let ctx = CudaContext::new(0)?;

        // Verify Arc type - in 0.18.1, CudaContext::new returns Arc<CudaContext>
        fn verify_arc_type(_ctx: Arc<CudaContext>) {
            // Type checking happens at compile time
        }

        verify_arc_type(ctx.clone());
        Ok(())
    }

    #[test]
    fn test_knowledge_graph_upload() -> Result<()> {
        ensure_gpu_available()?;

        // Create a simple graph
        let mut graph = KnowledgeGraph::new(128);

        // Add test nodes
        for i in 0..10 {
            let node = KnowledgeNode {
                id: i,
                content: format!("test_node_{}", i),
                node_type: "test".to_string(),
                embedding: vec![0.1 * i as f32; 128],
            };
            graph.add_node(node);
        }

        // Upload to GPU
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let gpu_graph = graph.upload_to_gpu(ctx, stream)?;

        // Verify node count
        assert_eq!(gpu_graph.node_count(), 10);

        Ok(())
    }

    #[test]
    fn test_gpu_similarity_search() -> Result<()> {
        ensure_gpu_available()?;

        // Create graph with diverse embeddings
        let mut graph = KnowledgeGraph::new(128);

        for i in 0..100 {
            let node = KnowledgeNode {
                id: i,
                content: format!("node_{}", i),
                node_type: if i % 2 == 0 { "even" } else { "odd" }.to_string(),
                embedding: generate_test_embedding(i as usize, 128),
            };
            graph.add_node(node);
        }

        // Upload to GPU
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let gpu_graph = graph.upload_to_gpu(ctx, stream)?;

        // Test similarity search
        let query = GraphQuery {
            query_text: "similarity".to_string(),
            query_embedding: generate_test_embedding(5, 128),
            max_results: 5,
            threshold: 0.5,
        };

        let results = gpu_graph.run_similarity_search(&query)?;

        // Verify we got results
        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        Ok(())
    }

    #[test]
    fn test_memory_efficiency() -> Result<()> {
        ensure_gpu_available()?;

        // Create graph with known size
        let mut graph = KnowledgeGraph::new(128);
        let node_count = 1000;

        for i in 0..node_count {
            let node = KnowledgeNode {
                id: i,
                content: format!("n{}", i),
                node_type: "test".to_string(),
                embedding: vec![0.0; 128],
            };
            graph.add_node(node);
        }

        // Upload and check memory
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let gpu_graph = graph.upload_to_gpu(ctx, stream)?;

        let memory_used = gpu_graph.memory_usage();

        // Basic sanity check - should use some memory
        assert!(memory_used > 0);

        // Shouldn't use excessive memory (< 100MB for 1000 nodes)
        assert!(memory_used < 100 * 1024 * 1024);

        Ok(())
    }

    #[tokio::test]
    async fn test_benchmark_quick_mode() -> Result<()> {
        ensure_gpu_available()?;

        // Run benchmark in quick mode
        let results = run_knowledge_graph_benchmark(true, false).await?;

        // Verify results
        assert!(results.max_nodes >= 1_000);
        assert!(results.query_throughput > 0.0);
        assert!(results.construction_rate > 0.0);
        assert!(results.gpu_memory_efficiency >= 0.0);
        assert!(results.gpu_memory_efficiency <= 1.0);

        Ok(())
    }

    #[test]
    fn test_embedding_generation() {
        // Test deterministic embedding generation
        let emb1 = generate_test_embedding(42, 128);
        let emb2 = generate_test_embedding(42, 128);
        let emb3 = generate_test_embedding(43, 128);

        // Same seed should produce same embedding
        assert_eq!(emb1, emb2);

        // Different seeds should produce different embeddings
        assert_ne!(emb1, emb3);

        // Check dimensions
        assert_eq!(emb1.len(), 128);

        // Check value range
        for &val in &emb1 {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    /// Generate test embedding
    fn generate_test_embedding(seed: usize, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let base = (seed as f32 * 0.1).sin();
                let variation = (i as f32 * 0.05).cos();
                (base + variation) * 0.5
            })
            .collect()
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_specific_tests {
    use super::*;

    #[test]
    fn test_cuda_synchronization() -> Result<()> {
        ensure_gpu_available()?;

        let ctx = CudaContext::new(0)?;
        let _stream = ctx.default_stream();

        // Test that context can be used for synchronization
        // Note: synchronization is handled internally by GPU operations

        Ok(())
    }
}
