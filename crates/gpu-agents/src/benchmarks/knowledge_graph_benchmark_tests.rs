//! Tests for GPU knowledge graph benchmarks
//!
//! Following TDD approach from rust.md with comprehensive GPU testing

use super::*;
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use cudarc::driver::CudaDevice;

    /// Test helper to verify GPU is available
    fn ensure_gpu_available() -> Result<()> {
        match CudaDevice::new(0) {
            Ok(_) => Ok(()),
            Err(e) => {
                eprintln!("GPU not available for testing: {}", e);
                Err(anyhow::anyhow!("GPU required for these tests"))
            }
        }
    }

    #[tokio::test]
    async fn test_gpu_knowledge_graph_creation() -> Result<()> {
        ensure_gpu_available()?;

        let perf = test_gpu_knowledge_graph_performance(100).await?;

        assert!(perf.success);
        assert_eq!(perf.node_count, 100);
        assert!(perf.gpu_memory_used_mb > 0.0);
        assert!(perf.query_throughput > 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_knowledge_graph_scaling() -> Result<()> {
        ensure_gpu_available()?;

        let small_perf = test_gpu_knowledge_graph_performance(100).await?;
        let large_perf = test_gpu_knowledge_graph_performance(1000).await?;

        // Larger graph should use more memory
        assert!(large_perf.gpu_memory_used_mb > small_perf.gpu_memory_used_mb);

        // Both should succeed
        assert!(small_perf.success);
        assert!(large_perf.success);

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_similarity_search_performance() -> Result<()> {
        ensure_gpu_available()?;

        let perf = test_gpu_knowledge_graph_performance(1000).await?;

        // GPU similarity search should be fast
        assert!(
            perf.query_throughput > 100.0,
            "Expected >100 queries/sec on GPU"
        );
        assert!(
            perf.gpu_kernel_time_ms > 0.0,
            "Should have GPU kernel execution time"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_quick_mode_benchmark() -> Result<()> {
        ensure_gpu_available()?;

        let results = run_knowledge_graph_benchmark(true, false).await?;

        assert!(results.max_nodes >= 1_000);
        assert!(results.query_throughput > 0.0);
        assert!(results.construction_rate > 0.0);
        assert!(results.gpu_memory_efficiency > 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_memory_tracking() -> Result<()> {
        ensure_gpu_available()?;

        let perf = test_gpu_knowledge_graph_performance(500).await?;

        // Verify GPU memory metrics
        assert!(perf.gpu_memory_used_mb > 0.0);
        assert!(
            perf.gpu_memory_used_mb < 1000.0,
            "Shouldn't use excessive memory for 500 nodes"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_batch_operations() -> Result<()> {
        ensure_gpu_available()?;

        // Test batch query performance
        let batch_sizes = vec![1, 10, 100];
        let mut previous_throughput = 0.0;

        for batch_size in batch_sizes {
            let perf = test_gpu_batch_query_performance(1000, batch_size).await?;

            // Larger batches should have better throughput
            assert!(perf.query_throughput >= previous_throughput);
            previous_throughput = perf.query_throughput;
        }

        Ok(())
    }

    /// Helper function to test batch query performance
    async fn test_gpu_batch_query_performance(
        node_count: usize,
        batch_size: usize,
    ) -> Result<KnowledgeGraphPerformance> {
        // Create and populate graph
        let mut graph = KnowledgeGraph::new();

        for i in 0..node_count {
            let node = KnowledgeNode {
                id: i as u32,
                content: format!("node_{}", i),
                node_type: "test".to_string(),
                embedding: vec![0.1 * (i as f32 % 10.0); 128],
            };
            graph.add_node(node);
        }

        // Upload to GPU
        let device = Arc::new(CudaDevice::new(0)?);
        let gpu_graph = graph.upload_to_gpu(device)?;

        // Test batch queries
        let start = Instant::now();
        let query_count = 100;

        for i in (0..query_count).step_by(batch_size) {
            let batch_end = (i + batch_size).min(query_count);
            let batch_queries: Vec<_> = (i..batch_end)
                .map(|j| GraphQuery {
                    query_type: "similarity".to_string(),
                    embedding: vec![0.2 * (j as f32 % 5.0); 128],
                    max_results: 10,
                })
                .collect();

            // Process batch on GPU
            for query in &batch_queries {
                let _ = gpu_graph.run_similarity_search(query)?;
            }
        }

        let duration = start.elapsed();
        let query_throughput = query_count as f64 / duration.as_secs_f64();

        Ok(KnowledgeGraphPerformance {
            node_count,
            query_throughput,
            construction_rate: 0.0, // Not measured in this test
            gpu_memory_used_mb: gpu_graph.memory_usage() as f64 / (1024.0 * 1024.0),
            gpu_kernel_time_ms: 0.0, // Would need GPU profiling
            success: true,
        })
    }
}
