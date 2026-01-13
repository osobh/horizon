//! TDD Step 3: Arc device handling tests
//!
//! Tests to verify Arc<CudaContext> is handled correctly throughout the codebase

use anyhow::Result;
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that benchmark constructors handle Arc<CudaContext> correctly
    #[test]
    fn test_benchmark_arc_device_handling() -> Result<()> {
        // Test that benchmarks expect device_id and create Arc<CudaContext> internally
        // This should NOT double-wrap Arc

        // GPU Evolution Benchmark
        if cudarc::driver::CudaContext::new(0).is_ok() {
            let config = crate::benchmarks::EvolutionBenchmarkConfig::default();
            let benchmark = crate::benchmarks::GpuEvolutionBenchmark::new(0, config);
            assert!(
                benchmark.is_ok(),
                "GpuEvolutionBenchmark should create successfully"
            );
        }

        // GPU Knowledge Graph Benchmark
        if cudarc::driver::CudaContext::new(0).is_ok() {
            let config = crate::benchmarks::KnowledgeGraphBenchmarkConfig::default();
            let benchmark = crate::benchmarks::GpuKnowledgeGraphBenchmark::new(0, config);
            assert!(
                benchmark.is_ok(),
                "GpuKnowledgeGraphBenchmark should create successfully"
            );
        }

        // GPU Streaming Benchmark
        if cudarc::driver::CudaContext::new(0).is_ok() {
            let config = crate::benchmarks::StreamingBenchmarkConfig::default();
            let benchmark = crate::benchmarks::GpuStreamingBenchmark::new(0, config);
            assert!(
                benchmark.is_ok(),
                "GpuStreamingBenchmark should create successfully"
            );
        }

        Ok(())
    }

    /// Test that streaming modules handle Arc<CudaContext> correctly
    #[test]
    fn test_streaming_arc_device_handling() -> Result<()> {
        if let Ok(device) = cudarc::driver::CudaContext::new(0) {
            // CudaContext::new returns Arc<CudaContext>
            let device_arc = device; // Already Arc<CudaContext>

            // Test GpuBufferPool accepts Arc<CudaContext>
            let pool = crate::streaming::GpuBufferPool::new(device_arc.clone(), 2, 1024);
            assert!(pool.is_ok(), "GpuBufferPool should accept Arc<CudaContext>");

            // Test GpuStreamProcessor accepts Arc<CudaContext>
            let config = crate::streaming::GpuStreamConfig::default();
            let processor = crate::streaming::GpuStreamProcessor::new(device_arc.clone(), config);
            assert!(
                processor.is_ok(),
                "GpuStreamProcessor should accept Arc<CudaContext>"
            );

            // Test GpuStreamPipeline accepts Arc<CudaContext>
            let config = crate::streaming::GpuStreamConfig::default();
            let pipeline = crate::streaming::GpuStreamPipeline::new(device_arc, config);
            assert!(
                pipeline.is_ok(),
                "GpuStreamPipeline should accept Arc<CudaContext>"
            );
        }

        Ok(())
    }

    /// Test that GPU swarm handles Arc<CudaContext> correctly
    #[test]
    fn test_gpu_swarm_arc_device_handling() -> Result<()> {
        if let Ok(device) = cudarc::driver::CudaContext::new(0) {
            // Test that GpuSwarm stores Arc<CudaContext> correctly
            let config = crate::GpuSwarmConfig {
                device_id: 0,
                agent_count: 1000,
                enable_cpu_fallback: true,
                memory_pool_size_gb: 2.0,
                evolution_params: crate::evolution::EvolutionParameters::default(),
            };

            // The GpuSwarm should handle device creation internally
            // and not require double Arc wrapping
            let swarm = crate::GpuSwarm::new(config);
            assert!(
                swarm.is_ok(),
                "GpuSwarm should create with proper Arc handling"
            );
        }

        Ok(())
    }

    /// Test that knowledge graph handles Arc<CudaContext> correctly  
    #[test]
    fn test_knowledge_graph_arc_device_handling() -> Result<()> {
        if let Ok(device) = cudarc::driver::CudaContext::new(0) {
            let device_arc = device; // Already Arc<CudaContext>

            // Test EnhancedGpuKnowledgeGraph accepts Arc<CudaContext>
            let graph = crate::knowledge::EnhancedGpuKnowledgeGraph::new(
                device_arc.clone(),
                1000, // node_count
                768,  // embedding_dim
            );
            assert!(
                graph.is_ok(),
                "EnhancedGpuKnowledgeGraph should accept Arc<CudaContext>"
            );

            // Test CsrGraph creation with Arc<CudaContext>
            let adjacency_list = vec![(0u32, vec![(1u32, 1.0f32)])];
            let csr_graph = crate::knowledge::CsrGraph::from_adjacency_list(
                device_arc,
                &adjacency_list,
                2, // node_count
            );
            assert!(csr_graph.is_ok(), "CsrGraph should accept Arc<CudaContext>");
        }

        Ok(())
    }

    /// Test Arc device type consistency
    #[test]
    fn test_arc_device_type_consistency() -> Result<()> {
        if let Ok(device) = cudarc::driver::CudaContext::new(0) {
            // Verify that CudaContext::new returns Arc<CudaContext>
            let _arc_device: Arc<cudarc::driver::CudaContext> = device;

            // This should compile without issues - proving the type is correct
            println!("Arc<CudaContext> type handling is consistent");
        }

        Ok(())
    }
}
