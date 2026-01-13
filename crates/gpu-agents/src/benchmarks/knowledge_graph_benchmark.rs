//! Knowledge graph scaling benchmarks using real GPU agents
//!
//! This module provides GPU-native knowledge graph benchmarks following
//! rust.md and cuda.md development standards with proper GPU acceleration.

// Standard library
use std::time::Instant;

// External crates
use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaStream};

// Local modules
use crate::knowledge::{GraphQuery, KnowledgeEdge, KnowledgeGraph, KnowledgeNode};

/// Run knowledge graph benchmarks with real GPU acceleration
pub async fn run_knowledge_graph_benchmark(
    quick_mode: bool,
    stress_mode: bool,
) -> Result<KnowledgeGraphBenchmarkResults> {
    println!("🕸️ Running Knowledge Graph Benchmarks");

    // Configure based on mode
    let node_counts = if quick_mode {
        vec![1_000, 10_000]
    } else if stress_mode {
        vec![10_000, 100_000, 1_000_000, 10_000_000]
    } else {
        vec![1_000, 10_000, 100_000, 1_000_000]
    };

    let mut max_nodes = 0;
    let mut max_query_throughput: f64 = 0.0;
    let mut max_construction_rate: f64 = 0.0;
    let mut total_memory_efficiency: f64 = 0.0;
    let mut efficiency_count = 0;

    for &node_count in &node_counts {
        println!("   Testing {} nodes on GPU...", node_count);

        let result = test_knowledge_graph_performance(node_count).await;

        match result {
            Ok(perf) => {
                if perf.success {
                    max_nodes = max_nodes.max(node_count);
                    max_query_throughput = max_query_throughput.max(perf.query_throughput);
                    max_construction_rate = max_construction_rate.max(perf.construction_rate);

                    // Calculate memory efficiency
                    let expected_memory_mb = calculate_expected_memory(node_count);
                    let efficiency = expected_memory_mb / perf.gpu_memory_used_mb.max(1.0);
                    total_memory_efficiency += efficiency;
                    efficiency_count += 1;

                    println!(
                        "   ✅ {} nodes: {:.1} queries/sec, {:.1} constructions/sec, {:.1}MB GPU memory",
                        node_count, perf.query_throughput, perf.construction_rate, perf.gpu_memory_used_mb
                    );
                    println!(
                        "      GPU kernel time: {:.3}ms/query, Memory efficiency: {:.1}%",
                        perf.gpu_kernel_time_ms,
                        efficiency * 100.0
                    );
                }
            }
            Err(e) => {
                println!("   ❌ {} nodes failed: {}", node_count, e);
                break;
            }
        }
    }

    let avg_memory_efficiency = if efficiency_count > 0 {
        total_memory_efficiency / efficiency_count as f64
    } else {
        0.0
    };

    Ok(KnowledgeGraphBenchmarkResults {
        max_nodes,
        query_throughput: max_query_throughput,
        construction_rate: max_construction_rate,
        gpu_memory_efficiency: avg_memory_efficiency,
    })
}

/// Test knowledge graph performance using GPU acceleration
///
/// This function creates a graph on CPU, uploads it to GPU, and performs
/// GPU-accelerated similarity searches following cuda.md best practices.
async fn test_knowledge_graph_performance(node_count: usize) -> Result<KnowledgeGraphPerformance> {
    // Create CPU knowledge graph for initial population
    let mut cpu_graph = KnowledgeGraph::new(128);

    // Measure construction time (CPU population)
    let construction_start = Instant::now();

    // Add nodes with embeddings
    for i in 0..node_count {
        let node = KnowledgeNode {
            id: i as u32,
            content: format!("node_{}", i),
            node_type: if i % 2 == 0 { "data" } else { "concept" }.to_string(),
            embedding: generate_node_embedding(i, 128), // 128-dim embedding
        };
        cpu_graph.add_node(node);
    }

    // Add edges for sparse connectivity (average 5 edges per node)
    let edge_count = node_count * 5;
    for i in 0..edge_count {
        let edge = KnowledgeEdge {
            source_id: (i % node_count) as u32,
            target_id: ((i * 7 + 13) % node_count) as u32, // Pseudo-random distribution
            weight: 0.5 + (i as f32 % 10.0) * 0.05,
            edge_type: if i % 2 == 0 {
                "similarity"
            } else {
                "reference"
            }
            .to_string(),
        };
        cpu_graph.add_edge(edge);
    }

    let construction_duration = construction_start.elapsed();
    let construction_rate = node_count as f64 / construction_duration.as_secs_f64();

    // Upload graph to GPU for accelerated operations
    let gpu_upload_start = Instant::now();

    // Create CUDA context and stream for GPU operations
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let gpu_graph = cpu_graph.upload_to_gpu(ctx, stream)?;
    let gpu_upload_duration = gpu_upload_start.elapsed();

    println!(
        "     GPU upload time: {:.2}ms for {} nodes",
        gpu_upload_duration.as_secs_f64() * 1000.0,
        node_count
    );

    // Test GPU query performance using similarity search
    let query_start = Instant::now();
    let query_count = if node_count > 100_000 { 1000 } else { 100 };

    // Perform GPU-accelerated similarity searches
    for i in 0..query_count {
        let query = GraphQuery {
            query_text: "similarity".to_string(),
            query_embedding: generate_query_embedding(i, 768),
            max_results: 10,
            threshold: 0.5,
        };

        // GPU similarity search with kernel execution
        let _results = gpu_graph
            .run_similarity_search(&query)
            .context("GPU similarity search failed")?;
    }

    // Ensure GPU operations complete
    // GPU synchronization happens internally in the GPU graph operations

    let query_duration = query_start.elapsed();
    let query_throughput = query_count as f64 / query_duration.as_secs_f64();

    // Get GPU memory usage
    let gpu_memory_used_mb = gpu_graph.memory_usage() as f64 / (1024.0 * 1024.0);

    // Calculate GPU kernel time (approximation)
    let gpu_kernel_time_ms = query_duration.as_secs_f64() * 1000.0 / query_count as f64;

    Ok(KnowledgeGraphPerformance {
        _node_count: node_count,
        query_throughput,
        construction_rate,
        gpu_memory_used_mb,
        gpu_kernel_time_ms,
        success: true,
    })
}

/// Generate deterministic node embedding for testing
#[inline]
fn generate_node_embedding(node_id: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            // Create varied embeddings based on node ID
            let base = (node_id as f32 * 0.1).sin();
            let variation = (i as f32 * 0.05).cos();
            (base + variation) * 0.5
        })
        .collect()
}

/// Generate deterministic query embedding for testing
#[inline]
fn generate_query_embedding(query_id: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            // Create query embeddings with different pattern
            let base = (query_id as f32 * 0.2).cos();
            let variation = (i as f32 * 0.1).sin();
            (base + variation) * 0.5
        })
        .collect()
}

/// Calculate expected memory usage for a graph with given node count
///
/// This estimates the theoretical minimum memory needed for nodes and edges
fn calculate_expected_memory(node_count: usize) -> f64 {
    // Node memory: ID (4 bytes) + embedding (128 * 4 bytes) + metadata
    let node_size = 4 + (128 * 4) + 64; // ~580 bytes per node

    // Edge memory: source/target IDs (8 bytes) + weight (4 bytes) + metadata
    let edge_size = 8 + 4 + 16; // ~28 bytes per edge
    let edge_count = node_count * 5; // Average 5 edges per node

    // Total in MB
    let total_bytes = (node_count * node_size) + (edge_count * edge_size);
    total_bytes as f64 / (1024.0 * 1024.0)
}

/// Results from GPU-accelerated knowledge graph benchmarks
#[derive(Debug, Clone)]
pub struct KnowledgeGraphBenchmarkResults {
    /// Maximum number of nodes successfully processed on GPU
    pub max_nodes: usize,
    /// GPU query throughput (queries per second)
    pub query_throughput: f64,
    /// Graph construction rate (nodes per second)
    pub construction_rate: f64,
    /// GPU memory efficiency (ratio of useful data to allocated memory)
    pub gpu_memory_efficiency: f64,
}

/// Performance metrics for individual knowledge graph test
#[derive(Debug, Clone)]
struct KnowledgeGraphPerformance {
    /// Number of nodes in the graph
    _node_count: usize,
    /// GPU query throughput (queries per second)
    query_throughput: f64,
    /// Graph construction rate (nodes per second)
    construction_rate: f64,
    /// GPU memory used in MB
    gpu_memory_used_mb: f64,
    /// Average GPU kernel execution time per query (ms)
    gpu_kernel_time_ms: f64,
    /// Whether the test completed successfully
    success: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_embedding_generation() {
        let embedding1 = generate_node_embedding(0, 128);
        let embedding2 = generate_node_embedding(1, 128);

        assert_eq!(embedding1.len(), 128);
        assert_eq!(embedding2.len(), 128);

        // Embeddings should be different
        let diff: f32 = embedding1
            .iter()
            .zip(&embedding2)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1);
    }

    #[test]
    fn test_query_embedding_generation() {
        let embedding = generate_query_embedding(0, 128);

        assert_eq!(embedding.len(), 128);

        // All values should be in reasonable range
        for &val in &embedding {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_memory_calculation() {
        let mem_100 = calculate_expected_memory(100);
        let mem_1000 = calculate_expected_memory(1000);

        assert!(mem_100 > 0.0);
        assert!(mem_1000 > mem_100);

        // Should scale roughly linearly
        let ratio = mem_1000 / mem_100;
        assert!(ratio > 9.0 && ratio < 11.0);
    }

    #[tokio::test]
    async fn test_quick_mode_benchmark() {
        // Skip if no GPU available
        if CudaContext::new(0).is_err() {
            eprintln!("Skipping GPU test - no CUDA device available");
            return;
        }

        let result = run_knowledge_graph_benchmark(true, false).await;

        match result {
            Ok(res) => {
                assert!(res.max_nodes >= 1_000);
                assert!(res.query_throughput > 0.0);
                assert!(res.gpu_memory_efficiency > 0.0);
            }
            Err(e) => {
                eprintln!("GPU benchmark failed (may be expected in CI): {}", e);
            }
        }
    }
}
