//! GPU Knowledge Graph benchmarks

use anyhow::Result;
use rand::Rng;
use std::sync::Arc;
use std::time::Instant;

use crate::knowledge::{
    CsrGraph, EnhancedGpuKnowledgeGraph, KnowledgeEdge, KnowledgeGraph, KnowledgeNode,
};

/// Knowledge graph benchmark configuration
#[derive(Debug, Clone)]
pub struct KnowledgeGraphBenchmarkConfig {
    /// Node counts to test
    pub node_counts: Vec<usize>,
    /// Average edges per node
    pub avg_edges_per_node: Vec<f32>,
    /// Embedding dimensions
    pub embedding_dim: usize,
    /// Query batch sizes
    pub query_batch_sizes: Vec<usize>,
    /// Number of queries per test
    pub num_queries: usize,
}

impl Default for KnowledgeGraphBenchmarkConfig {
    fn default() -> Self {
        Self {
            node_counts: vec![1_000, 10_000, 100_000, 1_000_000],
            avg_edges_per_node: vec![5.0, 10.0, 50.0],
            embedding_dim: 768,
            query_batch_sizes: vec![1, 10, 100, 1000],
            num_queries: 1000,
        }
    }
}

/// Knowledge graph benchmark results
#[derive(Debug, Clone)]
pub struct KnowledgeGraphBenchmarkResults {
    pub node_count: usize,
    pub edge_count: usize,
    pub build_time_ms: f64,
    pub query_latency_us: f64,
    pub queries_per_second: f64,
    pub pagerank_time_ms: f64,
    pub bfs_time_ms: f64,
    pub memory_usage_mb: f64,
}

/// GPU Knowledge Graph benchmark suite
pub struct GpuKnowledgeGraphBenchmark {
    device: Arc<cudarc::driver::CudaDevice>,
    config: KnowledgeGraphBenchmarkConfig,
}

impl GpuKnowledgeGraphBenchmark {
    /// Create new benchmark suite
    pub fn new(device_id: i32, config: KnowledgeGraphBenchmarkConfig) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(device_id as usize)?;
        Ok(Self { device, config })
    }

    /// Run all benchmarks
    pub async fn run_all(&mut self) -> Result<Vec<KnowledgeGraphBenchmarkResults>> {
        let mut results = Vec::new();

        for &node_count in &self.config.node_counts {
            for &avg_edges in &self.config.avg_edges_per_node {
                println!(
                    "\nBenchmarking: nodes={}, avg_edges={}",
                    node_count, avg_edges
                );

                match self.benchmark_graph(node_count, avg_edges).await {
                    Ok(result) => {
                        println!("  Build time: {:.2}ms", result.build_time_ms);
                        println!("  Query latency: {:.2}μs", result.query_latency_us);
                        println!("  Queries/sec: {:.0}", result.queries_per_second);
                        println!("  PageRank: {:.2}ms", result.pagerank_time_ms);
                        println!("  Memory: {:.2}MB", result.memory_usage_mb);
                        results.push(result);
                    }
                    Err(e) => {
                        eprintln!("  Benchmark failed: {}", e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Benchmark graph with specific parameters
    async fn benchmark_graph(
        &self,
        node_count: usize,
        avg_edges_per_node: f32,
    ) -> Result<KnowledgeGraphBenchmarkResults> {
        // Generate random graph
        let (cpu_graph, edge_count) = self.generate_random_graph(node_count, avg_edges_per_node)?;

        // Build GPU graph
        let build_start = Instant::now();
        let mut gpu_graph = EnhancedGpuKnowledgeGraph::new(
            self.device.clone(),
            node_count,
            self.config.embedding_dim,
        )?;

        // Convert adjacency list for CSR
        let adjacency_list = self.build_adjacency_list(&cpu_graph);
        let _csr_graph =
            CsrGraph::from_adjacency_list(self.device.clone(), &adjacency_list, node_count)?;

        // Build spatial index
        gpu_graph.build_spatial_index()?;
        let build_time = build_start.elapsed();

        // Benchmark queries
        let query_latency = self.benchmark_queries(&gpu_graph).await?;

        // Benchmark PageRank
        let pagerank_start = Instant::now();
        let _ = gpu_graph.pagerank(10, 0.85)?;
        let pagerank_time = pagerank_start.elapsed();

        // Benchmark BFS
        let bfs_start = Instant::now();
        if node_count > 1 {
            let _ = gpu_graph.shortest_path(0, (node_count - 1) as u32)?;
        }
        let bfs_time = bfs_start.elapsed();

        // Calculate memory usage
        let memory_usage_mb = self.calculate_memory_usage(node_count, edge_count);

        Ok(KnowledgeGraphBenchmarkResults {
            node_count,
            edge_count,
            build_time_ms: build_time.as_secs_f64() * 1000.0,
            query_latency_us: query_latency * 1_000_000.0,
            queries_per_second: 1.0 / query_latency,
            pagerank_time_ms: pagerank_time.as_secs_f64() * 1000.0,
            bfs_time_ms: bfs_time.as_secs_f64() * 1000.0,
            memory_usage_mb,
        })
    }

    /// Generate random graph
    fn generate_random_graph(
        &self,
        node_count: usize,
        avg_edges_per_node: f32,
    ) -> Result<(KnowledgeGraph, usize)> {
        let mut graph = KnowledgeGraph::new(self.config.embedding_dim);
        let mut rng = rand::thread_rng();
        let mut total_edges = 0;

        // Add nodes
        for i in 0..node_count {
            let embedding: Vec<f32> = (0..self.config.embedding_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();

            let node = KnowledgeNode {
                id: i as u32,
                content: format!("Node {}", i),
                node_type: "test".to_string(),
                embedding,
            };
            graph.add_node(node);
        }

        // Add edges
        for i in 0..node_count {
            let num_edges = rng.gen_range(0.0..avg_edges_per_node * 2.0) as usize;

            for _ in 0..num_edges {
                let target = rng.gen_range(0..node_count) as u32;
                if target != i as u32 {
                    let edge = KnowledgeEdge {
                        source_id: i as u32,
                        target_id: target,
                        edge_type: "connects".to_string(),
                        weight: rng.gen_range(0.0..1.0),
                    };
                    graph.add_edge(edge);
                    total_edges += 1;
                }
            }
        }

        Ok((graph, total_edges))
    }

    /// Build adjacency list from graph
    fn build_adjacency_list(&self, graph: &KnowledgeGraph) -> Vec<(u32, Vec<(u32, f32)>)> {
        // This is a simplified version - real implementation would extract from graph
        let mut adjacency: Vec<(u32, Vec<(u32, f32)>)> = Vec::new();

        // Generate some dummy edges for testing
        let mut rng = rand::thread_rng();
        for i in 0..graph.node_count() {
            let num_edges = rng.gen_range(0..10);
            let mut edges = Vec::new();

            for _ in 0..num_edges {
                let target = rng.gen_range(0..graph.node_count()) as u32;
                if target != i as u32 {
                    edges.push((target, rng.gen_range(0.0..1.0)));
                }
            }

            adjacency.push((i as u32, edges));
        }

        adjacency
    }

    /// Benchmark query performance
    async fn benchmark_queries(&self, gpu_graph: &EnhancedGpuKnowledgeGraph) -> Result<f64> {
        let mut rng = rand::thread_rng();
        let mut total_time = 0.0;

        for _ in 0..self.config.num_queries {
            // Generate random query embedding
            let query_embedding: Vec<f32> = (0..self.config.embedding_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();

            let start = Instant::now();
            let _ = gpu_graph.ann_search(&query_embedding, 10)?;
            total_time += start.elapsed().as_secs_f64();
        }

        Ok(total_time / self.config.num_queries as f64)
    }

    /// Calculate memory usage
    fn calculate_memory_usage(&self, node_count: usize, edge_count: usize) -> f64 {
        let node_memory = node_count * self.config.embedding_dim * std::mem::size_of::<f32>();
        let edge_memory =
            edge_count * (2 * std::mem::size_of::<u32>() + std::mem::size_of::<f32>());
        let index_memory = node_count * std::mem::size_of::<u32>(); // Simplified

        (node_memory + edge_memory + index_memory) as f64 / (1024.0 * 1024.0)
    }

    /// Benchmark query batch sizes
    pub async fn benchmark_query_batches(&self) -> Result<()> {
        println!("\n=== Query Batch Size Benchmarks ===");

        let node_count = 100_000;
        let avg_edges = 10.0;

        // Build graph
        let (_cpu_graph, _) = self.generate_random_graph(node_count, avg_edges)?;
        let mut gpu_graph = EnhancedGpuKnowledgeGraph::new(
            self.device.clone(),
            node_count,
            self.config.embedding_dim,
        )?;
        gpu_graph.build_spatial_index()?;

        for &batch_size in &self.config.query_batch_sizes {
            println!("\nBatch size: {}", batch_size);

            let mut rng = rand::thread_rng();
            let queries: Vec<Vec<f32>> = (0..batch_size)
                .map(|_| {
                    (0..self.config.embedding_dim)
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect()
                })
                .collect();

            let start = Instant::now();

            for query in &queries {
                let _ = gpu_graph.ann_search(query, 10)?;
            }

            let elapsed = start.elapsed();
            let throughput = batch_size as f64 / elapsed.as_secs_f64();

            println!(
                "  Time: {:.2}ms, Throughput: {:.0} queries/sec",
                elapsed.as_millis(),
                throughput
            );
        }

        Ok(())
    }

    /// Benchmark graph algorithms
    pub async fn benchmark_algorithms(&self) -> Result<()> {
        println!("\n=== Graph Algorithm Benchmarks ===");

        let sizes = vec![1_000, 10_000, 100_000];

        for size in sizes {
            println!("\nGraph size: {} nodes", size);

            let (_cpu_graph, _) = self.generate_random_graph(size, 10.0)?;
            let gpu_graph = EnhancedGpuKnowledgeGraph::new(
                self.device.clone(),
                size,
                self.config.embedding_dim,
            )?;

            // PageRank
            let start = Instant::now();
            let _ = gpu_graph.pagerank(10, 0.85)?;
            println!("  PageRank (10 iter): {:.2}ms", start.elapsed().as_millis());

            // Community detection
            let start = Instant::now();
            let _ = gpu_graph.detect_communities()?;
            println!(
                "  Community detection: {:.2}ms",
                start.elapsed().as_millis()
            );

            // BFS paths
            if size > 100 {
                let start = Instant::now();
                for _ in 0..10 {
                    let src = rand::thread_rng().gen_range(0..size) as u32;
                    let dst = rand::thread_rng().gen_range(0..size) as u32;
                    let _ = gpu_graph.shortest_path(src, dst)?;
                }
                println!(
                    "  BFS (10 paths): {:.2}ms avg",
                    start.elapsed().as_millis() as f64 / 10.0
                );
            }
        }

        Ok(())
    }
}

/// Run knowledge graph benchmarks
pub async fn run_knowledge_graph_benchmarks() -> Result<()> {
    println!("=== GPU Knowledge Graph Benchmarks ===");

    let config = KnowledgeGraphBenchmarkConfig::default();
    let mut benchmark = GpuKnowledgeGraphBenchmark::new(0, config)?;

    // Run comprehensive benchmarks
    let results = benchmark.run_all().await?;

    // Print summary
    println!("\n=== Summary ===");
    for result in &results {
        println!(
            "Nodes: {}, Edges: {}, Build: {:.2}ms, Query: {:.2}μs, QPS: {:.0}",
            result.node_count,
            result.edge_count,
            result.build_time_ms,
            result.query_latency_us,
            result.queries_per_second
        );
    }

    // Run specialized benchmarks
    benchmark.benchmark_query_batches().await?;
    benchmark.benchmark_algorithms().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_knowledge_graph_benchmark_small() {
        let config = KnowledgeGraphBenchmarkConfig {
            node_counts: vec![100],
            avg_edges_per_node: vec![5.0],
            num_queries: 10,
            ..Default::default()
        };

        if let Ok(mut benchmark) = GpuKnowledgeGraphBenchmark::new(0, config) {
            let results = benchmark.run_all().await?;
            assert!(!results.is_empty());
            assert!(results[0].build_time_ms > 0.0);
            assert!(results[0].query_latency_us > 0.0);
        }
    }
}
