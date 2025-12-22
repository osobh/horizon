//! Storage benchmark suite for GPU agent swarms
//!
//! Measures storage performance under various real-world scenarios
//! following rust.md and cuda.md best practices

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

use crate::storage::{GpuAgentData, GpuAgentStorage, GpuKnowledgeGraph, GpuStorageConfig};

/// Storage benchmark scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageBenchmarkScenario {
    /// Burst write of many agents
    BurstWrite,
    /// Random access pattern
    RandomAccess,
    /// Hot/cold access pattern (20% hot, 80% cold)
    HotColdPattern,
    /// Full swarm checkpoint/restore
    SwarmCheckpoint,
    /// Knowledge graph updates
    KnowledgeGraphUpdate,
    /// Concurrent swarm access
    ConcurrentSwarmAccess,
    /// Memory pressure test
    MemoryPressure,
    /// Evolution generation persistence
    EvolutionPersistence,
}

impl StorageBenchmarkScenario {
    pub fn all() -> Vec<Self> {
        vec![
            Self::BurstWrite,
            Self::RandomAccess,
            Self::HotColdPattern,
            Self::SwarmCheckpoint,
            Self::KnowledgeGraphUpdate,
            Self::ConcurrentSwarmAccess,
            Self::MemoryPressure,
            Self::EvolutionPersistence,
        ]
    }
}

/// Storage benchmark configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StorageBenchmarkConfig {
    pub scenario: StorageBenchmarkScenario,
    pub agent_count: usize,
    pub iterations: usize,
    pub enable_gpu_cache: bool,
    pub cache_size_mb: usize,
    pub concurrent_tasks: usize,
    pub agent_state_size: usize,
    pub hot_agent_percentage: usize,
    pub measure_gpu_memory: bool,
    pub warmup_iterations: usize,
    pub use_compression: bool,
    pub storage_path: Option<String>,
}

impl Default for StorageBenchmarkConfig {
    fn default() -> Self {
        Self {
            scenario: StorageBenchmarkScenario::BurstWrite,
            agent_count: 10_000,
            iterations: 100,
            enable_gpu_cache: true,
            cache_size_mb: 1024,
            concurrent_tasks: 4,
            agent_state_size: 256,
            hot_agent_percentage: 20,
            measure_gpu_memory: true,
            warmup_iterations: 10,
            use_compression: false,
            storage_path: None,
        }
    }
}

/// Storage benchmark metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageBenchmarkMetrics {
    pub total_agents_stored: u64,
    pub total_agents_retrieved: u64,
    pub avg_store_latency_ms: f64,
    pub avg_retrieve_latency_ms: f64,
    pub p99_store_latency_ms: f64,
    pub p99_retrieve_latency_ms: f64,
    pub cache_hit_rate: f64,
    pub store_throughput_agents_per_sec: f64,
    pub retrieve_throughput_agents_per_sec: f64,
    pub concurrent_throughput: f64,
    pub hot_agent_avg_latency_ms: f64,
    pub cold_agent_avg_latency_ms: f64,
    pub peak_gpu_memory_mb: f64,
    pub gpu_memory_efficiency: f64,
    pub contention_rate: f64,
    pub compression_ratio: f64,
}

/// Storage benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageBenchmarkResults {
    pub scenario: StorageBenchmarkScenario,
    pub config: String, // Serialized config
    pub metrics: StorageBenchmarkMetrics,
    pub duration: Duration,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub gpu_info: String,
}

/// Storage benchmark runner
pub struct StorageBenchmark {
    config: StorageBenchmarkConfig,
    storage: Arc<GpuAgentStorage>,
    metrics: Arc<tokio::sync::Mutex<StorageBenchmarkMetrics>>,
}

impl StorageBenchmark {
    /// Create new storage benchmark
    pub fn new(config: StorageBenchmarkConfig) -> Result<Self> {
        // Configure storage
        let storage_config = if let Some(path) = &config.storage_path {
            GpuStorageConfig::with_base_path(path)
        } else {
            GpuStorageConfig::development() // Use temp dir for benchmarks
        };

        let mut storage_config = storage_config;
        storage_config.enable_gpu_cache = config.enable_gpu_cache;
        storage_config.cache_size_mb = config.cache_size_mb;
        storage_config.enable_compression = config.use_compression;

        let storage = GpuAgentStorage::new(storage_config)?;

        Ok(Self {
            config,
            storage: Arc::new(storage),
            metrics: Arc::new(tokio::sync::Mutex::new(StorageBenchmarkMetrics::default())),
        })
    }

    /// Run the benchmark
    pub async fn run(&self) -> Result<StorageBenchmarkResults> {
        println!("🔧 Running Storage Benchmark: {:?}", self.config.scenario);
        println!(
            "   Agents: {}, Iterations: {}",
            self.config.agent_count, self.config.iterations
        );

        // Warmup
        if self.config.warmup_iterations > 0 {
            println!(
                "   Warming up with {} iterations...",
                self.config.warmup_iterations
            );
            self.warmup().await?;
        }

        let start = Instant::now();

        // Run scenario
        match self.config.scenario {
            StorageBenchmarkScenario::BurstWrite => self.run_burst_write().await?,
            StorageBenchmarkScenario::RandomAccess => self.run_random_access().await?,
            StorageBenchmarkScenario::HotColdPattern => self.run_hot_cold_pattern().await?,
            StorageBenchmarkScenario::SwarmCheckpoint => self.run_swarm_checkpoint().await?,
            StorageBenchmarkScenario::KnowledgeGraphUpdate => self.run_knowledge_graph().await?,
            StorageBenchmarkScenario::ConcurrentSwarmAccess => self.run_concurrent_access().await?,
            StorageBenchmarkScenario::MemoryPressure => self.run_memory_pressure().await?,
            StorageBenchmarkScenario::EvolutionPersistence => {
                self.run_evolution_persistence().await?
            }
        }

        let duration = start.elapsed();

        // Get final metrics
        let metrics = self.metrics.lock().await.clone();

        // Get GPU info
        let gpu_info = if let Ok(_device) = cudarc::driver::CudaDevice::new(0) {
            format!("CUDA Device 0")
        } else {
            "No GPU".to_string()
        };

        Ok(StorageBenchmarkResults {
            scenario: self.config.scenario,
            config: serde_json::to_string(&self.config)?,
            metrics,
            duration,
            timestamp: chrono::Utc::now(),
            gpu_info,
        })
    }

    /// Warmup iterations
    async fn warmup(&self) -> Result<()> {
        for i in 0..self.config.warmup_iterations {
            let agent = self.create_test_agent(&format!("warmup_{}", i));
            self.storage.store_agent(&agent.id, &agent).await?;
            let _ = self.storage.retrieve_agent(&agent.id).await?;
        }
        Ok(())
    }

    /// Burst write scenario
    async fn run_burst_write(&self) -> Result<()> {
        let mut store_latencies = Vec::new();

        for i in 0..self.config.agent_count {
            let agent = self.create_test_agent(&format!("burst_{}", i));

            let start = Instant::now();
            self.storage.store_agent(&agent.id, &agent).await?;
            let latency = start.elapsed();

            store_latencies.push(latency.as_secs_f64() * 1000.0);

            let mut metrics = self.metrics.lock().await;
            metrics.total_agents_stored += 1;
        }

        // Calculate metrics
        self.update_latency_metrics(&store_latencies, true).await;

        Ok(())
    }

    /// Random access scenario
    async fn run_random_access(&self) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // First, populate storage
        for i in 0..self.config.agent_count {
            let agent = self.create_test_agent(&format!("random_{}", i));
            self.storage.store_agent(&agent.id, &agent).await?;
        }

        let mut retrieve_latencies = Vec::new();

        // Random access pattern
        for _ in 0..self.config.iterations {
            let id = rng.gen_range(0..self.config.agent_count);
            let agent_id = format!("random_{}", id);

            let start = Instant::now();
            let _ = self.storage.retrieve_agent(&agent_id).await?;
            let latency = start.elapsed();

            retrieve_latencies.push(latency.as_secs_f64() * 1000.0);

            let mut metrics = self.metrics.lock().await;
            metrics.total_agents_retrieved += 1;
        }

        self.update_latency_metrics(&retrieve_latencies, false)
            .await;

        Ok(())
    }

    /// Hot/cold access pattern
    async fn run_hot_cold_pattern(&self) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Populate storage
        for i in 0..self.config.agent_count {
            let agent = self.create_test_agent(&format!("hotcold_{}", i));
            self.storage.store_agent(&agent.id, &agent).await?;
        }

        // Determine hot agents (first 20%)
        let hot_count = self.config.agent_count * self.config.hot_agent_percentage / 100;

        // Pre-warm cache with hot agents
        for i in 0..hot_count {
            self.storage.cache_agent(&format!("hotcold_{}", i)).await?;
        }

        let mut hot_latencies = Vec::new();
        let mut cold_latencies = Vec::new();
        let cache_stats_start = self.storage.cache_stats().await?;

        // Access pattern: 80% hot agents, 20% cold agents
        for _ in 0..self.config.iterations {
            let (agent_id, is_hot) = if rng.gen_bool(0.8) {
                // Access hot agent
                let id = rng.gen_range(0..hot_count);
                (format!("hotcold_{}", id), true)
            } else {
                // Access cold agent
                let id = rng.gen_range(hot_count..self.config.agent_count);
                (format!("hotcold_{}", id), false)
            };

            let start = Instant::now();
            let _ = self.storage.retrieve_agent(&agent_id).await?;
            let latency = start.elapsed().as_secs_f64() * 1000.0;

            if is_hot {
                hot_latencies.push(latency);
            } else {
                cold_latencies.push(latency);
            }
        }

        let cache_stats_end = self.storage.cache_stats().await?;

        // Update metrics
        let mut metrics = self.metrics.lock().await;
        metrics.hot_agent_avg_latency_ms =
            hot_latencies.iter().sum::<f64>() / hot_latencies.len() as f64;
        metrics.cold_agent_avg_latency_ms =
            cold_latencies.iter().sum::<f64>() / cold_latencies.len() as f64;

        let total_accesses = cache_stats_end.cache_hits + cache_stats_end.cache_misses
            - cache_stats_start.cache_hits
            - cache_stats_start.cache_misses;
        let hits = cache_stats_end.cache_hits - cache_stats_start.cache_hits;
        metrics.cache_hit_rate = hits as f64 / total_accesses as f64;

        Ok(())
    }

    /// Concurrent access scenario
    async fn run_concurrent_access(&self) -> Result<()> {
        // Populate storage
        for i in 0..self.config.agent_count {
            let agent = self.create_test_agent(&format!("concurrent_{}", i));
            self.storage.store_agent(&agent.id, &agent).await?;
        }

        let storage = self.storage.clone();
        let agent_count = self.config.agent_count;
        let iterations_per_task = self.config.iterations / self.config.concurrent_tasks;

        let start = Instant::now();
        let mut tasks = JoinSet::new();

        // Spawn concurrent tasks
        for task_id in 0..self.config.concurrent_tasks {
            let storage_clone = storage.clone();

            tasks.spawn(async move {
                let mut task_latencies = Vec::new();

                for i in 0..iterations_per_task {
                    let agent_id = format!(
                        "concurrent_{}",
                        (task_id * iterations_per_task + i) % agent_count
                    );

                    let op_start = Instant::now();
                    let _ = storage_clone.retrieve_agent(&agent_id).await?;
                    task_latencies.push(op_start.elapsed().as_secs_f64() * 1000.0);
                }

                Ok::<Vec<f64>, anyhow::Error>(task_latencies)
            });
        }

        // Collect results
        let mut all_latencies = Vec::new();
        while let Some(result) = tasks.join_next().await {
            let latencies = result??;
            all_latencies.extend(latencies);
        }

        let duration = start.elapsed();

        // Calculate concurrent throughput
        let mut metrics = self.metrics.lock().await;
        metrics.concurrent_throughput = all_latencies.len() as f64 / duration.as_secs_f64();
        metrics.total_agents_retrieved += all_latencies.len() as u64;

        Ok(())
    }

    /// Swarm checkpoint scenario
    async fn run_swarm_checkpoint(&self) -> Result<()> {
        // Create a swarm worth of agents
        let mut agents = Vec::new();
        for i in 0..self.config.agent_count {
            agents.push(self.create_test_agent(&format!("swarm_{}", i)));
        }

        // Checkpoint (store all agents)
        let checkpoint_start = Instant::now();
        for agent in &agents {
            self.storage.store_agent(&agent.id, agent).await?;
        }
        let checkpoint_duration = checkpoint_start.elapsed();

        // Clear cache to simulate cold restore
        // Note: In real implementation, we'd have a cache clear method

        // Restore (retrieve all agents)
        let restore_start = Instant::now();
        for agent in &agents {
            let _ = self.storage.retrieve_agent(&agent.id).await?;
        }
        let restore_duration = restore_start.elapsed();

        // Update metrics
        let mut metrics = self.metrics.lock().await;
        metrics.total_agents_stored = agents.len() as u64;
        metrics.total_agents_retrieved = agents.len() as u64;
        metrics.store_throughput_agents_per_sec =
            agents.len() as f64 / checkpoint_duration.as_secs_f64();
        metrics.retrieve_throughput_agents_per_sec =
            agents.len() as f64 / restore_duration.as_secs_f64();

        Ok(())
    }

    /// Knowledge graph update scenario
    async fn run_knowledge_graph(&self) -> Result<()> {
        use crate::storage::{GraphEdge as StorageGraphEdge, GraphNode as StorageGraphNode};

        let mut store_latencies = Vec::new();
        let mut retrieve_latencies = Vec::new();

        for i in 0..self.config.iterations {
            // Create a knowledge graph
            let mut nodes = Vec::new();
            let mut edges = Vec::new();

            let node_count = 100; // Smaller graphs, many iterations

            for j in 0..node_count {
                nodes.push(StorageGraphNode {
                    id: format!("node_{}_{}", i, j),
                    embedding: vec![j as f32 * 0.1; 768],
                    metadata: HashMap::new(),
                });
            }

            for j in 0..node_count - 1 {
                edges.push(StorageGraphEdge {
                    source: format!("node_{}_{}", i, j),
                    target: format!("node_{}_{}", i, j + 1),
                    weight: 0.5,
                    edge_type: "sequence".to_string(),
                });
            }

            let graph = GpuKnowledgeGraph { nodes, edges };
            let graph_id = format!("graph_{}", i);

            // Store graph
            let store_start = Instant::now();
            self.storage
                .store_knowledge_graph(&graph_id, &graph)
                .await?;
            store_latencies.push(store_start.elapsed().as_secs_f64() * 1000.0);

            // Retrieve graph
            let retrieve_start = Instant::now();
            let _ = self.storage.retrieve_knowledge_graph(&graph_id).await?;
            retrieve_latencies.push(retrieve_start.elapsed().as_secs_f64() * 1000.0);
        }

        // Update metrics for graphs
        let mut metrics = self.metrics.lock().await;
        metrics.avg_store_latency_ms =
            store_latencies.iter().sum::<f64>() / store_latencies.len() as f64;
        metrics.avg_retrieve_latency_ms =
            retrieve_latencies.iter().sum::<f64>() / retrieve_latencies.len() as f64;

        Ok(())
    }

    /// Memory pressure scenario
    async fn run_memory_pressure(&self) -> Result<()> {
        // Create agents with large state vectors
        let large_state_size = self.config.agent_state_size * 4; // 4x normal size

        for i in 0..self.config.agent_count {
            let mut agent = self.create_test_agent(&format!("pressure_{}", i));
            agent.state = vec![i as f32; large_state_size];
            agent.memory = vec![0.0; large_state_size / 2];

            self.storage.store_agent(&agent.id, &agent).await?;

            // Periodically check memory usage
            if i % 1000 == 0 && self.config.measure_gpu_memory {
                // In real implementation, we'd query GPU memory usage
                // For now, we estimate based on agent size
                let estimated_mb = (i * large_state_size * 4) as f64 / (1024.0 * 1024.0);

                let mut metrics = self.metrics.lock().await;
                metrics.peak_gpu_memory_mb = metrics.peak_gpu_memory_mb.max(estimated_mb);
            }
        }

        // Calculate memory efficiency
        let mut metrics = self.metrics.lock().await;
        let expected_size =
            (self.config.agent_count * large_state_size * 4) as f64 / (1024.0 * 1024.0);
        metrics.gpu_memory_efficiency = expected_size / metrics.peak_gpu_memory_mb.max(1.0);

        Ok(())
    }

    /// Evolution persistence scenario
    async fn run_evolution_persistence(&self) -> Result<()> {
        let generations = 10;
        let population_size = self.config.agent_count / generations;

        for gen in 0..generations {
            // Store generation
            let gen_start = Instant::now();

            for i in 0..population_size {
                let mut agent = self.create_test_agent(&format!("gen{}_{}", gen, i));
                agent.generation = gen as u64;
                agent.fitness = rand::random::<f64>();

                self.storage.store_agent(&agent.id, &agent).await?;
            }

            // Retrieve best agents from previous generation
            if gen > 0 {
                for i in 0..population_size / 10 {
                    // Top 10% of previous generation
                    let agent_id = format!("gen{}_{}", gen - 1, i);
                    let _ = self.storage.retrieve_agent(&agent_id).await?;
                }
            }

            let gen_duration = gen_start.elapsed();
            println!(
                "   Generation {} persisted in {:.2}ms",
                gen,
                gen_duration.as_secs_f64() * 1000.0
            );
        }

        let mut metrics = self.metrics.lock().await;
        metrics.total_agents_stored = (generations * population_size) as u64;

        Ok(())
    }

    /// Create test agent
    fn create_test_agent(&self, id: &str) -> GpuAgentData {
        GpuAgentData {
            id: id.to_string(),
            state: vec![rand::random(); self.config.agent_state_size],
            memory: vec![0.0; self.config.agent_state_size / 2],
            generation: 0,
            fitness: rand::random(),
            metadata: HashMap::new(),
        }
    }

    /// Update latency metrics
    async fn update_latency_metrics(&self, latencies: &[f64], is_store: bool) {
        if latencies.is_empty() {
            return;
        }

        let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;

        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b)?);
        let p99_idx = (sorted.len() as f64 * 0.99) as usize;
        let p99 = sorted[p99_idx.min(sorted.len() - 1)];

        let mut metrics = self.metrics.lock().await;

        if is_store {
            metrics.avg_store_latency_ms = avg;
            metrics.p99_store_latency_ms = p99;
            metrics.store_throughput_agents_per_sec = 1000.0 / avg; // Convert ms to agents/sec
        } else {
            metrics.avg_retrieve_latency_ms = avg;
            metrics.p99_retrieve_latency_ms = p99;
            metrics.retrieve_throughput_agents_per_sec = 1000.0 / avg;
        }
    }
}

/// Benchmark report
pub struct StorageBenchmarkReport {
    pub summary: String,
    pub warnings: Vec<String>,
}

impl StorageBenchmarkReport {
    pub fn from_metrics(metrics: StorageBenchmarkMetrics) -> Self {
        let mut summary = String::new();
        let mut warnings = Vec::new();

        summary.push_str("📊 Storage Benchmark Results\n");
        summary.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

        summary.push_str(&format!(
            "Total Agents Stored: {:>12}\n",
            format_number(metrics.total_agents_stored)
        ));
        summary.push_str(&format!(
            "Total Agents Retrieved: {:>9}\n",
            format_number(metrics.total_agents_retrieved)
        ));
        summary.push_str(&format!(
            "Cache Hit Rate: {:>17.1}%\n",
            metrics.cache_hit_rate * 100.0
        ));

        summary.push_str("\nLatency Metrics:\n");
        summary.push_str(&format!(
            "  Store Avg: {:>10.2}ms  P99: {:.2}ms\n",
            metrics.avg_store_latency_ms, metrics.p99_store_latency_ms
        ));
        summary.push_str(&format!(
            "  Retrieve Avg: {:>7.2}ms  P99: {:.2}ms\n",
            metrics.avg_retrieve_latency_ms, metrics.p99_retrieve_latency_ms
        ));

        summary.push_str("\nThroughput:\n");
        summary.push_str(&format!(
            "  Store: {:>14} agents/sec\n",
            format_number(metrics.store_throughput_agents_per_sec as u64)
        ));
        summary.push_str(&format!(
            "  Retrieve: {:>11} agents/sec\n",
            format_number(metrics.retrieve_throughput_agents_per_sec as u64)
        ));

        if metrics.concurrent_throughput > 0.0 {
            summary.push_str(&format!(
                "  Concurrent: {:>9} agents/sec\n",
                format_number(metrics.concurrent_throughput as u64)
            ));
        }

        // Warnings
        if metrics.cache_hit_rate < 0.5 {
            warnings.push("⚠️  Low cache hit rate - consider increasing cache size".to_string());
        }

        if metrics.avg_store_latency_ms > 10.0 {
            warnings.push("⚠️  High store latency - check disk I/O performance".to_string());
        }

        if metrics.contention_rate > 0.2 {
            warnings.push(
                "⚠️  High contention rate - consider optimizing concurrent access".to_string(),
            );
        }

        Self { summary, warnings }
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

/// Compare benchmark scenarios
pub fn compare_scenarios(results: Vec<StorageBenchmarkResults>) -> ScenarioComparison {
    let mut best_write_throughput = 0.0;
    let mut best_read_throughput = 0.0;
    let mut best_write_scenario = StorageBenchmarkScenario::BurstWrite;
    let mut best_read_scenario = StorageBenchmarkScenario::RandomAccess;

    for result in &results {
        if result.metrics.store_throughput_agents_per_sec > best_write_throughput {
            best_write_throughput = result.metrics.store_throughput_agents_per_sec;
            best_write_scenario = result.scenario;
        }

        if result.metrics.retrieve_throughput_agents_per_sec > best_read_throughput {
            best_read_throughput = result.metrics.retrieve_throughput_agents_per_sec;
            best_read_scenario = result.scenario;
        }
    }

    ScenarioComparison {
        best_write_scenario,
        best_write_throughput,
        best_read_scenario,
        best_read_throughput,
    }
}

pub struct ScenarioComparison {
    pub best_write_scenario: StorageBenchmarkScenario,
    pub best_write_throughput: f64,
    pub best_read_scenario: StorageBenchmarkScenario,
    pub best_read_throughput: f64,
}

/// Format large numbers with commas
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;

    for c in s.chars().rev() {
        if count == 3 {
            result.push(',');
            count = 0;
        }
        result.push(c);
        count += 1;
    }

    result.chars().rev().collect()
}

#[cfg(test)]
mod storage_benchmark_tests;
