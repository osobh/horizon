//! GPU-accelerated evaluation pipeline for agent fitness assessment

use crate::{
    AgentGenome, BenchmarkResult, EvaluationMetrics, EvaluationResult, EvolutionStreamingError,
};
use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use stratoswarm_streaming::{StreamChunk, StreamProcessor, StreamStats};
use tokio::sync::Semaphore;

pub mod benchmarks;
pub mod gpu;

pub use benchmarks::*;
pub use gpu::*;

/// GPU batch evaluator for high-throughput agent evaluation
pub struct GpuBatchEvaluator {
    id: String,
    batch_size: usize,
    timeout: Duration,
    benchmark_suite: BenchmarkSuite,
    gpu_executor: GpuBatchExecutor,
    stats: Arc<EvaluationStats>,
}

/// Thread-safe statistics for evaluation
///
/// Cache-line aligned (64 bytes) to prevent false sharing when
/// multiple evaluation threads update counters concurrently.
#[repr(C, align(64))]
#[derive(Debug, Default)]
struct EvaluationStats {
    agents_evaluated: AtomicU64,
    batches_processed: AtomicU64,
    evaluation_time_ns: AtomicU64,
    gpu_time_ns: AtomicU64,
    compilation_failures: AtomicU64,
    execution_failures: AtomicU64,
    // Padding to fill cache line (6 * 8 = 48 bytes, need 16 more)
    _padding: [u8; 16],
}

/// Evaluation configuration
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub timeout: Duration,
    pub memory_limit: u64,
    pub measure_power: bool,
    pub enable_profiling: bool,
    pub max_concurrent_evaluations: usize,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(10),
            memory_limit: 1024 * 1024 * 1024, // 1GB
            measure_power: false,
            enable_profiling: false,
            max_concurrent_evaluations: 64,
        }
    }
}

impl GpuBatchEvaluator {
    /// Create a new GPU batch evaluator
    pub fn new(id: String) -> Self {
        Self {
            id,
            batch_size: 64,
            timeout: Duration::from_secs(30),
            benchmark_suite: BenchmarkSuite::default(),
            gpu_executor: GpuBatchExecutor::new(),
            stats: Arc::new(EvaluationStats::default()),
        }
    }

    /// Configure batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Configure timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Configure benchmark suite
    pub fn with_benchmark_suite(mut self, suite: BenchmarkSuite) -> Self {
        self.benchmark_suite = suite;
        self
    }

    /// Get benchmark suite reference
    pub fn benchmark_suite(&self) -> &BenchmarkSuite {
        &self.benchmark_suite
    }

    /// Get current statistics snapshot
    pub fn get_stats_snapshot(&self) -> StreamStats {
        let agents = self.stats.agents_evaluated.load(Ordering::Relaxed);
        let batches = self.stats.batches_processed.load(Ordering::Relaxed);
        let eval_time_ns = self.stats.evaluation_time_ns.load(Ordering::Relaxed);
        let _gpu_time_ns = self.stats.gpu_time_ns.load(Ordering::Relaxed);
        let failures = self.stats.compilation_failures.load(Ordering::Relaxed)
            + self.stats.execution_failures.load(Ordering::Relaxed);

        let throughput_agents_per_sec = if eval_time_ns > 0 {
            (agents as f64) / ((eval_time_ns as f64) / 1_000_000_000.0)
        } else {
            0.0
        };

        StreamStats {
            chunks_processed: batches,
            bytes_processed: agents * 1000, // Rough estimate
            processing_time_ms: eval_time_ns / 1_000_000,
            throughput_mbps: throughput_agents_per_sec / 1_000.0, // Agents/sec to "MB/s" equivalent
            errors: failures,
        }
    }

    /// Evaluate a batch of agents
    pub async fn evaluate_batch(
        &self,
        agents: Vec<AgentGenome>,
    ) -> Result<Vec<EvaluationResult>, EvolutionStreamingError> {
        let start_time = Instant::now();
        let batch_size = agents.len();

        // Compile agents for GPU execution
        let compiled_agents = self.compile_agents(agents).await?;

        // Prepare GPU batch
        let gpu_batch = self
            .gpu_executor
            .prepare_batch(&compiled_agents)
            .await
            .map_err(|e| {
                EvolutionStreamingError::EvaluationFailed(format!(
                    "Failed to prepare GPU batch: {}",
                    e
                ))
            })?;

        // Run benchmarks
        let mut all_results = Vec::new();

        for benchmark in self.benchmark_suite.benchmarks() {
            let benchmark_start = Instant::now();

            // Execute batch on GPU
            let execution_results = gpu_batch
                .execute_benchmark(benchmark.as_ref(), self.timeout)
                .await
                .map_err(|e| {
                    EvolutionStreamingError::EvaluationFailed(format!(
                        "GPU execution failed: {}",
                        e
                    ))
                })?;

            let benchmark_time = benchmark_start.elapsed();
            self.stats
                .gpu_time_ns
                .fetch_add(benchmark_time.as_nanos() as u64, Ordering::Relaxed);

            // Process results
            for (idx, execution_result) in execution_results.into_iter().enumerate() {
                if let Some(existing_result) = all_results
                    .iter_mut()
                    .find(|r: &&mut EvaluationResult| r.agent_id == compiled_agents[idx].id)
                {
                    // Add benchmark result to existing evaluation
                    existing_result
                        .metrics
                        .benchmark_results
                        .push(BenchmarkResult {
                            name: benchmark.name(),
                            score: execution_result.score,
                            execution_time: execution_result.execution_time.as_nanos() as u64,
                            passed: execution_result.passed,
                        });
                } else {
                    // Create new evaluation result
                    let mut eval_result =
                        EvaluationResult::new(compiled_agents[idx].id, execution_result.score);
                    eval_result.execution_time = execution_result.execution_time.as_nanos() as u64;
                    eval_result.memory_usage = execution_result.memory_usage;
                    eval_result.metrics.benchmark_results.push(BenchmarkResult {
                        name: benchmark.name(),
                        score: execution_result.score,
                        execution_time: execution_result.execution_time.as_nanos() as u64,
                        passed: execution_result.passed,
                    });
                    all_results.push(eval_result);
                }
            }
        }

        // Calculate final fitness scores
        for result in &mut all_results {
            result.fitness = self.calculate_fitness(&result.metrics);
        }

        let total_time = start_time.elapsed().as_nanos() as u64;
        self.stats
            .evaluation_time_ns
            .fetch_add(total_time, Ordering::Relaxed);
        self.stats
            .agents_evaluated
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.stats.batches_processed.fetch_add(1, Ordering::Relaxed);

        Ok(all_results)
    }

    /// Compile agents for GPU execution
    async fn compile_agents(
        &self,
        agents: Vec<AgentGenome>,
    ) -> Result<Vec<CompiledAgent>, EvolutionStreamingError> {
        let mut compiled = Vec::new();

        for agent in agents {
            match self.compile_single_agent(&agent).await {
                Ok(compiled_agent) => compiled.push(compiled_agent),
                Err(e) => {
                    self.stats
                        .compilation_failures
                        .fetch_add(1, Ordering::Relaxed);
                    eprintln!("Failed to compile agent {}: {}", agent.id, e);
                    // Continue with other agents
                }
            }
        }

        if compiled.is_empty() {
            return Err(EvolutionStreamingError::EvaluationFailed(
                "No agents could be compiled".to_string(),
            ));
        }

        Ok(compiled)
    }

    /// Compile single agent
    async fn compile_single_agent(
        &self,
        agent: &AgentGenome,
    ) -> Result<CompiledAgent, EvolutionStreamingError> {
        // Simulate compilation process
        tokio::time::sleep(Duration::from_millis(1)).await; // Simulate compilation time

        // Basic validation
        if agent.code.is_empty() {
            return Err(EvolutionStreamingError::EvaluationFailed(
                "Agent code is empty".to_string(),
            ));
        }

        Ok(CompiledAgent {
            id: agent.id,
            kernel_code: agent.code.clone(),
            parameters: agent.parameters.clone(),
            compilation_time: Duration::from_millis(10), // Simulated
        })
    }

    /// Calculate fitness from evaluation metrics
    fn calculate_fitness(&self, metrics: &EvaluationMetrics) -> f64 {
        if metrics.benchmark_results.is_empty() {
            return 0.0;
        }

        let avg_score = metrics
            .benchmark_results
            .iter()
            .map(|r| r.score)
            .sum::<f64>()
            / metrics.benchmark_results.len() as f64;

        let pass_rate = metrics
            .benchmark_results
            .iter()
            .filter(|r| r.passed)
            .count() as f64
            / metrics.benchmark_results.len() as f64;

        // Combined fitness: 70% average score + 30% pass rate
        avg_score * 0.7 + pass_rate * 0.3
    }
}

#[async_trait]
impl StreamProcessor for GpuBatchEvaluator {
    async fn process(
        &mut self,
        chunk: StreamChunk,
    ) -> Result<StreamChunk, stratoswarm_streaming::StreamingError> {
        // Deserialize agent from chunk
        let agent_data = String::from_utf8_lossy(&chunk.data);
        let agent: AgentGenome = serde_json::from_str(&agent_data).map_err(|e| {
            stratoswarm_streaming::StreamingError::ProcessingFailed {
                reason: format!("Failed to deserialize agent: {e}"),
            }
        })?;

        // Evaluate single agent
        let results = self.evaluate_batch(vec![agent]).await.map_err(|e| {
            stratoswarm_streaming::StreamingError::ProcessingFailed {
                reason: e.to_string(),
            }
        })?;

        // Serialize result
        let result_data = serde_json::to_string(&results).map_err(|e| {
            stratoswarm_streaming::StreamingError::ProcessingFailed {
                reason: format!("Failed to serialize evaluation results: {e}"),
            }
        })?;

        Ok(StreamChunk::new(
            bytes::Bytes::from(result_data),
            chunk.sequence,
            chunk.metadata.source_id,
        ))
    }

    async fn process_batch(
        &mut self,
        chunks: Vec<StreamChunk>,
    ) -> Result<Vec<StreamChunk>, stratoswarm_streaming::StreamingError> {
        let mut agents = Vec::new();

        // Deserialize all agents
        for chunk in &chunks {
            let agent_data = String::from_utf8_lossy(&chunk.data);
            let agent: AgentGenome = serde_json::from_str(&agent_data).map_err(|e| {
                stratoswarm_streaming::StreamingError::ProcessingFailed {
                    reason: format!("Failed to deserialize agent: {e}"),
                }
            })?;
            agents.push(agent);
        }

        // Evaluate batch
        let results = self.evaluate_batch(agents).await.map_err(|e| {
            stratoswarm_streaming::StreamingError::ProcessingFailed {
                reason: e.to_string(),
            }
        })?;

        // Create result chunks
        let mut result_chunks = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            if i < results.len() {
                let result_data = serde_json::to_string(&results[i]).map_err(|e| {
                    stratoswarm_streaming::StreamingError::ProcessingFailed {
                        reason: format!("Failed to serialize evaluation result: {e}"),
                    }
                })?;

                result_chunks.push(StreamChunk::new(
                    bytes::Bytes::from(result_data),
                    chunk.sequence,
                    chunk.metadata.source_id.clone(),
                ));
            }
        }

        Ok(result_chunks)
    }

    async fn stats(&self) -> Result<StreamStats, stratoswarm_streaming::StreamingError> {
        Ok(self.get_stats_snapshot())
    }

    fn processor_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_batch_evaluator_creation() {
        let evaluator = GpuBatchEvaluator::new("test-evaluator".to_string());

        assert_eq!(evaluator.processor_id(), "test-evaluator");
        assert_eq!(evaluator.batch_size, 64);
        assert_eq!(evaluator.timeout, Duration::from_secs(30));
    }

    #[tokio::test]
    async fn test_evaluator_configuration() {
        let evaluator = GpuBatchEvaluator::new("config-test".to_string())
            .with_batch_size(32)
            .with_timeout(Duration::from_secs(60));

        assert_eq!(evaluator.batch_size, 32);
        assert_eq!(evaluator.timeout, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_single_agent_compilation() {
        let evaluator = GpuBatchEvaluator::new("compile-test".to_string());
        let agent = AgentGenome::new("fn compute() { return 42; }".to_string(), vec![1.0, 2.0]);

        let compiled = evaluator.compile_single_agent(&agent).await.unwrap();
        assert_eq!(compiled.id, agent.id);
        assert_eq!(compiled.kernel_code, agent.code);
        assert_eq!(compiled.parameters, agent.parameters);
    }

    #[tokio::test]
    async fn test_empty_code_compilation_failure() {
        let evaluator = GpuBatchEvaluator::new("empty-test".to_string());
        let agent = AgentGenome::new("".to_string(), vec![]);

        let result = evaluator.compile_single_agent(&agent).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fitness_calculation() {
        let evaluator = GpuBatchEvaluator::new("fitness-test".to_string());

        let mut metrics = EvaluationMetrics::default();
        metrics.benchmark_results = vec![
            BenchmarkResult {
                name: "test1".to_string(),
                score: 0.8,
                execution_time: 1000,
                passed: true,
            },
            BenchmarkResult {
                name: "test2".to_string(),
                score: 0.6,
                execution_time: 2000,
                passed: true,
            },
        ];

        let fitness = evaluator.calculate_fitness(&metrics);
        assert!(fitness > 0.0);
        assert!(fitness <= 1.0);

        // 70% of average score (0.7) + 30% of pass rate (1.0) = 0.49 + 0.3 = 0.79
        assert!((fitness - 0.79).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_evaluation_stats() {
        let evaluator = GpuBatchEvaluator::new("stats-test".to_string());

        // Initial stats
        let initial_stats = evaluator.get_stats_snapshot();
        assert_eq!(initial_stats.chunks_processed, 0);
        assert_eq!(initial_stats.errors, 0);

        // Process some agents
        let agents = vec![
            AgentGenome::new("fn test1() { 1 }".to_string(), vec![1.0]),
            AgentGenome::new("fn test2() { 2 }".to_string(), vec![2.0]),
        ];

        let _results = evaluator.evaluate_batch(agents).await.unwrap();

        let final_stats = evaluator.get_stats_snapshot();
        assert_eq!(final_stats.chunks_processed, 1); // One batch processed
        assert!(final_stats.processing_time_ms > 0);
    }

    #[test]
    fn test_evaluation_config_default() {
        let config = EvaluationConfig::default();

        assert_eq!(config.timeout, Duration::from_secs(10));
        assert_eq!(config.memory_limit, 1024 * 1024 * 1024);
        assert!(!config.measure_power);
        assert!(!config.enable_profiling);
        assert_eq!(config.max_concurrent_evaluations, 64);
    }

    #[tokio::test]
    async fn test_batch_evaluation_multiple_agents() {
        let evaluator = GpuBatchEvaluator::new("batch-test".to_string());

        let agents = vec![
            AgentGenome::new("fn kernel1() { return 1; }".to_string(), vec![1.0, 2.0]),
            AgentGenome::new("fn kernel2() { return 2; }".to_string(), vec![3.0, 4.0]),
            AgentGenome::new("fn kernel3() { return 3; }".to_string(), vec![5.0, 6.0]),
        ];

        let results = evaluator.evaluate_batch(agents.clone()).await.unwrap();
        assert_eq!(results.len(), 3);

        // Each result should have the corresponding agent ID
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.agent_id, agents[i].id);
            assert!(result.fitness >= 0.0);
            assert!(result.fitness <= 1.0);
            assert!(result.execution_time > 0);
        }
    }

    #[tokio::test]
    async fn test_compile_agents_mixed_success() {
        let evaluator = GpuBatchEvaluator::new("compile-mixed-test".to_string());

        let agents = vec![
            AgentGenome::new("valid code".to_string(), vec![1.0]),
            AgentGenome::new("".to_string(), vec![2.0]), // Empty code - should fail
            AgentGenome::new("more valid code".to_string(), vec![3.0]),
        ];

        let compiled = evaluator.compile_agents(agents).await.unwrap();
        // Should only have 2 compiled agents (the valid ones)
        assert_eq!(compiled.len(), 2);

        // Check compilation failure stats
        let stats = evaluator.get_stats_snapshot();
        assert_eq!(stats.errors, 1); // One compilation failure
    }

    #[tokio::test]
    async fn test_compile_agents_all_fail() {
        let evaluator = GpuBatchEvaluator::new("compile-fail-test".to_string());

        let agents = vec![
            AgentGenome::new("".to_string(), vec![1.0]),
            AgentGenome::new("".to_string(), vec![2.0]),
        ];

        let result = evaluator.compile_agents(agents).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No agents could be compiled"));
    }

    #[tokio::test]
    async fn test_fitness_calculation_edge_cases() {
        let evaluator = GpuBatchEvaluator::new("fitness-edge-test".to_string());

        // Empty benchmark results
        let empty_metrics = EvaluationMetrics::default();
        let fitness = evaluator.calculate_fitness(&empty_metrics);
        assert_eq!(fitness, 0.0);

        // All failing benchmarks
        let mut failing_metrics = EvaluationMetrics::default();
        failing_metrics.benchmark_results = vec![
            BenchmarkResult {
                name: "fail1".to_string(),
                score: 0.5,
                execution_time: 1000,
                passed: false,
            },
            BenchmarkResult {
                name: "fail2".to_string(),
                score: 0.7,
                execution_time: 2000,
                passed: false,
            },
        ];

        let failing_fitness = evaluator.calculate_fitness(&failing_metrics);
        // 70% of average score (0.6) + 30% of pass rate (0.0) = 0.42
        assert!((failing_fitness - 0.42).abs() < 0.01);

        // Perfect benchmarks
        let mut perfect_metrics = EvaluationMetrics::default();
        perfect_metrics.benchmark_results = vec![
            BenchmarkResult {
                name: "perfect1".to_string(),
                score: 1.0,
                execution_time: 500,
                passed: true,
            },
            BenchmarkResult {
                name: "perfect2".to_string(),
                score: 1.0,
                execution_time: 600,
                passed: true,
            },
        ];

        let perfect_fitness = evaluator.calculate_fitness(&perfect_metrics);
        // 70% of average score (1.0) + 30% of pass rate (1.0) = 1.0
        assert!((perfect_fitness - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_evaluator_with_custom_benchmark_suite() {
        let custom_suite = BenchmarkSuite::new().add_benchmark(Box::new(
            PerformanceBenchmark::new("custom_bench".to_string(), 1000.0),
        ));

        let evaluator = GpuBatchEvaluator::new("custom-bench-test".to_string())
            .with_benchmark_suite(custom_suite);

        // Verify benchmark suite is set
        assert_eq!(evaluator.benchmark_suite().benchmarks().len(), 1);

        let agent = AgentGenome::new("fn test() { 42 }".to_string(), vec![1.0]);
        let results = evaluator.evaluate_batch(vec![agent]).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_stream_processor_interface() {
        let mut evaluator = GpuBatchEvaluator::new("stream-test".to_string());

        let agent = AgentGenome::new("fn test() { 100 }".to_string(), vec![1.0, 2.0]);
        let agent_json = serde_json::to_string(&agent).unwrap();

        let chunk = StreamChunk::new(bytes::Bytes::from(agent_json), 1, "test-source".to_string());

        let result = evaluator.process(chunk).await.unwrap();
        assert!(!result.data.is_empty());
        assert_eq!(result.sequence, 1);

        let stats = evaluator.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
    }

    #[tokio::test]
    async fn test_stream_processor_batch_interface() {
        let mut evaluator = GpuBatchEvaluator::new("stream-batch-test".to_string());

        let agents = vec![
            AgentGenome::new("fn test1() { 1 }".to_string(), vec![1.0]),
            AgentGenome::new("fn test2() { 2 }".to_string(), vec![2.0]),
        ];

        let chunks: Vec<StreamChunk> = agents
            .iter()
            .enumerate()
            .map(|(i, agent)| {
                let json = serde_json::to_string(agent).unwrap();
                StreamChunk::new(bytes::Bytes::from(json), i as u64, format!("source-{i}"))
            })
            .collect();

        let results = evaluator.process_batch(chunks).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_evaluation_stats_accumulation() {
        let evaluator = GpuBatchEvaluator::new("stats-accum-test".to_string());

        // Process first batch
        let agents1 = vec![AgentGenome::new("fn test1() {}".to_string(), vec![1.0])];
        evaluator.evaluate_batch(agents1).await.unwrap();

        let stats1 = evaluator.get_stats_snapshot();
        assert_eq!(stats1.chunks_processed, 1);

        // Process second batch
        let agents2 = vec![
            AgentGenome::new("fn test2() {}".to_string(), vec![2.0]),
            AgentGenome::new("fn test3() {}".to_string(), vec![3.0]),
        ];
        evaluator.evaluate_batch(agents2).await.unwrap();

        let stats2 = evaluator.get_stats_snapshot();
        assert_eq!(stats2.chunks_processed, 2); // Two batches processed

        // Check that processing time accumulated
        assert!(stats2.processing_time_ms > stats1.processing_time_ms);
    }

    #[tokio::test]
    async fn test_concurrent_evaluations() {
        use std::sync::Arc;
        use tokio::task;

        let evaluator = Arc::new(GpuBatchEvaluator::new("concurrent-test".to_string()));
        let mut handles = vec![];

        // Launch multiple concurrent evaluations
        for i in 0..5 {
            let evaluator_clone = evaluator.clone();
            let handle = task::spawn(async move {
                let agent =
                    AgentGenome::new(format!("fn concurrent_{} {{ {} }}", i, i), vec![i as f32]);
                evaluator_clone.evaluate_batch(vec![agent]).await
            });
            handles.push(handle);
        }

        // Wait for all evaluations to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
            let results = result.unwrap();
            assert_eq!(results.len(), 1);
        }

        // Verify final stats
        let stats = evaluator.get_stats_snapshot();
        assert_eq!(stats.chunks_processed, 5);
    }

    #[tokio::test]
    async fn test_invalid_json_deserialization() {
        let mut evaluator = GpuBatchEvaluator::new("invalid-json-test".to_string());

        let chunk = StreamChunk::new(bytes::Bytes::from("invalid json {"), 1, "test".to_string());

        let result = evaluator.process(chunk).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to deserialize agent"));
    }

    #[tokio::test]
    async fn test_evaluation_config_custom() {
        let config = EvaluationConfig {
            timeout: Duration::from_secs(5),
            memory_limit: 512 * 1024 * 1024, // 512MB
            measure_power: true,
            enable_profiling: true,
            max_concurrent_evaluations: 32,
        };

        assert_eq!(config.timeout, Duration::from_secs(5));
        assert_eq!(config.memory_limit, 512 * 1024 * 1024);
        assert!(config.measure_power);
        assert!(config.enable_profiling);
        assert_eq!(config.max_concurrent_evaluations, 32);
    }

    #[tokio::test]
    async fn test_benchmark_result_aggregation() {
        let evaluator = GpuBatchEvaluator::new("aggregation-test".to_string());

        // Test that multiple benchmarks aggregate correctly
        let agent = AgentGenome::new("fn multi_benchmark() {}".to_string(), vec![1.0]);
        let results = evaluator.evaluate_batch(vec![agent.clone()]).await.unwrap();

        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert_eq!(result.agent_id, agent.id);

        // Should have results from all benchmarks in the default suite
        assert!(!result.metrics.benchmark_results.is_empty());

        // Fitness should be calculated from all benchmark results
        assert!(result.fitness >= 0.0);
        assert!(result.fitness <= 1.0);
    }

    #[tokio::test]
    async fn test_compilation_time_tracking() {
        let evaluator = GpuBatchEvaluator::new("compile-time-test".to_string());
        let agent = AgentGenome::new("fn time_test() {}".to_string(), vec![1.0]);

        let compiled = evaluator.compile_single_agent(&agent).await.unwrap();
        assert!(compiled.compilation_time > Duration::ZERO);
        assert_eq!(compiled.compilation_time, Duration::from_millis(10)); // Simulated time
    }

    #[tokio::test]
    async fn test_memory_usage_tracking() {
        let evaluator = GpuBatchEvaluator::new("memory-test".to_string());
        let agent = AgentGenome::new("fn memory_test() {}".to_string(), vec![1.0]);

        let results = evaluator.evaluate_batch(vec![agent]).await.unwrap();
        assert_eq!(results.len(), 1);

        // Memory usage should be tracked (set by GPU executor mock)
        assert!(results[0].memory_usage >= 0);
    }

    #[tokio::test]
    async fn test_throughput_calculation() {
        let evaluator = GpuBatchEvaluator::new("throughput-test".to_string());

        // Process agents to generate timing data
        let agents = vec![
            AgentGenome::new("fn throughput1() {}".to_string(), vec![1.0]),
            AgentGenome::new("fn throughput2() {}".to_string(), vec![2.0]),
        ];

        evaluator.evaluate_batch(agents).await.unwrap();

        let stats = evaluator.get_stats_snapshot();
        assert!(stats.throughput_mbps >= 0.0);

        // With processing time > 0, throughput should be calculated
        if stats.processing_time_ms > 0 {
            assert!(stats.throughput_mbps > 0.0);
        }
    }

    #[tokio::test]
    async fn test_gpu_time_tracking() {
        let evaluator = GpuBatchEvaluator::new("gpu-time-test".to_string());
        let agent = AgentGenome::new("fn gpu_time_test() {}".to_string(), vec![1.0]);

        // Initial GPU time should be zero
        let initial_stats = evaluator.get_stats_snapshot();

        evaluator.evaluate_batch(vec![agent]).await.unwrap();

        // After evaluation, GPU time should have accumulated
        // (This is tracked internally, not directly exposed in stats)
        let final_stats = evaluator.get_stats_snapshot();
        assert!(final_stats.processing_time_ms >= initial_stats.processing_time_ms);
    }

    #[tokio::test]
    async fn test_empty_batch_evaluation() {
        let evaluator = GpuBatchEvaluator::new("empty-batch-test".to_string());

        let result = evaluator.evaluate_batch(vec![]).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No agents could be compiled"));
    }

    #[tokio::test]
    async fn test_large_batch_evaluation() {
        let evaluator = GpuBatchEvaluator::new("large-batch-test".to_string()).with_batch_size(100);

        let agents: Vec<AgentGenome> = (0..50)
            .map(|i| AgentGenome::new(format!("fn batch_{}() {{ {} }}", i, i), vec![i as f32]))
            .collect();

        let results = evaluator.evaluate_batch(agents.clone()).await.unwrap();
        assert_eq!(results.len(), agents.len());

        // Verify each agent was evaluated
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.agent_id, agents[i].id);
            assert!(result.fitness >= 0.0);
        }
    }

    #[tokio::test]
    async fn test_evaluation_timeout_configuration() {
        let short_timeout = Duration::from_millis(1);
        let evaluator =
            GpuBatchEvaluator::new("timeout-test".to_string()).with_timeout(short_timeout);

        assert_eq!(evaluator.timeout, short_timeout);

        // The timeout is passed to GPU execution, but since we're using mocks,
        // we can't easily test actual timeout behavior
        let agent = AgentGenome::new("fn timeout_test() {}".to_string(), vec![1.0]);
        let results = evaluator.evaluate_batch(vec![agent]).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_stream_processor_error_handling() {
        let mut evaluator = GpuBatchEvaluator::new("error-test".to_string());

        // Test with malformed chunk data
        let bad_chunk = StreamChunk::new(
            bytes::Bytes::from("not json"),
            1,
            "error-source".to_string(),
        );

        let result = evaluator.process(bad_chunk).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_batch_stream_processing_partial_results() {
        let mut evaluator = GpuBatchEvaluator::new("partial-test".to_string());

        // Mix of valid and invalid agent data
        let chunks = vec![
            StreamChunk::new(
                bytes::Bytes::from(
                    serde_json::to_string(&AgentGenome::new("valid".to_string(), vec![1.0]))
                        .unwrap(),
                ),
                1,
                "source1".to_string(),
            ),
            StreamChunk::new(bytes::Bytes::from("invalid json"), 2, "source2".to_string()),
        ];

        let result = evaluator.process_batch(chunks).await;
        assert!(result.is_err()); // Should fail due to invalid JSON
    }

    #[test]
    fn test_evaluation_stats_default() {
        let stats = EvaluationStats::default();
        assert_eq!(stats.agents_evaluated.load(Ordering::Relaxed), 0);
        assert_eq!(stats.batches_processed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.evaluation_time_ns.load(Ordering::Relaxed), 0);
        assert_eq!(stats.gpu_time_ns.load(Ordering::Relaxed), 0);
        assert_eq!(stats.compilation_failures.load(Ordering::Relaxed), 0);
        assert_eq!(stats.execution_failures.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_processor_id_consistency() {
        let evaluator = GpuBatchEvaluator::new("consistency-test-123".to_string());
        assert_eq!(evaluator.processor_id(), "consistency-test-123");

        // ID should remain consistent after configuration changes
        let configured_evaluator = evaluator
            .with_batch_size(128)
            .with_timeout(Duration::from_secs(60));
        assert_eq!(configured_evaluator.processor_id(), "consistency-test-123");
    }
}
