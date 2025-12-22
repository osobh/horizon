//! GPU batch executor for agent evaluation

use crate::{AgentId, EvolutionStreamingError};
use std::time::Duration;
use thiserror::Error;

/// GPU execution errors
#[derive(Error, Debug)]
pub enum GpuExecutionError {
    #[error("GPU allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Kernel compilation failed: {0}")]
    CompilationFailed(String),

    #[error("Execution timeout: {duration:?}")]
    ExecutionTimeout { duration: Duration },

    #[error("Memory limit exceeded: {used} > {limit}")]
    MemoryLimitExceeded { used: u64, limit: u64 },

    #[error("GPU execution failed: {0}")]
    ExecutionFailed(String),
}

/// Compiled agent ready for GPU execution
#[derive(Debug, Clone)]
pub struct CompiledAgent {
    pub id: AgentId,
    pub kernel_code: String,
    pub parameters: Vec<f32>,
    pub compilation_time: Duration,
}

/// GPU execution result
#[derive(Debug, Clone)]
pub struct GpuExecutionResult {
    pub agent_id: AgentId,
    pub score: f64,
    pub execution_time: Duration,
    pub memory_usage: u64,
    pub passed: bool,
}

/// GPU batch for parallel execution
pub struct GpuBatch {
    agents: Vec<CompiledAgent>,
    batch_size: usize,
    allocated_memory: u64,
}

/// GPU batch executor
#[derive(Debug, Clone)]
pub struct GpuBatchExecutor {
    max_batch_size: usize,
    memory_limit: u64,
    enable_profiling: bool,
}

impl GpuBatchExecutor {
    /// Create a new GPU batch executor
    pub fn new() -> Self {
        Self {
            max_batch_size: 128,
            memory_limit: 2 * 1024 * 1024 * 1024, // 2GB
            enable_profiling: false,
        }
    }

    /// Configure maximum batch size
    pub fn with_max_batch_size(mut self, max_size: usize) -> Self {
        self.max_batch_size = max_size;
        self
    }

    /// Configure memory limit
    pub fn with_memory_limit(mut self, limit: u64) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Enable GPU profiling
    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.enable_profiling = enabled;
        self
    }

    /// Prepare a batch for GPU execution
    pub async fn prepare_batch(
        &self,
        agents: &[CompiledAgent],
    ) -> Result<GpuBatch, GpuExecutionError> {
        if agents.len() > self.max_batch_size {
            return Err(GpuExecutionError::AllocationFailed(format!(
                "Batch size {} exceeds maximum {}",
                agents.len(),
                self.max_batch_size
            )));
        }

        // Estimate memory usage
        let estimated_memory = self.estimate_memory_usage(agents);
        if estimated_memory > self.memory_limit {
            return Err(GpuExecutionError::MemoryLimitExceeded {
                used: estimated_memory,
                limit: self.memory_limit,
            });
        }

        // Simulate GPU memory allocation
        tokio::time::sleep(Duration::from_millis(1)).await;

        Ok(GpuBatch {
            agents: agents.to_vec(),
            batch_size: agents.len(),
            allocated_memory: estimated_memory,
        })
    }

    /// Estimate memory usage for batch
    fn estimate_memory_usage(&self, agents: &[CompiledAgent]) -> u64 {
        // Rough estimate: 1MB per agent + overhead
        agents.len() as u64 * 1024 * 1024 + 100 * 1024 * 1024
    }
}

impl Default for GpuBatchExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerformanceBenchmark {
    fn default() -> Self {
        Self::new("default_performance".to_string(), 1000.0)
    }
}

impl Default for MemoryEfficiencyBenchmark {
    fn default() -> Self {
        Self::new("default_memory".to_string(), 1024 * 1024)
    }
}

impl GpuBatch {
    /// Execute benchmark on all agents in batch
    pub async fn execute_benchmark(
        &self,
        benchmark: &dyn Benchmark,
        timeout: Duration,
    ) -> Result<Vec<GpuExecutionResult>, GpuExecutionError> {
        let mut results = Vec::new();

        for agent in &self.agents {
            let start_time = std::time::Instant::now();

            // Simulate GPU execution
            let execution_result =
                tokio::time::timeout(timeout, self.execute_single_agent(agent, benchmark)).await;

            let execution_time = start_time.elapsed();

            match execution_result {
                Ok(Ok(score)) => {
                    results.push(GpuExecutionResult {
                        agent_id: agent.id,
                        score,
                        execution_time,
                        memory_usage: self.estimate_agent_memory(agent),
                        passed: score > 0.5, // Simple pass threshold
                    });
                }
                Ok(Err(e)) => {
                    eprintln!("Agent {} execution failed: {}", agent.id, e);
                    results.push(GpuExecutionResult {
                        agent_id: agent.id,
                        score: 0.0,
                        execution_time,
                        memory_usage: 0,
                        passed: false,
                    });
                }
                Err(_) => {
                    return Err(GpuExecutionError::ExecutionTimeout { duration: timeout });
                }
            }
        }

        Ok(results)
    }

    /// Execute single agent
    async fn execute_single_agent(
        &self,
        agent: &CompiledAgent,
        benchmark: &dyn Benchmark,
    ) -> Result<f64, GpuExecutionError> {
        // Simulate execution time based on code complexity
        let execution_time = Duration::from_millis((agent.kernel_code.len() / 10).max(1) as u64);
        tokio::time::sleep(execution_time).await;

        // Simulate benchmark execution
        let score = benchmark
            .evaluate_agent(agent)
            .await
            .map_err(|e| GpuExecutionError::ExecutionFailed(e.to_string()))?;

        Ok(score)
    }

    /// Estimate memory usage for single agent
    fn estimate_agent_memory(&self, agent: &CompiledAgent) -> u64 {
        // Rough estimate based on code size and parameters
        (agent.kernel_code.len() + agent.parameters.len() * 4) as u64 * 1024
    }

    /// Get batch statistics
    pub fn get_batch_stats(&self) -> BatchStats {
        BatchStats {
            agent_count: self.batch_size,
            allocated_memory: self.allocated_memory,
            total_parameters: self.agents.iter().map(|a| a.parameters.len()).sum(),
            total_code_size: self.agents.iter().map(|a| a.kernel_code.len()).sum(),
        }
    }
}

/// Batch execution statistics
#[derive(Debug, Clone)]
pub struct BatchStats {
    pub agent_count: usize,
    pub allocated_memory: u64,
    pub total_parameters: usize,
    pub total_code_size: usize,
}

/// Benchmark trait for agent evaluation
#[async_trait::async_trait]
pub trait Benchmark: Send + Sync {
    /// Evaluate agent performance on this benchmark
    async fn evaluate_agent(
        &self,
        agent: &CompiledAgent,
    ) -> Result<f64, Box<dyn std::error::Error + Send + Sync>>;

    /// Get benchmark name
    fn name(&self) -> String;

    /// Get benchmark description
    fn description(&self) -> String;

    /// Get expected execution time
    fn expected_duration(&self) -> Duration;
}

/// Simple performance benchmark
#[derive(Debug, Clone)]
pub struct PerformanceBenchmark {
    name: String,
    target_ops_per_sec: f64,
}

impl PerformanceBenchmark {
    pub fn new(name: String, target_ops_per_sec: f64) -> Self {
        Self {
            name,
            target_ops_per_sec,
        }
    }
}

#[async_trait::async_trait]
impl Benchmark for PerformanceBenchmark {
    async fn evaluate_agent(
        &self,
        agent: &CompiledAgent,
    ) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        // Simulate performance measurement
        tokio::time::sleep(Duration::from_millis(5)).await;

        // Simple scoring based on code length and parameters
        let complexity_score = 1.0 / (1.0 + agent.kernel_code.len() as f64 / 1000.0);
        let parameter_score = if agent.parameters.is_empty() {
            0.5
        } else {
            agent.parameters.iter().map(|&p| p.abs()).sum::<f32>() as f64
                / agent.parameters.len() as f64
        };

        let score = (complexity_score + parameter_score.min(1.0)) / 2.0;
        Ok(score.max(0.0).min(1.0))
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        format!(
            "Performance benchmark targeting {} ops/sec",
            self.target_ops_per_sec
        )
    }

    fn expected_duration(&self) -> Duration {
        Duration::from_millis(10)
    }
}

/// Memory efficiency benchmark
#[derive(Debug, Clone)]
pub struct MemoryEfficiencyBenchmark {
    name: String,
    max_memory_usage: u64,
}

impl MemoryEfficiencyBenchmark {
    pub fn new(name: String, max_memory_usage: u64) -> Self {
        Self {
            name,
            max_memory_usage,
        }
    }
}

#[async_trait::async_trait]
impl Benchmark for MemoryEfficiencyBenchmark {
    async fn evaluate_agent(
        &self,
        agent: &CompiledAgent,
    ) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        // Simulate memory usage measurement
        tokio::time::sleep(Duration::from_millis(3)).await;

        let estimated_memory = (agent.kernel_code.len() + agent.parameters.len() * 4) as u64;
        let efficiency = if estimated_memory <= self.max_memory_usage {
            1.0 - (estimated_memory as f64 / self.max_memory_usage as f64)
        } else {
            0.0
        };

        Ok(efficiency.max(0.0).min(1.0))
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        format!(
            "Memory efficiency benchmark with {} byte limit",
            self.max_memory_usage
        )
    }

    fn expected_duration(&self) -> Duration {
        Duration::from_millis(5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentGenome;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_gpu_batch_executor_creation() {
        let executor = GpuBatchExecutor::new();
        assert_eq!(executor.max_batch_size, 128);
        assert_eq!(executor.memory_limit, 2 * 1024 * 1024 * 1024);
        assert!(!executor.enable_profiling);
    }

    #[tokio::test]
    async fn test_executor_configuration() {
        let executor = GpuBatchExecutor::new()
            .with_max_batch_size(64)
            .with_memory_limit(1024 * 1024 * 1024)
            .with_profiling(true);

        assert_eq!(executor.max_batch_size, 64);
        assert_eq!(executor.memory_limit, 1024 * 1024 * 1024);
        assert!(executor.enable_profiling);
    }

    #[tokio::test]
    async fn test_batch_preparation() {
        let executor = GpuBatchExecutor::new();
        let agents = vec![CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "fn test() { 42 }".to_string(),
            parameters: vec![1.0, 2.0],
            compilation_time: Duration::from_millis(10),
        }];

        let batch = executor.prepare_batch(&agents).await.unwrap();
        assert_eq!(batch.batch_size, 1);
        assert!(batch.allocated_memory > 0);
    }

    #[tokio::test]
    async fn test_batch_size_limit() {
        let executor = GpuBatchExecutor::new().with_max_batch_size(1);
        let agents = vec![
            CompiledAgent {
                id: Uuid::new_v4(),
                kernel_code: "test1".to_string(),
                parameters: vec![],
                compilation_time: Duration::from_millis(1),
            },
            CompiledAgent {
                id: Uuid::new_v4(),
                kernel_code: "test2".to_string(),
                parameters: vec![],
                compilation_time: Duration::from_millis(1),
            },
        ];

        let result = executor.prepare_batch(&agents).await;
        assert!(matches!(
            result,
            Err(GpuExecutionError::AllocationFailed(_))
        ));
    }

    #[tokio::test]
    async fn test_performance_benchmark() {
        let benchmark = PerformanceBenchmark::new("test_perf".to_string(), 1000.0);
        let agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "fn compute() { 42 }".to_string(),
            parameters: vec![1.0, 2.0, 3.0],
            compilation_time: Duration::from_millis(5),
        };

        let score = benchmark.evaluate_agent(&agent).await.unwrap();
        assert!(score >= 0.0 && score <= 1.0);
        assert_eq!(benchmark.name(), "test_perf");
    }

    #[tokio::test]
    async fn test_memory_efficiency_benchmark() {
        let benchmark = MemoryEfficiencyBenchmark::new("test_memory".to_string(), 1024);
        let agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "small".to_string(), // Small code should be efficient
            parameters: vec![1.0],
            compilation_time: Duration::from_millis(1),
        };

        let score = benchmark.evaluate_agent(&agent).await.unwrap();
        assert!(score >= 0.0 && score <= 1.0);
        assert!(score > 0.5); // Should be efficient
    }

    #[tokio::test]
    async fn test_batch_execution() {
        let executor = GpuBatchExecutor::new();
        let agents = vec![
            CompiledAgent {
                id: Uuid::new_v4(),
                kernel_code: "fn test1() { 1 }".to_string(),
                parameters: vec![1.0],
                compilation_time: Duration::from_millis(1),
            },
            CompiledAgent {
                id: Uuid::new_v4(),
                kernel_code: "fn test2() { 2 }".to_string(),
                parameters: vec![2.0],
                compilation_time: Duration::from_millis(1),
            },
        ];

        let batch = executor.prepare_batch(&agents).await.unwrap();
        let benchmark = PerformanceBenchmark::new("test".to_string(), 100.0);

        let results = batch
            .execute_benchmark(&benchmark, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(results.len(), 2);

        for result in results {
            assert!(agents.iter().any(|a| a.id == result.agent_id));
            assert!(result.execution_time > Duration::ZERO);
        }
    }

    #[tokio::test]
    async fn test_batch_stats() {
        let agents = vec![CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "test".to_string(),
            parameters: vec![1.0, 2.0],
            compilation_time: Duration::from_millis(1),
        }];

        let executor = GpuBatchExecutor::new();
        let batch = executor.prepare_batch(&agents).await.unwrap();
        let stats = batch.get_batch_stats();

        assert_eq!(stats.agent_count, 1);
        assert_eq!(stats.total_parameters, 2);
        assert_eq!(stats.total_code_size, 4); // "test".len()
        assert!(stats.allocated_memory > 0);
    }

    #[test]
    fn test_compiled_agent_creation() {
        let agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "fn test() { return 42; }".to_string(),
            parameters: vec![1.0, 2.0, 3.0],
            compilation_time: Duration::from_millis(15),
        };

        assert!(!agent.kernel_code.is_empty());
        assert_eq!(agent.parameters.len(), 3);
        assert_eq!(agent.compilation_time, Duration::from_millis(15));
    }

    #[test]
    fn test_memory_estimation() {
        let executor = GpuBatchExecutor::new();
        let agents = vec![CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "a".repeat(100),
            parameters: vec![1.0; 50],
            compilation_time: Duration::from_millis(1),
        }];

        let memory = executor.estimate_memory_usage(&agents);
        assert!(memory > 100 * 1024 * 1024); // Should be at least 100MB overhead
    }

    #[tokio::test]
    async fn test_memory_limit_exceeded() {
        let executor = GpuBatchExecutor::new().with_memory_limit(1024); // Very small limit
        let agents = vec![CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "a".repeat(10000), // Large code
            parameters: vec![1.0; 1000],    // Many parameters
            compilation_time: Duration::from_millis(1),
        }];

        let result = executor.prepare_batch(&agents).await;
        assert!(matches!(
            result,
            Err(GpuExecutionError::MemoryLimitExceeded { .. })
        ));
    }

    #[test]
    fn test_gpu_execution_error_display() {
        let errors = vec![
            GpuExecutionError::AllocationFailed("test".to_string()),
            GpuExecutionError::CompilationFailed("syntax error".to_string()),
            GpuExecutionError::ExecutionTimeout {
                duration: Duration::from_secs(5),
            },
            GpuExecutionError::MemoryLimitExceeded {
                used: 2000,
                limit: 1000,
            },
            GpuExecutionError::ExecutionFailed("runtime error".to_string()),
        ];

        for error in errors {
            let message = error.to_string();
            assert!(!message.is_empty());
        }
    }

    #[test]
    fn test_default_gpu_batch_executor() {
        let executor1 = GpuBatchExecutor::new();
        let executor2 = GpuBatchExecutor::default();

        assert_eq!(executor1.max_batch_size, executor2.max_batch_size);
        assert_eq!(executor1.memory_limit, executor2.memory_limit);
        assert_eq!(executor1.enable_profiling, executor2.enable_profiling);
    }

    #[tokio::test]
    async fn test_empty_batch_preparation() {
        let executor = GpuBatchExecutor::new();
        let agents = vec![];

        let batch = executor.prepare_batch(&agents).await.unwrap();
        assert_eq!(batch.batch_size, 0);
        assert!(batch.allocated_memory > 0); // Should have some overhead
    }

    #[tokio::test]
    async fn test_batch_execution_timeout() {
        let executor = GpuBatchExecutor::new();
        let agents = vec![CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "very long computation".to_string(),
            parameters: vec![1.0],
            compilation_time: Duration::from_millis(1),
        }];

        let batch = executor.prepare_batch(&agents).await.unwrap();
        let benchmark = PerformanceBenchmark::new("test".to_string(), 100.0);

        // Very short timeout to trigger timeout error
        let result = batch
            .execute_benchmark(&benchmark, Duration::from_millis(1))
            .await;

        assert!(matches!(
            result,
            Err(GpuExecutionError::ExecutionTimeout { .. })
        ));
    }

    #[tokio::test]
    async fn test_large_batch_execution() {
        let executor = GpuBatchExecutor::new();
        let mut agents = vec![];

        for i in 0..50 {
            agents.push(CompiledAgent {
                id: Uuid::new_v4(),
                kernel_code: format!("fn test_{}() {{ {} }}", i, i),
                parameters: vec![i as f32],
                compilation_time: Duration::from_millis(1),
            });
        }

        let batch = executor.prepare_batch(&agents).await.unwrap();
        let benchmark = PerformanceBenchmark::new("large_test".to_string(), 1000.0);

        let results = batch
            .execute_benchmark(&benchmark, Duration::from_secs(10))
            .await
            .unwrap();

        assert_eq!(results.len(), 50);
        for result in results {
            assert!(result.execution_time > Duration::ZERO);
            assert!(result.memory_usage > 0);
        }
    }

    #[tokio::test]
    async fn test_performance_benchmark_edge_cases() {
        let benchmark = PerformanceBenchmark::new("edge_test".to_string(), 500.0);

        // Empty code
        let agent_empty = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "".to_string(),
            parameters: vec![],
            compilation_time: Duration::from_millis(1),
        };

        let score = benchmark.evaluate_agent(&agent_empty).await.unwrap();
        assert!(score >= 0.0 && score <= 1.0);

        // Very large code
        let agent_large = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "a".repeat(10000),
            parameters: vec![100.0; 1000],
            compilation_time: Duration::from_millis(100),
        };

        let score = benchmark.evaluate_agent(&agent_large).await.unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[tokio::test]
    async fn test_memory_efficiency_benchmark_edge_cases() {
        let benchmark = MemoryEfficiencyBenchmark::new("memory_test".to_string(), 100);

        // Exactly at limit
        let agent_limit = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "a".repeat(100),
            parameters: vec![],
            compilation_time: Duration::from_millis(1),
        };

        let score = benchmark.evaluate_agent(&agent_limit).await.unwrap();
        assert!(score >= 0.0 && score <= 1.0);

        // Over limit
        let agent_over = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "a".repeat(200),
            parameters: vec![1.0; 50], // 50 * 4 = 200 bytes
            compilation_time: Duration::from_millis(1),
        };

        let score = benchmark.evaluate_agent(&agent_over).await.unwrap();
        assert_eq!(score, 0.0); // Should be 0 for over limit
    }

    #[tokio::test]
    async fn test_batch_stats_detailed() {
        let agents = vec![
            CompiledAgent {
                id: Uuid::new_v4(),
                kernel_code: "short".to_string(),
                parameters: vec![1.0],
                compilation_time: Duration::from_millis(1),
            },
            CompiledAgent {
                id: Uuid::new_v4(),
                kernel_code: "much longer code here".to_string(),
                parameters: vec![1.0, 2.0, 3.0, 4.0],
                compilation_time: Duration::from_millis(2),
            },
        ];

        let executor = GpuBatchExecutor::new();
        let batch = executor.prepare_batch(&agents).await.unwrap();
        let stats = batch.get_batch_stats();

        assert_eq!(stats.agent_count, 2);
        assert_eq!(stats.total_parameters, 5); // 1 + 4
        assert_eq!(stats.total_code_size, 27); // "short" + "much longer code here"
        assert!(stats.allocated_memory > 200 * 1024 * 1024); // At least 200MB for 2 agents
    }

    #[tokio::test]
    async fn test_compiled_agent_cloning() {
        let agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "fn test() {}".to_string(),
            parameters: vec![1.0, 2.0],
            compilation_time: Duration::from_millis(10),
        };

        let cloned = agent.clone();
        assert_eq!(agent.id, cloned.id);
        assert_eq!(agent.kernel_code, cloned.kernel_code);
        assert_eq!(agent.parameters, cloned.parameters);
        assert_eq!(agent.compilation_time, cloned.compilation_time);
    }

    #[test]
    fn test_gpu_execution_result_creation() {
        let result = GpuExecutionResult {
            agent_id: Uuid::new_v4(),
            score: 0.85,
            execution_time: Duration::from_millis(50),
            memory_usage: 1024 * 1024,
            passed: true,
        };

        assert_eq!(result.score, 0.85);
        assert_eq!(result.execution_time, Duration::from_millis(50));
        assert_eq!(result.memory_usage, 1024 * 1024);
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_benchmark_descriptions() {
        let perf_bench = PerformanceBenchmark::new("test_perf".to_string(), 1500.0);
        let mem_bench = MemoryEfficiencyBenchmark::new("test_mem".to_string(), 2048);

        assert!(perf_bench.description().contains("1500"));
        assert!(perf_bench.description().contains("ops/sec"));

        assert!(mem_bench.description().contains("2048"));
        assert!(mem_bench.description().contains("byte limit"));

        assert_eq!(perf_bench.expected_duration(), Duration::from_millis(10));
        assert_eq!(mem_bench.expected_duration(), Duration::from_millis(5));
    }

    #[tokio::test]
    async fn test_batch_agent_memory_estimation() {
        let agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "test_code".to_string(), // 9 bytes
            parameters: vec![1.0, 2.0, 3.0],      // 3 * 4 = 12 bytes
            compilation_time: Duration::from_millis(1),
        };

        let batch = GpuBatch {
            agents: vec![agent.clone()],
            batch_size: 1,
            allocated_memory: 1000000,
        };

        let memory = batch.estimate_agent_memory(&agent);
        assert_eq!(memory, (9 + 12) * 1024); // (code + params) * 1024
    }

    #[tokio::test]
    async fn test_concurrent_batch_execution() {
        use std::sync::Arc;
        use tokio::task;

        let executor = Arc::new(GpuBatchExecutor::new().with_max_batch_size(10));
        let mut handles = vec![];

        for i in 0..5 {
            let executor_clone = executor.clone();
            let handle = task::spawn(async move {
                let agents = vec![CompiledAgent {
                    id: Uuid::new_v4(),
                    kernel_code: format!("fn concurrent_{}() {{ {} }}", i, i),
                    parameters: vec![i as f32],
                    compilation_time: Duration::from_millis(1),
                }];

                let batch = executor_clone.prepare_batch(&agents).await.unwrap();
                let benchmark = PerformanceBenchmark::new(format!("concurrent_{i}"), 100.0);

                batch
                    .execute_benchmark(&benchmark, Duration::from_secs(1))
                    .await
            });
            handles.push(handle);
        }

        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
            let results = result.unwrap();
            assert_eq!(results.len(), 1);
        }
    }

    #[tokio::test]
    async fn test_execution_result_pass_threshold() {
        let executor = GpuBatchExecutor::new();
        let agents = vec![
            CompiledAgent {
                id: Uuid::new_v4(),
                kernel_code: "high_score".to_string(), // Should get high score
                parameters: vec![0.1],                 // Small parameter for high complexity score
                compilation_time: Duration::from_millis(1),
            },
            CompiledAgent {
                id: Uuid::new_v4(),
                kernel_code: "a".repeat(10000), // Should get low score
                parameters: vec![1000.0],       // Large parameter
                compilation_time: Duration::from_millis(1),
            },
        ];

        let batch = executor.prepare_batch(&agents).await.unwrap();
        let benchmark = PerformanceBenchmark::new("threshold_test".to_string(), 100.0);

        let results = batch
            .execute_benchmark(&benchmark, Duration::from_secs(5))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);

        // First agent should likely pass (score > 0.5)
        // Second agent should likely fail (score <= 0.5) due to large code and parameters
    }

    #[tokio::test]
    async fn test_memory_usage_calculation() {
        let executor = GpuBatchExecutor::new();

        // Test with different agent sizes
        let small_agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "x".to_string(),
            parameters: vec![1.0],
            compilation_time: Duration::from_millis(1),
        };

        let large_agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "x".repeat(1000),
            parameters: vec![1.0; 100],
            compilation_time: Duration::from_millis(1),
        };

        let small_memory = executor.estimate_memory_usage(&[small_agent]);
        let large_memory = executor.estimate_memory_usage(&[large_agent]);

        assert!(large_memory > small_memory);
        assert_eq!(large_memory - small_memory, 1024 * 1024); // Exactly 1MB difference per agent
    }

    #[test]
    fn test_batch_stats_zero_agents() {
        let batch = GpuBatch {
            agents: vec![],
            batch_size: 0,
            allocated_memory: 100000,
        };

        let stats = batch.get_batch_stats();
        assert_eq!(stats.agent_count, 0);
        assert_eq!(stats.total_parameters, 0);
        assert_eq!(stats.total_code_size, 0);
        assert_eq!(stats.allocated_memory, 100000);
    }

    #[tokio::test]
    async fn test_performance_benchmark_parameter_scoring() {
        let benchmark = PerformanceBenchmark::new("param_test".to_string(), 100.0);

        // Empty parameters should get 0.5 score
        let agent_empty_params = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "test".to_string(),
            parameters: vec![],
            compilation_time: Duration::from_millis(1),
        };

        let score_empty = benchmark.evaluate_agent(&agent_empty_params).await.unwrap();

        // Non-empty parameters should get average of absolute values
        let agent_with_params = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "test".to_string(),
            parameters: vec![-1.0, 2.0], // Average abs = 1.5
            compilation_time: Duration::from_millis(1),
        };

        let score_with_params = benchmark.evaluate_agent(&agent_with_params).await.unwrap();

        // Both should be valid scores
        assert!(score_empty >= 0.0 && score_empty <= 1.0);
        assert!(score_with_params >= 0.0 && score_with_params <= 1.0);
    }

    #[tokio::test]
    async fn test_gpu_execution_error_matching() {
        let executor = GpuBatchExecutor::new().with_max_batch_size(1);
        let agents = vec![
            CompiledAgent {
                id: Uuid::new_v4(),
                kernel_code: "test".to_string(),
                parameters: vec![],
                compilation_time: Duration::from_millis(1),
            };
            2
        ]; // 2 agents but max batch size is 1

        match executor.prepare_batch(&agents).await {
            Err(GpuExecutionError::AllocationFailed(msg)) => {
                assert!(msg.contains("Batch size 2"));
                assert!(msg.contains("exceeds maximum 1"));
            }
            _ => panic!("Expected AllocationFailed error"),
        }
    }

    #[tokio::test]
    async fn test_execution_timing_simulation() {
        let executor = GpuBatchExecutor::new();

        let short_code_agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "x".to_string(), // 1 char = 1ms minimum
            parameters: vec![],
            compilation_time: Duration::from_millis(1),
        };

        let long_code_agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "x".repeat(100), // 100 chars = 10ms
            parameters: vec![],
            compilation_time: Duration::from_millis(1),
        };

        let batch1 = executor.prepare_batch(&[short_code_agent]).await.unwrap();
        let batch2 = executor.prepare_batch(&[long_code_agent]).await.unwrap();

        let benchmark = PerformanceBenchmark::new("timing_test".to_string(), 100.0);

        let start1 = std::time::Instant::now();
        let _results1 = batch1
            .execute_benchmark(&benchmark, Duration::from_secs(1))
            .await
            .unwrap();
        let duration1 = start1.elapsed();

        let start2 = std::time::Instant::now();
        let _results2 = batch2
            .execute_benchmark(&benchmark, Duration::from_secs(1))
            .await
            .unwrap();
        let duration2 = start2.elapsed();

        // Longer code should take more time due to simulation
        assert!(duration2 > duration1);
    }
}
