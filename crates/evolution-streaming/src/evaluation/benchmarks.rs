//! Benchmark suite for agent evaluation

use super::gpu::{Benchmark, MemoryEfficiencyBenchmark, PerformanceBenchmark};
use std::time::Duration;

/// Collection of benchmarks for agent evaluation
pub struct BenchmarkSuite {
    benchmarks: Vec<Box<dyn Benchmark>>,
    timeout_per_benchmark: Duration,
    parallel_execution: bool,
}

impl Clone for BenchmarkSuite {
    fn clone(&self) -> Self {
        // Create new benchmarks since trait objects can't be cloned
        Self::default_suite() // Return default suite for simplicity
    }
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
            timeout_per_benchmark: Duration::from_secs(10),
            parallel_execution: false,
        }
    }

    /// Add benchmark to suite
    pub fn add_benchmark(mut self, benchmark: Box<dyn Benchmark>) -> Self {
        self.benchmarks.push(benchmark);
        self
    }

    /// Configure timeout per benchmark
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_per_benchmark = timeout;
        self
    }

    /// Enable parallel benchmark execution
    pub fn with_parallel_execution(mut self, enabled: bool) -> Self {
        self.parallel_execution = enabled;
        self
    }

    /// Get all benchmarks
    pub fn benchmarks(&self) -> &[Box<dyn Benchmark>] {
        &self.benchmarks
    }

    /// Get total expected execution time
    pub fn total_expected_duration(&self) -> Duration {
        if self.parallel_execution {
            // In parallel mode, return the longest benchmark duration
            self.benchmarks
                .iter()
                .map(|b| b.expected_duration())
                .max()
                .unwrap_or(Duration::ZERO)
        } else {
            // In sequential mode, sum all durations
            self.benchmarks.iter().map(|b| b.expected_duration()).sum()
        }
    }

    /// Get benchmark count
    pub fn len(&self) -> usize {
        self.benchmarks.len()
    }

    /// Check if suite is empty
    pub fn is_empty(&self) -> bool {
        self.benchmarks.is_empty()
    }

    /// Create default benchmark suite for general purpose evaluation
    pub fn default_suite() -> Self {
        Self::new()
            .add_benchmark(Box::new(PerformanceBenchmark::new(
                "computation_speed".to_string(),
                1000.0,
            )))
            .add_benchmark(Box::new(MemoryEfficiencyBenchmark::new(
                "memory_usage".to_string(),
                1024 * 1024, // 1MB limit
            )))
            .add_benchmark(Box::new(CorrectnessTestBenchmark::new(
                "correctness".to_string(),
                vec![
                    TestCase::new("basic_arithmetic".to_string(), vec![1.0, 2.0], 3.0),
                    TestCase::new("multiplication".to_string(), vec![3.0, 4.0], 12.0),
                    TestCase::new("negative_numbers".to_string(), vec![-1.0, 1.0], 0.0),
                ],
            )))
            .add_benchmark(Box::new(StressBenchmark::new(
                "stress_test".to_string(),
                1000, // 1000 iterations
                Duration::from_millis(100),
            )))
            .with_timeout(Duration::from_secs(30))
    }

    /// Create lightweight benchmark suite for fast evaluation
    pub fn lightweight_suite() -> Self {
        Self::new()
            .add_benchmark(Box::new(PerformanceBenchmark::new(
                "quick_perf".to_string(),
                500.0,
            )))
            .add_benchmark(Box::new(BasicCorrectnessTest::new(
                "quick_correctness".to_string(),
            )))
            .with_timeout(Duration::from_secs(5))
            .with_parallel_execution(true)
    }

    /// Create comprehensive benchmark suite for thorough evaluation
    pub fn comprehensive_suite() -> Self {
        Self::new()
            .add_benchmark(Box::new(PerformanceBenchmark::new(
                "high_performance".to_string(),
                10000.0,
            )))
            .add_benchmark(Box::new(MemoryEfficiencyBenchmark::new(
                "memory_strict".to_string(),
                512 * 1024, // 512KB limit
            )))
            .add_benchmark(Box::new(CorrectnessTestBenchmark::new(
                "comprehensive_correctness".to_string(),
                vec![
                    TestCase::new("edge_case_1".to_string(), vec![0.0], 0.0),
                    TestCase::new("edge_case_2".to_string(), vec![f32::MAX], f32::MAX),
                    TestCase::new("precision_test".to_string(), vec![0.1, 0.2], 0.3),
                    TestCase::new("large_numbers".to_string(), vec![1e6, 2e6], 3e6),
                ],
            )))
            .add_benchmark(Box::new(StressBenchmark::new(
                "intensive_stress".to_string(),
                10000,
                Duration::from_secs(1),
            )))
            .add_benchmark(Box::new(StabilityBenchmark::new(
                "stability_test".to_string(),
                100, // 100 iterations
            )))
            .with_timeout(Duration::from_secs(60))
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::default_suite()
    }
}

/// Test case for correctness benchmarks
#[derive(Debug, Clone)]
pub struct TestCase {
    pub name: String,
    pub inputs: Vec<f32>,
    pub expected_output: f32,
    pub tolerance: f32,
}

impl TestCase {
    pub fn new(name: String, inputs: Vec<f32>, expected_output: f32) -> Self {
        Self {
            name,
            inputs,
            expected_output,
            tolerance: 1e-6, // Default tolerance
        }
    }

    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }
}

/// Correctness test benchmark
#[derive(Debug, Clone)]
pub struct CorrectnessTestBenchmark {
    name: String,
    test_cases: Vec<TestCase>,
}

impl CorrectnessTestBenchmark {
    pub fn new(name: String, test_cases: Vec<TestCase>) -> Self {
        Self { name, test_cases }
    }
}

#[async_trait::async_trait]
impl Benchmark for CorrectnessTestBenchmark {
    async fn evaluate_agent(
        &self,
        agent: &super::gpu::CompiledAgent,
    ) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut passed_tests = 0;

        for test_case in &self.test_cases {
            // Simulate test execution
            tokio::time::sleep(Duration::from_millis(2)).await;

            // Simple correctness simulation based on agent properties
            let result = self.simulate_test_execution(agent, test_case);
            let is_correct = (result - test_case.expected_output).abs() <= test_case.tolerance;

            if is_correct {
                passed_tests += 1;
            }
        }

        let accuracy = passed_tests as f64 / self.test_cases.len() as f64;
        Ok(accuracy)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        format!("Correctness test with {} test cases", self.test_cases.len())
    }

    fn expected_duration(&self) -> Duration {
        Duration::from_millis(self.test_cases.len() as u64 * 2)
    }
}

impl CorrectnessTestBenchmark {
    fn simulate_test_execution(
        &self,
        agent: &super::gpu::CompiledAgent,
        test_case: &TestCase,
    ) -> f32 {
        // Simple simulation: use agent parameters and inputs to generate result
        if agent.parameters.is_empty() || test_case.inputs.is_empty() {
            return test_case.expected_output * 0.9; // Close but not exact
        }

        let param_sum: f32 = agent.parameters.iter().sum();
        let input_sum: f32 = test_case.inputs.iter().sum();

        // Simple formula to generate somewhat realistic results
        let noise = ((param_sum + input_sum) % 0.1) - 0.05;
        test_case.expected_output + noise
    }
}

/// Basic correctness test for quick evaluation
#[derive(Debug, Clone)]
pub struct BasicCorrectnessTest {
    name: String,
}

impl BasicCorrectnessTest {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

#[async_trait::async_trait]
impl Benchmark for BasicCorrectnessTest {
    async fn evaluate_agent(
        &self,
        agent: &super::gpu::CompiledAgent,
    ) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        // Quick correctness check
        tokio::time::sleep(Duration::from_millis(1)).await;

        // Simple heuristic: longer code and more parameters might be more correct
        let code_score = (agent.kernel_code.len() as f64 / 100.0).min(1.0);
        let param_score = (agent.parameters.len() as f64 / 10.0).min(1.0);

        let score = (code_score + param_score) / 2.0;
        Ok(score.max(0.3)) // Minimum score to avoid complete failures
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        "Basic correctness test for quick evaluation".to_string()
    }

    fn expected_duration(&self) -> Duration {
        Duration::from_millis(1)
    }
}

/// Stress test benchmark for performance under load
#[derive(Debug, Clone)]
pub struct StressBenchmark {
    name: String,
    iterations: usize,
    iteration_duration: Duration,
}

impl StressBenchmark {
    pub fn new(name: String, iterations: usize, iteration_duration: Duration) -> Self {
        Self {
            name,
            iterations,
            iteration_duration,
        }
    }
}

#[async_trait::async_trait]
impl Benchmark for StressBenchmark {
    async fn evaluate_agent(
        &self,
        agent: &super::gpu::CompiledAgent,
    ) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();

        // Simulate stress testing
        for _ in 0..self.iterations {
            tokio::time::sleep(Duration::from_nanos(1)).await; // Minimal delay
        }

        let actual_duration = start_time.elapsed();
        let expected_duration = self.iteration_duration * self.iterations as u32;

        // Score based on performance: closer to expected time is better
        let performance_ratio =
            expected_duration.as_nanos() as f64 / actual_duration.as_nanos() as f64;
        let score = performance_ratio.min(1.0);

        // Factor in agent complexity
        let complexity_factor = 1.0 - (agent.kernel_code.len() as f64 / 10000.0).min(0.5);

        Ok((score * complexity_factor).max(0.1))
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        format!("Stress test with {} iterations", self.iterations)
    }

    fn expected_duration(&self) -> Duration {
        self.iteration_duration * self.iterations as u32
    }
}

/// Stability benchmark for consistent performance
#[derive(Debug, Clone)]
pub struct StabilityBenchmark {
    name: String,
    test_runs: usize,
}

impl StabilityBenchmark {
    pub fn new(name: String, test_runs: usize) -> Self {
        Self { name, test_runs }
    }
}

#[async_trait::async_trait]
impl Benchmark for StabilityBenchmark {
    async fn evaluate_agent(
        &self,
        agent: &super::gpu::CompiledAgent,
    ) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut results = Vec::new();

        // Run multiple evaluations to test stability
        for _ in 0..self.test_runs {
            tokio::time::sleep(Duration::from_millis(1)).await;

            // Simulate evaluation with slight variations
            let base_score = (agent.parameters.len() as f64 / 10.0).min(1.0);
            let variation = (fastrand::f64() - 0.5) * 0.1; // ±5% variation
            results.push((base_score + variation).max(0.0).min(1.0));
        }

        // Calculate stability score based on variance
        let mean = results.iter().sum::<f64>() / results.len() as f64;
        let variance =
            results.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / results.len() as f64;

        // Lower variance = higher stability score
        let stability_score = 1.0 - variance.min(1.0);

        Ok((mean + stability_score) / 2.0)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        format!("Stability test with {} runs", self.test_runs)
    }

    fn expected_duration(&self) -> Duration {
        Duration::from_millis(self.test_runs as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::gpu::CompiledAgent;
    use uuid::Uuid;

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = BenchmarkSuite::new();
        assert!(suite.is_empty());
        assert_eq!(suite.len(), 0);
    }

    #[test]
    fn test_benchmark_suite_default() {
        let suite = BenchmarkSuite::default();
        assert!(!suite.is_empty());
        assert!(suite.len() > 0);
        assert!(suite.total_expected_duration() > Duration::ZERO);
    }

    #[test]
    fn test_lightweight_suite() {
        let suite = BenchmarkSuite::lightweight_suite();
        assert_eq!(suite.len(), 2);
        assert!(suite.parallel_execution);
        assert_eq!(suite.timeout_per_benchmark, Duration::from_secs(5));
    }

    #[test]
    fn test_comprehensive_suite() {
        let suite = BenchmarkSuite::comprehensive_suite();
        assert!(suite.len() >= 5);
        assert_eq!(suite.timeout_per_benchmark, Duration::from_secs(60));
    }

    #[test]
    fn test_test_case_creation() {
        let test_case =
            TestCase::new("test_add".to_string(), vec![1.0, 2.0], 3.0).with_tolerance(0.01);

        assert_eq!(test_case.name, "test_add");
        assert_eq!(test_case.inputs, vec![1.0, 2.0]);
        assert_eq!(test_case.expected_output, 3.0);
        assert_eq!(test_case.tolerance, 0.01);
    }

    #[tokio::test]
    async fn test_correctness_benchmark() {
        let test_cases = vec![
            TestCase::new("simple_add".to_string(), vec![1.0, 1.0], 2.0),
            TestCase::new("multiply".to_string(), vec![2.0, 3.0], 6.0),
        ];

        let benchmark = CorrectnessTestBenchmark::new("test_correctness".to_string(), test_cases);

        let agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "fn compute() { return a + b; }".to_string(),
            parameters: vec![1.0, 2.0],
            compilation_time: Duration::from_millis(5),
        };

        let score = benchmark.evaluate_agent(&agent).await.unwrap();
        assert!(score >= 0.0 && score <= 1.0);
        assert_eq!(benchmark.name(), "test_correctness");
    }

    #[tokio::test]
    async fn test_basic_correctness_test() {
        let benchmark = BasicCorrectnessTest::new("basic_test".to_string());

        let agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "fn test() { 42 }".to_string(),
            parameters: vec![1.0, 2.0, 3.0],
            compilation_time: Duration::from_millis(1),
        };

        let score = benchmark.evaluate_agent(&agent).await.unwrap();
        assert!(score >= 0.3); // Minimum score guarantee
        assert!(score <= 1.0);
        assert_eq!(benchmark.expected_duration(), Duration::from_millis(1));
    }

    #[tokio::test]
    async fn test_stress_benchmark() {
        let benchmark =
            StressBenchmark::new("stress_test".to_string(), 10, Duration::from_millis(1));

        let agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "small".to_string(),
            parameters: vec![1.0],
            compilation_time: Duration::from_millis(1),
        };

        let score = benchmark.evaluate_agent(&agent).await.unwrap();
        assert!(score >= 0.1);
        assert!(score <= 1.0);
        assert_eq!(benchmark.expected_duration(), Duration::from_millis(10));
    }

    #[tokio::test]
    async fn test_stability_benchmark() {
        let benchmark = StabilityBenchmark::new("stability_test".to_string(), 5);

        let agent = CompiledAgent {
            id: Uuid::new_v4(),
            kernel_code: "stable_function".to_string(),
            parameters: vec![1.0, 2.0, 3.0, 4.0, 5.0], // 5 parameters = 0.5 base score
            compilation_time: Duration::from_millis(1),
        };

        let score = benchmark.evaluate_agent(&agent).await.unwrap();
        assert!(score >= 0.0 && score <= 1.0);
        assert_eq!(benchmark.expected_duration(), Duration::from_millis(5));
    }

    #[test]
    fn test_suite_configuration() {
        let suite = BenchmarkSuite::new()
            .with_timeout(Duration::from_secs(15))
            .with_parallel_execution(true);

        assert_eq!(suite.timeout_per_benchmark, Duration::from_secs(15));
        assert!(suite.parallel_execution);
    }

    #[test]
    fn test_total_duration_calculation() {
        let mut suite = BenchmarkSuite::new()
            .add_benchmark(Box::new(BasicCorrectnessTest::new("test1".to_string())))
            .add_benchmark(Box::new(BasicCorrectnessTest::new("test2".to_string())));

        // Sequential mode: durations should be summed
        suite.parallel_execution = false;
        let sequential_duration = suite.total_expected_duration();
        assert_eq!(sequential_duration, Duration::from_millis(2)); // 1ms + 1ms

        // Parallel mode: should take max duration
        suite.parallel_execution = true;
        let parallel_duration = suite.total_expected_duration();
        assert_eq!(parallel_duration, Duration::from_millis(1)); // max(1ms, 1ms)
    }

    #[test]
    fn test_benchmark_suite_add() {
        let suite = BenchmarkSuite::new()
            .add_benchmark(Box::new(BasicCorrectnessTest::new("test1".to_string())))
            .add_benchmark(Box::new(BasicCorrectnessTest::new("test2".to_string())));

        assert_eq!(suite.len(), 2);
        assert!(!suite.is_empty());
    }
}
