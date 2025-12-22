//! Scenario benchmark runner for integrating scenarios with the benchmark system

use super::{BenchmarkPhase, ProgressWriter};
use crate::scenarios::{ScenarioConfig, ScenarioRunner};
use std::time::{Duration, Instant};

/// Benchmark metrics for a scenario run
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScenarioBenchmarkMetrics {
    pub throughput: f64,
    pub latency_ms: f64,
    pub gpu_utilization: f64,
    pub memory_usage_mb: f64,
    pub total_steps: u64,
    pub completed: bool,
}

/// Result of running a scenario as a benchmark
#[derive(Debug, Clone)]
pub struct ScenarioBenchmarkResult {
    pub scenario_id: String,
    pub metrics: ScenarioBenchmarkMetrics,
    pub duration: Duration,
    pub error: Option<String>,
}

/// Runner for executing scenarios as benchmarks
pub struct ScenarioBenchmarkRunner {
    device_id: i32,
}

impl ScenarioBenchmarkRunner {
    /// Create a new scenario benchmark runner
    pub fn new() -> Result<Self, anyhow::Error> {
        Ok(Self { device_id: 0 })
    }

    /// Run a scenario and collect benchmark metrics
    pub fn run_scenario_benchmark(
        &self,
        scenario: &ScenarioConfig,
    ) -> Result<ScenarioBenchmarkResult, anyhow::Error> {
        let start_time = Instant::now();

        // Create scenario runner
        let runner = ScenarioRunner::new(self.device_id)
            .map_err(|e| anyhow::anyhow!("Failed to create scenario runner: {}", e))?;

        // Run scenario
        let scenario_result = runner
            .run_scenario(scenario)
            .map_err(|e| anyhow::anyhow!("Failed to run scenario: {}", e))?;

        // Calculate metrics from scenario result
        let total_duration = start_time.elapsed();
        let total_steps = (scenario_result.duration.as_secs_f64() * 60.0) as u64; // Assuming 60 FPS
        let throughput = (scenario_result.agent_count as f64 * total_steps as f64)
            / total_duration.as_secs_f64();

        // Extract metrics from performance metrics
        let mut gpu_utilization = 75.0; // Default placeholder
        let mut memory_usage_mb = 512.0; // Default placeholder

        for metric in &scenario_result.metrics {
            match metric.name.as_str() {
                "gpu_utilization" => gpu_utilization = metric.value,
                "memory_usage_mb" => memory_usage_mb = metric.value,
                _ => {}
            }
        }

        Ok(ScenarioBenchmarkResult {
            scenario_id: scenario_result.scenario_id,
            metrics: ScenarioBenchmarkMetrics {
                throughput,
                latency_ms: (1000.0 / 60.0), // Frame time in ms
                gpu_utilization,
                memory_usage_mb,
                total_steps,
                completed: scenario_result.errors.is_empty(),
            },
            duration: total_duration,
            error: scenario_result.errors.first().cloned(),
        })
    }

    /// Run scenario benchmark with progress tracking
    pub fn run_scenario_benchmark_with_progress(
        &self,
        scenario: &ScenarioConfig,
        progress_writer: &ProgressWriter,
    ) -> Result<ScenarioBenchmarkResult, anyhow::Error> {
        progress_writer.log_phase(BenchmarkPhase::Custom("Scenario Execution".to_string()))?;
        progress_writer.log_progress(0.0)?;

        let result = self.run_scenario_benchmark(scenario)?;

        progress_writer.log_progress(1.0)?;
        Ok(result)
    }
}

/// Suite of scenarios to run together
pub struct ScenarioSuite {
    pub name: String,
    pub scenarios: Vec<ScenarioConfig>,
}

impl ScenarioSuite {
    /// Create a new scenario suite
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            scenarios: Vec::new(),
        }
    }

    /// Add a scenario to the suite
    pub fn add_scenario(mut self, scenario: ScenarioConfig) -> Self {
        self.scenarios.push(scenario);
        self
    }
}

/// Results from running a scenario suite
#[derive(Debug, Clone)]
pub struct ScenarioSuiteResults {
    pub suite_name: String,
    pub scenario_results: Vec<ScenarioBenchmarkResult>,
    pub summary: SuiteSummary,
}

/// Summary of suite execution
#[derive(Debug, Clone)]
pub struct SuiteSummary {
    pub total_duration: Duration,
    pub successful_scenarios: usize,
    pub failed_scenarios: usize,
    pub average_throughput: f64,
}

impl ScenarioBenchmarkRunner {
    /// Run a suite of scenarios
    pub fn run_scenario_suite(
        &self,
        suite: &ScenarioSuite,
    ) -> Result<ScenarioSuiteResults, anyhow::Error> {
        let start_time = Instant::now();
        let mut results = Vec::new();
        let mut successful = 0;
        let mut total_throughput = 0.0;

        for scenario in &suite.scenarios {
            match self.run_scenario_benchmark(scenario) {
                Ok(result) => {
                    if result.metrics.completed {
                        successful += 1;
                    }
                    total_throughput += result.metrics.throughput;
                    results.push(result);
                }
                Err(e) => {
                    results.push(ScenarioBenchmarkResult {
                        scenario_id: scenario.id.clone(),
                        metrics: ScenarioBenchmarkMetrics {
                            throughput: 0.0,
                            latency_ms: 0.0,
                            gpu_utilization: 0.0,
                            memory_usage_mb: 0.0,
                            total_steps: 0,
                            completed: false,
                        },
                        duration: Duration::from_secs(0),
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        let total_duration = start_time.elapsed();
        let failed = results.len() - successful;
        let average_throughput = if results.is_empty() {
            0.0
        } else {
            total_throughput / results.len() as f64
        };

        Ok(ScenarioSuiteResults {
            suite_name: suite.name.clone(),
            scenario_results: results,
            summary: SuiteSummary {
                total_duration,
                successful_scenarios: successful,
                failed_scenarios: failed,
                average_throughput,
            },
        })
    }
}

/// Metrics for scenario comparison
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComparisonMetric {
    Throughput,
    Latency,
    GpuUtilization,
    MemoryUsage,
}

/// Comparison between two scenarios
#[derive(Debug, Clone)]
pub struct ScenarioComparison {
    pub baseline_id: String,
    pub comparison_id: String,
    pub metrics: Vec<(ComparisonMetric, f64, f64)>, // (metric, baseline, comparison)
}

impl ScenarioComparison {
    /// Check if comparison has a specific metric
    pub fn has_metric(&self, metric: ComparisonMetric) -> bool {
        self.metrics.iter().any(|(m, _, _)| *m == metric)
    }

    /// Calculate improvement percentage for throughput
    pub fn improvement_percentage(&self) -> f64 {
        for (metric, baseline, comparison) in &self.metrics {
            if *metric == ComparisonMetric::Throughput {
                return ((comparison - baseline) / baseline) * 100.0;
            }
        }
        0.0
    }
}

impl ScenarioBenchmarkRunner {
    /// Compare two scenarios
    pub fn compare_scenarios(
        &self,
        baseline: &ScenarioConfig,
        comparison: &ScenarioConfig,
    ) -> Result<ScenarioComparison, anyhow::Error> {
        let baseline_result = self.run_scenario_benchmark(baseline)?;
        let comparison_result = self.run_scenario_benchmark(comparison)?;

        let metrics = vec![
            (
                ComparisonMetric::Throughput,
                baseline_result.metrics.throughput,
                comparison_result.metrics.throughput,
            ),
            (
                ComparisonMetric::Latency,
                baseline_result.metrics.latency_ms,
                comparison_result.metrics.latency_ms,
            ),
            (
                ComparisonMetric::GpuUtilization,
                baseline_result.metrics.gpu_utilization,
                comparison_result.metrics.gpu_utilization,
            ),
            (
                ComparisonMetric::MemoryUsage,
                baseline_result.metrics.memory_usage_mb,
                comparison_result.metrics.memory_usage_mb,
            ),
        ];

        Ok(ScenarioComparison {
            baseline_id: baseline.id.clone(),
            comparison_id: comparison.id.clone(),
            metrics,
        })
    }
}

/// Convert scenario config to benchmark config
pub fn scenario_to_benchmark_config(scenario: &ScenarioConfig) -> BenchmarkConfig {
    BenchmarkConfig {
        benchmark_name: scenario.id.clone(),
        agent_count: scenario.agent_count,
        duration: scenario.duration,
        seed: scenario.seed,
    }
}

/// Simple benchmark config for conversion
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub benchmark_name: String,
    pub agent_count: usize,
    pub duration: Duration,
    pub seed: Option<u64>,
}

/// Report format options
#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    Html,
    Json,
    Markdown,
}

/// Benchmark report for scenarios
pub struct ScenarioBenchmarkReport {
    result: ScenarioBenchmarkResult,
}

impl ScenarioBenchmarkReport {
    /// Create report from result
    pub fn from_result(result: &ScenarioBenchmarkResult) -> Self {
        Self {
            result: result.clone(),
        }
    }

    /// Save report to file
    pub fn save(&self, path: &std::path::Path, format: ReportFormat) -> Result<(), anyhow::Error> {
        let content = match format {
            ReportFormat::Html => self.generate_html(),
            ReportFormat::Json => serde_json::to_string_pretty(&self.result.metrics)?,
            ReportFormat::Markdown => self.generate_markdown(),
        };

        std::fs::write(path, content)?;
        Ok(())
    }

    fn generate_html(&self) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Scenario Benchmark Report - {}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metrics {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .label {{ font-weight: bold; display: inline-block; width: 200px; }}
    </style>
</head>
<body>
    <h1>Scenario Benchmark Report</h1>
    <h2>Scenario: {}</h2>
    <div class="metrics">
        <div class="metric"><span class="label">Throughput:</span> {:.2} agents/sec</div>
        <div class="metric"><span class="label">Latency:</span> {:.2} ms</div>
        <div class="metric"><span class="label">GPU Utilization:</span> {:.1}%</div>
        <div class="metric"><span class="label">Memory Usage:</span> {:.1} MB</div>
        <div class="metric"><span class="label">Total Steps:</span> {}</div>
        <div class="metric"><span class="label">Completed:</span> {}</div>
    </div>
</body>
</html>"#,
            self.result.scenario_id,
            self.result.scenario_id,
            self.result.metrics.throughput,
            self.result.metrics.latency_ms,
            self.result.metrics.gpu_utilization,
            self.result.metrics.memory_usage_mb,
            self.result.metrics.total_steps,
            self.result.metrics.completed
        )
    }

    fn generate_markdown(&self) -> String {
        format!(
            r#"# Scenario Benchmark Report

## Scenario: {}

### Metrics
- **Throughput**: {:.2} agents/sec
- **Latency**: {:.2} ms
- **GPU Utilization**: {:.1}%
- **Memory Usage**: {:.1} MB
- **Total Steps**: {}
- **Completed**: {}

### Duration
Total execution time: {:.2} seconds
"#,
            self.result.scenario_id,
            self.result.metrics.throughput,
            self.result.metrics.latency_ms,
            self.result.metrics.gpu_utilization,
            self.result.metrics.memory_usage_mb,
            self.result.metrics.total_steps,
            self.result.metrics.completed,
            self.result.duration.as_secs_f64()
        )
    }
}
