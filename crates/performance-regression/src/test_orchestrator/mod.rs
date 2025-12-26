//! Test orchestrator module
//!
//! This module provides modular test orchestration capabilities split into
//! logical components for better maintainability and organization.

pub mod ci_integration;
pub mod config;
pub mod results;
pub mod scheduler;
pub mod test_strategies;

#[cfg(test)]
mod tests;

// Re-export public types for backward compatibility
pub use ci_integration::CiIntegration;
pub use config::*;
pub use results::*;
pub use scheduler::TestScheduler;
pub use test_strategies::TestStrategyExecutor;

// Main orchestrator implementation
use crate::error::{PerformanceRegressionError, PerformanceRegressionResult};
use crate::metrics_collector::{MetricDataPoint, MetricStatistics, MetricType};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info};

// Re-export uuid for compatibility
use uuid;

/// Test orchestrator
pub struct TestOrchestrator {
    config: TestOrchestratorConfig,
    test_queue: Arc<RwLock<Vec<ScheduledTest>>>,
    execution_results: Arc<RwLock<HashMap<String, TestExecutionResult>>>,
    active_tests: Arc<RwLock<HashMap<String, TestExecutionResult>>>,
    scheduler_handle: Option<tokio::task::JoinHandle<()>>,
}

impl TestOrchestrator {
    /// Create new test orchestrator
    pub fn new(config: TestOrchestratorConfig) -> Self {
        Self {
            config,
            test_queue: Arc::new(RwLock::new(Vec::new())),
            execution_results: Arc::new(RwLock::new(HashMap::new())),
            active_tests: Arc::new(RwLock::new(HashMap::new())),
            scheduler_handle: None,
        }
    }

    /// Orchestrate test execution
    pub async fn orchestrate_tests(
        &self,
        test_strategies: Vec<TestStrategy>,
    ) -> PerformanceRegressionResult<TestReport> {
        info!("Orchestrating {} tests", test_strategies.len());

        let mut results = Vec::new();
        let start_time = Utc::now();

        if self.config.parallel_execution {
            // Execute tests in parallel
            let mut handles = Vec::new();
            let semaphore = Arc::new(tokio::sync::Semaphore::new(
                self.config.max_concurrent_tests,
            ));

            for strategy in test_strategies {
                let permit = semaphore.clone().acquire_owned().await?;

                let test_id = uuid::Uuid::new_v4().to_string();
                let start_time = Utc::now();

                let mut result = TestExecutionResult {
                    test_id: test_id.clone(),
                    strategy: strategy.clone(),
                    start_time,
                    end_time: start_time,
                    status: TestStatus::Running,
                    metrics: HashMap::new(),
                    statistics: HashMap::new(),
                    insights: TestInsights {
                        bottlenecks: Vec::new(),
                        improvements: Vec::new(),
                        warnings: Vec::new(),
                        recommendations: Vec::new(),
                        success_rate: 0.0,
                        avg_response_time_ms: 0.0,
                        peak_resource_usage: ResourceUsage {
                            cpu_percent: 0.0,
                            memory_mb: 0.0,
                            disk_iops: 0.0,
                            network_mbps: 0.0,
                        },
                    },
                    error_details: None,
                };

                self.active_tests
                    .write()
                    .await
                    .insert(test_id.clone(), result.clone());

                let active_tests = self.active_tests.clone();
                let execution_results = self.execution_results.clone();

                let handle = tokio::spawn(async move {
                    // Execute test using strategy executor
                    let execution_result = Self::execute_strategy_internal(strategy).await;

                    // Update result based on execution
                    match execution_result {
                        Ok((metrics, insights)) => {
                            result.metrics = metrics;
                            result.insights = insights;
                            result.status = TestStatus::Passed;
                        }
                        Err(e) => {
                            result.status = TestStatus::Failed;
                            result.error_details = Some(e.to_string());
                        }
                    }

                    result.end_time = Utc::now();

                    // Remove from active tests and add to results
                    active_tests.write().await.remove(&test_id);
                    execution_results
                        .write()
                        .await
                        .insert(test_id.clone(), result.clone());

                    drop(permit);
                    Ok::<TestExecutionResult, PerformanceRegressionError>(result)
                });
                handles.push(handle);
            }

            for handle in handles {
                match handle.await? {
                    Ok(result) => results.push(result),
                    Err(e) => error!("Test execution failed: {}", e),
                }
            }
        } else {
            // Execute tests sequentially
            for strategy in test_strategies {
                match self.execute_test_strategy(strategy).await {
                    Ok(result) => results.push(result),
                    Err(e) => error!("Test execution failed: {}", e),
                }
            }
        }

        let end_time = Utc::now();
        let report = self
            .generate_test_report_internal(results, start_time, end_time)
            .await?;

        Ok(report)
    }

    /// Schedule test execution
    pub async fn schedule_tests(
        &mut self,
        test_strategies: Vec<TestStrategy>,
        cron_expression: Option<String>,
    ) -> PerformanceRegressionResult<()> {
        if !self.config.scheduling_config.enabled {
            return Err(PerformanceRegressionError::ConfigurationError {
                parameter: "scheduling_config.enabled".to_string(),
                message: "Test scheduling is not enabled".to_string(),
            });
        }

        let cron_expr = cron_expression
            .or(self.config.scheduling_config.cron_expression.clone())
            .ok_or_else(|| PerformanceRegressionError::ConfigurationError {
                parameter: "cron_expression".to_string(),
                message: "No cron expression provided".to_string(),
            })?;

        // Add tests to queue
        let mut queue = self.test_queue.write().await;
        for strategy in test_strategies {
            let scheduled_test = ScheduledTest {
                id: uuid::Uuid::new_v4().to_string(),
                strategy,
                scheduled_time: Utc::now(), // Will be updated by scheduler
                retry_count: 0,
            };
            queue.push(scheduled_test);
        }

        // Start scheduler if not running
        if self.scheduler_handle.is_none() {
            let queue_clone = self.test_queue.clone();
            let config_clone = self.config.clone();

            self.scheduler_handle = Some(tokio::spawn(async move {
                TestScheduler::run_scheduler(queue_clone, config_clone, cron_expr).await;
            }));
        }

        Ok(())
    }

    /// Execute load test
    pub async fn execute_load_test(
        &self,
        config: LoadTestConfig,
    ) -> PerformanceRegressionResult<TestExecutionResult> {
        let strategy = TestStrategy::LoadTest(config);
        self.execute_test_strategy(strategy).await
    }

    /// Execute stress test
    pub async fn execute_stress_test(
        &self,
        config: StressTestConfig,
    ) -> PerformanceRegressionResult<TestExecutionResult> {
        let strategy = TestStrategy::StressTest(config);
        self.execute_test_strategy(strategy).await
    }

    /// Collect test metrics
    pub async fn collect_test_metrics(
        &self,
        test_id: &str,
    ) -> PerformanceRegressionResult<HashMap<MetricType, Vec<MetricDataPoint>>> {
        let active_tests = self.active_tests.read().await;
        if let Some(test) = active_tests.get(test_id) {
            Ok(test.metrics.clone())
        } else {
            let results = self.execution_results.read().await;
            if let Some(test) = results.get(test_id) {
                Ok(test.metrics.clone())
            } else {
                Err(PerformanceRegressionError::TestNotFound {
                    test_id: test_id.to_string(),
                })
            }
        }
    }

    /// Generate test report
    pub async fn generate_test_report(
        &self,
        test_ids: Vec<String>,
    ) -> PerformanceRegressionResult<TestReport> {
        let results = self.execution_results.read().await;
        let mut test_results = Vec::new();

        for test_id in test_ids {
            if let Some(result) = results.get(&test_id) {
                test_results.push(result.clone());
            }
        }

        if test_results.is_empty() {
            return Err(PerformanceRegressionError::NoTestResults);
        }

        let start_time = test_results.iter().map(|r| r.start_time).min().unwrap();
        let end_time = test_results.iter().map(|r| r.end_time).max().unwrap();

        self.generate_test_report_internal(test_results, start_time, end_time)
            .await
    }

    /// Integrate with CI/CD
    pub async fn integrate_with_ci(
        &self,
        report: TestReport,
        ci_metadata: CiMetadata,
    ) -> PerformanceRegressionResult<()> {
        if !self.config.ci_integration.enabled {
            return Ok(());
        }

        let mut report_with_metadata = report;
        report_with_metadata.ci_metadata = Some(ci_metadata);

        // Format report based on configuration
        let formatted_report = CiIntegration::format_report(
            &report_with_metadata,
            &self.config.ci_integration.report_format,
        )?;

        // Send webhook notification if configured
        if let Some(webhook_url) = &self.config.ci_integration.webhook_url {
            if report_with_metadata.summary.failed_tests > 0
                || self.config.ci_integration.notify_on_failure
            {
                CiIntegration::send_webhook_notification(webhook_url, &formatted_report).await?;
            }
        }

        // Platform-specific integration
        CiIntegration::execute_platform_integration(
            &self.config.ci_integration.platform,
            &report_with_metadata,
        )
        .await?;

        Ok(())
    }

    // Internal methods
    async fn execute_test_strategy(
        &self,
        strategy: TestStrategy,
    ) -> PerformanceRegressionResult<TestExecutionResult> {
        let test_id = uuid::Uuid::new_v4().to_string();
        let start_time = Utc::now();

        let mut result = TestExecutionResult {
            test_id: test_id.clone(),
            strategy: strategy.clone(),
            start_time,
            end_time: start_time, // Will be updated
            status: TestStatus::Running,
            metrics: HashMap::new(),
            statistics: HashMap::new(),
            insights: TestInsights {
                bottlenecks: Vec::new(),
                improvements: Vec::new(),
                warnings: Vec::new(),
                recommendations: Vec::new(),
                success_rate: 0.0,
                avg_response_time_ms: 0.0,
                peak_resource_usage: ResourceUsage {
                    cpu_percent: 0.0,
                    memory_mb: 0.0,
                    disk_iops: 0.0,
                    network_mbps: 0.0,
                },
            },
            error_details: None,
        };

        // Add to active tests
        self.active_tests
            .write()
            .await
            .insert(test_id.clone(), result.clone());

        // Execute test based on strategy
        let execution_result = Self::execute_strategy_internal(strategy).await;

        // Update result based on execution
        match execution_result {
            Ok((metrics, insights)) => {
                result.metrics = metrics;
                result.insights = insights;
                result.status = TestStatus::Passed;

                // Calculate statistics
                for (metric_type, data_points) in &result.metrics {
                    if let Some(stats) = self.calculate_statistics(data_points) {
                        result.statistics.insert(metric_type.clone(), stats);
                    }
                }
            }
            Err(e) => {
                result.status = TestStatus::Failed;
                result.error_details = Some(e.to_string());
            }
        }

        result.end_time = Utc::now();

        // Remove from active tests and add to results
        self.active_tests.write().await.remove(&test_id);
        self.execution_results
            .write()
            .await
            .insert(test_id.clone(), result.clone());

        Ok(result)
    }

    async fn execute_strategy_internal(
        strategy: TestStrategy,
    ) -> PerformanceRegressionResult<(HashMap<MetricType, Vec<MetricDataPoint>>, TestInsights)>
    {
        match strategy {
            TestStrategy::LoadTest(config) => TestStrategyExecutor::run_load_test(config).await,
            TestStrategy::StressTest(config) => TestStrategyExecutor::run_stress_test(config).await,
            TestStrategy::EnduranceTest(config) => {
                TestStrategyExecutor::run_endurance_test(config).await
            }
            TestStrategy::SpikeTest(config) => TestStrategyExecutor::run_spike_test(config).await,
            TestStrategy::VolumeTest(config) => TestStrategyExecutor::run_volume_test(config).await,
        }
    }

    async fn generate_test_report_internal(
        &self,
        results: Vec<TestExecutionResult>,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> PerformanceRegressionResult<TestReport> {
        let total_tests = results.len();
        let passed_tests = results
            .iter()
            .filter(|r| r.status == TestStatus::Passed)
            .count();
        let failed_tests = results
            .iter()
            .filter(|r| r.status == TestStatus::Failed)
            .count();

        let total_duration = (end_time - start_time).num_seconds() as f64;
        let avg_duration = if total_tests > 0 {
            results
                .iter()
                .map(|r| (r.end_time - r.start_time).num_seconds() as f64)
                .sum::<f64>()
                / total_tests as f64
        } else {
            0.0
        };

        let summary = TestSummary {
            total_tests,
            passed_tests,
            failed_tests,
            success_rate: if total_tests > 0 {
                (passed_tests as f64 / total_tests as f64) * 100.0
            } else {
                0.0
            },
            avg_duration_seconds: avg_duration,
            total_duration_seconds: total_duration,
        };

        Ok(TestReport {
            report_id: uuid::Uuid::new_v4().to_string(),
            generated_at: Utc::now(),
            results,
            summary,
            ci_metadata: None,
        })
    }

    fn calculate_statistics(&self, data_points: &[MetricDataPoint]) -> Option<MetricStatistics> {
        if data_points.is_empty() {
            return None;
        }

        let values: Vec<f64> = data_points.iter().map(|dp| dp.value.0).collect();
        let count = values.len() as f64;
        let sum: f64 = values.iter().sum();
        let average = sum / count;

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let variance: f64 = values.iter().map(|v| (v - average).powi(2)).sum::<f64>() / count;
        let std_deviation = variance.sqrt();

        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p95_index = ((count * 0.95) as usize).min(sorted_values.len() - 1);
        let p99_index = ((count * 0.99) as usize).min(sorted_values.len() - 1);

        Some(MetricStatistics {
            metric_type: data_points[0].metric_type.clone(),
            average: ordered_float::OrderedFloat(average),
            minimum: ordered_float::OrderedFloat(min),
            maximum: ordered_float::OrderedFloat(max),
            std_deviation: ordered_float::OrderedFloat(std_deviation),
            p95: ordered_float::OrderedFloat(sorted_values[p95_index]),
            p99: ordered_float::OrderedFloat(sorted_values[p99_index]),
            sample_count: data_points.len(),
            window_start: data_points.first().unwrap().timestamp,
            window_end: data_points.last().unwrap().timestamp,
        })
    }
}
