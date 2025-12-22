//! Canary testing framework for gradual deployments

use crate::error::{OperationalError, OperationalResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Canary testing system for gradual deployment validation
#[derive(Debug)]
pub struct CanaryTesting {
    /// Active canary tests
    active_tests: HashMap<String, CanaryTest>,
    /// Test results history
    results_history: Vec<CanaryResult>,
    /// Default test configuration
    default_config: CanaryConfig,
}

/// Canary test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryConfig {
    /// Test identifier
    pub test_id: String,
    /// Deployment being tested
    pub deployment_id: String,
    /// Traffic percentage for canary (0-100)
    pub traffic_percentage: u8,
    /// Test duration
    pub duration: Duration,
    /// Success criteria
    pub success_criteria: SuccessCriteria,
    /// Rollback threshold
    pub rollback_threshold: RollbackThreshold,
    /// Test stages
    pub stages: Vec<CanaryStage>,
}

/// Success criteria for canary tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    /// Minimum success rate percentage
    pub min_success_rate: f32,
    /// Maximum error rate percentage
    pub max_error_rate: f32,
    /// Maximum response time (ms)
    pub max_response_time_ms: f32,
    /// Minimum throughput (requests/sec)
    pub min_throughput: f32,
}

/// Rollback threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackThreshold {
    /// Error rate threshold for automatic rollback
    pub error_rate_threshold: f32,
    /// Response time threshold for rollback (ms)
    pub response_time_threshold_ms: f32,
    /// Consecutive failures before rollback
    pub consecutive_failures: u32,
}

/// Canary test stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryStage {
    /// Stage name
    pub name: String,
    /// Traffic percentage for this stage
    pub traffic_percentage: u8,
    /// Stage duration
    pub duration: Duration,
    /// Auto-promotion criteria
    pub auto_promote: bool,
}

/// Active canary test
#[derive(Debug, Clone)]
struct CanaryTest {
    config: CanaryConfig,
    status: CanaryStatus,
    started_at: DateTime<Utc>,
    current_stage: usize,
    metrics: TestMetrics,
}

/// Canary test status
#[derive(Debug, Clone, PartialEq)]
enum CanaryStatus {
    Running,
    Paused,
    Completed,
    Failed,
    RolledBack,
}

/// Test metrics for canary analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TestMetrics {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average response time (ms)
    pub avg_response_time_ms: f32,
    /// 95th percentile response time (ms)
    pub p95_response_time_ms: f32,
    /// Throughput (requests/sec)
    pub throughput_rps: f32,
    /// Error rate percentage
    pub error_rate: f32,
}

/// Canary test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryResult {
    /// Test identifier
    pub test_id: String,
    /// Deployment identifier
    pub deployment_id: String,
    /// Test outcome
    pub outcome: TestOutcome,
    /// Final metrics
    pub metrics: TestMetrics,
    /// Test duration
    pub duration: Duration,
    /// Started timestamp
    pub started_at: DateTime<Utc>,
    /// Completed timestamp
    pub completed_at: DateTime<Utc>,
    /// Failure reason (if any)
    pub failure_reason: Option<String>,
}

/// Test outcome
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestOutcome {
    Success,
    Failure,
    RolledBack,
    Cancelled,
}

impl CanaryTesting {
    /// Create a new canary testing system
    pub fn new() -> Self {
        Self {
            active_tests: HashMap::new(),
            results_history: Vec::new(),
            default_config: CanaryConfig::default(),
        }
    }

    /// Start a canary test
    pub async fn start_test(&mut self, config: CanaryConfig) -> OperationalResult<String> {
        let test_id = config.test_id.clone();

        // Validate configuration
        self.validate_config(&config)?;

        // Check if test already exists
        if self.active_tests.contains_key(&test_id) {
            return Err(OperationalError::ConfigurationError(format!(
                "Canary test {} already active",
                test_id
            )));
        }

        let test = CanaryTest {
            config: config.clone(),
            status: CanaryStatus::Running,
            started_at: Utc::now(),
            current_stage: 0,
            metrics: TestMetrics::default(),
        };

        self.active_tests.insert(test_id.clone(), test);

        // Start the test execution
        self.execute_test_stage(&test_id).await?;

        Ok(test_id)
    }

    /// Stop a canary test
    pub async fn stop_test(&mut self, test_id: &str) -> OperationalResult<CanaryResult> {
        let test = self.active_tests.remove(test_id).ok_or_else(|| {
            OperationalError::ConfigurationError(format!("Canary test {} not found", test_id))
        })?;

        let result = CanaryResult {
            test_id: test.config.test_id.clone(),
            deployment_id: test.config.deployment_id.clone(),
            outcome: match test.status {
                CanaryStatus::Completed => TestOutcome::Success,
                CanaryStatus::Failed => TestOutcome::Failure,
                CanaryStatus::RolledBack => TestOutcome::RolledBack,
                _ => TestOutcome::Cancelled,
            },
            metrics: test.metrics,
            duration: Utc::now()
                .signed_duration_since(test.started_at)
                .to_std()
                .unwrap_or(Duration::ZERO),
            started_at: test.started_at,
            completed_at: Utc::now(),
            failure_reason: None,
        };

        self.results_history.push(result.clone());
        Ok(result)
    }

    /// Get active test status
    pub fn get_test_status(&self, test_id: &str) -> Option<CanaryStatus> {
        self.active_tests
            .get(test_id)
            .map(|test| test.status.clone())
    }

    /// List active tests
    pub fn list_active_tests(&self) -> Vec<String> {
        self.active_tests.keys().cloned().collect()
    }

    /// Get test results history
    pub fn get_results_history(&self) -> &[CanaryResult] {
        &self.results_history
    }

    /// Update test metrics
    pub fn update_test_metrics(
        &mut self,
        test_id: &str,
        metrics: TestMetrics,
    ) -> OperationalResult<()> {
        // First check if test exists and get config
        let config = if let Some(test) = self.active_tests.get(test_id) {
            test.config.clone()
        } else {
            return Err(OperationalError::ConfigurationError(format!(
                "Canary test {} not found",
                test_id
            )));
        };

        // Now get mutable reference and update
        if let Some(test) = self.active_tests.get_mut(test_id) {
            test.metrics = metrics.clone();

            // Check if test should be rolled back
            if Self::should_rollback_static(&config, &metrics) {
                test.status = CanaryStatus::RolledBack;
            }
        }

        Ok(())
    }

    /// Promote canary to next stage
    pub async fn promote_to_next_stage(&mut self, test_id: &str) -> OperationalResult<bool> {
        if let Some(test) = self.active_tests.get_mut(test_id) {
            if test.current_stage + 1 < test.config.stages.len() {
                test.current_stage += 1;
                self.execute_test_stage(test_id).await?;
                Ok(true)
            } else {
                test.status = CanaryStatus::Completed;
                Ok(false) // No more stages
            }
        } else {
            Err(OperationalError::ConfigurationError(format!(
                "Canary test {} not found",
                test_id
            )))
        }
    }

    /// Get current test metrics
    pub fn get_test_metrics(&self, test_id: &str) -> Option<&TestMetrics> {
        self.active_tests.get(test_id).map(|test| &test.metrics)
    }

    // Private helper methods

    fn validate_config(&self, config: &CanaryConfig) -> OperationalResult<()> {
        if config.test_id.is_empty() {
            return Err(OperationalError::ConfigurationError(
                "Test ID cannot be empty".to_string(),
            ));
        }

        if config.deployment_id.is_empty() {
            return Err(OperationalError::ConfigurationError(
                "Deployment ID cannot be empty".to_string(),
            ));
        }

        if config.traffic_percentage > 100 {
            return Err(OperationalError::ConfigurationError(
                "Traffic percentage cannot exceed 100".to_string(),
            ));
        }

        if config.stages.is_empty() {
            return Err(OperationalError::ConfigurationError(
                "At least one canary stage is required".to_string(),
            ));
        }

        Ok(())
    }

    async fn execute_test_stage(&mut self, _test_id: &str) -> OperationalResult<()> {
        // Mock implementation for test stage execution
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    fn should_rollback(&self, config: &CanaryConfig, metrics: &TestMetrics) -> bool {
        Self::should_rollback_static(config, metrics)
    }

    fn should_rollback_static(config: &CanaryConfig, metrics: &TestMetrics) -> bool {
        let threshold = &config.rollback_threshold;

        metrics.error_rate > threshold.error_rate_threshold
            || metrics.avg_response_time_ms > threshold.response_time_threshold_ms
    }
}

impl Default for CanaryTesting {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self {
            test_id: "canary-test".to_string(),
            deployment_id: "deployment".to_string(),
            traffic_percentage: 10,
            duration: Duration::from_secs(300),
            success_criteria: SuccessCriteria::default(),
            rollback_threshold: RollbackThreshold::default(),
            stages: vec![
                CanaryStage {
                    name: "stage-1".to_string(),
                    traffic_percentage: 5,
                    duration: Duration::from_secs(120),
                    auto_promote: false,
                },
                CanaryStage {
                    name: "stage-2".to_string(),
                    traffic_percentage: 10,
                    duration: Duration::from_secs(180),
                    auto_promote: false,
                },
            ],
        }
    }
}

impl Default for SuccessCriteria {
    fn default() -> Self {
        Self {
            min_success_rate: 95.0,
            max_error_rate: 5.0,
            max_response_time_ms: 1000.0,
            min_throughput: 100.0,
        }
    }
}

impl Default for RollbackThreshold {
    fn default() -> Self {
        Self {
            error_rate_threshold: 10.0,
            response_time_threshold_ms: 2000.0,
            consecutive_failures: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> CanaryConfig {
        CanaryConfig {
            test_id: "test-canary".to_string(),
            deployment_id: "test-deployment".to_string(),
            traffic_percentage: 20,
            duration: Duration::from_secs(60),
            success_criteria: SuccessCriteria::default(),
            rollback_threshold: RollbackThreshold::default(),
            stages: vec![CanaryStage {
                name: "initial".to_string(),
                traffic_percentage: 10,
                duration: Duration::from_secs(30),
                auto_promote: false,
            }],
        }
    }

    #[test]
    fn test_canary_testing_creation() {
        let canary = CanaryTesting::new();
        assert!(canary.active_tests.is_empty());
        assert!(canary.results_history.is_empty());
    }

    #[test]
    fn test_config_serialization() {
        let config = create_test_config();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: CanaryConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config.test_id, deserialized.test_id);
        assert_eq!(config.deployment_id, deserialized.deployment_id);
        assert_eq!(config.traffic_percentage, deserialized.traffic_percentage);
    }

    #[test]
    fn test_test_metrics_serialization() {
        let metrics = TestMetrics {
            total_requests: 1000,
            successful_requests: 950,
            failed_requests: 50,
            avg_response_time_ms: 200.5,
            p95_response_time_ms: 800.0,
            throughput_rps: 100.0,
            error_rate: 5.0,
        };

        let serialized = serde_json::to_string(&metrics).unwrap();
        let deserialized: TestMetrics = serde_json::from_str(&serialized).unwrap();
        assert_eq!(metrics.total_requests, deserialized.total_requests);
        assert_eq!(
            metrics.successful_requests,
            deserialized.successful_requests
        );
        assert_eq!(
            metrics.avg_response_time_ms,
            deserialized.avg_response_time_ms
        );
    }

    #[test]
    fn test_config_validation() {
        let canary = CanaryTesting::new();

        // Valid config
        let valid_config = create_test_config();
        assert!(canary.validate_config(&valid_config).is_ok());

        // Empty test ID
        let mut invalid_config = create_test_config();
        invalid_config.test_id = String::new();
        assert!(canary.validate_config(&invalid_config).is_err());

        // Empty deployment ID
        let mut invalid_config = create_test_config();
        invalid_config.deployment_id = String::new();
        assert!(canary.validate_config(&invalid_config).is_err());

        // Invalid traffic percentage
        let mut invalid_config = create_test_config();
        invalid_config.traffic_percentage = 150;
        assert!(canary.validate_config(&invalid_config).is_err());

        // No stages
        let mut invalid_config = create_test_config();
        invalid_config.stages.clear();
        assert!(canary.validate_config(&invalid_config).is_err());
    }

    #[tokio::test]
    async fn test_canary_test_lifecycle() {
        let mut canary = CanaryTesting::new();
        let config = create_test_config();
        let test_id = config.test_id.clone();

        // Start test
        let result = canary.start_test(config).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_id);

        // Check test is active
        let active_tests = canary.list_active_tests();
        assert_eq!(active_tests.len(), 1);
        assert_eq!(active_tests[0], test_id);

        // Check status
        let status = canary.get_test_status(&test_id);
        assert_eq!(status, Some(CanaryStatus::Running));

        // Stop test
        let result = canary.stop_test(&test_id).await;
        assert!(result.is_ok());

        // Check test is no longer active
        let active_tests = canary.list_active_tests();
        assert!(active_tests.is_empty());

        // Check result is in history
        let history = canary.get_results_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].test_id, test_id);
    }

    #[tokio::test]
    async fn test_duplicate_test_start() {
        let mut canary = CanaryTesting::new();
        let config = create_test_config();

        // Start first test
        let result1 = canary.start_test(config.clone()).await;
        assert!(result1.is_ok());

        // Try to start duplicate test
        let result2 = canary.start_test(config).await;
        assert!(result2.is_err());
        assert!(matches!(
            result2.unwrap_err(),
            OperationalError::ConfigurationError(_)
        ));
    }

    #[tokio::test]
    async fn test_test_metrics_update() {
        let mut canary = CanaryTesting::new();
        let config = create_test_config();
        let test_id = config.test_id.clone();

        canary.start_test(config).await.unwrap();

        let metrics = TestMetrics {
            total_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            avg_response_time_ms: 150.0,
            p95_response_time_ms: 300.0,
            throughput_rps: 50.0,
            error_rate: 5.0,
        };

        let result = canary.update_test_metrics(&test_id, metrics.clone());
        assert!(result.is_ok());

        let retrieved_metrics = canary.get_test_metrics(&test_id);
        assert!(retrieved_metrics.is_some());
        assert_eq!(retrieved_metrics.unwrap().total_requests, 100);
    }

    #[tokio::test]
    async fn test_stage_promotion() {
        let mut canary = CanaryTesting::new();
        let mut config = create_test_config();

        // Add multiple stages
        config.stages.push(CanaryStage {
            name: "final".to_string(),
            traffic_percentage: 50,
            duration: Duration::from_secs(60),
            auto_promote: false,
        });

        let test_id = config.test_id.clone();
        canary.start_test(config).await.unwrap();

        // Promote to next stage
        let result = canary.promote_to_next_stage(&test_id).await;
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should return true for successful promotion

        // Try to promote beyond available stages
        let result = canary.promote_to_next_stage(&test_id).await;
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should return false when no more stages

        // Check status is completed
        let status = canary.get_test_status(&test_id);
        assert_eq!(status, Some(CanaryStatus::Completed));
    }

    #[tokio::test]
    async fn test_rollback_detection() {
        let mut canary = CanaryTesting::new();
        let config = create_test_config();
        let test_id = config.test_id.clone();

        canary.start_test(config).await.unwrap();

        // Update with metrics that should trigger rollback
        let bad_metrics = TestMetrics {
            total_requests: 100,
            successful_requests: 85,
            failed_requests: 15,
            avg_response_time_ms: 3000.0, // Exceeds threshold
            p95_response_time_ms: 5000.0,
            throughput_rps: 10.0,
            error_rate: 15.0, // Exceeds threshold
        };

        canary.update_test_metrics(&test_id, bad_metrics).unwrap();

        // Check status changed to rolled back
        let status = canary.get_test_status(&test_id);
        assert_eq!(status, Some(CanaryStatus::RolledBack));
    }

    #[test]
    fn test_test_outcome_serialization() {
        let outcomes = vec![
            TestOutcome::Success,
            TestOutcome::Failure,
            TestOutcome::RolledBack,
            TestOutcome::Cancelled,
        ];

        for outcome in outcomes {
            let serialized = serde_json::to_string(&outcome).unwrap();
            let deserialized: TestOutcome = serde_json::from_str(&serialized).unwrap();
            assert_eq!(outcome, deserialized);
        }
    }

    #[test]
    fn test_canary_result_serialization() {
        let result = CanaryResult {
            test_id: "test-123".to_string(),
            deployment_id: "deploy-123".to_string(),
            outcome: TestOutcome::Success,
            metrics: TestMetrics::default(),
            duration: Duration::from_secs(300),
            started_at: Utc::now(),
            completed_at: Utc::now(),
            failure_reason: None,
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: CanaryResult = serde_json::from_str(&serialized).unwrap();
        assert_eq!(result.test_id, deserialized.test_id);
        assert_eq!(result.outcome, deserialized.outcome);
    }

    #[test]
    fn test_default_implementations() {
        let _canary = CanaryTesting::default();
        let _config = CanaryConfig::default();
        let _criteria = SuccessCriteria::default();
        let _threshold = RollbackThreshold::default();
        let _metrics = TestMetrics::default();
    }

    #[tokio::test]
    async fn test_nonexistent_test_operations() {
        let mut canary = CanaryTesting::new();

        // Stop non-existent test
        let result = canary.stop_test("nonexistent").await;
        assert!(result.is_err());

        // Update metrics for non-existent test
        let result = canary.update_test_metrics("nonexistent", TestMetrics::default());
        assert!(result.is_err());

        // Promote non-existent test
        let result = canary.promote_to_next_stage("nonexistent").await;
        assert!(result.is_err());

        // Get status of non-existent test
        let status = canary.get_test_status("nonexistent");
        assert!(status.is_none());

        // Get metrics of non-existent test
        let metrics = canary.get_test_metrics("nonexistent");
        assert!(metrics.is_none());
    }

    #[test]
    fn test_canary_stage_validation() {
        let stage = CanaryStage {
            name: "test-stage".to_string(),
            traffic_percentage: 25,
            duration: Duration::from_secs(300),
            auto_promote: true,
        };

        assert_eq!(stage.name, "test-stage");
        assert_eq!(stage.traffic_percentage, 25);
        assert_eq!(stage.duration.as_secs(), 300);
        assert!(stage.auto_promote);
    }

    #[test]
    fn test_success_criteria_thresholds() {
        let criteria = SuccessCriteria {
            min_success_rate: 99.5,
            max_error_rate: 0.5,
            max_response_time_ms: 100.0,
            min_throughput: 1000.0,
        };

        assert_eq!(criteria.min_success_rate, 99.5);
        assert_eq!(criteria.max_error_rate, 0.5);
        assert_eq!(criteria.max_response_time_ms, 100.0);
        assert_eq!(criteria.min_throughput, 1000.0);
    }

    #[test]
    fn test_rollback_threshold_edge_cases() {
        let threshold = RollbackThreshold {
            error_rate_threshold: 0.0,
            response_time_threshold_ms: 0.0,
            consecutive_failures: u32::MAX,
        };

        assert_eq!(threshold.error_rate_threshold, 0.0);
        assert_eq!(threshold.response_time_threshold_ms, 0.0);
        assert_eq!(threshold.consecutive_failures, u32::MAX);
    }

    #[test]
    fn test_test_metrics_calculations() {
        let metrics = TestMetrics {
            total_requests: 1000,
            successful_requests: 950,
            failed_requests: 50,
            avg_response_time_ms: 45.5,
            p95_response_time_ms: 95.0,
            throughput_rps: 100.0,
            error_rate: 5.0,
        };

        // Verify consistency
        assert_eq!(
            metrics.total_requests,
            metrics.successful_requests + metrics.failed_requests
        );
        assert!(metrics.p95_response_time_ms >= metrics.avg_response_time_ms);
        assert_eq!(
            metrics.error_rate,
            (metrics.failed_requests as f32 / metrics.total_requests as f32) * 100.0
        );
    }

    #[test]
    fn test_canary_status_transitions() {
        let statuses = vec![
            CanaryStatus::Running,
            CanaryStatus::Paused,
            CanaryStatus::Completed,
            CanaryStatus::Failed,
            CanaryStatus::RolledBack,
        ];

        for status in statuses {
            let _name = format!("{:?}", status);
        }
    }

    #[tokio::test]
    async fn test_concurrent_canary_tests() {
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let canary = Arc::new(Mutex::new(CanaryTesting::new()));
        let mut handles = vec![];

        for i in 0..5 {
            let canary_clone = canary.clone();
            let handle = tokio::spawn(async move {
                let config = CanaryConfig {
                    test_id: format!("concurrent-test-{}", i),
                    deployment_id: format!("deploy-{}", i),
                    traffic_percentage: 10,
                    duration: Duration::from_secs(60),
                    stages: vec![CanaryStage {
                        name: "test".to_string(),
                        traffic_percentage: 10,
                        duration: Duration::from_secs(60),
                        auto_promote: false,
                    }],
                    success_criteria: SuccessCriteria::default(),
                    rollback_threshold: RollbackThreshold::default(),
                };

                let mut canary_locked = canary_clone.lock().await;
                canary_locked.start_test(config).await
            });
            handles.push(handle);
        }

        // Wait for all handles without futures crate
        for handle in handles {
            let result = handle.await;
            assert!(result.is_ok());
            assert!(result.unwrap().is_ok());
        }

        let canary_locked = canary.lock().await;
        assert_eq!(canary_locked.list_active_tests().len(), 5);
    }

    #[test]
    fn test_canary_config_with_empty_stages() {
        let config = CanaryConfig {
            test_id: "test".to_string(),
            deployment_id: "deploy".to_string(),
            traffic_percentage: 10,
            duration: Duration::from_secs(60),
            stages: vec![], // Empty stages
            success_criteria: SuccessCriteria::default(),
            rollback_threshold: RollbackThreshold::default(),
        };

        // Empty stages should still be valid for struct creation
        assert_eq!(config.stages.len(), 0);
    }

    #[test]
    fn test_canary_result_with_failure_reason() {
        let result = CanaryResult {
            test_id: "test-123".to_string(),
            deployment_id: "deploy-123".to_string(),
            outcome: TestOutcome::Failure,
            metrics: TestMetrics::default(),
            duration: Duration::from_secs(60),
            started_at: Utc::now(),
            completed_at: Utc::now(),
            failure_reason: Some("High error rate detected: 15% > 5%".to_string()),
        };

        assert_eq!(result.outcome, TestOutcome::Failure);
        assert!(result.failure_reason.is_some());
        assert!(result.failure_reason.unwrap().contains("High error rate"));
    }

    #[test]
    fn test_test_metrics_edge_values() {
        let metrics_zero = TestMetrics {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_response_time_ms: 0.0,
            p95_response_time_ms: 0.0,
            throughput_rps: 0.0,
            error_rate: 0.0,
        };

        let metrics_max = TestMetrics {
            total_requests: u64::MAX,
            successful_requests: u64::MAX,
            failed_requests: 0,
            avg_response_time_ms: f32::MAX,
            p95_response_time_ms: f32::MAX,
            throughput_rps: f32::MAX,
            error_rate: 100.0,
        };

        // Should handle edge values without panic
        let _ = format!("{:?}", metrics_zero);
        let _ = format!("{:?}", metrics_max);
    }
}
