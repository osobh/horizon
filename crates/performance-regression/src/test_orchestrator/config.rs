//! Configuration types and settings for test orchestration
//!
//! This module contains all configuration structs, enums, and default implementations
//! for controlling test orchestrator behavior, CI/CD integration, and scheduling.

use serde::{Deserialize, Serialize};

/// Test orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestOrchestratorConfig {
    /// Maximum concurrent test executions
    pub max_concurrent_tests: usize,
    /// Test execution timeout in seconds
    pub test_timeout_seconds: u64,
    /// Enable parallel test execution
    pub parallel_execution: bool,
    /// CI/CD integration settings
    pub ci_integration: CiIntegrationConfig,
    /// Test scheduling configuration
    pub scheduling_config: SchedulingConfig,
    /// Test result retention in hours
    pub result_retention_hours: u64,
}

impl Default for TestOrchestratorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tests: 4,
            test_timeout_seconds: 3600,
            parallel_execution: true,
            ci_integration: CiIntegrationConfig::default(),
            scheduling_config: SchedulingConfig::default(),
            result_retention_hours: 168, // 7 days
        }
    }
}

/// CI/CD integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiIntegrationConfig {
    /// Enable CI/CD integration
    pub enabled: bool,
    /// CI/CD platform type
    pub platform: CiPlatform,
    /// Webhook URL for notifications
    pub webhook_url: Option<String>,
    /// Enable automatic failure notifications
    pub notify_on_failure: bool,
    /// Report format
    pub report_format: ReportFormat,
}

impl Default for CiIntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            platform: CiPlatform::GitHub,
            webhook_url: None,
            notify_on_failure: true,
            report_format: ReportFormat::Json,
        }
    }
}

/// Supported CI/CD platforms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CiPlatform {
    /// GitHub Actions
    GitHub,
    /// GitLab CI
    GitLab,
    /// Jenkins
    Jenkins,
    /// CircleCI
    CircleCI,
    /// Generic webhook
    Generic,
}

/// Test report format
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportFormat {
    /// JSON format
    Json,
    /// JUnit XML format
    JUnit,
    /// Markdown format
    Markdown,
    /// HTML format
    Html,
}

/// Test scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingConfig {
    /// Enable scheduled test execution
    pub enabled: bool,
    /// Cron expression for scheduling
    pub cron_expression: Option<String>,
    /// Maximum scheduled test queue size
    pub max_queue_size: usize,
    /// Retry failed tests
    pub retry_on_failure: bool,
    /// Maximum retry attempts
    pub max_retries: u32,
}

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cron_expression: None,
            max_queue_size: 100,
            retry_on_failure: true,
            max_retries: 3,
        }
    }
}

/// Test execution strategy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestStrategy {
    /// Load testing - sustained load
    LoadTest(LoadTestConfig),
    /// Stress testing - increasing load until failure
    StressTest(StressTestConfig),
    /// Endurance testing - extended duration
    EnduranceTest(EnduranceTestConfig),
    /// Spike testing - sudden load changes
    SpikeTest(SpikeTestConfig),
    /// Volume testing - large data volumes
    VolumeTest(VolumeTestConfig),
}

/// Load test configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadTestConfig {
    /// Target requests per second
    pub target_rps: f64,
    /// Test duration in seconds
    pub duration_seconds: u64,
    /// Ramp-up time in seconds
    pub ramp_up_seconds: u64,
    /// Number of virtual users
    pub virtual_users: usize,
}

/// Stress test configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StressTestConfig {
    /// Initial requests per second
    pub initial_rps: f64,
    /// RPS increment per step
    pub rps_increment: f64,
    /// Step duration in seconds
    pub step_duration_seconds: u64,
    /// Maximum RPS limit
    pub max_rps: f64,
    /// Failure threshold (error rate)
    pub failure_threshold: f64,
}

/// Endurance test configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnduranceTestConfig {
    /// Target requests per second
    pub target_rps: f64,
    /// Test duration in hours
    pub duration_hours: u64,
    /// Memory leak detection threshold (MB/hour)
    pub memory_leak_threshold: f64,
    /// Performance degradation threshold (%)
    pub degradation_threshold: f64,
}

/// Spike test configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikeTestConfig {
    /// Base requests per second
    pub base_rps: f64,
    /// Spike requests per second
    pub spike_rps: f64,
    /// Spike duration in seconds
    pub spike_duration_seconds: u64,
    /// Recovery time in seconds
    pub recovery_time_seconds: u64,
}

/// Volume test configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VolumeTestConfig {
    /// Data volume in megabytes
    pub data_volume_mb: f64,
    /// Concurrent operations
    pub concurrent_operations: usize,
    /// Operation timeout in seconds
    pub operation_timeout_seconds: u64,
}
