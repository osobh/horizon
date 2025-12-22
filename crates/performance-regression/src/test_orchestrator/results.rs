//! Test execution results and reporting types
//!
//! This module contains all types related to test execution results,
//! test reports, insights, and metadata.

use crate::metrics_collector::{MetricDataPoint, MetricStatistics, MetricType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::config::TestStrategy;

/// Test execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionResult {
    /// Test ID
    pub test_id: String,
    /// Test strategy used
    pub strategy: TestStrategy,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: DateTime<Utc>,
    /// Test status
    pub status: TestStatus,
    /// Collected metrics
    pub metrics: HashMap<MetricType, Vec<MetricDataPoint>>,
    /// Aggregated statistics
    pub statistics: HashMap<MetricType, MetricStatistics>,
    /// Test insights
    pub insights: TestInsights,
    /// Error details if failed
    pub error_details: Option<String>,
}

/// Test execution status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestStatus {
    /// Test passed
    Passed,
    /// Test failed
    Failed,
    /// Test partially passed
    PartiallyPassed,
    /// Test was skipped
    Skipped,
    /// Test is running
    Running,
    /// Test is queued
    Queued,
}

/// Test execution insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestInsights {
    /// Performance bottlenecks detected
    pub bottlenecks: Vec<String>,
    /// Performance improvements detected
    pub improvements: Vec<String>,
    /// Warnings
    pub warnings: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Success rate percentage
    pub success_rate: f64,
    /// Average response time
    pub avg_response_time_ms: f64,
    /// Peak resource usage
    pub peak_resource_usage: ResourceUsage,
}

/// Resource usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Peak CPU usage percentage
    pub cpu_percent: f64,
    /// Peak memory usage in MB
    pub memory_mb: f64,
    /// Peak disk IOPS
    pub disk_iops: f64,
    /// Peak network bandwidth in Mbps
    pub network_mbps: f64,
}

/// Test report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    /// Report ID
    pub report_id: String,
    /// Generation time
    pub generated_at: DateTime<Utc>,
    /// Test results
    pub results: Vec<TestExecutionResult>,
    /// Overall summary
    pub summary: TestSummary,
    /// CI/CD metadata
    pub ci_metadata: Option<CiMetadata>,
}

/// Test summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    /// Total tests executed
    pub total_tests: usize,
    /// Passed tests
    pub passed_tests: usize,
    /// Failed tests
    pub failed_tests: usize,
    /// Overall success rate
    pub success_rate: f64,
    /// Average test duration
    pub avg_duration_seconds: f64,
    /// Total execution time
    pub total_duration_seconds: f64,
}

/// CI/CD metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiMetadata {
    /// Build ID
    pub build_id: String,
    /// Commit SHA
    pub commit_sha: String,
    /// Branch name
    pub branch: String,
    /// Pull request number
    pub pr_number: Option<u32>,
    /// Build URL
    pub build_url: Option<String>,
}

/// Scheduled test
#[derive(Debug, Clone)]
pub struct ScheduledTest {
    /// Test ID
    pub id: String,
    /// Test strategy
    pub strategy: TestStrategy,
    /// Scheduled time
    pub scheduled_time: DateTime<Utc>,
    /// Retry count
    pub retry_count: u32,
}
