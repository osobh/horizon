//! Type definitions for empirical validation system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Configuration for empirical validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Benchmark suite to use (e.g., "SWE-bench", "Polyglot")
    pub benchmark_suite: String,
    /// Number of tasks to evaluate
    pub num_tasks: usize,
    /// Timeout per task
    pub task_timeout: Duration,
    /// Minimum success rate to consider improvement
    pub min_success_rate: f64,
    /// Whether to use staged evaluation
    pub staged_evaluation: bool,
    /// Confidence threshold for statistical significance
    pub confidence_threshold: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            benchmark_suite: "SWE-bench".to_string(),
            num_tasks: 10,
            task_timeout: Duration::from_secs(300),
            min_success_rate: 0.4,
            staged_evaluation: true,
            confidence_threshold: 0.95,
        }
    }
}

/// A benchmark task to evaluate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTask {
    /// Unique task ID
    pub id: String,
    /// Repository URL or path
    pub repository: String,
    /// Task description or issue
    pub description: String,
    /// Programming language
    pub language: String,
    /// Test command to verify solution
    pub test_command: String,
    /// Expected difficulty (0.0 to 1.0)
    pub difficulty: f64,
    /// Task category
    pub category: TaskCategory,
}

/// Categories of benchmark tasks
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskCategory {
    /// Bug fixing tasks
    BugFix,
    /// Feature implementation
    Feature,
    /// Refactoring tasks
    Refactoring,
    /// Documentation tasks
    Documentation,
    /// Performance optimization
    Performance,
}

/// Result of a single task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task ID
    pub task_id: String,
    /// Whether the task was completed successfully
    pub success: bool,
    /// Execution time
    pub execution_time: Duration,
    /// Error message if failed
    pub error: Option<String>,
    /// Code changes made
    pub changes: Vec<CodeChange>,
    /// Resources used
    pub resource_usage: ResourceUsage,
}

/// A code change made during task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChange {
    /// File path
    pub file: String,
    /// Type of change
    pub change_type: ChangeType,
    /// Number of lines added
    pub lines_added: usize,
    /// Number of lines removed
    pub lines_removed: usize,
}

/// Types of code changes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeType {
    /// File created
    Created,
    /// File modified
    Modified,
    /// File deleted
    Deleted,
}

/// Resource usage during task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU time used
    pub cpu_time: Duration,
    /// Peak memory usage in bytes
    pub peak_memory: u64,
    /// Number of tool invocations
    pub tool_invocations: HashMap<String, usize>,
}

/// Result of empirical validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Agent ID being validated
    pub agent_id: String,
    /// Overall success rate
    pub success_rate: f64,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Individual task results
    pub task_results: Vec<TaskResult>,
    /// Statistical significance
    pub statistical_significance: StatisticalSignificance,
    /// Comparison with baseline
    pub baseline_comparison: Option<BaselineComparison>,
}

/// Performance metrics from validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average execution time per task
    pub avg_execution_time: Duration,
    /// Success rate by category
    pub success_by_category: HashMap<TaskCategory, f64>,
    /// Error patterns
    pub error_patterns: HashMap<String, usize>,
    /// Code change statistics
    pub code_change_stats: CodeChangeStats,
    /// Resource efficiency
    pub resource_efficiency: f64,
}

/// Statistics about code changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChangeStats {
    /// Average files changed per task
    pub avg_files_changed: f64,
    /// Average lines changed per task
    pub avg_lines_changed: f64,
    /// Change type distribution
    pub change_type_distribution: HashMap<ChangeType, usize>,
}

/// Statistical significance of results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSignificance {
    /// P-value from statistical test
    pub p_value: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Effect size
    pub effect_size: f64,
    /// Whether results are statistically significant
    pub is_significant: bool,
}

/// Comparison with baseline performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    /// Baseline success rate
    pub baseline_success_rate: f64,
    /// Improvement over baseline
    pub improvement: f64,
    /// Relative improvement percentage
    pub relative_improvement: f64,
}

/// Comprehensive validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Validation configuration used
    pub config: ValidationConfig,
    /// Validation results
    pub results: ValidationResult,
    /// Timestamp of validation
    pub timestamp: u64,
    /// Recommendations based on results
    pub recommendations: Vec<String>,
    /// Whether the agent shows improvement
    pub shows_improvement: bool,
}
