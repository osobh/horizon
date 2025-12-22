//! Empirical Validation Framework for Darwin Gödel Machine
//!
//! This module implements empirical validation of agent modifications through
//! benchmark evaluation, replacing the impractical formal proof requirements
//! of the original Gödel Machine.

pub mod benchmark;
pub mod evaluator;
pub mod metrics;
pub mod types;

pub use benchmark::{BenchmarkSuite, TaskExecutor};
pub use evaluator::EmpiricalEvaluator;
pub use metrics::MetricsCalculator;
pub use types::{
    BaselineComparison, BenchmarkTask, ChangeType, CodeChange, CodeChangeStats, PerformanceMetrics,
    ResourceUsage, StatisticalSignificance, TaskCategory, TaskResult, ValidationConfig,
    ValidationReport, ValidationResult,
};

#[cfg(test)]
mod tests;
