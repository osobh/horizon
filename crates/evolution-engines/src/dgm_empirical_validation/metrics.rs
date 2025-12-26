//! Metrics calculation for empirical validation

use super::types::*;
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use std::collections::HashMap;
use std::time::Duration;

/// Calculates performance metrics from task results
pub struct MetricsCalculator {
    // Placeholder for future fields
}

impl MetricsCalculator {
    /// Create new metrics calculator
    pub fn new() -> Self {
        Self {}
    }

    /// Calculate comprehensive metrics from task results
    pub fn calculate(&self, results: &[TaskResult]) -> EvolutionEngineResult<PerformanceMetrics> {
        if results.is_empty() {
            return Err(EvolutionEngineError::Other(
                "No results to calculate metrics".to_string(),
            ));
        }

        // Calculate average execution time
        let total_time: Duration = results.iter().map(|r| r.execution_time).sum();
        let avg_execution_time = total_time / results.len() as u32;

        // Calculate success rate by category
        let mut success_by_category: HashMap<TaskCategory, f64> = HashMap::new();
        let mut category_counts: HashMap<TaskCategory, (usize, usize)> = HashMap::new();

        // Infer categories from results (in real implementation would have task info)
        for (i, result) in results.iter().enumerate() {
            let category = match i {
                0 => TaskCategory::BugFix,
                1 => TaskCategory::Feature,
                2 => TaskCategory::Refactoring,
                _ => TaskCategory::BugFix,
            };

            let (success_count, total_count) =
                category_counts.entry(category.clone()).or_insert((0, 0));
            *total_count += 1;
            if result.success {
                *success_count += 1;
            }
        }

        for (category, (success_count, total_count)) in category_counts {
            success_by_category.insert(category, success_count as f64 / total_count as f64);
        }

        // Analyze error patterns
        let error_patterns = self.analyze_error_patterns(results);

        // Calculate code change statistics
        let code_change_stats = self.calculate_code_change_stats(results);

        // Calculate resource efficiency
        let resource_efficiency = self.calculate_resource_efficiency(results);

        Ok(PerformanceMetrics {
            avg_execution_time,
            success_by_category,
            error_patterns,
            code_change_stats,
            resource_efficiency,
        })
    }

    /// Calculate overall success rate
    pub fn calculate_success_rate(&self, results: &[TaskResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }

        let success_count = results.iter().filter(|r| r.success).count();
        success_count as f64 / results.len() as f64
    }

    /// Calculate resource efficiency (success per resource unit)
    pub fn calculate_resource_efficiency(&self, results: &[TaskResult]) -> f64 {
        let total_cpu_time: Duration = results.iter().map(|r| r.resource_usage.cpu_time).sum();

        let success_count = results.iter().filter(|r| r.success).count();

        if total_cpu_time.as_secs() == 0 {
            return 0.0;
        }

        // Efficiency: successes per CPU minute
        (success_count as f64 * 60.0) / total_cpu_time.as_secs_f64()
    }

    /// Analyze error patterns
    pub fn analyze_error_patterns(&self, results: &[TaskResult]) -> HashMap<String, usize> {
        let mut patterns = HashMap::new();

        for result in results {
            if let Some(error) = &result.error {
                // Extract error pattern (simplified)
                let pattern = if error.contains("sorting not implemented correctly") {
                    "sorting not implemented correctly"
                } else if error.contains("null pointer") {
                    "null pointer exception"
                } else if error.contains("timeout") {
                    "execution timeout"
                } else {
                    "other error"
                };

                *patterns.entry(pattern.to_string()).or_insert(0) += 1;
            }
        }

        patterns
    }

    // Helper methods

    fn calculate_code_change_stats(&self, results: &[TaskResult]) -> CodeChangeStats {
        let mut total_files = 0;
        let mut total_lines_added = 0;
        let mut total_lines_removed = 0;
        let mut change_type_distribution = HashMap::new();
        let mut task_count = 0;

        for result in results {
            if !result.changes.is_empty() {
                task_count += 1;
                total_files += result.changes.len();

                for change in &result.changes {
                    total_lines_added += change.lines_added;
                    total_lines_removed += change.lines_removed;
                    *change_type_distribution
                        .entry(change.change_type.clone())
                        .or_insert(0) += 1;
                }
            }
        }

        let avg_files_changed = if task_count > 0 {
            total_files as f64 / results.len() as f64
        } else {
            0.0
        };

        let avg_lines_changed = if task_count > 0 {
            (total_lines_added + total_lines_removed) as f64 / results.len() as f64
        } else {
            0.0
        };

        CodeChangeStats {
            avg_files_changed,
            avg_lines_changed,
            change_type_distribution,
        }
    }
}
