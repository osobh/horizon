//! Main empirical evaluation logic

use super::benchmark::{BenchmarkSuite, TaskExecutor};
use super::metrics::MetricsCalculator;
use super::types::*;
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use std::time::{SystemTime, UNIX_EPOCH};

/// Performs empirical validation of agent modifications
pub struct EmpiricalEvaluator {
    config: ValidationConfig,
    executor: TaskExecutor,
    calculator: MetricsCalculator,
}

impl EmpiricalEvaluator {
    /// Create new empirical evaluator
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            executor: TaskExecutor::new(),
            calculator: MetricsCalculator::new(),
        }
    }

    /// Evaluate an agent on benchmark tasks
    pub fn evaluate(
        &self,
        agent_id: &str,
        agent_code: &str,
        tasks: &[&BenchmarkTask],
    ) -> EvolutionEngineResult<ValidationResult> {
        let mut task_results = Vec::new();
        let mut completed_tasks = 0;

        for task in tasks {
            // Check if we should continue with staged evaluation
            if self.config.staged_evaluation && completed_tasks > 0 {
                let current_success_rate = self.calculator.calculate_success_rate(&task_results);
                if !self.should_continue_evaluation(current_success_rate, completed_tasks) {
                    break;
                }
            }

            // Execute task
            let result = self
                .executor
                .execute(task, agent_code, self.config.task_timeout)?;
            task_results.push(result);
            completed_tasks += 1;
        }

        // Calculate metrics
        let metrics = self.calculator.calculate(&task_results)?;
        let success_rate = self.calculator.calculate_success_rate(&task_results);

        // Calculate statistical significance
        let statistical_significance = self.calculate_statistical_significance(
            &task_results,
            0.5, // Baseline success rate
        )?;

        // Compare with baseline if available
        let baseline_comparison = if success_rate > 0.5 {
            Some(self.compare_with_baseline(success_rate, 0.5))
        } else {
            None
        };

        Ok(ValidationResult {
            agent_id: agent_id.to_string(),
            success_rate,
            metrics,
            task_results,
            statistical_significance,
            baseline_comparison,
        })
    }

    /// Check if evaluation should continue (for staged evaluation)
    pub fn should_continue_evaluation(
        &self,
        current_success_rate: f64,
        completed_tasks: usize,
    ) -> bool {
        // Early stopping if performance is too low
        if completed_tasks >= 5 && current_success_rate < self.config.min_success_rate * 0.5 {
            return false;
        }

        // Continue if above minimum threshold
        current_success_rate >= self.config.min_success_rate
    }

    /// Calculate statistical significance of results
    pub fn calculate_statistical_significance(
        &self,
        results: &[TaskResult],
        baseline_success_rate: f64,
    ) -> EvolutionEngineResult<StatisticalSignificance> {
        let success_count = results.iter().filter(|r| r.success).count();
        let total_count = results.len();
        let observed_rate = success_count as f64 / total_count as f64;

        // Simple binomial test (in real implementation would use proper statistics)
        let variance = baseline_success_rate * (1.0 - baseline_success_rate) / total_count as f64;
        let std_dev = variance.sqrt();
        let z_score = (observed_rate - baseline_success_rate) / std_dev;

        // Convert z-score to p-value (simplified)
        let p_value = if z_score.abs() > 2.0 {
            0.05 / (z_score.abs() - 1.0)
        } else {
            0.5 - 0.2 * z_score.abs()
        };

        // Calculate confidence interval
        let margin = 1.96 * std_dev; // 95% confidence
        let confidence_interval = (
            (observed_rate - margin).max(0.0),
            (observed_rate + margin).min(1.0),
        );

        // Calculate effect size (Cohen's d)
        let effect_size = (observed_rate - baseline_success_rate) / std_dev;

        Ok(StatisticalSignificance {
            p_value,
            confidence_interval,
            effect_size,
            is_significant: p_value < 0.05,
        })
    }

    /// Compare performance with baseline
    pub fn compare_with_baseline(
        &self,
        current_rate: f64,
        baseline_rate: f64,
    ) -> BaselineComparison {
        let improvement = current_rate - baseline_rate;
        let relative_improvement = if baseline_rate > 0.0 {
            (improvement / baseline_rate) * 100.0
        } else {
            0.0
        };

        BaselineComparison {
            baseline_success_rate: baseline_rate,
            improvement,
            relative_improvement,
        }
    }

    /// Generate comprehensive validation report
    pub fn generate_report(
        &self,
        results: &ValidationResult,
    ) -> EvolutionEngineResult<ValidationReport> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| EvolutionEngineError::Other(format!("Time error: {}", e)))?
            .as_secs();

        // Generate recommendations based on results
        let mut recommendations = Vec::new();

        if results.success_rate < 0.5 {
            recommendations
                .push("Consider improving error handling and recovery mechanisms".to_string());
            recommendations
                .push("Focus on improving performance on failed task categories".to_string());
        }

        if results.metrics.resource_efficiency < 0.5 {
            recommendations.push("Optimize resource usage to improve efficiency".to_string());
        }

        // Check for specific error patterns
        for (pattern, count) in &results.metrics.error_patterns {
            if *count > 2 {
                recommendations.push(format!("Address recurring error pattern: {}", pattern));
            }
        }

        // Determine if agent shows improvement
        let shows_improvement = results
            .baseline_comparison
            .as_ref()
            .map(|c| c.improvement > 0.0 && results.statistical_significance.is_significant)
            .unwrap_or(false);

        if shows_improvement {
            recommendations.push(
                "Agent shows statistically significant improvement - consider deployment"
                    .to_string(),
            );
        }

        Ok(ValidationReport {
            config: self.config.clone(),
            results: results.clone(),
            timestamp,
            recommendations,
            shows_improvement,
        })
    }
}
