//! Budget metrics and analytics

use serde::{Deserialize, Serialize};

/// Budget manager metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetMetrics {
    pub total_budgets: u64,
    pub active_budgets: u64,
    pub over_budget_count: u64,
    pub total_spend: f64,
    pub total_allocated: f64,
    pub alerts_triggered: u64,
    pub auto_actions_executed: u64,
    pub chargeback_reports_generated: u64,
}

impl BudgetMetrics {
    /// Calculate overall budget utilization
    pub fn utilization_rate(&self) -> f64 {
        if self.total_allocated > 0.0 {
            self.total_spend / self.total_allocated
        } else {
            0.0
        }
    }

    /// Get budget health score
    pub fn health_score(&self) -> f64 {
        let utilization = self.utilization_rate();
        let over_budget_rate = if self.active_budgets > 0 {
            self.over_budget_count as f64 / self.active_budgets as f64
        } else {
            0.0
        };
        
        // Simple health score calculation
        let score = 100.0 * (1.0 - over_budget_rate) * (1.0 - (utilization - 0.8).abs().min(0.2) * 5.0);
        score.max(0.0).min(100.0)
    }
}
