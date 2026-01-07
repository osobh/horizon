//! Main budget manager implementation

use crate::error::{CostOptimizationError, CostOptimizationResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{error, info, warn};
use uuid::Uuid;

use super::alerts::{AlertSeverity, BudgetAlert};
use super::budget::{Budget, BudgetStatus};
use super::chargeback::ChargebackReport;
use super::config::BudgetManagerConfig;
use super::metrics::BudgetMetrics;

/// Budget manager for tracking and controlling costs
pub struct BudgetManager {
    config: Arc<BudgetManagerConfig>,
    budgets: Arc<DashMap<String, Budget>>,
    budget_status: Arc<DashMap<String, BudgetStatus>>,
    alerts: Arc<RwLock<Vec<BudgetAlert>>>,
    metrics: Arc<RwLock<BudgetMetrics>>,
}

impl BudgetManager {
    /// Create new budget manager
    pub fn new(config: BudgetManagerConfig) -> CostOptimizationResult<Self> {
        Ok(Self {
            config: Arc::new(config),
            budgets: Arc::new(DashMap::new()),
            budget_status: Arc::new(DashMap::new()),
            alerts: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(BudgetMetrics::default())),
        })
    }

    /// Create a new budget
    pub fn create_budget(&self, budget: Budget) -> CostOptimizationResult<String> {
        let budget_id = budget.id.clone();

        // Initialize budget status
        let status = BudgetStatus {
            budget_id: budget_id.clone(),
            current_spend: 0.0,
            allocated_budget: budget.amount,
            remaining_budget: budget.amount,
            utilization_percent: 0.0,
            projected_spend: 0.0,
            period_start: Utc::now(),
            period_end: Utc::now(),
            trend: super::types::SpendTrend::Stable,
            days_remaining: 30,
            is_over_budget: false,
            last_updated: Utc::now(),
        };

        let amount = budget.amount;
        self.budgets.insert(budget_id.clone(), budget);
        self.budget_status.insert(budget_id.clone(), status);

        let mut metrics = self.metrics.write();
        metrics.total_budgets += 1;
        metrics.active_budgets += 1;
        metrics.total_allocated += amount;

        info!("Created budget: {}", budget_id);
        Ok(budget_id)
    }

    /// Update budget spend
    pub fn update_spend(&self, budget_id: &str, amount: f64) -> CostOptimizationResult<()> {
        let mut status = self.budget_status.get_mut(budget_id).ok_or_else(|| {
            CostOptimizationError::BudgetNotFound {
                budget_id: budget_id.to_string(),
            }
        })?;

        status.current_spend += amount;
        status.remaining_budget = status.allocated_budget - status.current_spend;
        status.utilization_percent = (status.current_spend / status.allocated_budget) * 100.0;
        status.is_over_budget = status.current_spend > status.allocated_budget;
        status.last_updated = Utc::now();

        // Check for alerts
        if let Some(budget) = self.budgets.get(budget_id) {
            for threshold in &budget.alert_thresholds {
                if status.utilization_percent >= *threshold {
                    self.trigger_alert(budget_id, *threshold, &status)?;
                }
            }
        }

        let mut metrics = self.metrics.write();
        metrics.total_spend += amount;
        if status.is_over_budget {
            metrics.over_budget_count += 1;
        }

        Ok(())
    }

    /// Get budget status
    pub fn get_budget_status(&self, budget_id: &str) -> CostOptimizationResult<BudgetStatus> {
        self.budget_status
            .get(budget_id)
            .map(|s| s.clone())
            .ok_or_else(|| CostOptimizationError::BudgetNotFound {
                budget_id: budget_id.to_string(),
            })
    }

    /// Generate chargeback report
    pub async fn generate_chargeback_report(
        &self,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> CostOptimizationResult<ChargebackReport> {
        info!(
            "Generating chargeback report for {} to {}",
            period_start, period_end
        );

        let report = ChargebackReport {
            id: Uuid::new_v4(),
            period_start,
            period_end,
            entries: vec![],
            total_amount: 0.0,
            adjustments: vec![],
            generated_at: Utc::now(),
            approved: false,
            approved_by: None,
        };

        self.metrics.write().chargeback_reports_generated += 1;

        Ok(report)
    }

    /// Get metrics
    pub fn get_metrics(&self) -> BudgetMetrics {
        self.metrics.read().clone()
    }

    // Helper methods

    fn trigger_alert(
        &self,
        budget_id: &str,
        threshold: f64,
        status: &BudgetStatus,
    ) -> CostOptimizationResult<()> {
        let alert = BudgetAlert {
            id: Uuid::new_v4(),
            budget_id: budget_id.to_string(),
            timestamp: Utc::now(),
            severity: if threshold >= 100.0 {
                AlertSeverity::Emergency
            } else if threshold >= 90.0 {
                AlertSeverity::Critical
            } else if threshold >= 80.0 {
                AlertSeverity::Warning
            } else {
                AlertSeverity::Info
            },
            threshold_triggered: threshold,
            current_spend: status.current_spend,
            budget_amount: status.allocated_budget,
            message: format!(
                "Budget {} has reached {}% utilization",
                budget_id, threshold
            ),
            recommendations: vec![],
            notification_sent: false,
            actions_taken: vec![],
            acknowledged: false,
            acknowledged_by: None,
            acknowledged_at: None,
        };

        self.alerts.write().push(alert);
        self.metrics.write().alerts_triggered += 1;

        warn!(
            "Budget alert triggered for {}: {}% threshold",
            budget_id, threshold
        );
        Ok(())
    }
}
