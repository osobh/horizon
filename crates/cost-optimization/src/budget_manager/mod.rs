//! Budget management module for cost tracking and control

mod alerts;
mod allocation;
mod budget;
mod chargeback;
mod config;
mod manager;
mod metrics;
mod types;

// Re-export commonly used types
pub use alerts::{AlertSeverity, AlertThreshold, AutoAction, BudgetAlert, NotificationChannel};
pub use allocation::{AllocationMethod, CostAllocation};
pub use budget::{Budget, BudgetPeriod, BudgetScope, BudgetStatus};
pub use chargeback::{Adjustment, AdjustmentType, ChargebackEntry, ChargebackReport, UsageDetail};
pub use config::BudgetManagerConfig;
pub use manager::BudgetManager;
pub use metrics::BudgetMetrics;
pub use types::{ActionType, SpendTrend};