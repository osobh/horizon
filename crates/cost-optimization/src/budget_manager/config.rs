//! Budget manager configuration

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Budget manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetManagerConfig {
    pub default_budget_period: super::budget::BudgetPeriod,
    pub alert_check_interval: Duration,
    pub forecast_horizon_days: u32,
    pub auto_actions_enabled: bool,
    pub notification_retry_attempts: u32,
    pub chargeback_enabled: bool,
    pub cost_allocation_enabled: bool,
}

impl Default for BudgetManagerConfig {
    fn default() -> Self {
        Self {
            default_budget_period: super::budget::BudgetPeriod::Monthly,
            alert_check_interval: Duration::from_secs(3600),
            forecast_horizon_days: 30,
            auto_actions_enabled: false,
            notification_retry_attempts: 3,
            chargeback_enabled: true,
            cost_allocation_enabled: true,
        }
    }
}
