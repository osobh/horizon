//! Budget definition and management

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Budget period
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetPeriod {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annual,
    Custom { days: u32 },
}

/// Budget scope
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetScope {
    Global,
    Department(String),
    Project(String),
    Team(String),
    Service(String),
    User(String),
    Custom(HashMap<String, String>),
}

/// Budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub id: String,
    pub name: String,
    pub amount: f64,
    pub period: BudgetPeriod,
    pub scope: BudgetScope,
    pub alert_thresholds: Vec<f64>,
    pub tags: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Budget status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetStatus {
    pub budget_id: String,
    pub current_spend: f64,
    pub allocated_budget: f64,
    pub remaining_budget: f64,
    pub utilization_percent: f64,
    pub projected_spend: f64,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub trend: super::types::SpendTrend,
    pub days_remaining: u32,
    pub is_over_budget: bool,
    pub last_updated: DateTime<Utc>,
}
