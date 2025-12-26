//! Cost allocation and tracking

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cost allocation method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationMethod {
    Direct,
    Proportional,
    ActivityBased,
    TagBased,
    Custom,
}

/// Cost allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAllocation {
    pub id: String,
    pub source_budget_id: String,
    pub target_budget_id: String,
    pub amount: f64,
    pub percentage: Option<f64>,
    pub allocation_method: AllocationMethod,
    pub tags: HashMap<String, String>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub description: String,
}
