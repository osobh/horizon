//! Chargeback reporting and adjustments

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Chargeback report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChargebackReport {
    pub id: Uuid,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub entries: Vec<ChargebackEntry>,
    pub total_amount: f64,
    pub adjustments: Vec<Adjustment>,
    pub generated_at: DateTime<Utc>,
    pub approved: bool,
    pub approved_by: Option<String>,
}

/// Chargeback entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChargebackEntry {
    pub department: String,
    pub project: Option<String>,
    pub user: Option<String>,
    pub resource_type: String,
    pub usage_details: Vec<UsageDetail>,
    pub total_cost: f64,
    pub allocated_cost: f64,
    pub tags: HashMap<String, String>,
}

/// Usage detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageDetail {
    pub resource_id: String,
    pub resource_name: String,
    pub quantity: f64,
    pub unit: String,
    pub unit_cost: f64,
    pub total_cost: f64,
    pub usage_hours: f64,
}

/// Cost adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adjustment {
    pub id: Uuid,
    pub adjustment_type: AdjustmentType,
    pub amount: f64,
    pub reason: String,
    pub applied_to: String,
    pub created_at: DateTime<Utc>,
}

/// Adjustment type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdjustmentType {
    Credit,
    Debit,
    Discount,
    Tax,
    Refund,
}
