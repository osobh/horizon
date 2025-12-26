//! Budget alerts and notifications

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThreshold {
    pub percentage: f64,
    pub severity: AlertSeverity,
    pub notification_channels: Vec<NotificationChannel>,
    pub auto_actions: Vec<AutoAction>,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email(String),
    Slack(String),
    PagerDuty(String),
    Webhook(String),
    SMS(String),
}

/// Automated actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoAction {
    pub action_type: super::types::ActionType,
    pub parameters: std::collections::HashMap<String, String>,
    pub delay_minutes: u32,
}

/// Budget alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAlert {
    pub id: Uuid,
    pub budget_id: String,
    pub timestamp: DateTime<Utc>,
    pub severity: AlertSeverity,
    pub threshold_triggered: f64,
    pub current_spend: f64,
    pub budget_amount: f64,
    pub message: String,
    pub recommendations: Vec<String>,
    pub notification_sent: bool,
    pub actions_taken: Vec<String>,
    pub acknowledged: bool,
    pub acknowledged_by: Option<String>,
    pub acknowledged_at: Option<DateTime<Utc>>,
}
