//! Anomaly detection for cost prediction

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Anomaly ID
    pub id: Uuid,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Actual value
    pub actual_value: f64,
    /// Expected value
    pub expected_value: f64,
    /// Deviation percentage
    pub deviation_percent: f64,
    /// Anomaly score (0-100)
    pub score: f64,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
}

/// Anomaly type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Spike in costs
    Spike,
    /// Drop in costs
    Drop,
    /// Gradual increase
    GradualIncrease,
    /// Unusual pattern
    UnusualPattern,
}
