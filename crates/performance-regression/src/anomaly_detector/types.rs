//! Core types for anomaly detection
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub value: f64,
    pub metadata: std::collections::HashMap<String, String>,
}
