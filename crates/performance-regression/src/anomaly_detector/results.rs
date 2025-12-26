//! Anomaly detection results
pub struct AnomalyResult {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub is_anomaly: bool,
    pub confidence_score: f64,
    pub deviation_score: f64,
    pub affected_metrics: Vec<String>,
    pub severity: String,
    pub anomaly_type: String,
    pub suggested_actions: Vec<String>,
}
