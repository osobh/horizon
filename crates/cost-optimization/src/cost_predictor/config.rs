//! Configuration for cost predictor

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Cost predictor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostPredictorConfig {
    /// Historical data retention period
    pub retention_period: Duration,
    /// Prediction confidence threshold
    pub confidence_threshold: f64,
    /// Anomaly detection sensitivity (0.5 - 1.0)
    pub anomaly_sensitivity: f64,
    /// Model update frequency
    pub model_update_frequency: Duration,
    /// Maximum prediction horizon
    pub max_prediction_horizon: Duration,
    /// Enable automatic model selection
    pub auto_model_selection: bool,
}

impl Default for CostPredictorConfig {
    fn default() -> Self {
        Self {
            retention_period: Duration::from_secs(86400 * 90), // 90 days
            confidence_threshold: 0.85,
            anomaly_sensitivity: 0.8,
            model_update_frequency: Duration::from_secs(3600), // 1 hour
            max_prediction_horizon: Duration::from_secs(86400 * 30), // 30 days
            auto_model_selection: true,
        }
    }
}
