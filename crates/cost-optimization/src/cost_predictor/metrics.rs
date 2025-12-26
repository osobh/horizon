//! Metrics for cost predictor

use serde::{Deserialize, Serialize};

/// Predictor metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictorMetrics {
    /// Total predictions made
    pub predictions_made: u64,
    /// Successful predictions
    pub successful_predictions: u64,
    /// Failed predictions
    pub failed_predictions: u64,
    /// Models trained
    pub models_trained: u64,
    /// Anomalies detected
    pub anomalies_detected: u64,
    /// Budget forecasts generated
    pub forecasts_generated: u64,
}

impl PredictorMetrics {
    /// Get prediction success rate
    pub fn success_rate(&self) -> f64 {
        if self.predictions_made == 0 {
            0.0
        } else {
            self.successful_predictions as f64 / self.predictions_made as f64
        }
    }
}
