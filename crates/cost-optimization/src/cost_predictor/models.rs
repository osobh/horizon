//! Prediction models and accuracy metrics

use serde::{Deserialize, Serialize};

/// Prediction model type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PredictionModel {
    /// Simple moving average
    MovingAverage,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// Linear regression
    LinearRegression,
    /// ARIMA model
    Arima,
    /// Machine learning ensemble
    Ensemble,
}

impl std::fmt::Display for PredictionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PredictionModel::MovingAverage => write!(f, "Moving Average"),
            PredictionModel::ExponentialSmoothing => write!(f, "Exponential Smoothing"),
            PredictionModel::LinearRegression => write!(f, "Linear Regression"),
            PredictionModel::Arima => write!(f, "ARIMA"),
            PredictionModel::Ensemble => write!(f, "Ensemble"),
        }
    }
}

/// Model accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAccuracy {
    /// Mean absolute error
    pub mae: f64,
    /// Mean squared error
    pub mse: f64,
    /// Root mean squared error
    pub rmse: f64,
    /// Mean absolute percentage error
    pub mape: f64,
    /// R-squared value
    pub r_squared: f64,
}

/// Trained model information
#[derive(Debug, Clone)]
pub struct TrainedModel {
    /// Model type
    pub model_type: PredictionModel,
    /// Model accuracy
    pub accuracy: ModelAccuracy,
    /// Training timestamp
    pub trained_at: chrono::DateTime<chrono::Utc>,
    /// Model parameters
    pub parameters: std::collections::HashMap<String, f64>,
}
