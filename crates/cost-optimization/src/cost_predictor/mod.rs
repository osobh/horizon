//! Cost prediction module for ML-based forecasting and trend analysis
//!
//! This module provides advanced cost prediction capabilities including:
//! - Time-series based cost forecasting
//! - ML models for cost trend analysis
//! - Anomaly detection in spending patterns
//! - Budget prediction and capacity planning

// Module declarations
mod anomaly;
mod budget;
mod config;
mod metrics;
mod models;
mod predictor;
mod recommendations;
mod risk;
mod types;

// Re-export commonly used types
pub use anomaly::{Anomaly, AnomalyType};
pub use budget::{BudgetForecast, RiskAssessment};
pub use config::CostPredictorConfig;
pub use metrics::PredictorMetrics;
pub use models::{ModelAccuracy, PredictionModel, TrainedModel};
pub use predictor::CostPredictor;
pub use recommendations::{ComplexityLevel, CostRecommendation, Priority};
pub use risk::{RiskFactor, RiskLevel};
pub use types::{
    ConfidenceInterval, CostMetricType, PredictedValue, PredictionRequest, PredictionResult,
    SeasonalPattern, Seasonality, TimeSeriesPoint, TrendAnalysis, TrendDirection,
};
