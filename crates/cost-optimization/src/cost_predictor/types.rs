//! Core types for cost prediction

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::anomaly::Anomaly;
use super::models::{ModelAccuracy, PredictionModel};

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Value
    pub value: f64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Cost metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CostMetricType {
    /// Total cost
    TotalCost,
    /// Compute cost
    ComputeCost,
    /// Storage cost
    StorageCost,
    /// Network cost
    NetworkCost,
    /// GPU cost
    GpuCost,
    /// Other costs
    OtherCost,
}

impl std::fmt::Display for CostMetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CostMetricType::TotalCost => write!(f, "Total Cost"),
            CostMetricType::ComputeCost => write!(f, "Compute Cost"),
            CostMetricType::StorageCost => write!(f, "Storage Cost"),
            CostMetricType::NetworkCost => write!(f, "Network Cost"),
            CostMetricType::GpuCost => write!(f, "GPU Cost"),
            CostMetricType::OtherCost => write!(f, "Other Cost"),
        }
    }
}

/// Seasonality type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Seasonality {
    /// No seasonality
    None,
    /// Daily patterns
    Daily,
    /// Weekly patterns
    Weekly,
    /// Monthly patterns
    Monthly,
    /// Quarterly patterns
    Quarterly,
}

/// Cost prediction request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRequest {
    /// Metric type to predict
    pub metric_type: CostMetricType,
    /// Prediction horizon
    pub horizon: Duration,
    /// Prediction model
    pub model: PredictionModel,
    /// Confidence level (0.0 - 1.0)
    pub confidence_level: f64,
    /// Include seasonality
    pub seasonality: Seasonality,
    /// Additional filters
    pub filters: HashMap<String, String>,
}

/// Cost prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Request details
    pub request: PredictionRequest,
    /// Predicted values
    pub predictions: Vec<PredictedValue>,
    /// Model accuracy metrics
    pub accuracy: ModelAccuracy,
    /// Anomalies detected
    pub anomalies: Vec<Anomaly>,
    /// Confidence intervals
    pub confidence_intervals: Vec<ConfidenceInterval>,
    /// Trend analysis
    pub trend: TrendAnalysis,
}

/// Predicted value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedValue {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Predicted value
    pub value: f64,
    /// Prediction confidence
    pub confidence: f64,
}

/// Confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Lower bound
    pub lower_bound: f64,
    /// Upper bound
    pub upper_bound: f64,
    /// Confidence level
    pub confidence_level: f64,
}

/// Trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength (0-1)
    pub strength: f64,
    /// Growth rate (percentage)
    pub growth_rate: f64,
    /// Seasonal patterns detected
    pub seasonal_patterns: Vec<SeasonalPattern>,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable/flat trend
    Stable,
    /// Volatile/unclear trend
    Volatile,
}

/// Seasonal pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    /// Pattern type
    pub pattern_type: Seasonality,
    /// Pattern strength
    pub strength: f64,
    /// Peak periods
    pub peak_periods: Vec<String>,
    /// Low periods
    pub low_periods: Vec<String>,
}
