//! Type definitions for the cost predictor module
//!
//! This module contains all data structures, enums, and configuration types
//! used throughout the cost prediction system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use uuid::Uuid;

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

/// Budget forecast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetForecast {
    /// Forecast ID
    pub id: Uuid,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Time period
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    /// Predicted total cost
    pub predicted_cost: f64,
    /// Cost breakdown by category
    pub cost_breakdown: HashMap<CostMetricType, f64>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    /// Recommendations
    pub recommendations: Vec<CostRecommendation>,
}

/// Risk assessment for budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk level
    pub risk_level: RiskLevel,
    /// Probability of budget overrun
    pub overrun_probability: f64,
    /// Expected variance
    pub expected_variance: f64,
    /// Risk factors
    pub risk_factors: Vec<RiskFactor>,
}

/// Risk level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Critical risk
    Critical,
}

/// Risk factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Factor name
    pub name: String,
    /// Impact score (0-100)
    pub impact: f64,
    /// Likelihood (0-1)
    pub likelihood: f64,
    /// Mitigation strategy
    pub mitigation: String,
}

/// Cost optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecommendation {
    /// Recommendation ID
    pub id: Uuid,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Potential savings
    pub potential_savings: f64,
    /// Implementation complexity
    pub complexity: ComplexityLevel,
    /// Priority
    pub priority: Priority,
}

/// Complexity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityLevel {
    /// Low complexity
    Low,
    /// Medium complexity
    Medium,
    /// High complexity
    High,
}

/// Priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority
    Low = 0,
    /// Medium priority
    Medium = 1,
    /// High priority
    High = 2,
    /// Critical priority
    Critical = 3,
}

/// Cost predictor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostPredictorConfig {
    /// Historical data retention period
    pub data_retention: Duration,
    /// Anomaly detection sensitivity (0-1)
    pub anomaly_sensitivity: f64,
    /// Minimum data points for prediction
    pub min_data_points: usize,
    /// Default prediction model
    pub default_model: PredictionModel,
    /// Enable real-time predictions
    pub enable_realtime: bool,
    /// Model update interval
    pub model_update_interval: Duration,
}

impl Default for CostPredictorConfig {
    fn default() -> Self {
        Self {
            data_retention: Duration::from_secs(86400 * 90), // 90 days
            anomaly_sensitivity: 0.8,
            min_data_points: 30,
            default_model: PredictionModel::Ensemble,
            enable_realtime: true,
            model_update_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Trained model state
#[derive(Debug, Clone)]
pub struct TrainedModel {
    /// Model type
    pub model_type: PredictionModel,
    /// Training timestamp
    pub trained_at: DateTime<Utc>,
    /// Model parameters
    pub parameters: ModelParameters,
    /// Accuracy metrics
    pub accuracy: ModelAccuracy,
}

/// Model parameters
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// Moving average window
    pub ma_window: Option<usize>,
    /// Smoothing factor
    pub smoothing_factor: Option<f64>,
    /// Regression coefficients
    pub coefficients: Option<Vec<f64>>,
    /// ARIMA parameters
    pub arima_params: Option<(usize, usize, usize)>,
}

/// Predictor metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictorMetrics {
    /// Total predictions made
    pub total_predictions: u64,
    /// Average prediction accuracy
    pub avg_accuracy: f64,
    /// Anomalies detected
    pub anomalies_detected: u64,
    /// Models trained
    pub models_trained: u64,
    /// Last model update
    pub last_model_update: Option<DateTime<Utc>>,
}