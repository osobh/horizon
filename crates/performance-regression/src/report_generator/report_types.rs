//! Report-related types and structures

use super::types::ReportOutputFormat;
use crate::metrics_collector::MetricType;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Report identifier
    pub id: String,
    /// Report title
    pub title: String,
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Time range covered
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
    /// Report format
    pub format: ReportOutputFormat,
    /// Report sections
    pub sections: Vec<ReportSection>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Report section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    /// Section title
    pub title: String,
    /// Section content type
    pub content_type: SectionContentType,
    /// Section data
    pub data: serde_json::Value,
}

/// Section content type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SectionContentType {
    /// Summary statistics
    Summary,
    /// Chart visualization
    Chart,
    /// Data table
    Table,
    /// Text content
    Text,
    /// Metric comparison
    Comparison,
}

/// Trend report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendReport {
    /// Report identifier
    pub id: String,
    /// Analysis period
    pub period: Duration,
    /// Detected trends
    pub trends: Vec<MetricTrend>,
    /// Predictions
    pub predictions: Vec<MetricPrediction>,
    /// Anomalies detected
    pub anomalies: Vec<TrendAnomaly>,
}

/// Metric trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTrend {
    /// Metric type
    pub metric_type: MetricType,
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Trend duration
    pub duration: Duration,
    /// Statistical confidence
    pub confidence: f64,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable/no trend
    Stable,
    /// Cyclic pattern
    Cyclic,
}

/// Metric prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPrediction {
    /// Metric type
    pub metric_type: MetricType,
    /// Prediction timestamp
    pub timestamp: DateTime<Utc>,
    /// Predicted value
    pub predicted_value: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Prediction method
    pub method: String,
}

/// Trend anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnomaly {
    /// Metric type
    pub metric_type: MetricType,
    /// Anomaly timestamp
    pub timestamp: DateTime<Utc>,
    /// Actual value
    pub actual_value: f64,
    /// Expected value
    pub expected_value: f64,
    /// Anomaly score
    pub score: f64,
}

/// Regression summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionSummary {
    /// Summary identifier
    pub id: String,
    /// Analysis period
    pub period: (DateTime<Utc>, DateTime<Utc>),
    /// Total regressions detected
    pub total_regressions: usize,
    /// Regressions by metric
    pub regressions_by_metric: HashMap<MetricType, Vec<RegressionDetail>>,
    /// Overall health score
    pub health_score: f64,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Regression detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetail {
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Baseline value
    pub baseline_value: f64,
    /// Current value
    pub current_value: f64,
    /// Degradation percentage
    pub degradation_percent: f64,
    /// Impact severity
    pub severity: RegressionSeverity,
}

/// Regression severity
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// Minor regression
    Low,
    /// Moderate regression
    Medium,
    /// Severe regression
    High,
    /// Critical regression
    Critical,
}
