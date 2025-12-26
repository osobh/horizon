use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ForecastPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lower_bound: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upper_bound: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ForecastResult {
    pub forecast_weeks: u8,
    pub points: Vec<ForecastPoint>,
    pub generated_at: DateTime<Utc>,
    pub model_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy_metrics: Option<AccuracyMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AccuracyMetrics {
    pub mape: f64,
    pub rmse: f64,
    pub mae: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct BacktestResult {
    pub train_size: usize,
    pub test_size: usize,
    pub metrics: AccuracyMetrics,
    pub predictions: Vec<ForecastPoint>,
    pub actuals: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecast_point_creation() {
        let point = ForecastPoint {
            timestamp: Utc::now(),
            value: 42.5,
            lower_bound: Some(40.0),
            upper_bound: Some(45.0),
        };
        assert_eq!(point.value, 42.5);
        assert_eq!(point.lower_bound, Some(40.0));
        assert_eq!(point.upper_bound, Some(45.0));
    }

    #[test]
    fn test_accuracy_metrics() {
        let metrics = AccuracyMetrics {
            mape: 0.12,
            rmse: 5.3,
            mae: 4.1,
        };
        assert_eq!(metrics.mape, 0.12);
        assert_eq!(metrics.rmse, 5.3);
        assert_eq!(metrics.mae, 4.1);
    }
}
