use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Trend direction
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Trend analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub direction: TrendDirection,
    pub growth_rate: Decimal, // Percentage: positive = increasing, negative = decreasing
    pub daily_average: Decimal,
    pub slope: f64, // Linear regression slope
    pub confidence: f64, // 0.0 to 1.0
}

impl TrendAnalysis {
    pub fn new(
        direction: TrendDirection,
        growth_rate: Decimal,
        daily_average: Decimal,
        slope: f64,
    ) -> Self {
        Self {
            direction,
            growth_rate,
            daily_average,
            slope,
            confidence: 0.0,
        }
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Forecast point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastPoint {
    pub date: DateTime<Utc>,
    pub forecasted_cost: Decimal,
    pub confidence: f64, // 0.0 to 1.0
}

impl ForecastPoint {
    pub fn new(date: DateTime<Utc>, forecasted_cost: Decimal, confidence: f64) -> Self {
        Self {
            date,
            forecasted_cost,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Cost forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostForecast {
    pub historical_start: DateTime<Utc>,
    pub historical_end: DateTime<Utc>,
    pub forecast_start: DateTime<Utc>,
    pub forecast_end: DateTime<Utc>,
    pub method: String, // e.g., "linear_regression"
    pub points: Vec<ForecastPoint>,
    pub avg_confidence: f64,
}

impl CostForecast {
    pub fn new(
        historical_start: DateTime<Utc>,
        historical_end: DateTime<Utc>,
        forecast_start: DateTime<Utc>,
        forecast_end: DateTime<Utc>,
        method: String,
        points: Vec<ForecastPoint>,
    ) -> Self {
        let avg_confidence = if points.is_empty() {
            0.0
        } else {
            points.iter().map(|p| p.confidence).sum::<f64>() / points.len() as f64
        };

        Self {
            historical_start,
            historical_end,
            forecast_start,
            forecast_end,
            method,
            points,
            avg_confidence,
        }
    }
}

/// Daily trend data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyTrend {
    pub date: DateTime<Utc>,
    pub total_cost: Decimal,
    pub gpu_cost: Decimal,
    pub cpu_cost: Decimal,
    pub network_cost: Decimal,
    pub storage_cost: Decimal,
    pub job_count: i64,
}

/// Monthly trend data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyTrend {
    pub month: DateTime<Utc>,
    pub total_cost: Decimal,
    pub gpu_cost: Decimal,
    pub cpu_cost: Decimal,
    pub network_cost: Decimal,
    pub storage_cost: Decimal,
    pub job_count: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_trend_analysis_creation() {
        let trend = TrendAnalysis::new(
            TrendDirection::Increasing,
            dec!(15.5),
            dec!(100.00),
            0.5,
        )
        .with_confidence(0.85);

        assert_eq!(trend.direction, TrendDirection::Increasing);
        assert_eq!(trend.growth_rate, dec!(15.5));
        assert_eq!(trend.daily_average, dec!(100.00));
        assert_eq!(trend.slope, 0.5);
        assert_eq!(trend.confidence, 0.85);
    }

    #[test]
    fn test_trend_confidence_clamping() {
        let trend = TrendAnalysis::new(
            TrendDirection::Stable,
            dec!(0.0),
            dec!(50.00),
            0.0,
        )
        .with_confidence(1.5);

        assert_eq!(trend.confidence, 1.0); // Clamped to 1.0
    }

    #[test]
    fn test_forecast_point() {
        let now = Utc::now();
        let point = ForecastPoint::new(now, dec!(125.50), 0.9);

        assert_eq!(point.forecasted_cost, dec!(125.50));
        assert_eq!(point.confidence, 0.9);
    }

    #[test]
    fn test_cost_forecast_avg_confidence() {
        let now = Utc::now();
        let points = vec![
            ForecastPoint::new(now, dec!(100.00), 0.9),
            ForecastPoint::new(now, dec!(110.00), 0.8),
            ForecastPoint::new(now, dec!(120.00), 0.7),
        ];

        let forecast = CostForecast::new(
            now,
            now,
            now,
            now,
            "linear_regression".to_string(),
            points,
        );

        assert!((forecast.avg_confidence - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_cost_forecast_empty_points() {
        let now = Utc::now();
        let forecast = CostForecast::new(
            now,
            now,
            now,
            now,
            "linear_regression".to_string(),
            vec![],
        );

        assert_eq!(forecast.avg_confidence, 0.0);
        assert_eq!(forecast.points.len(), 0);
    }
}
