use chrono::{Duration, Utc};
use rust_decimal::Decimal;

use crate::error::{HpcError, Result, ReporterErrorExt};
use crate::models::summary::HasCostBreakdown;
use crate::models::trend::{CostForecast, ForecastPoint};

#[cfg(test)]
use crate::models::summary::DailyCostSummary;

pub struct CostForecaster;

impl CostForecaster {
    pub fn new() -> Self {
        Self
    }

    /// Forecast costs using linear regression
    pub fn forecast<T>(
        &self,
        historical: &[T],
        days_ahead: usize,
    ) -> Result<CostForecast>
    where
        T: HasCostBreakdown,
    {
        if historical.is_empty() {
            return Err(HpcError::insufficient_data(1));
        }

        if historical.len() < 3 {
            return Err(HpcError::insufficient_data(3));
        }

        // Calculate linear regression
        let (slope, intercept) = self.linear_regression(historical);

        // Get date range
        let historical_start = Utc::now() - Duration::days(historical.len() as i64);
        let historical_end = Utc::now();
        let forecast_start = historical_end;
        let forecast_end = historical_end + Duration::days(days_ahead as i64);

        // Generate forecast points
        let mut points = Vec::new();
        let base_index = historical.len();

        for day in 1..=days_ahead {
            let forecast_date = forecast_start + Duration::days(day as i64);
            let x = (base_index + day) as f64;
            let forecasted_value = intercept + (slope * x);

            let forecasted_cost = if forecasted_value < 0.0 {
                Decimal::ZERO
            } else {
                Decimal::from_f64_retain(forecasted_value).unwrap_or(Decimal::ZERO)
            };

            // Calculate confidence - decreases with distance
            let confidence = self.calculate_forecast_confidence(historical, day);

            points.push(ForecastPoint::new(forecast_date, forecasted_cost, confidence));
        }

        Ok(CostForecast::new(
            historical_start,
            historical_end,
            forecast_start,
            forecast_end,
            "linear_regression".to_string(),
            points,
        ))
    }

    /// Simple linear regression
    fn linear_regression<T>(&self, data: &[T]) -> (f64, f64)
    where
        T: HasCostBreakdown,
    {
        let n = data.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, item) in data.iter().enumerate() {
            let x = i as f64;
            let y = item.total_cost().to_string().parse::<f64>().unwrap_or(0.0);
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }

    /// Calculate forecast confidence based on variance and distance
    fn calculate_forecast_confidence<T>(&self, historical: &[T], days_ahead: usize) -> f64
    where
        T: HasCostBreakdown,
    {
        // Base confidence from historical variance
        let variance = self.calculate_variance(historical);
        let base_confidence = if variance < 0.1 {
            0.9
        } else if variance < 0.3 {
            0.7
        } else {
            0.5
        };

        // Decay confidence based on forecast distance
        let decay_factor = 0.95_f64.powi(days_ahead as i32);

        (base_confidence * decay_factor).clamp(0.0, 1.0)
    }

    /// Calculate variance in historical data
    fn calculate_variance<T>(&self, data: &[T]) -> f64
    where
        T: HasCostBreakdown,
    {
        if data.len() < 2 {
            return 0.0;
        }

        let values: Vec<f64> = data
            .iter()
            .map(|d| d.total_cost().to_string().parse::<f64>().unwrap_or(0.0))
            .collect();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        // Coefficient of variation (normalized variance)
        if mean > 0.0 {
            (variance.sqrt() / mean).abs()
        } else {
            0.0
        }
    }
}

impl Default for CostForecaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_forecast_insufficient_data() {
        let forecaster = CostForecaster::new();
        let data: Vec<DailyCostSummary> = vec![];

        let result = forecaster.forecast(&data, 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_forecast_too_few_points() {
        let forecaster = CostForecaster::new();
        let now = Utc::now();
        let data = vec![
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(100.00),
                gpu_cost: dec!(80.00),
                cpu_cost: dec!(10.00),
                network_cost: dec!(5.00),
                storage_cost: dec!(5.00),
                job_count: 10,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(110.00),
                gpu_cost: dec!(88.00),
                cpu_cost: dec!(11.00),
                network_cost: dec!(5.50),
                storage_cost: dec!(5.50),
                job_count: 11,
            },
        ];

        let result = forecaster.forecast(&data, 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_forecast_linear_growth() {
        let forecaster = CostForecaster::new();
        let now = Utc::now();
        let data = vec![
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(100.00),
                gpu_cost: dec!(80.00),
                cpu_cost: dec!(10.00),
                network_cost: dec!(5.00),
                storage_cost: dec!(5.00),
                job_count: 10,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(110.00),
                gpu_cost: dec!(88.00),
                cpu_cost: dec!(11.00),
                network_cost: dec!(5.50),
                storage_cost: dec!(5.50),
                job_count: 11,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(120.00),
                gpu_cost: dec!(96.00),
                cpu_cost: dec!(12.00),
                network_cost: dec!(6.00),
                storage_cost: dec!(6.00),
                job_count: 12,
            },
        ];

        let forecast = forecaster.forecast(&data, 7).unwrap();

        assert_eq!(forecast.points.len(), 7);
        assert_eq!(forecast.method, "linear_regression");

        // First forecast point should be ~130 (continuing the +10 pattern)
        let first_forecast = forecast.points[0].forecasted_cost;
        assert!(first_forecast > dec!(115.00) && first_forecast < dec!(145.00));

        // Confidence should decrease with distance
        assert!(forecast.points[0].confidence > forecast.points[6].confidence);
    }

    #[test]
    fn test_forecast_stable_costs() {
        let forecaster = CostForecaster::new();
        let now = Utc::now();
        let data = vec![
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(100.00),
                gpu_cost: dec!(80.00),
                cpu_cost: dec!(10.00),
                network_cost: dec!(5.00),
                storage_cost: dec!(5.00),
                job_count: 10,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(100.50),
                gpu_cost: dec!(80.00),
                cpu_cost: dec!(10.00),
                network_cost: dec!(5.00),
                storage_cost: dec!(5.50),
                job_count: 10,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(99.50),
                gpu_cost: dec!(79.00),
                cpu_cost: dec!(10.00),
                network_cost: dec!(5.00),
                storage_cost: dec!(5.50),
                job_count: 10,
            },
        ];

        let forecast = forecaster.forecast(&data, 3).unwrap();

        // Forecasts should be close to 100
        for point in &forecast.points {
            assert!(point.forecasted_cost > dec!(90.00) && point.forecasted_cost < dec!(110.00));
        }
    }

    #[test]
    fn test_forecast_negative_prevention() {
        let forecaster = CostForecaster::new();
        let now = Utc::now();
        // Decreasing trend that would go negative
        let data = vec![
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(30.00),
                gpu_cost: dec!(24.00),
                cpu_cost: dec!(3.00),
                network_cost: dec!(1.50),
                storage_cost: dec!(1.50),
                job_count: 3,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(20.00),
                gpu_cost: dec!(16.00),
                cpu_cost: dec!(2.00),
                network_cost: dec!(1.00),
                storage_cost: dec!(1.00),
                job_count: 2,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(10.00),
                gpu_cost: dec!(8.00),
                cpu_cost: dec!(1.00),
                network_cost: dec!(0.50),
                storage_cost: dec!(0.50),
                job_count: 1,
            },
        ];

        let forecast = forecaster.forecast(&data, 5).unwrap();

        // All forecasts should be >= 0
        for point in &forecast.points {
            assert!(point.forecasted_cost >= Decimal::ZERO);
        }
    }
}
