use crate::error::{HpcError, Result, CapacityErrorExt};
use crate::forecaster::{backtest_forecast, EtsForecaster};
use crate::models::{BacktestResult, ForecastPoint, ForecastResult};
use chrono::{Duration, Utc};

/// Forecast service for generating GPU demand predictions
pub struct ForecastService {
    min_historical_days: usize,
}

impl ForecastService {
    /// Create a new forecast service
    ///
    /// # Arguments
    /// * `min_historical_days` - Minimum days of historical data required
    pub fn new(min_historical_days: usize) -> Self {
        Self {
            min_historical_days,
        }
    }

    /// Validate that we have sufficient historical data
    fn validate_data_sufficiency(&self, data: &[f64]) -> Result<()> {
        if data.len() < self.min_historical_days {
            return Err(HpcError::insufficient_data(format!(
                "Need at least {} days of data, have {} data points",
                self.min_historical_days,
                data.len()
            )));
        }
        Ok(())
    }

    /// Preprocess historical data (remove outliers, fill gaps)
    fn preprocess_data(&self, data: &[f64]) -> Vec<f64> {
        // For now, simple outlier removal using IQR method
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1_idx = sorted.len() / 4;
        let q3_idx = 3 * sorted.len() / 4;
        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        // Replace outliers with median
        let median = sorted[sorted.len() / 2];

        data.iter()
            .map(|&x| {
                if x < lower_bound || x > upper_bound {
                    median
                } else {
                    x
                }
            })
            .collect()
    }

    /// Generate GPU demand forecast
    ///
    /// # Arguments
    /// * `historical_data` - Historical time series data
    /// * `forecast_weeks` - Number of weeks to forecast
    /// * `include_intervals` - Whether to include confidence intervals
    ///
    /// # Returns
    /// ForecastResult with predictions and optional accuracy metrics
    pub fn forecast_gpu_demand(
        &self,
        historical_data: &[f64],
        forecast_weeks: u8,
        include_intervals: bool,
    ) -> Result<ForecastResult> {
        // Validate data sufficiency
        self.validate_data_sufficiency(historical_data)?;

        // Preprocess data
        let processed_data = self.preprocess_data(historical_data);

        // Train forecaster
        let mut forecaster = EtsForecaster::new();
        forecaster.train(&processed_data)?;

        // Calculate forecast horizon (days)
        let horizon = forecast_weeks as usize * 7;

        // Generate forecast
        let base_time = Utc::now();
        let points = if include_intervals {
            let forecast_with_intervals = forecaster.forecast_with_intervals(horizon, 0.95)?;

            forecast_with_intervals
                .point_forecast
                .iter()
                .enumerate()
                .map(|(i, &value)| ForecastPoint {
                    timestamp: base_time + Duration::days(i as i64 + 1),
                    value,
                    lower_bound: Some(forecast_with_intervals.lower_bound[i]),
                    upper_bound: Some(forecast_with_intervals.upper_bound[i]),
                })
                .collect()
        } else {
            let forecast_values = forecaster.forecast(horizon)?;

            forecast_values
                .iter()
                .enumerate()
                .map(|(i, &value)| ForecastPoint {
                    timestamp: base_time + Duration::days(i as i64 + 1),
                    value,
                    lower_bound: None,
                    upper_bound: None,
                })
                .collect()
        };

        // Calculate accuracy metrics using backtest (last 30 days as holdout)
        let accuracy_metrics = if processed_data.len() >= 30 {
            let train_size = processed_data.len() - 30;
            match backtest_forecast(&processed_data, train_size, 30) {
                Ok(backtest) => Some(backtest.metrics),
                Err(_) => None,
            }
        } else {
            None
        };

        Ok(ForecastResult {
            forecast_weeks,
            points,
            generated_at: Utc::now(),
            model_type: "ETS".to_string(),
            accuracy_metrics,
        })
    }

    /// Perform backtest validation on historical data
    ///
    /// # Arguments
    /// * `historical_data` - Historical time series data
    /// * `train_days` - Number of days to use for training
    /// * `test_days` - Number of days to forecast and validate
    ///
    /// # Returns
    /// BacktestResult with predictions, actuals, and accuracy metrics
    pub fn backtest(
        &self,
        historical_data: &[f64],
        train_days: usize,
        test_days: usize,
    ) -> Result<BacktestResult> {
        // Validate data sufficiency
        if historical_data.len() < train_days + test_days {
            return Err(HpcError::insufficient_data(format!(
                "Need {} data points, have {}",
                train_days + test_days,
                historical_data.len()
            )));
        }

        // Preprocess data
        let processed_data = self.preprocess_data(historical_data);

        // Perform backtest
        backtest_forecast(&processed_data, train_days, test_days)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(count: usize) -> Vec<f64> {
        (0..count)
            .map(|i| {
                let base = 50.0;
                let trend = i as f64 * 0.1;
                let seasonal = ((i as f64 * 2.0 * std::f64::consts::PI) / 7.0).sin() * 10.0;
                base + trend + seasonal
            })
            .collect()
    }

    #[test]
    fn test_forecast_service_creation() {
        let service = ForecastService::new(180);
        assert_eq!(service.min_historical_days, 180);
    }

    #[test]
    fn test_validate_data_sufficiency() {
        let service = ForecastService::new(180);

        let insufficient_data = generate_test_data(100);
        assert!(service.validate_data_sufficiency(&insufficient_data).is_err());

        let sufficient_data = generate_test_data(200);
        assert!(service.validate_data_sufficiency(&sufficient_data).is_ok());
    }

    #[test]
    fn test_preprocess_data() {
        let service = ForecastService::new(180);

        // Data with outliers
        let mut data = vec![10.0; 100];
        data[50] = 1000.0; // Outlier
        data[75] = -500.0; // Outlier

        let processed = service.preprocess_data(&data);

        // Outliers should be replaced
        assert!(processed[50] < 100.0);
        assert!(processed[75] > -100.0);
    }

    #[test]
    fn test_forecast_gpu_demand() {
        let service = ForecastService::new(180);
        let data = generate_test_data(200);

        let result = service.forecast_gpu_demand(&data, 13, false);

        assert!(result.is_ok());
        let forecast = result.unwrap();

        assert_eq!(forecast.forecast_weeks, 13);
        assert_eq!(forecast.points.len(), 91); // 13 weeks * 7 days
        assert_eq!(forecast.model_type, "ETS");

        // Check that forecasts are reasonable
        for point in &forecast.points {
            assert!(point.value > 0.0);
            assert!(point.value < 200.0); // Reasonable upper bound
        }
    }

    #[test]
    fn test_forecast_with_intervals() {
        let service = ForecastService::new(180);
        let data = generate_test_data(200);

        let result = service.forecast_gpu_demand(&data, 4, true);

        assert!(result.is_ok());
        let forecast = result.unwrap();

        // All points should have confidence intervals
        for point in &forecast.points {
            assert!(point.lower_bound.is_some());
            assert!(point.upper_bound.is_some());

            let lower = point.lower_bound.unwrap();
            let upper = point.upper_bound.unwrap();

            assert!(lower < point.value);
            assert!(point.value < upper);
        }
    }

    #[test]
    fn test_forecast_insufficient_data() {
        let service = ForecastService::new(180);
        let data = generate_test_data(100);

        let result = service.forecast_gpu_demand(&data, 13, false);

        assert!(result.is_err());
    }

    #[test]
    fn test_backtest() {
        let service = ForecastService::new(50);
        let data = generate_test_data(200);

        let result = service.backtest(&data, 150, 30);

        assert!(result.is_ok());
        let backtest = result.unwrap();

        assert_eq!(backtest.train_size, 150);
        assert_eq!(backtest.test_size, 30);
        assert_eq!(backtest.predictions.len(), 30);
        assert_eq!(backtest.actuals.len(), 30);

        // Metrics should be reasonable
        assert!(backtest.metrics.mape < 1.0); // Less than 100% error
        assert!(backtest.metrics.rmse > 0.0);
        assert!(backtest.metrics.mae > 0.0);
    }
}
