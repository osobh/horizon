use crate::error::{CapacityErrorExt, HpcError, Result};
use crate::forecaster::ets::EtsForecaster;
use crate::forecaster::metrics::{
    mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error,
};
use crate::models::{AccuracyMetrics, BacktestResult, ForecastPoint};
use chrono::{Duration, Utc};

/// Split time series data into train and test sets
///
/// # Arguments
/// * `data` - Full dataset
/// * `train_ratio` - Proportion of data for training (e.g., 0.8 for 80%)
///
/// # Returns
/// (train_data, test_data)
pub fn split_train_test(data: &[f64], train_ratio: f64) -> (Vec<f64>, Vec<f64>) {
    let train_size = (data.len() as f64 * train_ratio) as usize;
    let train = data[..train_size].to_vec();
    let test = data[train_size..].to_vec();
    (train, test)
}

/// Perform backtest validation on historical data
///
/// Trains model on first `train_size` points and tests on next `test_size` points
///
/// # Arguments
/// * `data` - Historical time series data
/// * `train_size` - Number of points to use for training
/// * `test_size` - Number of points to forecast and validate
///
/// # Returns
/// BacktestResult with predictions, actuals, and accuracy metrics
pub fn backtest_forecast(
    data: &[f64],
    train_size: usize,
    test_size: usize,
) -> Result<BacktestResult> {
    // Validate inputs
    if train_size + test_size > data.len() {
        return Err(HpcError::invalid_parameters(format!(
            "train_size ({}) + test_size ({}) exceeds data length ({})",
            train_size,
            test_size,
            data.len()
        )));
    }

    if train_size < 10 {
        return Err(HpcError::insufficient_data(format!(
            "Need at least 10 training points, got {}",
            train_size
        )));
    }

    // Split data
    let train_data = &data[..train_size];
    let test_data = &data[train_size..train_size + test_size];

    // Train model
    let mut forecaster = EtsForecaster::new();
    forecaster.train(train_data)?;

    // Generate forecasts
    let forecast_values = forecaster.forecast(test_size)?;

    // Calculate accuracy metrics
    let mape = mean_absolute_percentage_error(&forecast_values, test_data);
    let rmse = root_mean_squared_error(&forecast_values, test_data);
    let mae = mean_absolute_error(&forecast_values, test_data);

    // Create forecast points with timestamps
    let base_time = Utc::now();
    let predictions: Vec<ForecastPoint> = forecast_values
        .iter()
        .enumerate()
        .map(|(i, &value)| ForecastPoint {
            timestamp: base_time + Duration::days(i as i64),
            value,
            lower_bound: None,
            upper_bound: None,
        })
        .collect();

    Ok(BacktestResult {
        train_size,
        test_size,
        metrics: AccuracyMetrics { mape, rmse, mae },
        predictions,
        actuals: test_data.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_train_test_basic() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let (train, test) = split_train_test(&data, 0.7);

        assert_eq!(train.len(), 70);
        assert_eq!(test.len(), 30);
        assert_eq!(train[0], 1.0);
        assert_eq!(test[0], 71.0);
    }

    #[test]
    fn test_split_train_test_edge_cases() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let (train, test) = split_train_test(&data, 0.0);
        assert_eq!(train.len(), 0);
        assert_eq!(test.len(), 5);

        let (train, test) = split_train_test(&data, 1.0);
        assert_eq!(train.len(), 5);
        assert_eq!(test.len(), 0);
    }

    #[test]
    fn test_backtest_forecast_success() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = backtest_forecast(&data, 70, 20);

        assert!(result.is_ok());
        let backtest = result.unwrap();
        assert_eq!(backtest.train_size, 70);
        assert_eq!(backtest.test_size, 20);
        assert_eq!(backtest.predictions.len(), 20);
        assert_eq!(backtest.actuals.len(), 20);
    }

    #[test]
    fn test_backtest_insufficient_data() {
        let data = vec![1.0, 2.0, 3.0];
        let result = backtest_forecast(&data, 2, 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_backtest_invalid_sizes() {
        let data: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = backtest_forecast(&data, 40, 20);

        assert!(result.is_err());
    }
}
