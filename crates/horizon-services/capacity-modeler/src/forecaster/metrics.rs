/// Calculate Mean Absolute Percentage Error (MAPE)
///
/// MAPE = mean(|actual - forecast| / |actual|) * 100
///
/// # Arguments
/// * `forecast` - Predicted values
/// * `actual` - Actual observed values
///
/// # Returns
/// MAPE as a decimal (e.g., 0.15 = 15%)
pub fn mean_absolute_percentage_error(forecast: &[f64], actual: &[f64]) -> f64 {
    let n = forecast.len().min(actual.len());
    if n == 0 {
        return 0.0;
    }

    let sum: f64 = forecast
        .iter()
        .zip(actual.iter())
        .take(n)
        .map(|(f, a)| {
            if a.abs() < f64::EPSILON {
                // Avoid division by zero
                0.0
            } else {
                (a - f).abs() / a.abs()
            }
        })
        .sum();

    sum / n as f64
}

/// Calculate Root Mean Squared Error (RMSE)
///
/// RMSE = sqrt(mean((actual - forecast)^2))
///
/// # Arguments
/// * `forecast` - Predicted values
/// * `actual` - Actual observed values
///
/// # Returns
/// RMSE value
pub fn root_mean_squared_error(forecast: &[f64], actual: &[f64]) -> f64 {
    let n = forecast.len().min(actual.len());
    if n == 0 {
        return 0.0;
    }

    let sum_squared_errors: f64 = forecast
        .iter()
        .zip(actual.iter())
        .take(n)
        .map(|(f, a)| (a - f).powi(2))
        .sum();

    (sum_squared_errors / n as f64).sqrt()
}

/// Calculate Mean Absolute Error (MAE)
///
/// MAE = mean(|actual - forecast|)
///
/// # Arguments
/// * `forecast` - Predicted values
/// * `actual` - Actual observed values
///
/// # Returns
/// MAE value
pub fn mean_absolute_error(forecast: &[f64], actual: &[f64]) -> f64 {
    let n = forecast.len().min(actual.len());
    if n == 0 {
        return 0.0;
    }

    let sum_absolute_errors: f64 = forecast
        .iter()
        .zip(actual.iter())
        .take(n)
        .map(|(f, a)| (a - f).abs())
        .sum();

    sum_absolute_errors / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mape_basic() {
        let actual = vec![100.0, 200.0, 300.0];
        let forecast = vec![90.0, 210.0, 285.0];

        let mape = mean_absolute_percentage_error(&forecast, &actual);

        // Expected: (10/100 + 10/200 + 15/300) / 3 = (0.1 + 0.05 + 0.05) / 3 = 0.0667
        assert!((mape - 0.0667).abs() < 0.001);
    }

    #[test]
    fn test_rmse_basic() {
        let actual = vec![3.0, -0.5, 2.0, 7.0];
        let forecast = vec![2.5, 0.0, 2.0, 8.0];

        let rmse = root_mean_squared_error(&forecast, &actual);

        // Errors: 0.5, 0.5, 0.0, 1.0
        // Squared: 0.25, 0.25, 0.0, 1.0
        // Mean: 1.5 / 4 = 0.375
        // RMSE: sqrt(0.375) ≈ 0.612
        assert!((rmse - 0.612).abs() < 0.001);
    }

    #[test]
    fn test_mae_basic() {
        let actual = vec![10.0, 20.0, 30.0];
        let forecast = vec![8.0, 22.0, 28.0];

        let mae = mean_absolute_error(&forecast, &actual);

        // Errors: 2, 2, 2
        // MAE: 6 / 3 = 2
        assert_eq!(mae, 2.0);
    }

    #[test]
    fn test_empty_inputs() {
        let empty: Vec<f64> = vec![];
        let data = vec![1.0, 2.0, 3.0];

        assert_eq!(mean_absolute_percentage_error(&empty, &data), 0.0);
        assert_eq!(root_mean_squared_error(&empty, &data), 0.0);
        assert_eq!(mean_absolute_error(&empty, &data), 0.0);
    }

    #[test]
    fn test_zero_actual_values() {
        let actual = vec![0.0, 10.0, 0.0];
        let forecast = vec![5.0, 12.0, 3.0];

        let mape = mean_absolute_percentage_error(&forecast, &actual);
        assert!(mape.is_finite());

        let rmse = root_mean_squared_error(&forecast, &actual);
        assert!(rmse > 0.0);

        let mae = mean_absolute_error(&forecast, &actual);
        assert!(mae > 0.0);
    }
}
