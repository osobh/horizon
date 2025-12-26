use capacity_modeler::forecaster::metrics::{
    mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error,
};
use capacity_modeler::forecaster::validation::{backtest_forecast, split_train_test};

#[test]
fn test_mape_calculation() {
    let actual = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let forecast = vec![110.0, 190.0, 310.0, 390.0, 510.0];

    let mape = mean_absolute_percentage_error(&forecast, &actual);

    // MAPE = mean(|actual - forecast| / actual)
    // = mean(|10|/100, |10|/200, |10|/300, |10|/400, |10|/500)
    // = mean(0.1, 0.05, 0.033, 0.025, 0.02)
    // ≈ 0.0456
    assert!(
        (mape - 0.0456).abs() < 0.001,
        "MAPE should be approximately 0.0456, got {}",
        mape
    );
}

#[test]
fn test_rmse_calculation() {
    let actual = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let forecast = vec![12.0, 18.0, 32.0, 38.0, 52.0];

    let rmse = root_mean_squared_error(&forecast, &actual);

    // RMSE = sqrt(mean((actual - forecast)^2))
    // = sqrt(mean(4, 4, 4, 4, 4))
    // = sqrt(4)
    // = 2.0
    assert!(
        (rmse - 2.0).abs() < 0.001,
        "RMSE should be 2.0, got {}",
        rmse
    );
}

#[test]
fn test_mae_calculation() {
    let actual = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let forecast = vec![12.0, 18.0, 32.0, 38.0, 52.0];

    let mae = mean_absolute_error(&forecast, &actual);

    // MAE = mean(|actual - forecast|)
    // = mean(2, 2, 2, 2, 2)
    // = 2.0
    assert_eq!(mae, 2.0, "MAE should be 2.0");
}

#[test]
fn test_perfect_forecast_metrics() {
    let actual = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let forecast = actual.clone();

    assert_eq!(
        mean_absolute_percentage_error(&forecast, &actual),
        0.0,
        "Perfect forecast should have 0 MAPE"
    );
    assert_eq!(
        root_mean_squared_error(&forecast, &actual),
        0.0,
        "Perfect forecast should have 0 RMSE"
    );
    assert_eq!(
        mean_absolute_error(&forecast, &actual),
        0.0,
        "Perfect forecast should have 0 MAE"
    );
}

#[test]
fn test_train_test_split() {
    let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();

    let (train, test) = split_train_test(&data, 0.8);

    assert_eq!(train.len(), 80, "Train set should be 80% of data");
    assert_eq!(test.len(), 20, "Test set should be 20% of data");

    // Verify no data leakage
    assert_eq!(train[0], 1.0);
    assert_eq!(train[79], 80.0);
    assert_eq!(test[0], 81.0);
    assert_eq!(test[19], 100.0);
}

#[test]
fn test_backtest_forecast() {
    let data: Vec<f64> = (1..=200).map(|x| x as f64).collect();

    let result = backtest_forecast(&data, 150, 30);

    assert!(result.is_ok(), "Backtest should succeed");

    let backtest = result.unwrap();
    assert_eq!(backtest.train_size, 150);
    assert_eq!(backtest.test_size, 30);
    assert_eq!(backtest.predictions.len(), 30);
    assert_eq!(backtest.actuals.len(), 30);

    // Metrics should be reasonable (not NaN or infinite)
    assert!(backtest.metrics.mape.is_finite());
    assert!(backtest.metrics.rmse.is_finite());
    assert!(backtest.metrics.mae.is_finite());

    // For a simple increasing sequence, forecast should be reasonably accurate
    assert!(
        backtest.metrics.mape < 0.3,
        "MAPE should be < 30% for simple data, got {}",
        backtest.metrics.mape
    );
}

#[test]
fn test_backtest_with_insufficient_train_data() {
    let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();

    let result = backtest_forecast(&data, 5, 10); // Only 5 training points

    assert!(
        result.is_err(),
        "Backtest should fail with insufficient training data"
    );
}

#[test]
fn test_backtest_with_invalid_sizes() {
    let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();

    // Train + test > data length
    let result = backtest_forecast(&data, 80, 30);
    assert!(result.is_err(), "Should fail when sizes exceed data length");
}

#[test]
fn test_metrics_with_zero_actual() {
    let actual = vec![0.0, 10.0, 20.0];
    let forecast = vec![5.0, 12.0, 18.0];

    // MAPE should handle division by zero gracefully
    let mape = mean_absolute_percentage_error(&forecast, &actual);
    assert!(mape.is_finite(), "MAPE should handle zero values");

    // RMSE and MAE should work normally
    let rmse = root_mean_squared_error(&forecast, &actual);
    let mae = mean_absolute_error(&forecast, &actual);

    assert!(rmse > 0.0);
    assert!(mae > 0.0);
}

#[test]
fn test_metrics_with_mismatched_lengths() {
    let actual = vec![10.0, 20.0, 30.0];
    let forecast = vec![12.0, 18.0]; // Wrong length

    // Metrics should handle this gracefully (might panic or return special value)
    // For now, we expect it to use the minimum length
    let mae = mean_absolute_error(&forecast, &actual);
    assert!(mae.is_finite());
}
