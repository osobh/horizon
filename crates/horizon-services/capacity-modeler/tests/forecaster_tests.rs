use capacity_modeler::forecaster::ets::EtsForecaster;

// Helper function to generate test data
fn generate_seasonal_data(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 50.0;
            let trend = i as f64 * 0.1;
            let seasonal = ((i as f64 * 2.0 * std::f64::consts::PI) / 7.0).sin() * 10.0;
            base + trend + seasonal
        })
        .collect()
}

fn generate_simple_data(count: usize) -> Vec<f64> {
    (0..count).map(|i| (i + 1) as f64).collect()
}

#[test]
fn test_ets_creation() {
    let forecaster = EtsForecaster::new();
    assert!(!forecaster.is_trained());
}

#[test]
fn test_ets_training_with_simple_data() {
    let data = generate_simple_data(50);
    let mut forecaster = EtsForecaster::new();

    let result = forecaster.train(&data);
    assert!(result.is_ok(), "Training should succeed");
    assert!(forecaster.is_trained(), "Forecaster should be trained");
}

#[test]
fn test_ets_training_with_seasonal_data() {
    let data = generate_seasonal_data(180);
    let mut forecaster = EtsForecaster::new();

    let result = forecaster.train(&data);
    assert!(result.is_ok(), "Training should succeed with seasonal data");
    assert!(forecaster.is_trained());
}

#[test]
fn test_forecast_generation() {
    let data = generate_seasonal_data(180);
    let mut forecaster = EtsForecaster::new();
    forecaster.train(&data).unwrap();

    let forecast = forecaster.forecast(91).unwrap(); // 13 weeks * 7 days

    assert_eq!(forecast.len(), 91, "Should generate 91 forecast points");
    assert!(
        forecast.iter().all(|&v| v > 0.0),
        "All forecast values should be positive"
    );
}

#[test]
fn test_forecast_without_training() {
    let forecaster = EtsForecaster::new();
    let result = forecaster.forecast(10);
    assert!(result.is_err(), "Forecast without training should fail");
}

#[test]
fn test_confidence_intervals() {
    let data = generate_seasonal_data(180);
    let mut forecaster = EtsForecaster::new();
    forecaster.train(&data).unwrap();

    let result = forecaster.forecast_with_intervals(30, 0.95).unwrap();

    assert_eq!(result.point_forecast.len(), 30);
    assert_eq!(result.lower_bound.len(), 30);
    assert_eq!(result.upper_bound.len(), 30);

    // Verify intervals are valid (lower < forecast < upper)
    for i in 0..30 {
        assert!(
            result.lower_bound[i] < result.point_forecast[i],
            "Lower bound should be less than forecast"
        );
        assert!(
            result.point_forecast[i] < result.upper_bound[i],
            "Forecast should be less than upper bound"
        );
    }
}

#[test]
fn test_insufficient_data() {
    let data = vec![1.0, 2.0, 3.0]; // Too little data
    let mut forecaster = EtsForecaster::new();

    let result = forecaster.train(&data);
    assert!(
        result.is_err(),
        "Training with insufficient data should fail"
    );
}

#[test]
fn test_empty_data() {
    let data: Vec<f64> = vec![];
    let mut forecaster = EtsForecaster::new();

    let result = forecaster.train(&data);
    assert!(result.is_err(), "Training with empty data should fail");
}

#[test]
fn test_forecast_reasonable_values() {
    let data = generate_seasonal_data(180);
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let std_dev = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64)
        .sqrt();

    let mut forecaster = EtsForecaster::new();
    forecaster.train(&data).unwrap();

    let forecast = forecaster.forecast(91).unwrap();

    // Forecasts should be within reasonable bounds (mean +/- 5 std devs)
    for &value in &forecast {
        assert!(
            value > mean - 5.0 * std_dev && value < mean + 5.0 * std_dev,
            "Forecast value {} is outside reasonable bounds (mean={}, std={})",
            value,
            mean,
            std_dev
        );
    }
}

#[test]
fn test_multiple_forecasts() {
    let data = generate_seasonal_data(180);
    let mut forecaster = EtsForecaster::new();
    forecaster.train(&data).unwrap();

    let forecast1 = forecaster.forecast(30).unwrap();
    let forecast2 = forecaster.forecast(30).unwrap();

    // Same trained model should produce same forecasts
    assert_eq!(
        forecast1, forecast2,
        "Multiple forecasts should be identical"
    );
}
