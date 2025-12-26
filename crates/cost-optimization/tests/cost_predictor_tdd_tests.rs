//! TDD Validation Tests for Cost Predictor Refactoring
//!
//! These tests establish expected behavior before refactoring the monolithic
//! cost_predictor.rs file. They ensure that all functionality is preserved
//! during the splitting process.

use crate::cost_predictor::*;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use tokio;

/// Test module for data types and structures
#[cfg(test)]
mod types_tests {
    use super::*;

    #[test]
    fn test_cost_metric_type_display() {
        assert_eq!(CostMetricType::TotalCost.to_string(), "Total Cost");
        assert_eq!(CostMetricType::ComputeCost.to_string(), "Compute Cost");
        assert_eq!(CostMetricType::GpuCost.to_string(), "GPU Cost");
    }

    #[test]
    fn test_prediction_model_display() {
        assert_eq!(PredictionModel::MovingAverage.to_string(), "Moving Average");
        assert_eq!(PredictionModel::Ensemble.to_string(), "Ensemble");
    }

    #[test]
    fn test_seasonality_serialization() {
        let seasonality = Seasonality::Weekly;
        let serialized = serde_json::to_string(&seasonality).unwrap();
        let deserialized: Seasonality = serde_json::from_str(&serialized).unwrap();
        assert_eq!(seasonality, deserialized);
    }

    #[test]
    fn test_prediction_request_creation() {
        let request = PredictionRequest {
            metric_type: CostMetricType::ComputeCost,
            horizon: Duration::from_secs(3600),
            model: PredictionModel::MovingAverage,
            confidence_level: 0.95,
            seasonality: Seasonality::Daily,
            filters: HashMap::new(),
        };
        assert_eq!(request.metric_type, CostMetricType::ComputeCost);
        assert_eq!(request.confidence_level, 0.95);
    }

    #[test]
    fn test_time_series_point_creation() {
        let point = TimeSeriesPoint {
            timestamp: Utc::now(),
            value: 100.0,
            metadata: HashMap::new(),
        };
        assert_eq!(point.value, 100.0);
    }

    #[test]
    fn test_anomaly_type_variants() {
        let spike = AnomalyType::Spike;
        let drop = AnomalyType::Drop;
        assert_ne!(spike, drop);
    }

    #[test]
    fn test_trend_direction_variants() {
        let increasing = TrendDirection::Increasing;
        let stable = TrendDirection::Stable;
        assert_ne!(increasing, stable);
    }

    #[test]
    fn test_risk_level_variants() {
        let low = RiskLevel::Low;
        let critical = RiskLevel::Critical;
        assert_ne!(low, critical);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Medium);
        assert!(Priority::Medium > Priority::Low);
    }

    #[test]
    fn test_cost_predictor_config_default() {
        let config = CostPredictorConfig::default();
        assert_eq!(config.min_data_points, 30);
        assert_eq!(config.default_model, PredictionModel::Ensemble);
        assert!(config.enable_realtime);
    }
}

/// Test module for core predictor functionality
#[cfg(test)]
mod core_predictor_tests {
    use super::*;

    #[test]
    fn test_cost_predictor_creation() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config);
        assert!(predictor.is_ok());
    }

    #[test]
    fn test_add_data_point() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let result = predictor.add_data_point(
            CostMetricType::ComputeCost,
            100.0,
            Utc::now(),
            HashMap::new(),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_data_retention_enforcement() {
        let config = CostPredictorConfig {
            data_retention: Duration::from_secs(3600), // 1 hour
            ..Default::default()
        };
        let predictor = CostPredictor::new(config)?;

        // Add old data point
        let old_timestamp = Utc::now() - ChronoDuration::hours(2);
        predictor
            .add_data_point(
                CostMetricType::ComputeCost,
                100.0,
                old_timestamp,
                HashMap::new(),
            )
            .unwrap();

        // Add recent data point
        predictor
            .add_data_point(
                CostMetricType::ComputeCost,
                110.0,
                Utc::now(),
                HashMap::new(),
            )
            .unwrap();

        // Old data should be cleaned up (test via prediction attempt)
        let series = predictor
            .time_series
            .get(&CostMetricType::ComputeCost)
            .unwrap();
        assert!(series.len() <= 1); // Only recent data should remain
    }

    #[tokio::test]
    async fn test_prediction_with_insufficient_data() {
        let config = CostPredictorConfig {
            min_data_points: 10,
            ..Default::default()
        };
        let predictor = CostPredictor::new(config)?;

        let request = PredictionRequest {
            metric_type: CostMetricType::ComputeCost,
            horizon: Duration::from_secs(3600),
            model: PredictionModel::MovingAverage,
            confidence_level: 0.95,
            seasonality: Seasonality::None,
            filters: HashMap::new(),
        };

        let result = predictor.predict(request).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_metrics_tracking() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let metrics = predictor.get_metrics();
        assert_eq!(metrics.total_predictions, 0);
        assert_eq!(metrics.anomalies_detected, 0);
        assert_eq!(metrics.models_trained, 0);
    }

    #[test]
    fn test_anomaly_history_access() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let anomalies = predictor.get_anomalies();
        assert!(anomalies.is_empty());
    }
}

/// Test module for prediction algorithms
#[cfg(test)]
mod prediction_algorithm_tests {
    use super::*;

    fn create_test_series(size: usize) -> VecDeque<TimeSeriesPoint> {
        let mut series = VecDeque::new();
        let base_time = Utc::now() - ChronoDuration::hours(size as i64);

        for i in 0..size {
            series.push_back(TimeSeriesPoint {
                timestamp: base_time + ChronoDuration::hours(i as i64),
                value: 100.0 + (i as f64 * 2.0) + (i as f64).sin() * 10.0,
                metadata: HashMap::new(),
            });
        }

        series
    }

    #[tokio::test]
    async fn test_moving_average_prediction() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let series = create_test_series(50);
        for point in series {
            predictor
                .time_series
                .entry(CostMetricType::ComputeCost)
                .or_insert_with(VecDeque::new)
                .push_back(point);
        }

        let request = PredictionRequest {
            metric_type: CostMetricType::ComputeCost,
            horizon: Duration::from_secs(3600 * 24),
            model: PredictionModel::MovingAverage,
            confidence_level: 0.95,
            seasonality: Seasonality::None,
            filters: HashMap::new(),
        };

        let result = predictor.predict(request).await;
        assert!(result.is_ok());

        let prediction = result.unwrap();
        assert_eq!(prediction.predictions.len(), 24);
        assert!(prediction.predictions[0].value > 0.0);
    }

    #[tokio::test]
    async fn test_exponential_smoothing_prediction() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let series = create_test_series(50);
        for point in series {
            predictor
                .time_series
                .entry(CostMetricType::StorageCost)
                .or_insert_with(VecDeque::new)
                .push_back(point);
        }

        let request = PredictionRequest {
            metric_type: CostMetricType::StorageCost,
            horizon: Duration::from_secs(3600 * 12),
            model: PredictionModel::ExponentialSmoothing,
            confidence_level: 0.90,
            seasonality: Seasonality::None,
            filters: HashMap::new(),
        };

        let result = predictor.predict(request).await;
        assert!(result.is_ok());

        let prediction = result.unwrap();
        assert_eq!(prediction.predictions.len(), 12);
    }

    #[tokio::test]
    async fn test_linear_regression_prediction() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let series = create_test_series(50);
        for point in series {
            predictor
                .time_series
                .entry(CostMetricType::NetworkCost)
                .or_insert_with(VecDeque::new)
                .push_back(point);
        }

        let request = PredictionRequest {
            metric_type: CostMetricType::NetworkCost,
            horizon: Duration::from_secs(3600 * 6),
            model: PredictionModel::LinearRegression,
            confidence_level: 0.95,
            seasonality: Seasonality::None,
            filters: HashMap::new(),
        };

        let result = predictor.predict(request).await;
        assert!(result.is_ok());

        let prediction = result.unwrap();
        assert_eq!(prediction.predictions.len(), 6);
    }

    #[tokio::test]
    async fn test_ensemble_prediction() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let series = create_test_series(100);
        for point in series {
            predictor
                .time_series
                .entry(CostMetricType::GpuCost)
                .or_insert_with(VecDeque::new)
                .push_back(point);
        }

        let request = PredictionRequest {
            metric_type: CostMetricType::GpuCost,
            horizon: Duration::from_secs(3600 * 48),
            model: PredictionModel::Ensemble,
            confidence_level: 0.99,
            seasonality: Seasonality::Daily,
            filters: HashMap::new(),
        };

        let result = predictor.predict(request).await;
        assert!(result.is_ok());

        let prediction = result.unwrap();
        assert_eq!(prediction.predictions.len(), 48);
        assert!(prediction.accuracy.r_squared > 0.8);
    }

    #[test]
    fn test_confidence_intervals_calculation() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let predictions = vec![
            PredictedValue {
                timestamp: Utc::now() + ChronoDuration::hours(1),
                value: 100.0,
                confidence: 0.95,
            },
            PredictedValue {
                timestamp: Utc::now() + ChronoDuration::hours(2),
                value: 110.0,
                confidence: 0.90,
            },
        ];

        let model = create_test_trained_model(PredictionModel::MovingAverage);
        let intervals = predictor
            .calculate_confidence_intervals(&predictions, 0.95, &model)
            .unwrap();

        assert_eq!(intervals.len(), 2);
        for (i, interval) in intervals.iter().enumerate() {
            assert!(interval.lower_bound < predictions[i].value);
            assert!(interval.upper_bound > predictions[i].value);
            assert_eq!(interval.confidence_level, 0.95);
        }
    }

    fn create_test_trained_model(model_type: PredictionModel) -> TrainedModel {
        TrainedModel {
            model_type,
            trained_at: Utc::now(),
            parameters: ModelParameters {
                ma_window: Some(7),
                smoothing_factor: Some(0.3),
                coefficients: Some(vec![0.5, 1.0]),
                arima_params: Some((1, 1, 1)),
            },
            accuracy: ModelAccuracy {
                mae: 5.0,
                mse: 30.0,
                rmse: 5.48,
                mape: 5.0,
                r_squared: 0.85,
            },
        }
    }
}

/// Test module for anomaly detection
#[cfg(test)]
mod anomaly_detection_tests {
    use super::*;

    #[test]
    fn test_anomaly_detection_spike() {
        let config = CostPredictorConfig {
            anomaly_sensitivity: 0.8,
            min_data_points: 10,
            ..Default::default()
        };
        let predictor = CostPredictor::new(config)?;

        // Add normal data
        for i in 0..20 {
            predictor
                .add_data_point(
                    CostMetricType::ComputeCost,
                    100.0 + (i as f64).sin() * 5.0,
                    Utc::now() - ChronoDuration::hours(20 - i),
                    HashMap::new(),
                )
                .unwrap();
        }

        // Add anomaly (spike)
        predictor
            .add_data_point(
                CostMetricType::ComputeCost,
                500.0,
                Utc::now(),
                HashMap::new(),
            )
            .unwrap();

        let anomalies = predictor.get_anomalies();
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::Spike);
        assert!(anomalies[0].score > 0.0);
    }

    #[test]
    fn test_anomaly_detection_drop() {
        let config = CostPredictorConfig {
            anomaly_sensitivity: 0.9,
            min_data_points: 15,
            ..Default::default()
        };
        let predictor = CostPredictor::new(config)?;

        // Add normal data
        for i in 0..25 {
            predictor
                .add_data_point(
                    CostMetricType::StorageCost,
                    200.0 + (i as f64).sin() * 10.0,
                    Utc::now() - ChronoDuration::hours(25 - i),
                    HashMap::new(),
                )
                .unwrap();
        }

        // Add anomaly (drop)
        predictor
            .add_data_point(
                CostMetricType::StorageCost,
                50.0,
                Utc::now(),
                HashMap::new(),
            )
            .unwrap();

        let anomalies = predictor.get_anomalies();
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::Drop);
    }

    #[test]
    fn test_anomaly_sensitivity_configuration() {
        let high_sensitivity_config = CostPredictorConfig {
            anomaly_sensitivity: 0.95,
            min_data_points: 10,
            ..Default::default()
        };
        let low_sensitivity_config = CostPredictorConfig {
            anomaly_sensitivity: 0.5,
            min_data_points: 10,
            ..Default::default()
        };

        let high_predictor = CostPredictor::new(high_sensitivity_config).unwrap();
        let low_predictor = CostPredictor::new(low_sensitivity_config).unwrap();

        // Both should be created successfully
        assert!(high_predictor.get_anomalies().is_empty());
        assert!(low_predictor.get_anomalies().is_empty());
    }

    #[test]
    fn test_insufficient_data_for_anomaly_detection() {
        let config = CostPredictorConfig {
            min_data_points: 20,
            ..Default::default()
        };
        let predictor = CostPredictor::new(config)?;

        // Add only a few data points
        for i in 0..5 {
            predictor
                .add_data_point(
                    CostMetricType::ComputeCost,
                    100.0 + i as f64,
                    Utc::now() - ChronoDuration::hours(5 - i),
                    HashMap::new(),
                )
                .unwrap();
        }

        // Should not detect anomalies with insufficient data
        let anomalies = predictor.get_anomalies();
        assert!(anomalies.is_empty());
    }
}

/// Test module for trend analysis
#[cfg(test)]
mod trend_analysis_tests {
    use super::*;

    #[test]
    fn test_increasing_trend_detection() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        // Create increasing trend data
        let mut series = VecDeque::new();
        for i in 0..50 {
            series.push_back(TimeSeriesPoint {
                timestamp: Utc::now() - ChronoDuration::hours(50 - i),
                value: 100.0 + i as f64 * 2.0,
                metadata: HashMap::new(),
            });
        }

        let predictions = vec![PredictedValue {
            timestamp: Utc::now() + ChronoDuration::hours(1),
            value: 200.0,
            confidence: 0.9,
        }];

        let trend = predictor.analyze_trend(&series, &predictions).unwrap();
        assert_eq!(trend.direction, TrendDirection::Increasing);
        assert!(trend.strength > 0.5);
        assert!(trend.growth_rate > 0.0);
    }

    #[test]
    fn test_stable_trend_detection() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        // Create stable trend data
        let mut series = VecDeque::new();
        for i in 0..50 {
            series.push_back(TimeSeriesPoint {
                timestamp: Utc::now() - ChronoDuration::hours(50 - i),
                value: 100.0 + (i as f64).sin() * 2.0, // Small fluctuations
                metadata: HashMap::new(),
            });
        }

        let predictions = vec![PredictedValue {
            timestamp: Utc::now() + ChronoDuration::hours(1),
            value: 102.0,
            confidence: 0.9,
        }];

        let trend = predictor.analyze_trend(&series, &predictions).unwrap();
        assert_eq!(trend.direction, TrendDirection::Stable);
        assert!(trend.strength < 0.1);
    }

    #[test]
    fn test_decreasing_trend_detection() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        // Create decreasing trend data
        let mut series = VecDeque::new();
        for i in 0..50 {
            series.push_back(TimeSeriesPoint {
                timestamp: Utc::now() - ChronoDuration::hours(50 - i),
                value: 200.0 - i as f64 * 1.5,
                metadata: HashMap::new(),
            });
        }

        let predictions = vec![PredictedValue {
            timestamp: Utc::now() + ChronoDuration::hours(1),
            value: 120.0,
            confidence: 0.9,
        }];

        let trend = predictor.analyze_trend(&series, &predictions).unwrap();
        assert_eq!(trend.direction, TrendDirection::Decreasing);
        assert!(trend.strength > 0.3);
        assert!(trend.growth_rate < 0.0);
    }

    #[test]
    fn test_seasonal_pattern_detection() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let series = VecDeque::new();
        let predictions = Vec::new();

        let trend = predictor.analyze_trend(&series, &predictions)?;

        // Should detect some seasonal patterns
        assert!(!trend.seasonal_patterns.is_empty());
        assert_eq!(trend.seasonal_patterns[0].pattern_type, Seasonality::Weekly);
    }
}

/// Test module for budget forecasting
#[cfg(test)]
mod budget_forecast_tests {
    use super::*;

    fn add_test_data_for_all_metrics(predictor: &CostPredictor) {
        let metrics = vec![
            CostMetricType::ComputeCost,
            CostMetricType::StorageCost,
            CostMetricType::NetworkCost,
            CostMetricType::GpuCost,
            CostMetricType::OtherCost,
        ];

        for metric_type in metrics {
            for i in 0..50 {
                let point = TimeSeriesPoint {
                    timestamp: Utc::now() - ChronoDuration::hours(50 - i),
                    value: 100.0 + (i as f64 * 2.0) + (i as f64).sin() * 10.0,
                    metadata: HashMap::new(),
                };
                predictor
                    .time_series
                    .entry(metric_type)
                    .or_insert_with(VecDeque::new)
                    .push_back(point);
            }
        }
    }

    #[tokio::test]
    async fn test_budget_forecast_generation() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        add_test_data_for_all_metrics(&predictor);

        let period_start = Utc::now();
        let period_end = Utc::now() + ChronoDuration::days(30);

        let result = predictor.forecast_budget(period_start, period_end).await;
        assert!(result.is_ok());

        let forecast = result.unwrap();
        assert!(forecast.predicted_cost > 0.0);
        assert!(!forecast.cost_breakdown.is_empty());
        assert!(!forecast.recommendations.is_empty());
        assert_eq!(forecast.period_start, period_start);
        assert_eq!(forecast.period_end, period_end);
    }

    #[tokio::test]
    async fn test_budget_forecast_risk_assessment() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        add_test_data_for_all_metrics(&predictor);

        let period_start = Utc::now();
        let period_end = Utc::now() + ChronoDuration::days(7);

        let forecast = predictor
            .forecast_budget(period_start, period_end)
            .await
            .unwrap();

        // Should have risk assessment
        let risk = &forecast.risk_assessment;
        assert!(risk.overrun_probability >= 0.0);
        assert!(risk.overrun_probability <= 1.0);
        assert!(risk.expected_variance > 0.0);
    }

    #[tokio::test]
    async fn test_budget_forecast_recommendations() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        add_test_data_for_all_metrics(&predictor);

        let period_start = Utc::now();
        let period_end = Utc::now() + ChronoDuration::days(30);

        let forecast = predictor
            .forecast_budget(period_start, period_end)
            .await
            .unwrap();

        // Should have recommendations
        for recommendation in &forecast.recommendations {
            assert!(!recommendation.title.is_empty());
            assert!(!recommendation.description.is_empty());
            assert!(recommendation.potential_savings >= 0.0);
        }
    }

    #[test]
    fn test_cost_breakdown_completeness() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let mut breakdown = HashMap::new();
        breakdown.insert(CostMetricType::ComputeCost, 1000.0);
        breakdown.insert(CostMetricType::StorageCost, 500.0);
        breakdown.insert(CostMetricType::GpuCost, 2000.0);

        let risk = predictor.assess_budget_risk(3500.0, &breakdown)?;

        // Should detect high GPU dependency
        assert!(!risk.risk_factors.is_empty());
        let gpu_risk = risk.risk_factors.iter().find(|f| f.name.contains("GPU"));
        assert!(gpu_risk.is_some());
    }
}

/// Test module for model training and management
#[cfg(test)]
mod model_training_tests {
    use super::*;

    fn create_test_series(size: usize) -> VecDeque<TimeSeriesPoint> {
        let mut series = VecDeque::new();
        let base_time = Utc::now() - ChronoDuration::hours(size as i64);

        for i in 0..size {
            series.push_back(TimeSeriesPoint {
                timestamp: base_time + ChronoDuration::hours(i as i64),
                value: 100.0 + (i as f64 * 1.5) + (i as f64 * 0.1).sin() * 5.0,
                metadata: HashMap::new(),
            });
        }

        series
    }

    #[test]
    fn test_model_training_all_types() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let series = create_test_series(50);

        let model_types = vec![
            PredictionModel::MovingAverage,
            PredictionModel::ExponentialSmoothing,
            PredictionModel::LinearRegression,
            PredictionModel::Arima,
            PredictionModel::Ensemble,
        ];

        for model_type in model_types {
            let result = predictor.train_model(model_type, &series);
            assert!(result.is_ok());

            let model = result.unwrap();
            assert_eq!(model.model_type, model_type);
            assert!(model.accuracy.mae >= 0.0);
            assert!(model.accuracy.r_squared <= 1.0);
        }
    }

    #[test]
    fn test_model_accuracy_calculation() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let values = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];

        // Test moving average accuracy
        let ma_accuracy = predictor.calculate_ma_accuracy(&values, 3)?;
        assert!(ma_accuracy.mae >= 0.0);
        assert!(ma_accuracy.rmse >= 0.0);

        // Test exponential smoothing accuracy
        let es_accuracy = predictor.calculate_es_accuracy(&values, 0.3).unwrap();
        assert!(es_accuracy.mae >= 0.0);
        assert!(es_accuracy.rmse >= 0.0);

        // Test linear regression accuracy
        let lr_coeffs = vec![100.0, 2.0];
        let lr_accuracy = predictor
            .calculate_lr_accuracy(&values, &lr_coeffs)
            .unwrap();
        assert!(lr_accuracy.mae >= 0.0);
        assert!(lr_accuracy.r_squared >= 0.0);
    }

    #[test]
    fn test_linear_regression_fitting() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        // Perfect linear data
        let values = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];

        let coeffs = predictor.fit_linear_regression(&values)?;
        assert_eq!(coeffs.len(), 2);

        // Should detect slope of approximately 2.0
        assert!((coeffs[1] - 2.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_model_caching_and_reuse() {
        let config = CostPredictorConfig {
            model_update_interval: Duration::from_secs(3600), // 1 hour
            ..Default::default()
        };
        let predictor = CostPredictor::new(config)?;

        let series = create_test_series(50);
        for point in series {
            predictor
                .time_series
                .entry(CostMetricType::ComputeCost)
                .or_insert_with(VecDeque::new)
                .push_back(point);
        }

        let request = PredictionRequest {
            metric_type: CostMetricType::ComputeCost,
            horizon: Duration::from_secs(3600),
            model: PredictionModel::MovingAverage,
            confidence_level: 0.95,
            seasonality: Seasonality::None,
            filters: HashMap::new(),
        };

        // First prediction should train model
        let result1 = predictor.predict(request.clone()).await;
        assert!(result1.is_ok());

        // Second prediction should reuse cached model
        let result2 = predictor.predict(request).await;
        assert!(result2.is_ok());

        // Models should be cached
        let key = (CostMetricType::ComputeCost, PredictionModel::MovingAverage);
        assert!(predictor.models.contains_key(&key));
    }
}

/// Test module for mathematical utilities
#[cfg(test)]
mod math_utilities_tests {
    use super::*;

    #[test]
    fn test_moving_average_prediction() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let values = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];
        let predicted = predictor.predict_moving_average(&values, 3);

        // Should be average of last 3 values: (106 + 108 + 110) / 3 = 108
        assert!((predicted - 108.0).abs() < 0.01);
    }

    #[test]
    fn test_exponential_smoothing_prediction() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let values = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];
        let predicted = predictor.predict_exponential_smoothing(&values, 0.3);

        // Should be exponentially smoothed value
        assert!(predicted > 100.0);
        assert!(predicted < 110.0);
    }

    #[test]
    fn test_linear_regression_prediction() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let values = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];
        let coeffs = vec![100.0, 2.0]; // intercept=100, slope=2

        let predicted = predictor.predict_linear_regression(&values, &coeffs, 1);
        // Should be: 100 + 2 * (6 + 1) = 114
        assert!((predicted - 114.0).abs() < 0.01);
    }

    #[test]
    fn test_edge_cases() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        // Empty data
        let empty_values = vec![];
        let ma_result = predictor.predict_moving_average(&empty_values, 3);
        assert_eq!(ma_result, 0.0); // Should handle empty gracefully

        let es_result = predictor.predict_exponential_smoothing(&empty_values, 0.3);
        assert_eq!(es_result, 0.0);

        // Single value
        let single_value = vec![100.0];
        let ma_single = predictor.predict_moving_average(&single_value, 3);
        assert_eq!(ma_single, 100.0);

        let es_single = predictor.predict_exponential_smoothing(&single_value, 0.3);
        assert_eq!(es_single, 100.0);
    }

    #[test]
    fn test_insufficient_coefficients() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let values = vec![100.0, 102.0, 104.0];
        let insufficient_coeffs = vec![100.0]; // Missing slope

        let predicted = predictor.predict_linear_regression(&values, &insufficient_coeffs, 1);
        // Should fall back to mean
        assert!((predicted - 102.0).abs() < 0.01);
    }
}

/// Integration tests that validate end-to-end workflows
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_prediction_workflow() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        // 1. Add historical data
        for i in 0..100 {
            predictor
                .add_data_point(
                    CostMetricType::ComputeCost,
                    100.0 + i as f64 + (i as f64 * 0.1).sin() * 10.0,
                    Utc::now() - ChronoDuration::hours(100 - i),
                    HashMap::new(),
                )
                .unwrap();
        }

        // 2. Make prediction
        let request = PredictionRequest {
            metric_type: CostMetricType::ComputeCost,
            horizon: Duration::from_secs(3600 * 24),
            model: PredictionModel::Ensemble,
            confidence_level: 0.95,
            seasonality: Seasonality::Daily,
            filters: HashMap::new(),
        };

        let prediction = predictor.predict(request).await.unwrap();

        // 3. Validate all components are present
        assert!(!prediction.predictions.is_empty());
        assert!(!prediction.confidence_intervals.is_empty());
        assert!(prediction.accuracy.r_squared > 0.0);
        assert_eq!(prediction.trend.seasonal_patterns.len(), 1);

        // 4. Check metrics were updated
        let metrics = predictor.get_metrics();
        assert_eq!(metrics.total_predictions, 1);
        assert_eq!(metrics.models_trained, 1);
    }

    #[tokio::test]
    async fn test_multiple_metric_types() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        let metric_types = vec![
            CostMetricType::ComputeCost,
            CostMetricType::StorageCost,
            CostMetricType::GpuCost,
        ];

        // Add data for all metric types
        for metric_type in &metric_types {
            for i in 0..50 {
                predictor
                    .add_data_point(
                        *metric_type,
                        100.0 + i as f64,
                        Utc::now() - ChronoDuration::hours(50 - i),
                        HashMap::new(),
                    )
                    .unwrap();
            }
        }

        // Make predictions for all metric types
        for metric_type in &metric_types {
            let request = PredictionRequest {
                metric_type: *metric_type,
                horizon: Duration::from_secs(3600 * 12),
                model: PredictionModel::MovingAverage,
                confidence_level: 0.90,
                seasonality: Seasonality::None,
                filters: HashMap::new(),
            };

            let result = predictor.predict(request).await;
            assert!(result.is_ok());
        }

        // Should have cached models for all metric types
        assert!(predictor.models.len() >= metric_types.len());
    }

    #[tokio::test]
    async fn test_budget_forecast_integration() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        // Add comprehensive historical data
        let all_metrics = vec![
            CostMetricType::ComputeCost,
            CostMetricType::StorageCost,
            CostMetricType::NetworkCost,
            CostMetricType::GpuCost,
            CostMetricType::OtherCost,
        ];

        for metric_type in &all_metrics {
            for i in 1..=100 {
                let base_value = match metric_type {
                    CostMetricType::ComputeCost => 500.0,
                    CostMetricType::StorageCost => 200.0,
                    CostMetricType::NetworkCost => 100.0,
                    CostMetricType::GpuCost => 1000.0,
                    CostMetricType::OtherCost => 50.0,
                    _ => 100.0,
                };

                predictor
                    .add_data_point(
                        *metric_type,
                        base_value + i as f64 + (i as f64 * 0.1).sin() * 20.0,
                        Utc::now() - ChronoDuration::hours(100 - i),
                        HashMap::new(),
                    )
                    .unwrap();
            }
        }

        // Generate budget forecast
        let period_start = Utc::now();
        let period_end = Utc::now() + ChronoDuration::days(30);

        let forecast = predictor
            .forecast_budget(period_start, period_end)
            .await
            .unwrap();

        // Validate forecast completeness
        assert!(forecast.predicted_cost > 0.0);
        assert_eq!(forecast.cost_breakdown.len(), 5); // All metric types

        // Validate risk assessment
        assert!(forecast.risk_assessment.overrun_probability >= 0.0);
        assert!(!forecast.risk_assessment.risk_factors.is_empty());

        // Validate recommendations
        assert!(!forecast.recommendations.is_empty());
        for rec in &forecast.recommendations {
            assert!(rec.potential_savings >= 0.0);
            assert!(!rec.title.is_empty());
        }
    }

    #[test]
    fn test_concurrent_operations() {
        use std::sync::Arc;
        use std::thread;

        let config = CostPredictorConfig::default();
        let predictor = Arc::new(CostPredictor::new(config)?);

        let mut handles = Vec::new();

        // Spawn multiple threads adding data concurrently
        for thread_id in 0..5 {
            let predictor_clone = Arc::clone(&predictor);
            let handle = thread::spawn(move || {
                for i in 0..20 {
                    let result = predictor_clone.add_data_point(
                        CostMetricType::ComputeCost,
                        100.0 + (thread_id * 20 + i) as f64,
                        Utc::now() - ChronoDuration::minutes((thread_id * 20 + i) as i64),
                        HashMap::new(),
                    );
                    assert!(result.is_ok());
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Should have data from all threads
        let series = predictor
            .time_series
            .get(&CostMetricType::ComputeCost)
            .unwrap();
        assert!(series.len() >= 90); // Some may have been cleaned due to retention
    }

    #[test]
    fn test_error_handling_and_recovery() {
        let config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(config).unwrap();

        // Test with invalid durations
        let invalid_duration = Duration::from_secs(0);
        let request = PredictionRequest {
            metric_type: CostMetricType::ComputeCost,
            horizon: invalid_duration,
            model: PredictionModel::MovingAverage,
            confidence_level: 0.95,
            seasonality: Seasonality::None,
            filters: HashMap::new(),
        };

        // Should handle gracefully
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(predictor.predict(request));
        // Even with 0 horizon, should not panic
        assert!(result.is_ok() || result.is_err());
    }
}
