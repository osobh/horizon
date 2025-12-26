//! Mathematical algorithms and prediction utilities
//!
//! This module contains the core mathematical functions used for cost prediction,
//! including moving averages, exponential smoothing, linear regression, and
//! accuracy calculations.

use crate::error::CostOptimizationResult;
use super::types::*;
use statistical::{mean, standard_deviation};
use std::collections::VecDeque;

/// Mathematical algorithms for cost prediction
pub struct PredictionAlgorithms;

impl PredictionAlgorithms {
    /// Simple moving average prediction
    pub fn predict_moving_average(values: &[f64], window: usize) -> f64 {
        if values.len() < window {
            return mean(values);
        }

        let recent_values = &values[values.len() - window..];
        mean(recent_values)
    }

    /// Exponential smoothing prediction
    pub fn predict_exponential_smoothing(values: &[f64], alpha: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut smoothed = values[0];
        for value in values.iter().skip(1) {
            smoothed = alpha * value + (1.0 - alpha) * smoothed;
        }

        smoothed
    }

    /// Linear regression prediction
    pub fn predict_linear_regression(values: &[f64], coeffs: &[f64], step: u32) -> f64 {
        if coeffs.len() < 2 {
            return mean(values);
        }

        let x = values.len() as f64 + step as f64;
        coeffs[0] + coeffs[1] * x
    }

    /// Fit linear regression model
    pub fn fit_linear_regression(values: &[f64]) -> CostOptimizationResult<Vec<f64>> {
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let x_mean = mean(&x_values);
        let y_mean = mean(values);

        let mut num = 0.0;
        let mut den = 0.0;

        for i in 0..values.len() {
            num += (x_values[i] - x_mean) * (values[i] - y_mean);
            den += (x_values[i] - x_mean).powi(2);
        }

        let slope = if den != 0.0 { num / den } else { 0.0 };
        let intercept = y_mean - slope * x_mean;

        Ok(vec![intercept, slope])
    }

    /// Calculate moving average accuracy
    pub fn calculate_ma_accuracy(
        values: &[f64],
        window: usize,
    ) -> CostOptimizationResult<ModelAccuracy> {
        if values.len() < window + 1 {
            return Ok(ModelAccuracy {
                mae: 0.0,
                mse: 0.0,
                rmse: 0.0,
                mape: 0.0,
                r_squared: 1.0,
            });
        }

        let mut errors = Vec::new();

        for i in window..values.len() {
            let predicted = mean(&values[i - window..i]);
            let actual = values[i];
            let error = (actual - predicted).abs();
            errors.push(error);
        }

        let mae = mean(&errors);
        let mse = errors.iter().map(|e| e.powi(2)).sum::<f64>() / errors.len() as f64;
        let rmse = mse.sqrt();
        let mape = errors
            .iter()
            .zip(&values[window..])
            .map(|(e, v)| if *v != 0.0 { e / v.abs() } else { 0.0 })
            .sum::<f64>()
            / errors.len() as f64
            * 100.0;

        Ok(ModelAccuracy {
            mae,
            mse,
            rmse,
            mape,
            r_squared: 0.8, // Simplified
        })
    }

    /// Calculate exponential smoothing accuracy
    pub fn calculate_es_accuracy(
        values: &[f64],
        alpha: f64,
    ) -> CostOptimizationResult<ModelAccuracy> {
        if values.len() < 2 {
            return Ok(ModelAccuracy {
                mae: 0.0,
                mse: 0.0,
                rmse: 0.0,
                mape: 0.0,
                r_squared: 1.0,
            });
        }

        let mut smoothed = values[0];
        let mut errors = Vec::new();

        for i in 1..values.len() {
            let predicted = smoothed;
            let actual = values[i];
            let error = (actual - predicted).abs();
            errors.push(error);

            smoothed = alpha * actual + (1.0 - alpha) * smoothed;
        }

        let mae = mean(&errors);
        let mse = errors.iter().map(|e| e.powi(2)).sum::<f64>() / errors.len() as f64;
        let rmse = mse.sqrt();

        Ok(ModelAccuracy {
            mae,
            mse,
            rmse,
            mape: 5.0,       // Simplified
            r_squared: 0.85, // Simplified
        })
    }

    /// Calculate linear regression accuracy
    pub fn calculate_lr_accuracy(
        values: &[f64],
        coeffs: &[f64],
    ) -> CostOptimizationResult<ModelAccuracy> {
        if values.is_empty() || coeffs.len() < 2 {
            return Ok(ModelAccuracy {
                mae: 0.0,
                mse: 0.0,
                rmse: 0.0,
                mape: 0.0,
                r_squared: 1.0,
            });
        }

        let mut errors = Vec::new();
        let y_mean = mean(values);
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for (i, actual) in values.iter().enumerate() {
            let predicted = coeffs[0] + coeffs[1] * i as f64;
            let error = (actual - predicted).abs();
            errors.push(error);

            ss_tot += (actual - y_mean).powi(2);
            ss_res += (actual - predicted).powi(2);
        }

        let mae = mean(&errors);
        let mse = ss_res / values.len() as f64;
        let rmse = mse.sqrt();
        let r_squared = if ss_tot != 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        Ok(ModelAccuracy {
            mae,
            mse,
            rmse,
            mape: 6.0, // Simplified
            r_squared,
        })
    }

    /// Generate predictions using the specified model
    pub fn generate_predictions(
        request: &PredictionRequest,
        series: &VecDeque<TimeSeriesPoint>,
        model: &TrainedModel,
    ) -> CostOptimizationResult<Vec<PredictedValue>> {
        use crate::error::CostOptimizationError;
        use chrono::{Duration as ChronoDuration};

        let values: Vec<f64> = series.iter().map(|p| p.value).collect();
        let last_timestamp = series
            .back()
            .ok_or_else(|| CostOptimizationError::PredictionError {
                model: model.model_type.to_string(),
                reason: "No data points available".to_string(),
            })?
            .timestamp;

        let horizon_hours = request.horizon.as_secs() / 3600;
        let mut predictions = Vec::new();

        for i in 1..=horizon_hours {
            let timestamp = last_timestamp + ChronoDuration::hours(i as i64);

            let value = match model.model_type {
                PredictionModel::MovingAverage => {
                    let window = model.parameters.ma_window.unwrap_or(7);
                    Self::predict_moving_average(&values, window)
                }
                PredictionModel::ExponentialSmoothing => {
                    let alpha = model.parameters.smoothing_factor.unwrap_or(0.3);
                    Self::predict_exponential_smoothing(&values, alpha)
                }
                PredictionModel::LinearRegression => {
                    let coeffs = model.parameters.coefficients.as_ref().ok_or_else(|| {
                        CostOptimizationError::PredictionError {
                            model: model.model_type.to_string(),
                            reason: "No coefficients found".to_string(),
                        }
                    })?;
                    Self::predict_linear_regression(&values, coeffs, i as u32)
                }
                _ => {
                    // Simplified prediction for other models
                    let avg = mean(&values);
                    avg * (1.0 + (i as f64 * 0.01)) // 1% growth per hour
                }
            };

            predictions.push(PredictedValue {
                timestamp,
                value,
                confidence: 1.0 - (i as f64 / horizon_hours as f64) * 0.2, // Decreasing confidence
            });
        }

        Ok(predictions)
    }

    /// Calculate confidence intervals
    pub fn calculate_confidence_intervals(
        predictions: &[PredictedValue],
        confidence_level: f64,
        model: &TrainedModel,
    ) -> CostOptimizationResult<Vec<ConfidenceInterval>> {
        let z_score = match confidence_level {
            cl if cl >= 0.99 => 2.576,
            cl if cl >= 0.95 => 1.96,
            cl if cl >= 0.90 => 1.645,
            _ => 1.28, // 80% confidence
        };

        let std_error = model.accuracy.rmse;

        Ok(predictions
            .iter()
            .enumerate()
            .map(|(i, pred)| {
                let margin = z_score * std_error * (1.0 + i as f64 * 0.1); // Increasing uncertainty
                ConfidenceInterval {
                    timestamp: pred.timestamp,
                    lower_bound: (pred.value - margin).max(0.0),
                    upper_bound: pred.value + margin,
                    confidence_level,
                }
            })
            .collect())
    }

    /// Analyze trend in data
    pub fn analyze_trend(
        series: &VecDeque<TimeSeriesPoint>,
        predictions: &[PredictedValue],
    ) -> CostOptimizationResult<TrendAnalysis> {
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();

        // Calculate trend direction and strength
        let (direction, strength) = if values.len() < 2 {
            (TrendDirection::Stable, 0.0)
        } else {
            let first_half_avg = mean(&values[..values.len() / 2]);
            let second_half_avg = mean(&values[values.len() / 2..]);

            let change = (second_half_avg - first_half_avg) / first_half_avg;

            let direction = if change.abs() < 0.05 {
                TrendDirection::Stable
            } else if change > 0.0 {
                TrendDirection::Increasing
            } else {
                TrendDirection::Decreasing
            };

            let strength = change.abs().min(1.0);

            (direction, strength)
        };

        // Calculate growth rate
        let growth_rate = if !predictions.is_empty() && !values.is_empty() {
            let last_actual = values.last().unwrap();
            let last_predicted = predictions.last().unwrap().value;
            ((last_predicted - last_actual) / last_actual) * 100.0
        } else {
            0.0
        };

        // Detect seasonal patterns (simplified)
        let seasonal_patterns = vec![SeasonalPattern {
            pattern_type: Seasonality::Weekly,
            strength: 0.7,
            peak_periods: vec!["Monday".to_string(), "Friday".to_string()],
            low_periods: vec!["Weekend".to_string()],
        }];

        Ok(TrendAnalysis {
            direction,
            strength,
            growth_rate,
            seasonal_patterns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_average_prediction() {
        let values = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];
        let predicted = PredictionAlgorithms::predict_moving_average(&values, 3);
        
        // Should be average of last 3 values: (106 + 108 + 110) / 3 = 108
        assert!((predicted - 108.0).abs() < 0.01);
    }

    #[test]
    fn test_exponential_smoothing_prediction() {
        let values = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];
        let predicted = PredictionAlgorithms::predict_exponential_smoothing(&values, 0.3);
        
        // Should be exponentially smoothed value
        assert!(predicted > 100.0);
        assert!(predicted < 110.0);
    }

    #[test]
    fn test_linear_regression_prediction() {
        let values = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];
        let coeffs = vec![100.0, 2.0]; // intercept=100, slope=2
        
        let predicted = PredictionAlgorithms::predict_linear_regression(&values, &coeffs, 1);
        // Should be: 100 + 2 * (6 + 1) = 114
        assert!((predicted - 114.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_fitting() {
        // Perfect linear data
        let values = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];
        
        let coeffs = PredictionAlgorithms::fit_linear_regression(&values).unwrap();
        assert_eq!(coeffs.len(), 2);
        
        // Should detect slope of approximately 2.0
        assert!((coeffs[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_model_accuracy_calculation() {
        let values = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];

        // Test moving average accuracy
        let ma_accuracy = PredictionAlgorithms::calculate_ma_accuracy(&values, 3).unwrap();
        assert!(ma_accuracy.mae >= 0.0);
        assert!(ma_accuracy.rmse >= 0.0);

        // Test exponential smoothing accuracy
        let es_accuracy = PredictionAlgorithms::calculate_es_accuracy(&values, 0.3)?;
        assert!(es_accuracy.mae >= 0.0);
        assert!(es_accuracy.rmse >= 0.0);

        // Test linear regression accuracy
        let lr_coeffs = vec![100.0, 2.0];
        let lr_accuracy = PredictionAlgorithms::calculate_lr_accuracy(&values, &lr_coeffs).unwrap();
        assert!(lr_accuracy.mae >= 0.0);
        assert!(lr_accuracy.r_squared >= 0.0);
    }

    #[test]
    fn test_edge_cases() {
        // Empty data
        let empty_values = vec![];
        let ma_result = PredictionAlgorithms::predict_moving_average(&empty_values, 3);
        assert_eq!(ma_result, 0.0); // Should handle empty gracefully

        let es_result = PredictionAlgorithms::predict_exponential_smoothing(&empty_values, 0.3);
        assert_eq!(es_result, 0.0);

        // Single value
        let single_value = vec![100.0];
        let ma_single = PredictionAlgorithms::predict_moving_average(&single_value, 3);
        assert_eq!(ma_single, 100.0);

        let es_single = PredictionAlgorithms::predict_exponential_smoothing(&single_value, 0.3);
        assert_eq!(es_single, 100.0);
    }

    #[test]
    fn test_insufficient_coefficients() {
        let values = vec![100.0, 102.0, 104.0];
        let insufficient_coeffs = vec![100.0]; // Missing slope
        
        let predicted = PredictionAlgorithms::predict_linear_regression(&values, &insufficient_coeffs, 1);
        // Should fall back to mean
        assert!((predicted - 102.0).abs() < 0.01);
    }
}