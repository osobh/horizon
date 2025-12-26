//! Data analysis and trend detection functionality

use super::report_types::{
    MetricPrediction, MetricTrend, RegressionDetail, RegressionSeverity, TrendAnomaly,
    TrendDirection,
};
use crate::metrics_collector::{MetricDataPoint, MetricType};
use chrono::Duration;
use std::collections::HashMap;

/// Analyze trend in metric data
pub fn analyze_trend(metric_type: &MetricType, data: &[MetricDataPoint]) -> Option<MetricTrend> {
    if data.len() < 2 {
        return None;
    }

    // Simple linear regression for trend analysis
    let n = data.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;

    let start_time = data[0].timestamp.timestamp() as f64;

    for metric in data.iter() {
        let x = (metric.timestamp.timestamp() as f64 - start_time) / 3600.0; // Hours from start
        let y = metric.value.0;

        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let r_squared = ((n * sum_xy - sum_x * sum_y)
        / ((n * sum_x2 - sum_x * sum_x).sqrt() * (n * sum_y * sum_y - sum_y * sum_y).sqrt()))
    .powi(2);

    let direction = if slope.abs() < 0.01 {
        TrendDirection::Stable
    } else if slope > 0.0 {
        TrendDirection::Increasing
    } else {
        TrendDirection::Decreasing
    };

    let duration = data.last().unwrap().timestamp - data.first().unwrap().timestamp;

    Some(MetricTrend {
        metric_type: metric_type.clone(),
        direction,
        strength: slope.abs().min(1.0),
        duration,
        confidence: r_squared,
    })
}

/// Create prediction for metric
pub fn create_prediction(
    metric_type: &MetricType,
    data: &[MetricDataPoint],
    horizon_hours: f64,
) -> Option<MetricPrediction> {
    if data.len() < 3 {
        return None;
    }

    // Simple linear extrapolation
    let n = data.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;

    let start_time = data[0].timestamp.timestamp() as f64;

    for metric in data.iter() {
        let x = (metric.timestamp.timestamp() as f64 - start_time) / 3600.0;
        let y = metric.value.0;

        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    let last_time = data.last().unwrap().timestamp;
    let prediction_time = last_time + Duration::hours(horizon_hours as i64);
    let x_pred = (prediction_time.timestamp() as f64 - start_time) / 3600.0;
    let y_pred = slope * x_pred + intercept;

    // Simple confidence interval (±10% for demonstration)
    let confidence_margin = y_pred * 0.1;

    Some(MetricPrediction {
        metric_type: metric_type.clone(),
        timestamp: prediction_time,
        predicted_value: y_pred,
        confidence_interval: (y_pred - confidence_margin, y_pred + confidence_margin),
        method: "linear_regression".to_string(),
    })
}

/// Detect anomalies in metric data
pub fn detect_anomalies(metric_type: &MetricType, data: &[MetricDataPoint]) -> Vec<TrendAnomaly> {
    if data.len() < 5 {
        return Vec::new();
    }

    let mut anomalies = Vec::new();

    // Calculate rolling statistics
    let window_size = 5;
    for i in window_size..data.len() {
        let window = &data[i - window_size..i];
        let mean: f64 = window.iter().map(|m| m.value.0).sum::<f64>() / window_size as f64;
        let std_dev: f64 = {
            let variance = window
                .iter()
                .map(|m| (m.value.0 - mean).powi(2))
                .sum::<f64>()
                / window_size as f64;
            variance.sqrt()
        };

        let current = &data[i];
        let z_score = if std_dev > 0.0 {
            (current.value.0 - mean).abs() / std_dev
        } else {
            0.0
        };

        // Flag as anomaly if z-score > 2.5
        if z_score > 2.5 {
            anomalies.push(TrendAnomaly {
                metric_type: metric_type.clone(),
                timestamp: current.timestamp,
                actual_value: current.value.0,
                expected_value: mean,
                score: z_score / 3.0, // Normalize to 0-1 range
            });
        }
    }

    anomalies
}

/// Calculate health score based on regressions
pub fn calculate_health_score(
    regressions_by_metric: &HashMap<MetricType, Vec<RegressionDetail>>,
) -> f64 {
    if regressions_by_metric.is_empty() {
        return 1.0;
    }

    let mut total_weight = 0.0;
    let mut weighted_score = 0.0;

    for (_metric_type, regressions) in regressions_by_metric {
        for regression in regressions {
            let weight = match regression.severity {
                RegressionSeverity::Low => 0.25,
                RegressionSeverity::Medium => 0.5,
                RegressionSeverity::High => 0.75,
                RegressionSeverity::Critical => 1.0,
            };

            let impact = (regression.degradation_percent / 100.0).min(1.0);
            weighted_score += weight * impact;
            total_weight += weight;
        }
    }

    if total_weight > 0.0 {
        1.0 - (weighted_score / total_weight).min(1.0)
    } else {
        1.0
    }
}

/// Generate recommendations based on regressions
pub fn generate_recommendations(
    regressions_by_metric: &HashMap<MetricType, Vec<RegressionDetail>>,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    for (metric_type, regressions) in regressions_by_metric {
        let critical_count = regressions
            .iter()
            .filter(|r| r.severity == RegressionSeverity::Critical)
            .count();

        if critical_count > 0 {
            recommendations.push(format!(
                "URGENT: {} critical regressions detected for {:?}. Immediate investigation required.",
                critical_count, metric_type
            ));
        }

        let avg_degradation = regressions
            .iter()
            .map(|r| r.degradation_percent)
            .sum::<f64>()
            / regressions.len() as f64;

        if avg_degradation > 20.0 {
            recommendations.push(format!(
                "Performance for {:?} has degraded by {:.1}% on average. Consider optimization or scaling.",
                metric_type, avg_degradation
            ));
        }
    }

    if recommendations.is_empty() {
        recommendations.push("No significant performance issues detected.".to_string());
    }

    recommendations
}

/// Get unique metric types from data
pub fn get_unique_metric_types(metrics_data: &[MetricDataPoint]) -> Vec<String> {
    let mut types = std::collections::HashSet::new();
    for metric in metrics_data {
        types.insert(format!("{:?}", metric.metric_type));
    }
    types.into_iter().collect()
}

/// Aggregate chart data to limit points
pub fn aggregate_chart_data(
    metrics_data: &[MetricDataPoint],
    max_points: usize,
) -> Vec<MetricDataPoint> {
    if metrics_data.len() <= max_points {
        return metrics_data.to_vec();
    }

    // Simple downsampling - take every nth point
    let step = metrics_data.len() / max_points;
    metrics_data.iter().step_by(step.max(1)).cloned().collect()
}
