//! Statistical analysis utilities for usage data
//!
//! This module provides utilities for calculating various statistical measures
//! from resource usage snapshots.

use crate::error::{CostOptimizationError, CostOptimizationResult};
use crate::resource_tracker::ResourceSnapshot;
use crate::usage_analyzer::types::*;
use chrono::{Datelike, Timelike};
use statistical::{mean, standard_deviation};

/// Calculate percentile of a sorted slice
pub fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    let index = (p / 100.0 * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[index.min(sorted_data.len() - 1)]
}

/// Statistics calculator for usage analysis
pub struct StatisticsCalculator {
    /// Configuration
    config: UsageAnalyzerConfig,
}

impl StatisticsCalculator {
    /// Create a new statistics calculator
    pub fn new(config: UsageAnalyzerConfig) -> Self {
        Self { config }
    }

    /// Calculate comprehensive usage statistics from snapshots
    pub fn calculate_statistics(
        &self,
        snapshots: &[ResourceSnapshot],
    ) -> CostOptimizationResult<UsageStatistics> {
        let utilizations: Vec<f64> = snapshots.iter().map(|s| s.utilization).collect();

        if utilizations.is_empty() {
            return Err(CostOptimizationError::CalculationError {
                details: "No utilization data available".to_string(),
            });
        }

        // Basic statistics
        let avg = mean(&utilizations);
        let std_dev = standard_deviation(&utilizations, Some(avg));

        // Sort for percentile calculations
        let mut sorted = utilizations.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate percentiles
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        // Calculate idle percentage
        let idle_count = utilizations
            .iter()
            .filter(|&&u| u <= self.config.idle_threshold)
            .count();
        let idle_percentage = (idle_count as f64 / utilizations.len() as f64) * 100.0;

        // Find min and max
        let peak_utilization = utilizations
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_utilization = utilizations.iter().cloned().fold(f64::INFINITY, f64::min);

        Ok(UsageStatistics {
            avg_utilization: avg,
            peak_utilization,
            min_utilization,
            std_deviation: std_dev,
            p95_utilization: p95,
            p99_utilization: p99,
            idle_percentage,
            sample_count: utilizations.len(),
        })
    }

    /// Calculate statistics for a specific time window
    pub fn calculate_windowed_statistics(
        &self,
        snapshots: &[ResourceSnapshot],
        window_start: chrono::DateTime<chrono::Utc>,
        window_end: chrono::DateTime<chrono::Utc>,
    ) -> CostOptimizationResult<UsageStatistics> {
        let windowed_snapshots: Vec<&ResourceSnapshot> = snapshots
            .iter()
            .filter(|s| s.timestamp >= window_start && s.timestamp <= window_end)
            .collect();

        let windowed_snapshots: Vec<ResourceSnapshot> =
            windowed_snapshots.into_iter().cloned().collect();

        self.calculate_statistics(&windowed_snapshots)
    }

    /// Calculate rolling statistics with a moving window
    pub fn calculate_rolling_statistics(
        &self,
        snapshots: &[ResourceSnapshot],
        window_size: usize,
    ) -> CostOptimizationResult<Vec<(chrono::DateTime<chrono::Utc>, UsageStatistics)>> {
        if snapshots.len() < window_size {
            return Err(CostOptimizationError::CalculationError {
                details: format!(
                    "Not enough data points for rolling statistics: {} < {}",
                    snapshots.len(),
                    window_size
                ),
            });
        }

        let mut rolling_stats = Vec::new();

        for i in window_size..=snapshots.len() {
            let window = &snapshots[i - window_size..i];
            let stats = self.calculate_statistics(window)?;
            let timestamp = snapshots[i - 1].timestamp; // Use end of window as timestamp
            rolling_stats.push((timestamp, stats));
        }

        Ok(rolling_stats)
    }

    /// Calculate variance across different time dimensions
    pub fn calculate_temporal_variance(
        &self,
        snapshots: &[ResourceSnapshot],
    ) -> CostOptimizationResult<TemporalVariance> {
        if snapshots.is_empty() {
            return Err(CostOptimizationError::CalculationError {
                details: "Cannot calculate temporal variance with no snapshots".to_string(),
            });
        }

        // Group by hour of day
        let mut hourly_groups: std::collections::HashMap<u32, Vec<f64>> =
            std::collections::HashMap::new();
        for snapshot in snapshots {
            let hour = snapshot.timestamp.time().hour();
            hourly_groups
                .entry(hour)
                .or_default()
                .push(snapshot.utilization);
        }

        // Calculate hourly variance
        let hourly_means: Vec<f64> = (0..24)
            .map(|hour| {
                hourly_groups
                    .get(&hour)
                    .map(|values| mean(values))
                    .unwrap_or(0.0)
            })
            .collect();

        let hourly_variance = if hourly_means.len() > 1 {
            let hourly_mean = mean(&hourly_means);
            let variance_sum: f64 = hourly_means
                .iter()
                .map(|&x| (x - hourly_mean).powi(2))
                .sum();
            variance_sum / (hourly_means.len() - 1) as f64
        } else {
            0.0
        };

        // Group by day of week
        let mut daily_groups: std::collections::HashMap<u32, Vec<f64>> =
            std::collections::HashMap::new();
        for snapshot in snapshots {
            let day = snapshot.timestamp.weekday().num_days_from_monday();
            daily_groups
                .entry(day)
                .or_default()
                .push(snapshot.utilization);
        }

        // Calculate daily variance
        let daily_means: Vec<f64> = (0..7)
            .map(|day| {
                daily_groups
                    .get(&day)
                    .map(|values| mean(values))
                    .unwrap_or(0.0)
            })
            .collect();

        let daily_variance = if daily_means.len() > 1 {
            let daily_mean = mean(&daily_means);
            let variance_sum: f64 = daily_means.iter().map(|&x| (x - daily_mean).powi(2)).sum();
            variance_sum / (daily_means.len() - 1) as f64
        } else {
            0.0
        };

        Ok(TemporalVariance {
            hourly_variance,
            daily_variance,
            overall_variance: standard_deviation(
                &snapshots.iter().map(|s| s.utilization).collect::<Vec<_>>(),
                None,
            )
            .powi(2),
        })
    }

    /// Calculate correlation between different metrics
    pub fn calculate_correlation(
        &self,
        x_values: &[f64],
        y_values: &[f64],
    ) -> CostOptimizationResult<f64> {
        if x_values.len() != y_values.len() || x_values.is_empty() {
            return Err(CostOptimizationError::CalculationError {
                details: "Invalid data for correlation calculation".to_string(),
            });
        }

        let x_mean = mean(x_values);
        let y_mean = mean(y_values);

        let mut numerator = 0.0;
        let mut x_sum_sq = 0.0;
        let mut y_sum_sq = 0.0;

        for i in 0..x_values.len() {
            let x_diff = x_values[i] - x_mean;
            let y_diff = y_values[i] - y_mean;

            numerator += x_diff * y_diff;
            x_sum_sq += x_diff.powi(2);
            y_sum_sq += y_diff.powi(2);
        }

        let denominator = (x_sum_sq * y_sum_sq).sqrt();

        if denominator == 0.0 {
            Ok(0.0) // No correlation when one variable is constant
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Detect outliers using IQR method
    pub fn detect_outliers(&self, values: &[f64]) -> (Vec<usize>, OutlierStats) {
        if values.len() < 4 {
            return (Vec::new(), OutlierStats::default());
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1 = percentile(&sorted, 25.0);
        let q3 = percentile(&sorted, 75.0);
        let iqr = q3 - q1;

        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        let mut outlier_indices = Vec::new();
        let mut outlier_count = 0;
        let mut extreme_outlier_count = 0;

        for (i, &value) in values.iter().enumerate() {
            if value < lower_bound || value > upper_bound {
                outlier_indices.push(i);
                outlier_count += 1;

                // Extreme outliers (beyond 3 * IQR)
                if value < q1 - 3.0 * iqr || value > q3 + 3.0 * iqr {
                    extreme_outlier_count += 1;
                }
            }
        }

        let stats = OutlierStats {
            total_outliers: outlier_count,
            extreme_outliers: extreme_outlier_count,
            outlier_percentage: (outlier_count as f64 / values.len() as f64) * 100.0,
            lower_bound,
            upper_bound,
            q1,
            q3,
            iqr,
        };

        (outlier_indices, stats)
    }
}

/// Temporal variance statistics
#[derive(Debug, Clone)]
pub struct TemporalVariance {
    /// Variance across hours of the day
    pub hourly_variance: f64,
    /// Variance across days of the week
    pub daily_variance: f64,
    /// Overall variance
    pub overall_variance: f64,
}

/// Outlier detection statistics
#[derive(Debug, Clone, Default)]
pub struct OutlierStats {
    /// Total number of outliers
    pub total_outliers: usize,
    /// Number of extreme outliers
    pub extreme_outliers: usize,
    /// Percentage of values that are outliers
    pub outlier_percentage: f64,
    /// Lower outlier bound
    pub lower_bound: f64,
    /// Upper outlier bound
    pub upper_bound: f64,
    /// First quartile
    pub q1: f64,
    /// Third quartile
    pub q3: f64,
    /// Interquartile range
    pub iqr: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_tracker::ResourceType;
    use chrono::Utc;
    use std::collections::HashMap;

    fn create_test_snapshots(values: Vec<f64>) -> Vec<ResourceSnapshot> {
        let base_time = Utc::now() - chrono::Duration::hours(values.len() as i64);

        values
            .into_iter()
            .enumerate()
            .map(|(i, utilization)| ResourceSnapshot {
                timestamp: base_time + chrono::Duration::hours(i as i64),
                resource_type: ResourceType::Cpu,
                resource_id: "test-resource".to_string(),
                utilization,
                available: 100.0 - utilization,
                total: 100.0,
                metadata: HashMap::new(),
            })
            .collect()
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 50.0), 5.0);
        assert_eq!(percentile(&data, 100.0), 10.0);

        // Test with empty data
        let empty_data: Vec<f64> = vec![];
        assert_eq!(percentile(&empty_data, 50.0), 0.0);
    }

    #[test]
    fn test_basic_statistics_calculation() {
        let config = UsageAnalyzerConfig::default();
        let calculator = StatisticsCalculator::new(config);

        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let snapshots = create_test_snapshots(values);

        let stats = calculator.calculate_statistics(&snapshots)?;

        assert_eq!(stats.avg_utilization, 55.0);
        assert_eq!(stats.peak_utilization, 100.0);
        assert_eq!(stats.min_utilization, 10.0);
        assert_eq!(stats.sample_count, 10);
        assert!(stats.std_deviation > 0.0);
        assert!(stats.p95_utilization > stats.avg_utilization);
        assert!(stats.p99_utilization >= stats.p95_utilization);
    }

    #[test]
    fn test_idle_percentage_calculation() {
        let config = UsageAnalyzerConfig::default();
        let calculator = StatisticsCalculator::new(config);

        // Create data where 30% of values are below idle threshold (5.0)
        let mut values = vec![1.0, 2.0, 3.0]; // 3 idle values
        values.extend(vec![10.0, 20.0, 30.0, 40.0]); // 4 non-idle values

        let snapshots = create_test_snapshots(values);
        let stats = calculator.calculate_statistics(&snapshots)?;

        let expected_idle_percentage = (3.0 / 7.0) * 100.0; // ~42.86%
        assert!((stats.idle_percentage - expected_idle_percentage).abs() < 0.01);
    }

    #[test]
    fn test_windowed_statistics() {
        let config = UsageAnalyzerConfig::default();
        let calculator = StatisticsCalculator::new(config);

        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let snapshots = create_test_snapshots(values);

        let window_start = snapshots[1].timestamp; // Include snapshots 1-3
        let window_end = snapshots[3].timestamp;

        let stats = calculator
            .calculate_windowed_statistics(&snapshots, window_start, window_end)
            .unwrap();

        // Should only include values 20.0, 30.0, 40.0
        assert_eq!(stats.sample_count, 3);
        assert_eq!(stats.avg_utilization, 30.0);
    }

    #[test]
    fn test_rolling_statistics() {
        let config = UsageAnalyzerConfig::default();
        let calculator = StatisticsCalculator::new(config);

        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let snapshots = create_test_snapshots(values);

        let rolling_stats = calculator.calculate_rolling_statistics(&snapshots, 3)?;

        // Should have 3 rolling windows: [10,20,30], [20,30,40], [30,40,50]
        assert_eq!(rolling_stats.len(), 3);

        // First window average should be (10+20+30)/3 = 20
        assert_eq!(rolling_stats[0].1.avg_utilization, 20.0);

        // Last window average should be (30+40+50)/3 = 40
        assert_eq!(rolling_stats[2].1.avg_utilization, 40.0);
    }

    #[test]
    fn test_temporal_variance() {
        let config = UsageAnalyzerConfig::default();
        let calculator = StatisticsCalculator::new(config);

        // Create snapshots with varying patterns across hours
        let mut snapshots = Vec::new();
        let base_time = Utc::now() - chrono::Duration::hours(48);

        for i in 0..48 {
            // 2 days of hourly data
            let hour = i % 24;
            let utilization = if hour >= 9 && hour <= 17 {
                80.0 // High during work hours
            } else {
                20.0 // Low during off hours
            };

            snapshots.push(ResourceSnapshot {
                timestamp: base_time + chrono::Duration::hours(i as i64),
                resource_type: ResourceType::Cpu,
                resource_id: "test-resource".to_string(),
                utilization,
                available: 100.0 - utilization,
                total: 100.0,
                metadata: HashMap::new(),
            });
        }

        let variance = calculator.calculate_temporal_variance(&snapshots).unwrap();

        // Should detect high hourly variance due to work/off-hour pattern
        assert!(variance.hourly_variance > 0.0);
        assert!(variance.overall_variance > 0.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let config = UsageAnalyzerConfig::default();
        let calculator = StatisticsCalculator::new(config);

        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let correlation = calculator.calculate_correlation(&x, &y)?;
        assert!((correlation - 1.0).abs() < 0.001);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let correlation = calculator.calculate_correlation(&x, &y_neg).unwrap();
        assert!((correlation - (-1.0)).abs() < 0.001);

        // No correlation
        let y_random = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        let correlation = calculator.calculate_correlation(&x, &y_random).unwrap();
        assert!(correlation.abs() < 0.5); // Should be close to 0
    }

    #[test]
    fn test_outlier_detection() {
        let config = UsageAnalyzerConfig::default();
        let calculator = StatisticsCalculator::new(config);

        // Normal data with a few outliers
        let values = vec![
            10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0, // Normal range
            50.0, // Outlier
            13.0, 17.0, 19.0, 2.0, // Another outlier
        ];

        let (outlier_indices, stats) = calculator.detect_outliers(&values);

        // Should detect the outliers (50.0 and 2.0)
        assert!(outlier_indices.len() > 0);
        assert!(stats.total_outliers > 0);
        assert!(stats.outlier_percentage > 0.0);
        assert!(stats.iqr > 0.0);
    }

    #[test]
    fn test_empty_data_handling() {
        let config = UsageAnalyzerConfig::default();
        let calculator = StatisticsCalculator::new(config);

        let empty_snapshots = Vec::new();
        let result = calculator.calculate_statistics(&empty_snapshots);

        assert!(result.is_err());
        match result {
            Err(CostOptimizationError::CalculationError { details }) => {
                assert!(details.contains("No utilization data available"));
            }
            _ => panic!("Expected CalculationError"),
        }
    }

    #[test]
    fn test_insufficient_data_for_rolling_stats() {
        let config = UsageAnalyzerConfig::default();
        let calculator = StatisticsCalculator::new(config);

        let values = vec![10.0, 20.0]; // Only 2 values
        let snapshots = create_test_snapshots(values);

        let result = calculator.calculate_rolling_statistics(&snapshots, 5); // Window size larger than data

        assert!(result.is_err());
    }
}
