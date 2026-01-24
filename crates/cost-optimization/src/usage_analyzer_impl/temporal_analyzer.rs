//! Temporal analysis for usage patterns over time
//!
//! This module analyzes resource usage patterns across different time dimensions
//! including hourly, daily, and trend analysis.

use crate::error::CostOptimizationResult;
use crate::resource_tracker::ResourceSnapshot;
use crate::usage_analyzer::types::*;
use chrono::{Datelike, Timelike};
use std::time::Duration;

/// Temporal analyzer for time-based usage analysis
pub struct TemporalAnalyzer {
    /// Configuration
    config: UsageAnalyzerConfig,
}

impl TemporalAnalyzer {
    /// Create a new temporal analyzer
    pub fn new(config: UsageAnalyzerConfig) -> Self {
        Self { config }
    }

    /// Analyze temporal patterns in resource usage
    pub fn analyze_temporal_patterns(
        &self,
        snapshots: &[ResourceSnapshot],
    ) -> CostOptimizationResult<TemporalAnalysis> {
        // Initialize hourly pattern
        let mut hourly_usage = vec![
            HourlyUsage {
                hour: 0,
                avg_utilization: 0.0,
                samples: 0
            };
            24
        ];
        for i in 0..24 {
            hourly_usage[i].hour = i as u32;
        }

        // Initialize daily pattern
        let mut daily_usage = vec![
            DailyUsage {
                day_of_week: 0,
                avg_utilization: 0.0,
                samples: 0
            };
            7
        ];
        for i in 0..7 {
            daily_usage[i].day_of_week = (i + 1) as u32; // 1=Monday
        }

        // Process snapshots to build patterns
        for snapshot in snapshots {
            let hour = snapshot.timestamp.hour() as usize;
            let dow = snapshot.timestamp.weekday().num_days_from_monday() as usize;

            // Update hourly pattern
            hourly_usage[hour].avg_utilization += snapshot.utilization;
            hourly_usage[hour].samples += 1;

            // Update daily pattern
            daily_usage[dow].avg_utilization += snapshot.utilization;
            daily_usage[dow].samples += 1;
        }

        // Calculate averages
        for usage in &mut hourly_usage {
            if usage.samples > 0 {
                usage.avg_utilization /= usage.samples as f64;
            }
        }

        for usage in &mut daily_usage {
            if usage.samples > 0 {
                usage.avg_utilization /= usage.samples as f64;
            }
        }

        // Find peak times and low periods
        let (peak_times, low_periods) = self.find_peak_and_low_periods(snapshots)?;

        // Calculate usage trend
        let trend = self.calculate_usage_trend(snapshots);

        Ok(TemporalAnalysis {
            hourly_pattern: hourly_usage,
            daily_pattern: daily_usage,
            peak_times,
            low_periods,
            trend,
        })
    }

    /// Find peak and low usage periods in the data
    pub fn find_peak_and_low_periods(
        &self,
        snapshots: &[ResourceSnapshot],
    ) -> CostOptimizationResult<(Vec<PeakTime>, Vec<LowPeriod>)> {
        let mut peak_times = Vec::new();
        let mut low_periods = Vec::new();

        if snapshots.is_empty() {
            return Ok((peak_times, low_periods));
        }

        let threshold_high = 80.0;
        let threshold_low = self.config.idle_threshold;

        let mut in_peak = false;
        let mut in_low = false;
        let mut period_start = snapshots[0].timestamp;
        let mut period_max = 0.0;
        let mut period_sum = 0.0;
        let mut period_count = 0;

        for snapshot in snapshots {
            // Peak detection
            if snapshot.utilization >= threshold_high {
                if !in_peak {
                    in_peak = true;
                    period_start = snapshot.timestamp;
                    period_max = snapshot.utilization;
                } else {
                    period_max = period_max.max(snapshot.utilization);
                }
            } else if in_peak {
                in_peak = false;
                peak_times.push(PeakTime {
                    start: period_start,
                    end: snapshot.timestamp,
                    peak_utilization: period_max,
                    duration: (snapshot.timestamp - period_start)
                        .to_std()
                        .unwrap_or_default(),
                });
            }

            // Low period detection
            if snapshot.utilization <= threshold_low {
                if !in_low {
                    in_low = true;
                    period_start = snapshot.timestamp;
                    period_sum = snapshot.utilization;
                    period_count = 1;
                } else {
                    period_sum += snapshot.utilization;
                    period_count += 1;
                }
            } else if in_low {
                in_low = false;
                let duration = (snapshot.timestamp - period_start)
                    .to_std()
                    .unwrap_or_default();
                low_periods.push(LowPeriod {
                    start: period_start,
                    end: snapshot.timestamp,
                    avg_utilization: period_sum / period_count as f64,
                    can_shutdown: duration > Duration::from_secs(2 * 60 * 60), // 2 hours
                });
            }
        }

        Ok((peak_times, low_periods))
    }

    /// Calculate usage trend over time
    pub fn calculate_usage_trend(&self, snapshots: &[ResourceSnapshot]) -> UsageTrend {
        if snapshots.len() < 2 {
            return UsageTrend {
                direction: TrendDirection::Stable,
                rate: 0.0,
                confidence: 0.0,
            };
        }

        let trend_slope = self.calculate_trend_slope(snapshots);

        let direction = if trend_slope > 0.1 {
            TrendDirection::Increasing
        } else if trend_slope < -0.1 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        // Calculate confidence based on data consistency
        let confidence = self.calculate_trend_confidence(snapshots, trend_slope);

        UsageTrend {
            direction,
            rate: trend_slope * 100.0, // Convert to percentage
            confidence,
        }
    }

    /// Calculate trend slope using linear regression
    fn calculate_trend_slope(&self, snapshots: &[ResourceSnapshot]) -> f64 {
        if snapshots.len() < 2 {
            return 0.0;
        }

        let n = snapshots.len() as f64;
        let x_values: Vec<f64> = (0..snapshots.len()).map(|i| i as f64).collect();
        let y_values: Vec<f64> = snapshots.iter().map(|s| s.utilization).collect();

        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = y_values.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..snapshots.len() {
            numerator += (x_values[i] - x_mean) * (y_values[i] - y_mean);
            denominator += (x_values[i] - x_mean).powi(2);
        }

        if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Calculate confidence in the trend calculation
    fn calculate_trend_confidence(&self, snapshots: &[ResourceSnapshot], slope: f64) -> f64 {
        if snapshots.len() < 10 {
            return 0.3; // Low confidence with little data
        }

        // Calculate R-squared for trend line
        let y_values: Vec<f64> = snapshots.iter().map(|s| s.utilization).collect();
        let y_mean = y_values.iter().sum::<f64>() / y_values.len() as f64;

        let mut ss_res = 0.0; // Sum of squares of residuals
        let mut ss_tot = 0.0; // Total sum of squares

        for (i, &y) in y_values.iter().enumerate() {
            let y_pred = y_mean + slope * (i as f64 - (y_values.len() as f64 / 2.0));
            ss_res += (y - y_pred).powi(2);
            ss_tot += (y - y_mean).powi(2);
        }

        if ss_tot == 0.0 {
            return 0.5; // Neutral confidence
        }

        let r_squared = 1.0 - (ss_res / ss_tot);
        r_squared.max(0.0).min(1.0) // Clamp between 0 and 1
    }

    /// Detect seasonal patterns
    pub fn detect_seasonality(&self, snapshots: &[ResourceSnapshot]) -> Vec<String> {
        let mut patterns = Vec::new();

        if snapshots.len() < 48 {
            return patterns; // Need at least 2 days of hourly data
        }

        // Check for daily patterns
        if self.has_daily_pattern(snapshots) {
            patterns.push("Daily".to_string());
        }

        // Check for weekly patterns
        if self.has_weekly_pattern(snapshots) {
            patterns.push("Weekly".to_string());
        }

        patterns
    }

    /// Check for daily usage patterns
    fn has_daily_pattern(&self, snapshots: &[ResourceSnapshot]) -> bool {
        let mut hourly_usage = vec![Vec::new(); 24];

        // Group snapshots by hour
        for snapshot in snapshots {
            let hour = snapshot.timestamp.hour() as usize;
            hourly_usage[hour].push(snapshot.utilization);
        }

        // Calculate coefficient of variation across hours
        let hourly_means: Vec<f64> = hourly_usage
            .iter()
            .map(|hour_data| {
                if hour_data.is_empty() {
                    0.0
                } else {
                    hour_data.iter().sum::<f64>() / hour_data.len() as f64
                }
            })
            .collect();

        let overall_mean = hourly_means.iter().sum::<f64>() / 24.0;
        let variance = hourly_means
            .iter()
            .map(|&mean| (mean - overall_mean).powi(2))
            .sum::<f64>()
            / 24.0;

        let std_dev = variance.sqrt();
        let cv = if overall_mean > 0.0 {
            std_dev / overall_mean
        } else {
            0.0
        };

        cv > 0.2 // Significant variation across hours
    }

    /// Check for weekly usage patterns
    fn has_weekly_pattern(&self, snapshots: &[ResourceSnapshot]) -> bool {
        if snapshots.len() < 7 * 24 {
            return false; // Need at least a week of data
        }

        let mut daily_usage = vec![Vec::new(); 7];

        // Group snapshots by day of week
        for snapshot in snapshots {
            let dow = snapshot.timestamp.weekday().num_days_from_monday() as usize;
            daily_usage[dow].push(snapshot.utilization);
        }

        // Calculate coefficient of variation across days
        let daily_means: Vec<f64> = daily_usage
            .iter()
            .map(|day_data| {
                if day_data.is_empty() {
                    0.0
                } else {
                    day_data.iter().sum::<f64>() / day_data.len() as f64
                }
            })
            .collect();

        let overall_mean = daily_means.iter().sum::<f64>() / 7.0;
        let variance = daily_means
            .iter()
            .map(|&mean| (mean - overall_mean).powi(2))
            .sum::<f64>()
            / 7.0;

        let std_dev = variance.sqrt();
        let cv = if overall_mean > 0.0 {
            std_dev / overall_mean
        } else {
            0.0
        };

        cv > 0.15 // Significant variation across days
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_tracker::ResourceType;
    use chrono::Utc;
    use std::collections::HashMap;

    fn create_test_snapshots(count: usize, pattern: &str) -> Vec<ResourceSnapshot> {
        let mut snapshots = Vec::new();
        let base_time = Utc::now() - chrono::Duration::hours(count as i64);

        for i in 0..count {
            let utilization = match pattern {
                "daily_pattern" => {
                    let hour = i % 24;
                    if hour >= 9 && hour <= 17 {
                        80.0 // Work hours
                    } else {
                        20.0 // Off hours
                    }
                }
                "weekly_pattern" => {
                    let day = (i / 24) % 7;
                    if day < 5 {
                        70.0 // Weekdays
                    } else {
                        30.0 // Weekends
                    }
                }
                "spiky" => {
                    if i % 10 == 0 {
                        95.0
                    } else {
                        30.0
                    }
                }
                "growing" => 20.0 + (i as f64 * 0.1),
                _ => 50.0,
            };

            snapshots.push(ResourceSnapshot {
                timestamp: base_time + chrono::Duration::hours(i as i64),
                resource_type: ResourceType::Cpu,
                resource_id: "test-resource".to_string(),
                utilization: utilization.min(100.0).max(0.0),
                available: 100.0 - utilization.min(100.0).max(0.0),
                total: 100.0,
                metadata: HashMap::new(),
            });
        }

        snapshots
    }

    #[test]
    fn test_hourly_pattern_analysis() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = TemporalAnalyzer::new(config);

        let snapshots = create_test_snapshots(168, "daily_pattern"); // 1 week
        let result = analyzer.analyze_temporal_patterns(&snapshots).unwrap();

        assert_eq!(result.hourly_pattern.len(), 24);

        // Work hours should have higher utilization
        let work_hour_avg = result.hourly_pattern[12].avg_utilization; // Noon
        let off_hour_avg = result.hourly_pattern[2].avg_utilization; // 2 AM

        assert!(work_hour_avg > off_hour_avg);
    }

    #[test]
    fn test_daily_pattern_analysis() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = TemporalAnalyzer::new(config);

        let snapshots = create_test_snapshots(336, "weekly_pattern"); // 2 weeks
        let result = analyzer.analyze_temporal_patterns(&snapshots).unwrap();

        assert_eq!(result.daily_pattern.len(), 7);

        // Weekdays should have higher utilization than weekends
        let weekday_avg = result.daily_pattern[0].avg_utilization; // Monday
        let weekend_avg = result.daily_pattern[5].avg_utilization; // Saturday

        assert!(weekday_avg > weekend_avg);
    }

    #[test]
    fn test_peak_and_low_detection() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = TemporalAnalyzer::new(config);

        let snapshots = create_test_snapshots(100, "spiky");
        let (peaks, lows) = analyzer.find_peak_and_low_periods(&snapshots).unwrap();

        assert!(!peaks.is_empty());
        assert!(!lows.is_empty());

        // Verify peak characteristics
        for peak in &peaks {
            assert!(peak.peak_utilization >= 80.0);
            assert!(peak.duration.as_secs() > 0);
        }

        // Verify low period characteristics
        for low in &lows {
            assert!(low.avg_utilization <= 5.0);
        }
    }

    #[test]
    fn test_trend_calculation() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = TemporalAnalyzer::new(config);

        // Test growing trend
        let growing_snapshots = create_test_snapshots(100, "growing");
        let trend = analyzer.calculate_usage_trend(&growing_snapshots);

        assert_eq!(trend.direction, TrendDirection::Increasing);
        assert!(trend.rate > 0.0);
        assert!(trend.confidence > 0.0);
    }

    #[test]
    fn test_seasonality_detection() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = TemporalAnalyzer::new(config);

        // Test daily pattern detection
        let daily_snapshots = create_test_snapshots(168, "daily_pattern");
        let patterns = analyzer.detect_seasonality(&daily_snapshots);

        assert!(patterns.contains(&"Daily".to_string()));

        // Test weekly pattern detection
        let weekly_snapshots = create_test_snapshots(336, "weekly_pattern");
        let patterns = analyzer.detect_seasonality(&weekly_snapshots);

        assert!(patterns.contains(&"Weekly".to_string()));
    }

    #[test]
    fn test_empty_snapshots() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = TemporalAnalyzer::new(config);

        let empty_snapshots = Vec::new();
        let (peaks, lows) = analyzer.find_peak_and_low_periods(&empty_snapshots).unwrap();

        assert!(peaks.is_empty());
        assert!(lows.is_empty());
    }
}
