//! Pattern detection logic for usage analysis
//!
//! This module contains the core pattern detection algorithms that analyze
//! resource usage patterns and classify them into different categories.

use crate::error::CostOptimizationResult;
use crate::resource_tracker::ResourceSnapshot;
use crate::usage_analyzer::types::*;
use chrono::Timelike;
use statistical::{mean, standard_deviation};

/// Pattern detector for analyzing usage patterns
pub struct PatternDetector {
    /// Configuration
    config: UsageAnalyzerConfig,
}

impl PatternDetector {
    /// Create a new pattern detector
    pub fn new(config: UsageAnalyzerConfig) -> Self {
        Self { config }
    }

    /// Detect usage pattern from snapshots and statistics
    pub fn detect_pattern(
        &self,
        snapshots: &[ResourceSnapshot],
        statistics: &UsageStatistics,
    ) -> CostOptimizationResult<(UsagePattern, f64)> {
        // Check for idle pattern first (highest confidence when detected)
        if statistics.idle_percentage > 90.0 {
            return Ok((UsagePattern::Idle, 0.95));
        }

        // Check for constant patterns (high confidence for low variance)
        if statistics.std_deviation < 5.0 {
            if statistics.avg_utilization > 80.0 {
                return Ok((UsagePattern::ConstantHigh, 0.9));
            } else if statistics.avg_utilization < 20.0 {
                return Ok((UsagePattern::ConstantLow, 0.9));
            }
        }

        // Check for spiky pattern (high peaks compared to average)
        if statistics.p99_utilization > statistics.avg_utilization * 2.0 {
            return Ok((UsagePattern::Spiky, 0.85));
        }

        // Check for trending patterns
        let trend = self.calculate_trend(snapshots);
        if trend.abs() > 0.1 {
            if trend > 0.0 {
                return Ok((UsagePattern::Growing, 0.8));
            } else {
                return Ok((UsagePattern::Declining, 0.8));
            }
        }

        // Check for periodic pattern
        if self.has_periodic_pattern(snapshots) {
            return Ok((UsagePattern::Periodic, 0.75));
        }

        // Default to unpredictable if no clear pattern
        Ok((UsagePattern::Unpredictable, 0.5))
    }

    /// Calculate usage trend using linear regression
    pub fn calculate_trend(&self, snapshots: &[ResourceSnapshot]) -> f64 {
        if snapshots.len() < 2 {
            return 0.0;
        }

        let x_values: Vec<f64> = (0..snapshots.len()).map(|i| i as f64).collect();
        let y_values: Vec<f64> = snapshots.iter().map(|s| s.utilization).collect();

        let x_mean = mean(&x_values);
        let y_mean = mean(&y_values);

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

    /// Check for periodic pattern using simplified analysis
    pub fn has_periodic_pattern(&self, snapshots: &[ResourceSnapshot]) -> bool {
        // Need sufficient data for pattern detection
        if snapshots.len() < 48 {
            return false;
        }

        // Check for daily pattern by analyzing hourly averages
        let mut hourly_avg = vec![0.0; 24];
        let mut hourly_count = vec![0; 24];

        for snapshot in snapshots {
            let hour = snapshot.timestamp.hour() as usize;
            hourly_avg[hour] += snapshot.utilization;
            hourly_count[hour] += 1;
        }

        // Calculate hourly averages
        for i in 0..24 {
            if hourly_count[i] > 0 {
                hourly_avg[i] /= hourly_count[i] as f64;
            }
        }

        // Calculate standard deviation across hours
        let hourly_std = standard_deviation(&hourly_avg, None);

        // Significant hourly variation indicates periodic pattern
        hourly_std > 10.0
    }

    /// Get pattern model for a resource
    pub fn get_pattern_model(
        &self,
        resource_id: &str,
        pattern: UsagePattern,
        snapshots: &[ResourceSnapshot],
    ) -> PatternModel {
        let parameters = self.extract_pattern_parameters(&pattern, snapshots);

        PatternModel {
            resource_id: resource_id.to_string(),
            pattern,
            parameters,
            last_updated: chrono::Utc::now(),
        }
    }

    /// Extract pattern-specific parameters
    fn extract_pattern_parameters(
        &self,
        pattern: &UsagePattern,
        snapshots: &[ResourceSnapshot],
    ) -> PatternParameters {
        match pattern {
            UsagePattern::Periodic => {
                // Try to detect period length
                let period = self.detect_period_length(snapshots);
                PatternParameters {
                    period,
                    growth_rate: None,
                    spike_threshold: None,
                    baseline: Some(self.calculate_baseline(snapshots)),
                }
            }
            UsagePattern::Growing | UsagePattern::Declining => {
                let growth_rate = Some(self.calculate_trend(snapshots));
                PatternParameters {
                    period: None,
                    growth_rate,
                    spike_threshold: None,
                    baseline: Some(self.calculate_baseline(snapshots)),
                }
            }
            UsagePattern::Spiky => {
                let spike_threshold = self.calculate_spike_threshold(snapshots);
                PatternParameters {
                    period: None,
                    growth_rate: None,
                    spike_threshold: Some(spike_threshold),
                    baseline: Some(self.calculate_baseline(snapshots)),
                }
            }
            _ => PatternParameters {
                period: None,
                growth_rate: None,
                spike_threshold: None,
                baseline: Some(self.calculate_baseline(snapshots)),
            },
        }
    }

    /// Detect period length for periodic patterns
    fn detect_period_length(&self, snapshots: &[ResourceSnapshot]) -> Option<std::time::Duration> {
        // Simplified period detection - assume daily period for now
        // In a full implementation, this would use FFT or autocorrelation
        if snapshots.len() >= 24 {
            Some(std::time::Duration::from_secs(24 * 60 * 60)) // 24 hours
        } else {
            None
        }
    }

    /// Calculate baseline utilization
    fn calculate_baseline(&self, snapshots: &[ResourceSnapshot]) -> f64 {
        let utilizations: Vec<f64> = snapshots.iter().map(|s| s.utilization).collect();

        if utilizations.is_empty() {
            return 0.0;
        }

        // Use median as baseline to reduce impact of outliers
        let mut sorted = utilizations.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b)?);

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Calculate spike threshold for spiky patterns
    fn calculate_spike_threshold(&self, snapshots: &[ResourceSnapshot]) -> f64 {
        let utilizations: Vec<f64> = snapshots.iter().map(|s| s.utilization).collect();

        if utilizations.is_empty() {
            return 80.0; // Default threshold
        }

        let avg = mean(&utilizations);
        let std_dev = standard_deviation(&utilizations, Some(avg));

        // Threshold is 2 standard deviations above mean
        avg + 2.0 * std_dev
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
                "constant_high" => 85.0 + (i as f64).sin() * 2.0,
                "constant_low" => 15.0 + (i as f64).sin() * 2.0,
                "spiky" => {
                    if i % 10 == 0 {
                        95.0
                    } else {
                        30.0
                    }
                }
                "periodic" => 50.0 + (i as f64 * 0.26).sin() * 30.0, // ~24 hour period
                "growing" => 20.0 + (i as f64 * 0.5),
                "declining" => 80.0 - (i as f64 * 0.3),
                "idle" => {
                    if i % 50 == 0 {
                        5.0
                    } else {
                        1.0
                    }
                }
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

    fn create_test_statistics(snapshots: &[ResourceSnapshot]) -> UsageStatistics {
        let utilizations: Vec<f64> = snapshots.iter().map(|s| s.utilization).collect();
        let avg = mean(&utilizations);
        let std_dev = standard_deviation(&utilizations, Some(avg));

        let mut sorted = utilizations.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b)?);

        let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
        let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];

        let idle_count = utilizations.iter().filter(|&&u| u <= 5.0).count();
        let idle_percentage = (idle_count as f64 / utilizations.len() as f64) * 100.0;

        UsageStatistics {
            avg_utilization: avg,
            peak_utilization: utilizations
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            min_utilization: utilizations.iter().cloned().fold(f64::INFINITY, f64::min),
            std_deviation: std_dev,
            p95_utilization: p95,
            p99_utilization: p99,
            idle_percentage,
            sample_count: utilizations.len(),
        }
    }

    #[test]
    fn test_detect_constant_high_pattern() {
        let config = UsageAnalyzerConfig::default();
        let detector = PatternDetector::new(config);

        let snapshots = create_test_snapshots(100, "constant_high");
        let statistics = create_test_statistics(&snapshots);

        let (pattern, confidence) = detector.detect_pattern(&snapshots, &statistics)?;

        assert_eq!(pattern, UsagePattern::ConstantHigh);
        assert!(confidence > 0.8);
    }

    #[test]
    fn test_detect_spiky_pattern() {
        let config = UsageAnalyzerConfig::default();
        let detector = PatternDetector::new(config);

        let snapshots = create_test_snapshots(100, "spiky");
        let statistics = create_test_statistics(&snapshots);

        let (pattern, confidence) = detector.detect_pattern(&snapshots, &statistics)?;

        assert_eq!(pattern, UsagePattern::Spiky);
        assert!(confidence > 0.8);
    }

    #[test]
    fn test_detect_idle_pattern() {
        let config = UsageAnalyzerConfig::default();
        let detector = PatternDetector::new(config);

        let snapshots = create_test_snapshots(100, "idle");
        let statistics = create_test_statistics(&snapshots);

        let (pattern, confidence) = detector.detect_pattern(&snapshots, &statistics)?;

        assert_eq!(pattern, UsagePattern::Idle);
        assert!(confidence > 0.9);
    }

    #[test]
    fn test_trend_calculation() {
        let config = UsageAnalyzerConfig::default();
        let detector = PatternDetector::new(config);

        let growing_snapshots = create_test_snapshots(50, "growing");
        let trend = detector.calculate_trend(&growing_snapshots);
        assert!(trend > 0.0);

        let declining_snapshots = create_test_snapshots(50, "declining");
        let trend = detector.calculate_trend(&declining_snapshots);
        assert!(trend < 0.0);
    }

    #[test]
    fn test_periodic_pattern_detection() {
        let config = UsageAnalyzerConfig::default();
        let detector = PatternDetector::new(config);

        let periodic_snapshots = create_test_snapshots(100, "periodic");
        assert!(detector.has_periodic_pattern(&periodic_snapshots));

        let constant_snapshots = create_test_snapshots(100, "constant_high");
        assert!(!detector.has_periodic_pattern(&constant_snapshots));
    }

    #[test]
    fn test_pattern_model_creation() {
        let config = UsageAnalyzerConfig::default();
        let detector = PatternDetector::new(config);

        let snapshots = create_test_snapshots(100, "periodic");
        let model = detector.get_pattern_model("test-resource", UsagePattern::Periodic, &snapshots);

        assert_eq!(model.resource_id, "test-resource");
        assert_eq!(model.pattern, UsagePattern::Periodic);
        assert!(model.parameters.period.is_some());
        assert!(model.parameters.baseline.is_some());
    }
}
