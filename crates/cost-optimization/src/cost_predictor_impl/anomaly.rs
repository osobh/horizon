//! Anomaly detection functionality for cost prediction
//!
//! This module provides real-time and historical anomaly detection capabilities
//! for identifying unusual cost patterns and spending spikes.

use crate::error::{CostOptimizationError, CostOptimizationResult};
use super::types::*;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use statistical::{mean, standard_deviation};
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::warn;
use uuid::Uuid;

/// Anomaly detector for cost metrics
pub struct AnomalyDetector {
    /// Configuration
    config: Arc<CostPredictorConfig>,
    /// Anomaly history
    anomalies: Arc<RwLock<Vec<Anomaly>>>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new(config: Arc<CostPredictorConfig>) -> Self {
        Self {
            config,
            anomalies: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Detect anomaly in real-time
    pub fn detect_anomaly(
        &self,
        metric_type: CostMetricType,
        value: f64,
        timestamp: DateTime<Utc>,
        series: &VecDeque<TimeSeriesPoint>,
    ) -> CostOptimizationResult<()> {
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();

        if values.len() < 10 {
            return Ok(()); // Need more data for anomaly detection
        }

        let avg = mean(&values);
        let std_dev = standard_deviation(&values, Some(avg));

        let z_score = (value - avg) / std_dev;
        let threshold = 3.0 * (1.0 - self.config.anomaly_sensitivity);

        if z_score.abs() > threshold {
            let anomaly = Anomaly {
                id: Uuid::new_v4(),
                timestamp,
                actual_value: value,
                expected_value: avg,
                deviation_percent: ((value - avg) / avg * 100.0).abs(),
                score: (z_score.abs() / threshold * 100.0).min(100.0),
                anomaly_type: if value > avg {
                    AnomalyType::Spike
                } else {
                    AnomalyType::Drop
                },
            };

            let deviation_percent = anomaly.deviation_percent;
            self.anomalies.write().push(anomaly);

            warn!(
                "Anomaly detected: {} value={:.2} (expected={:.2}, deviation={:.1}%)",
                metric_type, value, avg, deviation_percent
            );
        }

        Ok(())
    }

    /// Detect historical anomalies
    pub fn detect_historical_anomalies(
        &self,
        series: &VecDeque<TimeSeriesPoint>,
    ) -> CostOptimizationResult<Vec<Anomaly>> {
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();

        if values.len() < 10 {
            return Ok(Vec::new());
        }

        let avg = mean(&values);
        let std_dev = standard_deviation(&values, Some(avg));
        let threshold = 2.5;

        let mut anomalies = Vec::new();

        for (i, point) in series.iter().enumerate() {
            let z_score = (point.value - avg) / std_dev;

            if z_score.abs() > threshold {
                anomalies.push(Anomaly {
                    id: Uuid::new_v4(),
                    timestamp: point.timestamp,
                    actual_value: point.value,
                    expected_value: avg,
                    deviation_percent: ((point.value - avg) / avg * 100.0).abs(),
                    score: (z_score.abs() / threshold * 100.0).min(100.0),
                    anomaly_type: if point.value > avg {
                        AnomalyType::Spike
                    } else {
                        AnomalyType::Drop
                    },
                });
            }
        }

        Ok(anomalies)
    }

    /// Detect gradual anomalies (trend changes)
    pub fn detect_gradual_anomalies(
        &self,
        series: &VecDeque<TimeSeriesPoint>,
        window_size: usize,
    ) -> CostOptimizationResult<Vec<Anomaly>> {
        if series.len() < window_size * 2 {
            return Ok(Vec::new());
        }

        let mut anomalies = Vec::new();
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();

        // Compare rolling windows to detect gradual changes
        for i in window_size..values.len() - window_size {
            let prev_window = &values[i - window_size..i];
            let curr_window = &values[i..i + window_size];

            let prev_avg = mean(prev_window);
            let curr_avg = mean(curr_window);

            let change_percent = ((curr_avg - prev_avg) / prev_avg * 100.0).abs();

            // Detect significant gradual changes
            if change_percent > 25.0 {
                let point = &series[i + window_size / 2]; // Middle of current window
                
                anomalies.push(Anomaly {
                    id: Uuid::new_v4(),
                    timestamp: point.timestamp,
                    actual_value: curr_avg,
                    expected_value: prev_avg,
                    deviation_percent: change_percent,
                    score: (change_percent / 50.0 * 100.0).min(100.0),
                    anomaly_type: AnomalyType::GradualIncrease,
                });
            }
        }

        Ok(anomalies)
    }

    /// Detect seasonal anomalies
    pub fn detect_seasonal_anomalies(
        &self,
        series: &VecDeque<TimeSeriesPoint>,
        seasonality: Seasonality,
    ) -> CostOptimizationResult<Vec<Anomaly>> {
        let period = match seasonality {
            Seasonality::Daily => 24,      // 24 hours
            Seasonality::Weekly => 168,    // 24 * 7 hours
            Seasonality::Monthly => 720,   // 24 * 30 hours
            Seasonality::Quarterly => 2160, // 24 * 90 hours
            Seasonality::None => return Ok(Vec::new()),
        };

        if series.len() < period * 2 {
            return Ok(Vec::new());
        }

        let mut anomalies = Vec::new();
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();

        // Look for deviations from seasonal patterns
        for i in period..values.len() {
            let current_value = values[i];
            let seasonal_value = values[i - period];
            let deviation_percent = ((current_value - seasonal_value) / seasonal_value * 100.0).abs();

            if deviation_percent > 50.0 {
                let point = &series[i];
                
                anomalies.push(Anomaly {
                    id: Uuid::new_v4(),
                    timestamp: point.timestamp,
                    actual_value: current_value,
                    expected_value: seasonal_value,
                    deviation_percent,
                    score: (deviation_percent / 100.0 * 100.0).min(100.0),
                    anomaly_type: AnomalyType::UnusualPattern,
                });
            }
        }

        Ok(anomalies)
    }

    /// Get anomaly history
    pub fn get_anomalies(&self) -> Vec<Anomaly> {
        self.anomalies.read().clone()
    }

    /// Clear old anomalies
    pub fn clear_old_anomalies(&self, cutoff: DateTime<Utc>) {
        self.anomalies.write().retain(|a| a.timestamp > cutoff);
    }

    /// Get anomaly count
    pub fn get_anomaly_count(&self) -> usize {
        self.anomalies.read().len()
    }

    /// Get anomalies by type
    pub fn get_anomalies_by_type(&self, anomaly_type: AnomalyType) -> Vec<Anomaly> {
        self.anomalies
            .read()
            .iter()
            .filter(|a| a.anomaly_type == anomaly_type)
            .cloned()
            .collect()
    }

    /// Get recent anomalies
    pub fn get_recent_anomalies(&self, hours: i64) -> Vec<Anomaly> {
        let cutoff = Utc::now() - chrono::Duration::hours(hours);
        self.anomalies
            .read()
            .iter()
            .filter(|a| a.timestamp > cutoff)
            .cloned()
            .collect()
    }

    /// Get anomaly statistics
    pub fn get_anomaly_statistics(&self) -> AnomalyStatistics {
        let anomalies = self.anomalies.read();
        
        let mut stats = AnomalyStatistics {
            total_anomalies: anomalies.len(),
            spikes: 0,
            drops: 0,
            gradual_increases: 0,
            unusual_patterns: 0,
            average_score: 0.0,
            highest_score: 0.0,
            latest_anomaly: None,
        };

        if anomalies.is_empty() {
            return stats;
        }

        let mut total_score = 0.0;
        let mut latest_timestamp = DateTime::<Utc>::from_timestamp(0, 0).unwrap();

        for anomaly in anomalies.iter() {
            match anomaly.anomaly_type {
                AnomalyType::Spike => stats.spikes += 1,
                AnomalyType::Drop => stats.drops += 1,
                AnomalyType::GradualIncrease => stats.gradual_increases += 1,
                AnomalyType::UnusualPattern => stats.unusual_patterns += 1,
            }

            total_score += anomaly.score;
            
            if anomaly.score > stats.highest_score {
                stats.highest_score = anomaly.score;
            }

            if anomaly.timestamp > latest_timestamp {
                latest_timestamp = anomaly.timestamp;
                stats.latest_anomaly = Some(anomaly.timestamp);
            }
        }

        stats.average_score = total_score / anomalies.len() as f64;

        stats
    }

    /// Classify anomaly severity
    pub fn classify_severity(&self, anomaly: &Anomaly) -> AnomalySeverity {
        match anomaly.score {
            s if s >= 90.0 => AnomalySeverity::Critical,
            s if s >= 75.0 => AnomalySeverity::High,
            s if s >= 50.0 => AnomalySeverity::Medium,
            _ => AnomalySeverity::Low,
        }
    }

    /// Add external anomaly
    pub fn add_anomaly(&self, anomaly: Anomaly) {
        self.anomalies.write().push(anomaly);
    }
}

/// Anomaly statistics
#[derive(Debug, Clone)]
pub struct AnomalyStatistics {
    /// Total number of anomalies
    pub total_anomalies: usize,
    /// Number of spikes
    pub spikes: usize,
    /// Number of drops
    pub drops: usize,
    /// Number of gradual increases
    pub gradual_increases: usize,
    /// Number of unusual patterns
    pub unusual_patterns: usize,
    /// Average anomaly score
    pub average_score: f64,
    /// Highest anomaly score
    pub highest_score: f64,
    /// Timestamp of latest anomaly
    pub latest_anomaly: Option<DateTime<Utc>>,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;
    
    fn create_test_series(size: usize, pattern: &str) -> VecDeque<TimeSeriesPoint> {
        let mut series = VecDeque::new();
        let base_time = Utc::now() - ChronoDuration::hours(size as i64);

        for i in 0..size {
            let value = match pattern {
                "normal" => 100.0 + (i as f64).sin() * 5.0,
                "with_spike" => {
                    if i == size / 2 {
                        500.0 // Spike in the middle
                    } else {
                        100.0 + (i as f64).sin() * 5.0
                    }
                }
                "with_drop" => {
                    if i == size / 2 {
                        20.0 // Drop in the middle
                    } else {
                        100.0 + (i as f64).sin() * 5.0
                    }
                }
                "gradual_increase" => 100.0 + i as f64 * 2.0,
                _ => 100.0,
            };

            series.push_back(TimeSeriesPoint {
                timestamp: base_time + ChronoDuration::hours(i as i64),
                value,
                metadata: std::collections::HashMap::new(),
            });
        }

        series
    }

    #[test]
    fn test_anomaly_detector_creation() {
        let config = Arc::new(CostPredictorConfig::default());
        let detector = AnomalyDetector::new(config);
        assert_eq!(detector.get_anomaly_count(), 0);
    }

    #[test]
    fn test_spike_detection() {
        let config = Arc::new(CostPredictorConfig {
            anomaly_sensitivity: 0.8,
            ..Default::default()
        });
        let detector = AnomalyDetector::new(config);

        let series = create_test_series(50, "normal");
        
        // Add spike
        let result = detector.detect_anomaly(
            CostMetricType::ComputeCost,
            500.0, // Spike value
            Utc::now(),
            &series,
        );

        assert!(result.is_ok());
        assert_eq!(detector.get_anomaly_count(), 1);

        let anomalies = detector.get_anomalies();
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::Spike);
        assert!(anomalies[0].score > 0.0);
    }

    #[test]
    fn test_drop_detection() {
        let config = Arc::new(CostPredictorConfig {
            anomaly_sensitivity: 0.9,
            ..Default::default()
        });
        let detector = AnomalyDetector::new(config);

        let series = create_test_series(30, "normal");
        
        // Add drop
        let result = detector.detect_anomaly(
            CostMetricType::StorageCost,
            10.0, // Drop value
            Utc::now(),
            &series,
        );

        assert!(result.is_ok());
        assert_eq!(detector.get_anomaly_count(), 1);

        let anomalies = detector.get_anomalies();
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::Drop);
    }

    #[test]
    fn test_historical_anomaly_detection() {
        let config = Arc::new(CostPredictorConfig::default());
        let detector = AnomalyDetector::new(config);

        let series = create_test_series(50, "with_spike");
        
        let anomalies = detector.detect_historical_anomalies(&series)?;
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::Spike);
    }

    #[test]
    fn test_gradual_anomaly_detection() {
        let config = Arc::new(CostPredictorConfig::default());
        let detector = AnomalyDetector::new(config);

        let series = create_test_series(100, "gradual_increase");
        
        let anomalies = detector.detect_gradual_anomalies(&series, 10)?;
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::GradualIncrease);
    }

    #[test]
    fn test_seasonal_anomaly_detection() {
        let config = Arc::new(CostPredictorConfig::default());
        let detector = AnomalyDetector::new(config);

        // Create series with enough data for daily seasonality
        let mut series = VecDeque::new();
        let base_time = Utc::now() - ChronoDuration::hours(100);

        for i in 0..100 {
            let value = if i == 50 {
                1000.0 // Seasonal anomaly
            } else {
                100.0 + (i % 24) as f64 * 5.0 // Daily pattern
            };

            series.push_back(TimeSeriesPoint {
                timestamp: base_time + ChronoDuration::hours(i as i64),
                value,
                metadata: std::collections::HashMap::new(),
            });
        }

        let anomalies = detector.detect_seasonal_anomalies(&series, Seasonality::Daily).unwrap();
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::UnusualPattern);
    }

    #[test]
    fn test_insufficient_data_handling() {
        let config = Arc::new(CostPredictorConfig::default());
        let detector = AnomalyDetector::new(config);

        let small_series = create_test_series(5, "normal");
        
        // Should not detect anomalies with insufficient data
        let result = detector.detect_anomaly(
            CostMetricType::ComputeCost,
            1000.0,
            Utc::now(),
            &small_series,
        );

        assert!(result.is_ok());
        assert_eq!(detector.get_anomaly_count(), 0);
    }

    #[test]
    fn test_anomaly_statistics() {
        let config = Arc::new(CostPredictorConfig::default());
        let detector = AnomalyDetector::new(config);

        // Add various types of anomalies
        let spike = Anomaly {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            actual_value: 500.0,
            expected_value: 100.0,
            deviation_percent: 400.0,
            score: 95.0,
            anomaly_type: AnomalyType::Spike,
        };

        let drop = Anomaly {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            actual_value: 20.0,
            expected_value: 100.0,
            deviation_percent: 80.0,
            score: 75.0,
            anomaly_type: AnomalyType::Drop,
        };

        detector.add_anomaly(spike);
        detector.add_anomaly(drop);

        let stats = detector.get_anomaly_statistics();
        assert_eq!(stats.total_anomalies, 2);
        assert_eq!(stats.spikes, 1);
        assert_eq!(stats.drops, 1);
        assert_eq!(stats.average_score, 85.0);
        assert_eq!(stats.highest_score, 95.0);
    }

    #[test]
    fn test_anomaly_filtering() {
        let config = Arc::new(CostPredictorConfig::default());
        let detector = AnomalyDetector::new(config);

        let spike = Anomaly {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            actual_value: 500.0,
            expected_value: 100.0,
            deviation_percent: 400.0,
            score: 95.0,
            anomaly_type: AnomalyType::Spike,
        };

        let drop = Anomaly {
            id: Uuid::new_v4(),
            timestamp: Utc::now() - ChronoDuration::hours(10),
            actual_value: 20.0,
            expected_value: 100.0,
            deviation_percent: 80.0,
            score: 75.0,
            anomaly_type: AnomalyType::Drop,
        };

        detector.add_anomaly(spike);
        detector.add_anomaly(drop);

        // Test filtering by type
        let spikes = detector.get_anomalies_by_type(AnomalyType::Spike);
        assert_eq!(spikes.len(), 1);

        // Test filtering by recency
        let recent = detector.get_recent_anomalies(5); // Last 5 hours
        assert_eq!(recent.len(), 1); // Only the recent spike
    }

    #[test]
    fn test_severity_classification() {
        let config = Arc::new(CostPredictorConfig::default());
        let detector = AnomalyDetector::new(config);

        let critical_anomaly = Anomaly {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            actual_value: 1000.0,
            expected_value: 100.0,
            deviation_percent: 900.0,
            score: 95.0,
            anomaly_type: AnomalyType::Spike,
        };

        let low_anomaly = Anomaly {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            actual_value: 120.0,
            expected_value: 100.0,
            deviation_percent: 20.0,
            score: 30.0,
            anomaly_type: AnomalyType::Spike,
        };

        assert_eq!(detector.classify_severity(&critical_anomaly), AnomalySeverity::Critical);
        assert_eq!(detector.classify_severity(&low_anomaly), AnomalySeverity::Low);
    }

    #[test]
    fn test_anomaly_cleanup() {
        let config = Arc::new(CostPredictorConfig::default());
        let detector = AnomalyDetector::new(config);

        let old_anomaly = Anomaly {
            id: Uuid::new_v4(),
            timestamp: Utc::now() - ChronoDuration::days(10),
            actual_value: 500.0,
            expected_value: 100.0,
            deviation_percent: 400.0,
            score: 95.0,
            anomaly_type: AnomalyType::Spike,
        };

        let recent_anomaly = Anomaly {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            actual_value: 300.0,
            expected_value: 100.0,
            deviation_percent: 200.0,
            score: 80.0,
            anomaly_type: AnomalyType::Spike,
        };

        detector.add_anomaly(old_anomaly);
        detector.add_anomaly(recent_anomaly);
        assert_eq!(detector.get_anomaly_count(), 2);

        // Clear anomalies older than 5 days
        let cutoff = Utc::now() - ChronoDuration::days(5);
        detector.clear_old_anomalies(cutoff);

        assert_eq!(detector.get_anomaly_count(), 1); // Only recent anomaly should remain
    }
}