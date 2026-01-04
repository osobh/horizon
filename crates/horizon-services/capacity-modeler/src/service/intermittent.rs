use crate::error::{CapacityErrorExt, HpcError, Result};
use crate::forecaster::UptimePattern;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Intermittent capacity estimator for nodes with variable availability
/// Calculates probabilistic capacity based on uptime patterns
#[derive(Debug, Clone)]
pub struct IntermittentCapacityEstimator {
    patterns: HashMap<String, UptimePattern>,
}

/// Capacity estimate for a specific time window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityEstimate {
    pub node_id: String,
    pub total_capacity: f64,
    pub available_capacity: f64,
    pub availability_probability: f64,
    pub confidence_level: f64,
}

/// Time window for capacity estimation
#[derive(Debug, Clone)]
pub struct TimeWindow {
    pub start_hour: u8,
    pub end_hour: u8,
}

impl TimeWindow {
    pub fn new(start_hour: u8, end_hour: u8) -> Result<Self> {
        if start_hour > 23 || end_hour > 23 {
            return Err(HpcError::invalid_parameters(
                "Hours must be between 0 and 23",
            ));
        }

        if end_hour <= start_hour {
            return Err(HpcError::invalid_parameters(
                "End hour must be after start hour",
            ));
        }

        Ok(Self {
            start_hour,
            end_hour,
        })
    }

    pub fn hours(&self) -> Vec<u8> {
        (self.start_hour..self.end_hour).collect()
    }
}

impl IntermittentCapacityEstimator {
    /// Create a new estimator
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }

    /// Add uptime pattern for a node
    pub fn add_pattern(&mut self, pattern: UptimePattern) {
        self.patterns.insert(pattern.node_id.clone(), pattern);
    }

    /// Estimate available capacity for a node at a specific hour
    pub fn estimate_capacity_at_hour(
        &self,
        node_id: &str,
        hour: u8,
        total_capacity: f64,
    ) -> Result<CapacityEstimate> {
        if hour > 23 {
            return Err(HpcError::invalid_parameters(
                "Hour must be between 0 and 23",
            ));
        }

        let pattern = self.patterns.get(node_id).ok_or_else(|| {
            HpcError::insufficient_data(format!("No uptime pattern for node {}", node_id))
        })?;

        let availability_probability = pattern.availability_probability_at_hour(hour);
        let available_capacity = total_capacity * availability_probability;
        let confidence_level = Self::calculate_confidence(pattern);

        Ok(CapacityEstimate {
            node_id: node_id.to_string(),
            total_capacity,
            available_capacity,
            availability_probability,
            confidence_level,
        })
    }

    /// Estimate capacity over a time window
    pub fn estimate_capacity_over_window(
        &self,
        node_id: &str,
        window: &TimeWindow,
        total_capacity: f64,
    ) -> Result<CapacityEstimate> {
        let pattern = self.patterns.get(node_id).ok_or_else(|| {
            HpcError::insufficient_data(format!("No uptime pattern for node {}", node_id))
        })?;

        let hours = window.hours();
        if hours.is_empty() {
            return Err(HpcError::invalid_parameters("Empty time window"));
        }

        // Calculate average availability over the window
        let total_prob: f64 = hours
            .iter()
            .map(|&h| pattern.availability_probability_at_hour(h))
            .sum();
        let avg_probability = total_prob / hours.len() as f64;

        let available_capacity = total_capacity * avg_probability;
        let confidence_level = Self::calculate_confidence(pattern);

        Ok(CapacityEstimate {
            node_id: node_id.to_string(),
            total_capacity,
            available_capacity,
            availability_probability: avg_probability,
            confidence_level,
        })
    }

    /// Calculate confidence level based on pattern data
    fn calculate_confidence(pattern: &UptimePattern) -> f64 {
        // Confidence increases with number of sessions
        let session_factor = (pattern.total_sessions as f64 / 100.0).min(1.0);

        // Confidence decreases with low uptime percentage
        let uptime_factor = pattern.avg_uptime_percent / 100.0;

        // Combined confidence
        (session_factor * 0.5 + uptime_factor * 0.5)
            .max(0.0)
            .min(1.0)
    }

    /// Get all node IDs with patterns
    pub fn node_ids(&self) -> Vec<String> {
        self.patterns.keys().cloned().collect()
    }
}

impl Default for IntermittentCapacityEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pattern(node_id: &str, reliability: f64) -> UptimePattern {
        let mut pattern = UptimePattern::new(node_id.to_string());
        pattern.reliability_score = reliability;
        pattern.avg_uptime_percent = reliability * 100.0;
        pattern.typical_online_hours = vec![9, 10, 11, 12, 13, 14, 15, 16, 17];
        pattern.total_sessions = 50;
        pattern
    }

    #[test]
    fn test_estimator_creation() {
        let estimator = IntermittentCapacityEstimator::new();
        assert_eq!(estimator.node_ids().len(), 0);
    }

    #[test]
    fn test_add_pattern() {
        let mut estimator = IntermittentCapacityEstimator::new();
        let pattern = create_test_pattern("node-1", 0.8);

        estimator.add_pattern(pattern);

        assert_eq!(estimator.node_ids().len(), 1);
        assert!(estimator.node_ids().contains(&"node-1".to_string()));
    }

    #[test]
    fn test_estimate_capacity_at_hour() {
        let mut estimator = IntermittentCapacityEstimator::new();
        let pattern = create_test_pattern("node-1", 0.8);
        estimator.add_pattern(pattern);

        let result = estimator.estimate_capacity_at_hour("node-1", 10, 100.0);

        assert!(result.is_ok());
        let estimate = result.unwrap();
        assert_eq!(estimate.node_id, "node-1");
        assert_eq!(estimate.total_capacity, 100.0);
        assert_eq!(estimate.available_capacity, 80.0); // 100.0 * 0.8
    }

    #[test]
    fn test_estimate_capacity_invalid_hour() {
        let mut estimator = IntermittentCapacityEstimator::new();
        let pattern = create_test_pattern("node-1", 0.8);
        estimator.add_pattern(pattern);

        let result = estimator.estimate_capacity_at_hour("node-1", 24, 100.0);

        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_capacity_nonexistent_node() {
        let estimator = IntermittentCapacityEstimator::new();

        let result = estimator.estimate_capacity_at_hour("nonexistent", 10, 100.0);

        assert!(result.is_err());
    }

    #[test]
    fn test_time_window_creation() {
        let window = TimeWindow::new(9, 17);

        assert!(window.is_ok());
        let w = window.unwrap();
        assert_eq!(w.start_hour, 9);
        assert_eq!(w.end_hour, 17);
    }

    #[test]
    fn test_time_window_invalid_hours() {
        // End hour > 23
        let result = TimeWindow::new(9, 24);
        assert!(result.is_err());

        // Start hour > 23
        let result = TimeWindow::new(24, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_time_window_end_before_start() {
        let result = TimeWindow::new(17, 9);
        assert!(result.is_err());
    }

    #[test]
    fn test_time_window_hours() {
        let window = TimeWindow::new(9, 12).unwrap();
        let hours = window.hours();

        assert_eq!(hours, vec![9, 10, 11]);
    }

    #[test]
    fn test_estimate_capacity_over_window() {
        let mut estimator = IntermittentCapacityEstimator::new();
        let pattern = create_test_pattern("node-1", 0.8);
        estimator.add_pattern(pattern);

        let window = TimeWindow::new(9, 17).unwrap();
        let result = estimator.estimate_capacity_over_window("node-1", &window, 100.0);

        assert!(result.is_ok());
        let estimate = result.unwrap();
        assert_eq!(estimate.node_id, "node-1");
        assert_eq!(estimate.total_capacity, 100.0);
        // During typical hours (9-17), availability should be ~0.8
        assert!(estimate.available_capacity > 70.0);
    }

    #[test]
    fn test_calculate_confidence() {
        let mut pattern = UptimePattern::new("node-1".to_string());
        pattern.total_sessions = 100;
        pattern.avg_uptime_percent = 80.0;

        let confidence = IntermittentCapacityEstimator::calculate_confidence(&pattern);

        // With 100 sessions and 80% uptime, confidence should be high
        assert!(confidence > 0.5);
        assert!(confidence <= 1.0);
    }

    #[test]
    fn test_calculate_confidence_low_sessions() {
        let mut pattern = UptimePattern::new("node-1".to_string());
        pattern.total_sessions = 10;
        pattern.avg_uptime_percent = 80.0;

        let confidence = IntermittentCapacityEstimator::calculate_confidence(&pattern);

        // With only 10 sessions, confidence should be lower
        assert!(confidence < 0.5);
    }

    #[test]
    fn test_multiple_nodes() {
        let mut estimator = IntermittentCapacityEstimator::new();
        estimator.add_pattern(create_test_pattern("node-1", 0.8));
        estimator.add_pattern(create_test_pattern("node-2", 0.6));
        estimator.add_pattern(create_test_pattern("node-3", 0.9));

        assert_eq!(estimator.node_ids().len(), 3);
    }

    #[test]
    fn test_capacity_estimate_serialization() {
        let estimate = CapacityEstimate {
            node_id: "node-1".to_string(),
            total_capacity: 100.0,
            available_capacity: 80.0,
            availability_probability: 0.8,
            confidence_level: 0.75,
        };

        let json = serde_json::to_string(&estimate).unwrap();
        let deserialized: CapacityEstimate = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.node_id, "node-1");
        assert_eq!(deserialized.total_capacity, 100.0);
    }
}
