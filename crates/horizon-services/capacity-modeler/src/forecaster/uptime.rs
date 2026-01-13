use crate::error::HpcError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Uptime pattern analyzer for intermittent nodes
/// Tracks historical uptime patterns to predict future availability
#[derive(Debug, Clone)]
pub struct UptimePatternAnalyzer {
    patterns: HashMap<String, UptimePattern>,
}

/// Uptime pattern for a specific node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UptimePattern {
    pub node_id: String,
    /// Average uptime percentage over last 30 days
    pub avg_uptime_percent: f64,
    /// Reliability score (0.0-1.0)
    pub reliability_score: f64,
    /// Typical hours when node is online (0-23)
    pub typical_online_hours: Vec<u8>,
    /// Average session duration in minutes
    pub avg_session_duration_minutes: f64,
    /// Total sessions in last 30 days
    pub total_sessions: u64,
}

impl UptimePattern {
    /// Create a new uptime pattern
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            avg_uptime_percent: 0.0,
            reliability_score: 0.0,
            typical_online_hours: Vec::new(),
            avg_session_duration_minutes: 0.0,
            total_sessions: 0,
        }
    }

    /// Calculate reliability score from uptime percentage
    pub fn calculate_reliability_score(uptime_percent: f64) -> f64 {
        // Cap at 1.0
        (uptime_percent / 100.0).min(1.0).max(0.0)
    }

    /// Check if node is likely online at given hour (0-23)
    pub fn is_likely_online_at_hour(&self, hour: u8) -> bool {
        if hour > 23 {
            return false;
        }
        self.typical_online_hours.contains(&hour)
    }

    /// Get availability probability at given hour (0.0-1.0)
    pub fn availability_probability_at_hour(&self, hour: u8) -> f64 {
        if self.is_likely_online_at_hour(hour) {
            self.reliability_score
        } else {
            // Small probability even outside typical hours
            self.reliability_score * 0.1
        }
    }
}

impl UptimePatternAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }

    /// Add or update a node's uptime pattern
    pub fn update_pattern(&mut self, pattern: UptimePattern) {
        self.patterns.insert(pattern.node_id.clone(), pattern);
    }

    /// Get pattern for a node
    pub fn get_pattern(&self, node_id: &str) -> Option<&UptimePattern> {
        self.patterns.get(node_id)
    }

    /// Calculate uptime pattern from historical data
    pub fn calculate_pattern_from_data(
        node_id: String,
        sessions: Vec<UptimeSession>,
    ) -> Result<UptimePattern, HpcError> {
        if sessions.is_empty() {
            return Ok(UptimePattern::new(node_id));
        }

        let total_sessions = sessions.len() as u64;

        // Calculate average session duration
        let total_minutes: u64 = sessions.iter().map(|s| s.duration_minutes).sum();
        let avg_duration = total_minutes as f64 / total_sessions as f64;

        // Calculate uptime percentage (assuming 30 days)
        let total_minutes_in_30_days = 30.0 * 24.0 * 60.0;
        let uptime_percent = (total_minutes as f64 / total_minutes_in_30_days) * 100.0;
        let reliability_score = UptimePattern::calculate_reliability_score(uptime_percent);

        // Find typical online hours
        let mut hour_counts: HashMap<u8, usize> = HashMap::new();
        for session in &sessions {
            for hour in session.online_hours.iter() {
                *hour_counts.entry(*hour).or_insert(0) += 1;
            }
        }

        // Select hours that appear in at least 30% of sessions
        let threshold = (total_sessions as f64 * 0.3) as usize;
        let mut typical_hours: Vec<u8> = hour_counts
            .iter()
            .filter(|(_, count)| **count >= threshold)
            .map(|(hour, _)| *hour)
            .collect();
        typical_hours.sort_unstable();

        Ok(UptimePattern {
            node_id,
            avg_uptime_percent: uptime_percent.min(100.0),
            reliability_score,
            typical_online_hours: typical_hours,
            avg_session_duration_minutes: avg_duration,
            total_sessions,
        })
    }

    /// Get all patterns
    pub fn all_patterns(&self) -> Vec<&UptimePattern> {
        self.patterns.values().collect()
    }

    /// Remove pattern for a node
    pub fn remove_pattern(&mut self, node_id: &str) -> Option<UptimePattern> {
        self.patterns.remove(node_id)
    }
}

impl Default for UptimePatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Uptime session for pattern calculation
#[derive(Debug, Clone)]
pub struct UptimeSession {
    pub duration_minutes: u64,
    pub online_hours: Vec<u8>,
}

impl UptimeSession {
    /// Create a new uptime session
    pub fn new(duration_minutes: u64, online_hours: Vec<u8>) -> Self {
        Self {
            duration_minutes,
            online_hours,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uptime_pattern_creation() {
        let pattern = UptimePattern::new("node-1".to_string());

        assert_eq!(pattern.node_id, "node-1");
        assert_eq!(pattern.avg_uptime_percent, 0.0);
        assert_eq!(pattern.reliability_score, 0.0);
        assert!(pattern.typical_online_hours.is_empty());
    }

    #[test]
    fn test_calculate_reliability_score() {
        assert_eq!(UptimePattern::calculate_reliability_score(100.0), 1.0);
        assert_eq!(UptimePattern::calculate_reliability_score(50.0), 0.5);
        assert_eq!(UptimePattern::calculate_reliability_score(0.0), 0.0);
    }

    #[test]
    fn test_calculate_reliability_score_capped() {
        // Should cap at 1.0
        assert_eq!(UptimePattern::calculate_reliability_score(150.0), 1.0);
    }

    #[test]
    fn test_is_likely_online_at_hour() {
        let mut pattern = UptimePattern::new("node-1".to_string());
        pattern.typical_online_hours = vec![9, 10, 11, 12, 13, 14, 15, 16, 17];

        assert!(pattern.is_likely_online_at_hour(10));
        assert!(pattern.is_likely_online_at_hour(14));
        assert!(!pattern.is_likely_online_at_hour(22));
        assert!(!pattern.is_likely_online_at_hour(3));
    }

    #[test]
    fn test_is_likely_online_invalid_hour() {
        let pattern = UptimePattern::new("node-1".to_string());
        assert!(!pattern.is_likely_online_at_hour(24));
        assert!(!pattern.is_likely_online_at_hour(25));
    }

    #[test]
    fn test_availability_probability_at_hour() {
        let mut pattern = UptimePattern::new("node-1".to_string());
        pattern.reliability_score = 0.8;
        pattern.typical_online_hours = vec![9, 10, 11, 12];

        // During typical hours, should return reliability score
        assert_eq!(pattern.availability_probability_at_hour(10), 0.8);

        // Outside typical hours, should return 10% of reliability score
        let prob = pattern.availability_probability_at_hour(22);
        assert!((prob - 0.08).abs() < 0.001);
    }

    #[test]
    fn test_uptime_pattern_analyzer_creation() {
        let analyzer = UptimePatternAnalyzer::new();
        assert_eq!(analyzer.all_patterns().len(), 0);
    }

    #[test]
    fn test_update_pattern() {
        let mut analyzer = UptimePatternAnalyzer::new();
        let pattern = UptimePattern::new("node-1".to_string());

        analyzer.update_pattern(pattern);

        assert_eq!(analyzer.all_patterns().len(), 1);
        assert!(analyzer.get_pattern("node-1").is_some());
    }

    #[test]
    fn test_get_pattern() {
        let mut analyzer = UptimePatternAnalyzer::new();
        let mut pattern = UptimePattern::new("node-1".to_string());
        pattern.reliability_score = 0.75;

        analyzer.update_pattern(pattern);

        let retrieved = analyzer.get_pattern("node-1").unwrap();
        assert_eq!(retrieved.reliability_score, 0.75);
    }

    #[test]
    fn test_get_nonexistent_pattern() {
        let analyzer = UptimePatternAnalyzer::new();
        assert!(analyzer.get_pattern("nonexistent").is_none());
    }

    #[test]
    fn test_calculate_pattern_from_empty_data() {
        let result =
            UptimePatternAnalyzer::calculate_pattern_from_data("node-1".to_string(), vec![]);

        assert!(result.is_ok());
        let pattern = result.unwrap();
        assert_eq!(pattern.total_sessions, 0);
        assert_eq!(pattern.avg_uptime_percent, 0.0);
    }

    #[test]
    fn test_calculate_pattern_from_single_session() {
        let sessions = vec![UptimeSession::new(480, vec![9, 10, 11, 12, 13, 14, 15, 16])];

        let result =
            UptimePatternAnalyzer::calculate_pattern_from_data("node-1".to_string(), sessions);

        assert!(result.is_ok());
        let pattern = result.unwrap();
        assert_eq!(pattern.total_sessions, 1);
        assert_eq!(pattern.avg_session_duration_minutes, 480.0);
    }

    #[test]
    fn test_calculate_pattern_from_multiple_sessions() {
        let sessions = vec![
            UptimeSession::new(480, vec![9, 10, 11, 12]),
            UptimeSession::new(420, vec![9, 10, 11]),
            UptimeSession::new(360, vec![9, 10]),
        ];

        let result =
            UptimePatternAnalyzer::calculate_pattern_from_data("node-1".to_string(), sessions);

        assert!(result.is_ok());
        let pattern = result.unwrap();
        assert_eq!(pattern.total_sessions, 3);

        // Average duration = (480 + 420 + 360) / 3 = 420
        assert_eq!(pattern.avg_session_duration_minutes, 420.0);
    }

    #[test]
    fn test_calculate_pattern_typical_hours() {
        // 3 sessions, each with hours [9, 10, 11]
        let sessions = vec![
            UptimeSession::new(180, vec![9, 10, 11]),
            UptimeSession::new(180, vec![9, 10, 11]),
            UptimeSession::new(180, vec![9, 10, 11]),
        ];

        let result =
            UptimePatternAnalyzer::calculate_pattern_from_data("node-1".to_string(), sessions);

        let pattern = result.unwrap();
        // All three hours appear in 100% of sessions (> 30% threshold)
        assert_eq!(pattern.typical_online_hours, vec![9, 10, 11]);
    }

    #[test]
    fn test_remove_pattern() {
        let mut analyzer = UptimePatternAnalyzer::new();
        analyzer.update_pattern(UptimePattern::new("node-1".to_string()));

        let removed = analyzer.remove_pattern("node-1");
        assert!(removed.is_some());
        assert!(analyzer.get_pattern("node-1").is_none());
    }

    #[test]
    fn test_all_patterns() {
        let mut analyzer = UptimePatternAnalyzer::new();
        analyzer.update_pattern(UptimePattern::new("node-1".to_string()));
        analyzer.update_pattern(UptimePattern::new("node-2".to_string()));
        analyzer.update_pattern(UptimePattern::new("node-3".to_string()));

        assert_eq!(analyzer.all_patterns().len(), 3);
    }
}
