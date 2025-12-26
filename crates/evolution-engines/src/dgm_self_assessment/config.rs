//! Configuration types for self-assessment

use serde::{Deserialize, Serialize};

/// Self-assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAssessmentConfig {
    /// History window size for tracking
    pub history_window: usize,
    /// Minimum improvement threshold
    pub min_improvement_threshold: f64,
    /// Assessment interval (generations)
    pub assessment_interval: u32,
    /// Pattern effectiveness threshold
    pub pattern_effectiveness_threshold: f64,
    /// Enable detailed logging
    pub enable_detailed_logging: bool,
}

impl Default for SelfAssessmentConfig {
    fn default() -> Self {
        Self {
            history_window: 50,
            min_improvement_threshold: 0.01,
            assessment_interval: 10,
            pattern_effectiveness_threshold: 0.6,
            enable_detailed_logging: false,
        }
    }
}
