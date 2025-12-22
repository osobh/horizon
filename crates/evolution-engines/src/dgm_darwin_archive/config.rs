//! Configuration for Darwin archive system

use serde::{Deserialize, Serialize};

/// Configuration for Darwin archive behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarwinArchiveConfig {
    /// Maximum archive size (0 = unlimited)
    pub max_size: usize,
    /// Minimum performance to be archived
    pub min_performance_threshold: f64,
    /// Weight for performance in parent selection
    pub performance_weight: f64,
    /// Weight for number of children in parent selection
    pub children_weight: f64,
    /// Weight for diversity contribution in parent selection
    pub diversity_weight: f64,
    /// Number of generations to track for stepping stones
    pub stepping_stone_window: usize,
    /// Enable detailed logging
    pub enable_logging: bool,
}

impl Default for DarwinArchiveConfig {
    fn default() -> Self {
        Self {
            max_size: 10000,
            min_performance_threshold: 0.0,
            performance_weight: 0.5,
            children_weight: 0.3,
            diversity_weight: 0.2,
            stepping_stone_window: 10,
            enable_logging: false,
        }
    }
}
