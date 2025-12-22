//! Search space definitions for ADAS

use serde::{Deserialize, Serialize};

/// Architecture search space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureSearchSpace {
    /// Memory capacity range [min, max]
    pub memory_capacity_range: (usize, usize),
    /// Processing units range [min, max]
    pub processing_units_range: (u32, u32),
    /// Network depth range [min, max]
    pub network_depth_range: (usize, usize),
    /// Network width range [min, max]
    pub network_width_range: (u32, u32),
}

impl Default for ArchitectureSearchSpace {
    fn default() -> Self {
        Self {
            memory_capacity_range: (1024, 1024 * 1024), // 1KB to 1MB
            processing_units_range: (1, 16),
            network_depth_range: (2, 10),
            network_width_range: (10, 1000),
        }
    }
}

/// Behavior search space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorSearchSpace {
    /// Exploration rate range [min, max]
    pub exploration_rate_range: (f64, f64),
    /// Learning rate range [min, max]
    pub learning_rate_range: (f64, f64),
    /// Risk tolerance range [min, max]
    pub risk_tolerance_range: (f64, f64),
}

impl Default for BehaviorSearchSpace {
    fn default() -> Self {
        Self {
            exploration_rate_range: (0.01, 0.5),
            learning_rate_range: (0.0001, 0.1),
            risk_tolerance_range: (0.0, 1.0),
        }
    }
}
