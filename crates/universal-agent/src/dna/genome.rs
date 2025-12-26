//! Genome structures for behavior and architecture configuration

use serde::{Deserialize, Serialize};

pub use horizon_agents_core::AutonomyLevel;

/// Behavioral parameters governing agent actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorGenome {
    /// Exploration vs exploitation balance (0.0 = pure exploitation, 1.0 = pure exploration)
    pub exploration_rate: f64,
    /// Learning rate for skill improvement (0.0 - 1.0)
    pub learning_rate: f64,
    /// Risk tolerance for autonomous actions (0.0 = conservative, 1.0 = aggressive)
    pub risk_tolerance: f64,
    /// Autonomy level (from existing AutonomyLevel)
    pub autonomy_level: AutonomyLevel,
    /// Response style preferences
    pub response_preferences: ResponsePreferences,
    /// Resource usage preferences
    pub resource_preferences: ResourcePreferences,
}

impl Default for BehaviorGenome {
    fn default() -> Self {
        Self {
            exploration_rate: 0.2,
            learning_rate: 0.1,
            risk_tolerance: 0.3,
            autonomy_level: AutonomyLevel::Medium,
            response_preferences: ResponsePreferences::default(),
            resource_preferences: ResourcePreferences::default(),
        }
    }
}

impl BehaviorGenome {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a conservative behavior profile
    pub fn conservative() -> Self {
        Self {
            exploration_rate: 0.1,
            learning_rate: 0.05,
            risk_tolerance: 0.1,
            autonomy_level: AutonomyLevel::Low,
            response_preferences: ResponsePreferences {
                verbosity: 0.7,
                action_bias: 0.2,
                confidence_threshold: 0.9,
            },
            resource_preferences: ResourcePreferences::default(),
        }
    }

    /// Create an aggressive behavior profile
    pub fn aggressive() -> Self {
        Self {
            exploration_rate: 0.5,
            learning_rate: 0.2,
            risk_tolerance: 0.7,
            autonomy_level: AutonomyLevel::High,
            response_preferences: ResponsePreferences {
                verbosity: 0.3,
                action_bias: 0.8,
                confidence_threshold: 0.6,
            },
            resource_preferences: ResourcePreferences::default(),
        }
    }

    /// Adjust exploration rate (clamped to 0.0 - 1.0)
    pub fn with_exploration_rate(mut self, rate: f64) -> Self {
        self.exploration_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Adjust learning rate (clamped to 0.0 - 1.0)
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Adjust risk tolerance (clamped to 0.0 - 1.0)
    pub fn with_risk_tolerance(mut self, tolerance: f64) -> Self {
        self.risk_tolerance = tolerance.clamp(0.0, 1.0);
        self
    }

    /// Set autonomy level
    pub fn with_autonomy_level(mut self, level: AutonomyLevel) -> Self {
        self.autonomy_level = level;
        self
    }

    /// Apply a behavior adjustment
    pub fn adjust(&mut self, parameter: &str, delta: f64) -> bool {
        match parameter {
            "exploration_rate" => {
                self.exploration_rate = (self.exploration_rate + delta).clamp(0.0, 1.0);
                true
            }
            "learning_rate" => {
                self.learning_rate = (self.learning_rate + delta).clamp(0.0, 1.0);
                true
            }
            "risk_tolerance" => {
                self.risk_tolerance = (self.risk_tolerance + delta).clamp(0.0, 1.0);
                true
            }
            "verbosity" => {
                self.response_preferences.verbosity =
                    (self.response_preferences.verbosity + delta).clamp(0.0, 1.0);
                true
            }
            "action_bias" => {
                self.response_preferences.action_bias =
                    (self.response_preferences.action_bias + delta).clamp(0.0, 1.0);
                true
            }
            "confidence_threshold" => {
                self.response_preferences.confidence_threshold =
                    (self.response_preferences.confidence_threshold + delta).clamp(0.0, 1.0);
                true
            }
            _ => false,
        }
    }
}

/// Response style preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsePreferences {
    /// Verbosity level (0.0 = terse, 1.0 = detailed)
    pub verbosity: f64,
    /// Prefer actions vs recommendations (0.0 = recommendations, 1.0 = actions)
    pub action_bias: f64,
    /// Confidence threshold for autonomous actions
    pub confidence_threshold: f64,
}

impl Default for ResponsePreferences {
    fn default() -> Self {
        Self {
            verbosity: 0.5,
            action_bias: 0.5,
            confidence_threshold: 0.8,
        }
    }
}

/// Resource usage preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePreferences {
    /// Memory efficiency vs speed tradeoff (0.0 = memory efficient, 1.0 = speed)
    pub memory_vs_speed: f64,
    /// Parallelism preference (1.0 = sequential, higher = more parallel)
    pub parallelism_factor: f64,
    /// Batch size preference (1 = no batching, higher = larger batches)
    pub batch_preference: f64,
}

impl Default for ResourcePreferences {
    fn default() -> Self {
        Self {
            memory_vs_speed: 0.5,
            parallelism_factor: 2.0,
            batch_preference: 1.0,
        }
    }
}

/// Architectural configuration for the agent runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureGenome {
    /// Memory capacity in bytes
    pub memory_capacity: usize,
    /// Processing units available
    pub processing_units: u32,
    /// Network topology for multi-agent scenarios
    pub network_topology: Vec<u32>,
    /// Context window size for LLM operations
    pub context_window: usize,
    /// Available tools (tool IDs)
    pub available_tools: Vec<String>,
    /// Maximum concurrent operations
    pub max_concurrent_ops: u32,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

impl Default for ArchitectureGenome {
    fn default() -> Self {
        Self {
            memory_capacity: 1024 * 1024 * 100, // 100MB
            processing_units: 4,
            network_topology: vec![],
            context_window: 8192,
            available_tools: vec![],
            max_concurrent_ops: 10,
            retry_config: RetryConfig::default(),
        }
    }
}

impl ArchitectureGenome {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_memory_capacity(mut self, capacity: usize) -> Self {
        self.memory_capacity = capacity;
        self
    }

    pub fn with_processing_units(mut self, units: u32) -> Self {
        self.processing_units = units;
        self
    }

    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.available_tools = tools;
        self
    }

    pub fn with_max_concurrent_ops(mut self, max: u32) -> Self {
        self.max_concurrent_ops = max;
        self
    }

    pub fn add_tool(&mut self, tool_id: String) {
        if !self.available_tools.contains(&tool_id) {
            self.available_tools.push(tool_id);
        }
    }

    pub fn remove_tool(&mut self, tool_id: &str) {
        self.available_tools.retain(|t| t != tool_id);
    }
}

/// Retry configuration for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Initial backoff in milliseconds
    pub initial_backoff_ms: u64,
    /// Maximum backoff in milliseconds
    pub max_backoff_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 10000,
            backoff_multiplier: 2.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavior_genome_defaults() {
        let genome = BehaviorGenome::default();
        assert_eq!(genome.exploration_rate, 0.2);
        assert_eq!(genome.learning_rate, 0.1);
        assert_eq!(genome.risk_tolerance, 0.3);
    }

    #[test]
    fn test_behavior_adjustment() {
        let mut genome = BehaviorGenome::default();
        assert!(genome.adjust("exploration_rate", 0.3));
        assert_eq!(genome.exploration_rate, 0.5);

        // Test clamping
        assert!(genome.adjust("exploration_rate", 1.0));
        assert_eq!(genome.exploration_rate, 1.0);

        // Unknown parameter
        assert!(!genome.adjust("unknown", 0.5));
    }

    #[test]
    fn test_architecture_genome() {
        let genome = ArchitectureGenome::new()
            .with_memory_capacity(1024 * 1024 * 200)
            .with_processing_units(8)
            .with_tools(vec!["tool1".to_string(), "tool2".to_string()]);

        assert_eq!(genome.memory_capacity, 1024 * 1024 * 200);
        assert_eq!(genome.processing_units, 8);
        assert_eq!(genome.available_tools.len(), 2);
    }
}
