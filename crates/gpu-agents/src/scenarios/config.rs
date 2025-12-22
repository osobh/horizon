//! Configuration structures for GPU agent swarm scenarios
//!
//! This module defines the configuration format for various agent scenarios
//! used in stress testing and benchmarking.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Main scenario configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScenarioConfig {
    /// Unique identifier for the scenario
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Description of what this scenario tests
    pub description: String,

    /// Type of scenario to run
    pub scenario_type: ScenarioType,

    /// Number of agents to simulate
    pub agent_count: usize,

    /// Duration to run the scenario
    pub duration: Duration,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Performance objectives
    pub objectives: Vec<PerformanceObjective>,
}

/// Types of scenarios available
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ScenarioType {
    /// Simple reactive agents
    Simple {
        behavior: SimpleBehavior,
        interaction_radius: f32,
        update_frequency: f32,
    },

    /// LLM-based reasoning agents
    Reasoning { config: ReasoningConfig },

    /// Knowledge graph agents with memory
    Knowledge { config: KnowledgeConfig },
}

/// Agent type classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AgentType {
    Simple,
    Reasoning,
    Knowledge,
}

/// Simple agent behaviors
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SimpleBehavior {
    /// Random movement
    RandomWalk,
    /// Move towards a target
    Seeking,
    /// Form groups
    Flocking,
    /// Avoid obstacles
    Avoidance,
    /// Mixed behaviors
    Composite,
}

/// Configuration for reasoning agents
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReasoningConfig {
    /// LLM model to use
    pub model: String,

    /// Maximum tokens per inference
    pub max_tokens: usize,

    /// Temperature for generation
    pub temperature: f32,

    /// Prompt complexity level
    pub prompt_complexity: PromptComplexity,

    /// Batch size for inference
    pub batch_size: usize,

    /// Decision frequency (decisions per second)
    pub decision_frequency: f32,
}

/// Configuration for knowledge graph agents
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeConfig {
    /// Initial knowledge graph size
    pub initial_nodes: usize,

    /// Maximum graph size per agent
    pub max_nodes: usize,

    /// Memory access patterns
    pub memory_pattern: MemoryPattern,

    /// Update frequency for knowledge
    pub update_frequency: f32,

    /// Sharing ratio between agents
    pub sharing_ratio: f32,
}

/// Memory access patterns for knowledge agents
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPattern {
    /// Random access
    Random,
    /// Recent items first
    Temporal,
    /// Most connected nodes
    Associative,
    /// Priority-based
    Hierarchical,
}

/// Prompt complexity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PromptComplexity {
    /// Simple one-shot prompts
    Simple,
    /// Chain-of-thought reasoning
    Moderate,
    /// Multi-step planning
    Complex,
    /// Full autonomous agent
    Advanced,
}

/// Performance objectives to measure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerformanceObjective {
    /// Name of the metric
    pub metric: String,

    /// Target value
    pub target: f64,

    /// Whether higher is better
    pub maximize: bool,
}

impl ScenarioConfig {
    /// Load configuration from YAML file
    pub fn from_yaml(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from TOML file
    pub fn from_toml(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.agent_count == 0 {
            return Err("Agent count must be greater than 0".into());
        }

        if self.duration.as_secs() == 0 {
            return Err("Duration must be greater than 0".into());
        }

        match &self.scenario_type {
            ScenarioType::Simple {
                interaction_radius,
                update_frequency,
                ..
            } => {
                if *interaction_radius <= 0.0 {
                    return Err("Interaction radius must be positive".into());
                }
                if *update_frequency <= 0.0 {
                    return Err("Update frequency must be positive".into());
                }
            }
            ScenarioType::Reasoning { config } => {
                if config.max_tokens == 0 {
                    return Err("Max tokens must be greater than 0".into());
                }
                if config.temperature < 0.0 || config.temperature > 2.0 {
                    return Err("Temperature must be between 0.0 and 2.0".into());
                }
                if config.batch_size == 0 {
                    return Err("Batch size must be greater than 0".into());
                }
                if config.decision_frequency <= 0.0 {
                    return Err("Decision frequency must be positive".into());
                }
            }
            ScenarioType::Knowledge { config } => {
                if config.max_nodes < config.initial_nodes {
                    return Err("Max nodes must be >= initial nodes".into());
                }
                if config.update_frequency <= 0.0 {
                    return Err("Update frequency must be positive".into());
                }
                if config.sharing_ratio < 0.0 || config.sharing_ratio > 1.0 {
                    return Err("Sharing ratio must be between 0.0 and 1.0".into());
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
