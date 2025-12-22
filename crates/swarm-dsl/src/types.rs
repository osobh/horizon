//! Types for agent specifications produced by the compiler

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Agent specification produced by the compiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    /// Agent name
    pub name: String,
    /// Agent type (e.g., WebAgent, ComputeAgent)
    pub agent_type: String,
    /// Number of replicas (min, max for auto-scaling)
    pub replicas: (u32, Option<u32>),
    /// Agent configuration
    pub config: AgentConfig,
    /// Outgoing connections to other agents
    pub connections: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Resource requirements
    pub resources: ResourceRequirements,
    /// Network configuration
    pub network: NetworkConfig,
    /// Personality traits
    pub personality: PersonalityTraits,
    /// Whether evolution is enabled
    pub evolution_enabled: bool,
    /// Tier preferences for resource allocation
    pub tier_preferences: Vec<String>,
}

/// Resource requirements for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU cores
    pub cpu: f64,
    /// Memory in bytes
    pub memory: u64,
    /// GPU requirement (optional)
    pub gpu: Option<f64>,
    /// Whether GPU is optional
    pub gpu_optional: bool,
}

/// Network configuration for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Ports to expose
    pub expose_ports: Vec<u16>,
    /// Enable service mesh
    pub enable_mesh: bool,
    /// Load balancing strategy
    pub load_balance_strategy: String,
}

/// Personality traits that affect agent behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityTraits {
    /// Risk tolerance (0-1)
    pub risk_tolerance: f32,
    /// Cooperation level (0-1)
    pub cooperation: f32,
    /// Exploration tendency (0-1)
    pub exploration: f32,
    /// Efficiency focus (0-1)
    pub efficiency_focus: f32,
    /// Stability preference (0-1)
    pub stability_preference: f32,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu: 1.0,
            memory: 1024 * 1024 * 1024, // 1GB
            gpu: None,
            gpu_optional: false,
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            expose_ports: Vec::new(),
            enable_mesh: false,
            load_balance_strategy: "round_robin".to_string(),
        }
    }
}

impl Default for PersonalityTraits {
    fn default() -> Self {
        Self {
            risk_tolerance: 0.5,
            cooperation: 0.5,
            exploration: 0.5,
            efficiency_focus: 0.5,
            stability_preference: 0.5,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            resources: ResourceRequirements::default(),
            network: NetworkConfig::default(),
            personality: PersonalityTraits::default(),
            evolution_enabled: false,
            tier_preferences: vec!["CPU".to_string(), "Memory".to_string()],
        }
    }
}
