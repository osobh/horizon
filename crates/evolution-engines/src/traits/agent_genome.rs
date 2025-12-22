//! Agent genome and related types

use exorust_agent_core::Goal;
use serde::{Deserialize, Serialize};

/// Agent genome representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGenome {
    /// Goal specification
    pub goal: Goal,
    /// Architecture parameters
    pub architecture: ArchitectureGenes,
    /// Behavioral parameters
    pub behavior: BehaviorGenes,
}

/// Architecture genes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureGenes {
    /// Memory capacity
    pub memory_capacity: usize,
    /// Processing units
    pub processing_units: u32,
    /// Network topology
    pub network_topology: Vec<u32>,
}

/// Behavior genes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorGenes {
    /// Exploration rate
    pub exploration_rate: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Risk tolerance
    pub risk_tolerance: f64,
}

impl Default for ArchitectureGenes {
    fn default() -> Self {
        Self {
            memory_capacity: 1024,
            processing_units: 1,
            network_topology: vec![10],
        }
    }
}

impl Default for BehaviorGenes {
    fn default() -> Self {
        Self {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        }
    }
}
