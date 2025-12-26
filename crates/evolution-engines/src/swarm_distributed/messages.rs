//! Message types for inter-node communication

use super::types::SwarmNode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Message types for inter-node communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedMessage {
    /// Heartbeat message to maintain node liveness
    Heartbeat {
        node_id: String,
        timestamp: u64,
        load: f64,
    },
    /// Particle migration message
    ParticleMigration {
        particles: Vec<MigrationParticle>,
        source_node: String,
        target_node: String,
    },
    /// Global best update message
    GlobalBestUpdate {
        best_particle: MigrationParticle,
        fitness: f64,
        generation: u32,
    },
    /// Node join request
    NodeJoin { node_info: SwarmNode },
    /// Node leave notification
    NodeLeave { node_id: String, reason: String },
    /// Checkpoint synchronization
    CheckpointSync {
        checkpoint_data: CheckpointData,
        generation: u32,
    },
}

/// Particle data for migration between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationParticle {
    /// Particle identifier
    pub id: String,
    /// Agent genome
    pub genome: crate::traits::AgentGenome,
    /// Particle velocity
    pub velocity: Vec<f64>,
    /// Personal best fitness
    pub personal_best_fitness: f64,
    /// Current fitness
    pub current_fitness: Option<f64>,
}

/// Checkpoint data for fault tolerance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    /// Generation number
    pub generation: u32,
    /// Global best particle
    pub global_best: Option<MigrationParticle>,
    /// Global best fitness
    pub global_best_fitness: Option<f64>,
    /// Node particle assignments
    pub node_assignments: HashMap<String, Vec<String>>, // node_id -> particle_ids
    /// Checkpoint timestamp
    pub timestamp: u64,
}

impl DistributedMessage {
    /// Get message type as string
    pub fn message_type(&self) -> &'static str {
        match self {
            DistributedMessage::Heartbeat { .. } => "heartbeat",
            DistributedMessage::ParticleMigration { .. } => "particle_migration",
            DistributedMessage::GlobalBestUpdate { .. } => "global_best_update",
            DistributedMessage::NodeJoin { .. } => "node_join",
            DistributedMessage::NodeLeave { .. } => "node_leave",
            DistributedMessage::CheckpointSync { .. } => "checkpoint_sync",
        }
    }

    /// Get source node ID if applicable
    pub fn source_node(&self) -> Option<&str> {
        match self {
            DistributedMessage::Heartbeat { node_id, .. } => Some(node_id),
            DistributedMessage::ParticleMigration { source_node, .. } => Some(source_node),
            DistributedMessage::NodeJoin { node_info } => Some(&node_info.node_id),
            DistributedMessage::NodeLeave { node_id, .. } => Some(node_id),
            _ => None,
        }
    }
}

impl MigrationParticle {
    /// Convert to velocity vector for optimization
    pub fn to_velocity_vector(&self) -> Vec<f64> {
        self.velocity.clone()
    }

    /// Create from evolvable agent (stub for now)
    pub fn from_agent(
        id: String,
        _agent: &crate::traits::MockEvolvableAgent,
        velocity: Vec<f64>,
        personal_best_fitness: f64,
    ) -> Self {
        // Create a mock genome for testing
        let genome = crate::traits::AgentGenome {
            goal: stratoswarm_agent_core::Goal::new(
                "mock".to_string(),
                stratoswarm_agent_core::GoalPriority::Normal,
            ),
            architecture: crate::traits::ArchitectureGenes {
                memory_capacity: 1024,
                processing_units: 1,
                network_topology: vec![10],
            },
            behavior: crate::traits::BehaviorGenes {
                exploration_rate: 0.5,
                learning_rate: 0.01,
                risk_tolerance: 0.5,
            },
        };

        Self {
            id,
            genome,
            velocity,
            personal_best_fitness,
            current_fitness: None,
        }
    }
}
