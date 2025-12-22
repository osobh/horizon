//! Recovery execution strategies

use super::checkpoint_manager::CheckpointManager;
use super::types::CheckpointSnapshot;
use crate::error::EvolutionEngineResult;
use crate::swarm_distributed::{Migration, MigrationPlan};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Recovery executor trait
pub trait RecoveryExecutor: Send + Sync {
    /// Execute recovery strategy
    fn execute_recovery(
        &self,
        failed_node: &str,
        affected_particles: &[String],
        checkpoint: Option<&CheckpointSnapshot>,
    ) -> EvolutionEngineResult<MigrationPlan>;
}

/// Redistribute recovery strategy - redistributes particles to other nodes
pub struct RedistributeRecovery {
    /// Available nodes for redistribution
    pub(crate) available_nodes: Vec<String>,
}

/// Checkpoint recovery strategy - restores from latest checkpoint
pub struct CheckpointRecovery {
    /// Checkpoint manager reference
    pub(crate) checkpoint_manager: Arc<RwLock<CheckpointManager>>,
}

/// Hybrid recovery strategy - combines redistribution and checkpointing
pub struct HybridRecovery {
    /// Redistribute recovery
    pub(crate) redistribute: RedistributeRecovery,
    /// Checkpoint recovery
    pub(crate) checkpoint: CheckpointRecovery,
}

impl RedistributeRecovery {
    pub fn new(available_nodes: Vec<String>) -> Self {
        Self { available_nodes }
    }
}

impl RecoveryExecutor for RedistributeRecovery {
    fn execute_recovery(
        &self,
        failed_node: &str,
        affected_particles: &[String],
        _checkpoint: Option<&CheckpointSnapshot>,
    ) -> EvolutionEngineResult<MigrationPlan> {
        let mut migrations = Vec::new();

        // Simple round-robin redistribution
        for (i, particle_id) in affected_particles.iter().enumerate() {
            if !self.available_nodes.is_empty() {
                let target_node = &self.available_nodes[i % self.available_nodes.len()];
                migrations.push(Migration {
                    particle_id: particle_id.clone(),
                    from_node: failed_node.to_string(),
                    to_node: target_node.clone(),
                    priority: 1.0,
                    expected_benefit: 0.8,
                });
            }
        }

        Ok(MigrationPlan {
            migrations,
            expected_improvement: 0.8,
            migration_cost: affected_particles.len() as f64 * 0.1,
        })
    }
}

impl CheckpointRecovery {
    pub fn new(checkpoint_manager: Arc<RwLock<CheckpointManager>>) -> Self {
        Self { checkpoint_manager }
    }
}

impl RecoveryExecutor for CheckpointRecovery {
    fn execute_recovery(
        &self,
        failed_node: &str,
        affected_particles: &[String],
        checkpoint: Option<&CheckpointSnapshot>,
    ) -> EvolutionEngineResult<MigrationPlan> {
        let mut migrations = Vec::new();

        if let Some(checkpoint) = checkpoint {
            // Find particles from checkpoint that were on the failed node
            if let Some(_node_state) = checkpoint.node_states.get(failed_node) {
                // Create migrations to restore particles from checkpoint
                for particle_id in affected_particles {
                    // In a real implementation, would find suitable target nodes
                    // For now, create placeholder migrations
                    migrations.push(Migration {
                        particle_id: particle_id.clone(),
                        from_node: failed_node.to_string(),
                        to_node: "recovery_node".to_string(),
                        priority: 1.0,
                        expected_benefit: 0.9,
                    });
                }
            }
        }

        Ok(MigrationPlan {
            migrations,
            expected_improvement: 0.9,
            migration_cost: affected_particles.len() as f64 * 0.05,
        })
    }
}

impl HybridRecovery {
    pub fn new(checkpoint_manager: Arc<RwLock<CheckpointManager>>) -> Self {
        Self {
            redistribute: RedistributeRecovery::new(vec![
                "backup_node1".to_string(),
                "backup_node2".to_string(),
            ]),
            checkpoint: CheckpointRecovery::new(checkpoint_manager),
        }
    }
}

impl RecoveryExecutor for HybridRecovery {
    fn execute_recovery(
        &self,
        failed_node: &str,
        affected_particles: &[String],
        checkpoint: Option<&CheckpointSnapshot>,
    ) -> EvolutionEngineResult<MigrationPlan> {
        // Try checkpoint recovery first
        let checkpoint_plan =
            self.checkpoint
                .execute_recovery(failed_node, affected_particles, checkpoint)?;

        // If checkpoint recovery doesn't cover all particles, use redistribution
        if checkpoint_plan.migrations.len() < affected_particles.len() {
            let redistribute_plan =
                self.redistribute
                    .execute_recovery(failed_node, affected_particles, checkpoint)?;

            // Combine both plans
            let mut combined_migrations = checkpoint_plan.migrations;
            combined_migrations.extend(redistribute_plan.migrations);

            Ok(MigrationPlan {
                migrations: combined_migrations,
                expected_improvement: 0.85,
                migration_cost: affected_particles.len() as f64 * 0.075,
            })
        } else {
            Ok(checkpoint_plan)
        }
    }
}
