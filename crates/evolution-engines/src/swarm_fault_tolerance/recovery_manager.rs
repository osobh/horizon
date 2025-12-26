//! Recovery management for failed nodes

use super::checkpoint_manager::CheckpointManager;
use super::fault_detector::FaultToleranceConfig;
use super::recovery_executors::RecoveryExecutor;
use super::types::{RecoveryEvent, RecoveryStatus};
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use crate::swarm_distributed::{MigrationPlan, RecoveryStrategy};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Recovery manager for handling node failures
pub struct RecoveryManager {
    /// Configuration
    pub(crate) config: FaultToleranceConfig,
    /// Checkpoint manager reference
    pub(crate) checkpoint_manager: Arc<RwLock<CheckpointManager>>,
    /// Recovery strategies
    pub(crate) recovery_strategies: HashMap<RecoveryStrategy, Box<dyn RecoveryExecutor>>,
    /// Recovery event history
    pub(crate) recovery_history: Vec<RecoveryEvent>,
}

impl RecoveryManager {
    /// Create new recovery manager
    pub async fn new(
        config: FaultToleranceConfig,
        checkpoint_manager: Arc<RwLock<CheckpointManager>>,
    ) -> EvolutionEngineResult<Self> {
        let mut recovery_strategies: HashMap<RecoveryStrategy, Box<dyn RecoveryExecutor>> =
            HashMap::new();

        // Add recovery strategies
        recovery_strategies.insert(
            RecoveryStrategy::Redistribute,
            Box::new(super::recovery_executors::RedistributeRecovery::new(vec![])),
        );
        recovery_strategies.insert(
            RecoveryStrategy::Checkpoint,
            Box::new(super::recovery_executors::CheckpointRecovery::new(
                checkpoint_manager.clone(),
            )),
        );
        recovery_strategies.insert(
            RecoveryStrategy::Hybrid,
            Box::new(super::recovery_executors::HybridRecovery::new(
                checkpoint_manager.clone(),
            )),
        );

        Ok(Self {
            config,
            checkpoint_manager,
            recovery_strategies,
            recovery_history: Vec::new(),
        })
    }

    /// Execute recovery for a failed node
    pub async fn execute_recovery(
        &mut self,
        failed_node: &str,
        affected_particles: &[String],
    ) -> EvolutionEngineResult<MigrationPlan> {
        let recovery_event = RecoveryEvent {
            id: uuid::Uuid::new_v4().to_string(),
            failed_node: failed_node.to_string(),
            strategy: self.config.recovery_strategy.clone(),
            failure_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            recovery_start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            recovery_completion_time: None,
            status: RecoveryStatus::InProgress,
            particles_affected: affected_particles.len(),
            success_rate: 0.0,
        };

        self.recovery_history.push(recovery_event.clone());

        // Get latest checkpoint if needed
        let checkpoint = if matches!(
            self.config.recovery_strategy,
            RecoveryStrategy::Checkpoint | RecoveryStrategy::Hybrid
        ) {
            self.checkpoint_manager
                .read()
                .await
                .get_latest_checkpoint()
                .await?
        } else {
            None
        };

        // Execute recovery strategy
        let recovery_executor = self
            .recovery_strategies
            .get(&self.config.recovery_strategy)
            .ok_or_else(|| EvolutionEngineError::InvalidConfiguration {
                message: format!(
                    "Unknown recovery strategy: {:?}",
                    self.config.recovery_strategy
                ),
            })?;

        let migration_plan = recovery_executor.execute_recovery(
            failed_node,
            affected_particles,
            checkpoint.as_ref(),
        )?;

        // Update recovery event
        let last_event = match self.recovery_history.last_mut() {
            Some(event) => event,
            None => return Err(EvolutionEngineError::Other("No recovery event found".to_string())),
        };
        last_event.recovery_completion_time = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        );
        last_event.status = RecoveryStatus::Completed;
        last_event.success_rate = 1.0; // Simplistic success rate

        Ok(migration_plan)
    }

    /// Get recovery history
    pub fn get_recovery_history(&self) -> &[RecoveryEvent] {
        &self.recovery_history
    }
}
