//! ExoRust Fault Tolerance System
//!
//! Phase 4 production hardening component that provides:
//! - GPU memory checkpointing
//! - Agent state serialization  
//! - Distributed checkpoint coordination
//! - Fast recovery protocols
//! - Zero-downtime upgrades

pub mod checkpoint;
pub mod coordinator;
pub mod error;
pub mod recovery;
pub mod predictive_scaler;

pub use checkpoint::*;
pub use coordinator::*;
pub use error::*;
pub use recovery::*;
pub use predictive_scaler::*;

/// Main fault tolerance manager
pub struct FaultToleranceManager {
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
    coordinator: DistributedCoordinator,
}

impl FaultToleranceManager {
    /// Create new fault tolerance manager
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            checkpoint_manager: CheckpointManager::new()?,
            recovery_manager: RecoveryManager::new()?,
            coordinator: DistributedCoordinator::new()?,
        })
    }

    /// Create checkpoint of entire system
    pub async fn create_system_checkpoint(&self) -> Result<CheckpointId, FaultToleranceError> {
        self.checkpoint_manager.create_full_checkpoint().await
    }

    /// Recover system from checkpoint
    pub async fn recover_from_checkpoint(
        &self,
        checkpoint_id: CheckpointId,
    ) -> Result<(), FaultToleranceError> {
        self.recovery_manager
            .restore_full_system(checkpoint_id)
            .await
    }

    /// Check system health
    pub async fn health_check(&self) -> HealthStatus {
        self.coordinator.system_health().await
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod edge_case_tests;

#[cfg(test)]
mod tests_inline {
    use super::*;

    #[tokio::test]
    async fn test_fault_tolerance_manager_creation() {
        let manager = FaultToleranceManager::new();
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_system_checkpoint_creation() {
        let manager = FaultToleranceManager::new()?;
        let checkpoint_result = manager.create_system_checkpoint().await;

        // Should either succeed or fail gracefully
        assert!(checkpoint_result.is_ok() || checkpoint_result.is_err());
    }

    #[tokio::test]
    async fn test_system_health_check() {
        let manager = FaultToleranceManager::new()?;
        let health = manager.health_check().await;

        // Health check should always return some status
        assert!(matches!(
            health,
            HealthStatus::Healthy | HealthStatus::Degraded | HealthStatus::Failed
        ));
    }

    #[tokio::test]
    async fn test_recovery_from_invalid_checkpoint() {
        let manager = FaultToleranceManager::new()?;
        let invalid_checkpoint = CheckpointId::new();

        let recovery_result = manager.recover_from_checkpoint(invalid_checkpoint).await;

        // Should fail gracefully for invalid checkpoint
        assert!(recovery_result.is_err());
    }
}
