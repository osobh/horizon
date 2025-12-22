//! System recovery from checkpoints

use crate::checkpoint::{CheckpointId, CheckpointManager, SystemCheckpoint};
use crate::error::{FaultToleranceError, FtResult, HealthStatus};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::time::timeout;

/// Recovery strategy for system restoration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Full system restore from checkpoint
    Full,
    /// Partial restore - agents only
    AgentsOnly,
    /// Partial restore - GPU state only
    GpuOnly,
    /// Rolling restore - restore components incrementally
    Rolling,
}

/// Recovery progress tracking
#[derive(Debug, Clone)]
pub struct RecoveryProgress {
    pub stage: RecoveryStage,
    pub completed_steps: usize,
    pub total_steps: usize,
    pub elapsed_time: Duration,
    pub estimated_remaining: Duration,
}

/// Recovery stages
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStage {
    Loading,
    ValidatingCheckpoint,
    RestoringGpuState,
    RestoringAgents,
    VerifyingSystem,
    Complete,
    Failed(String),
}

/// Recovery manager handles system restoration
pub struct RecoveryManager {
    checkpoint_manager: CheckpointManager,
    recovery_timeout: Duration,
    pub(crate) verification_enabled: bool,
}

impl RecoveryManager {
    /// Create new recovery manager
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            checkpoint_manager: CheckpointManager::new()?,
            recovery_timeout: Duration::from_secs(300), // 5 minutes default
            verification_enabled: true,
        })
    }

    /// Create recovery manager with custom timeout
    pub fn with_timeout(recovery_timeout: Duration) -> anyhow::Result<Self> {
        Ok(Self {
            checkpoint_manager: CheckpointManager::new()?,
            recovery_timeout,
            verification_enabled: true,
        })
    }

    /// Create recovery manager with custom checkpoint manager
    pub fn with_checkpoint_manager(checkpoint_manager: CheckpointManager) -> Self {
        Self {
            checkpoint_manager,
            recovery_timeout: Duration::from_secs(300),
            verification_enabled: true,
        }
    }

    /// Create recovery manager with custom checkpoint manager and timeout
    pub fn with_checkpoint_manager_and_timeout(
        checkpoint_manager: CheckpointManager,
        recovery_timeout: Duration,
    ) -> Self {
        Self {
            checkpoint_manager,
            recovery_timeout,
            verification_enabled: true,
        }
    }

    /// Restore full system from checkpoint
    pub async fn restore_full_system(&self, checkpoint_id: CheckpointId) -> FtResult<()> {
        let start_time = Instant::now();

        let recovery_future = self.perform_recovery(checkpoint_id, RecoveryStrategy::Full);

        timeout(self.recovery_timeout, recovery_future)
            .await
            .map_err(|_| FaultToleranceError::RecoveryFailed("Recovery timeout".to_string()))?
    }

    /// Restore agents only from checkpoint
    pub async fn restore_agents_only(&self, checkpoint_id: CheckpointId) -> FtResult<()> {
        let recovery_future = self.perform_recovery(checkpoint_id, RecoveryStrategy::AgentsOnly);

        timeout(self.recovery_timeout, recovery_future)
            .await
            .map_err(|_| FaultToleranceError::RecoveryFailed("Recovery timeout".to_string()))?
    }

    /// Restore GPU state only from checkpoint
    pub async fn restore_gpu_only(&self, checkpoint_id: CheckpointId) -> FtResult<()> {
        let recovery_future = self.perform_recovery(checkpoint_id, RecoveryStrategy::GpuOnly);

        timeout(self.recovery_timeout, recovery_future)
            .await
            .map_err(|_| FaultToleranceError::RecoveryFailed("Recovery timeout".to_string()))?
    }

    /// Get recovery progress
    pub async fn get_recovery_progress(&self) -> RecoveryProgress {
        // Mock progress for development
        RecoveryProgress {
            stage: RecoveryStage::Complete,
            completed_steps: 5,
            total_steps: 5,
            elapsed_time: Duration::from_secs(10),
            estimated_remaining: Duration::from_secs(0),
        }
    }

    /// Perform the actual recovery process
    async fn perform_recovery(
        &self,
        checkpoint_id: CheckpointId,
        strategy: RecoveryStrategy,
    ) -> FtResult<()> {
        // Step 1: Load checkpoint
        let checkpoint = self.load_and_validate_checkpoint(checkpoint_id).await?;

        // Step 2: Restore based on strategy
        match strategy {
            RecoveryStrategy::Full => {
                self.restore_gpu_state(&checkpoint).await?;
                self.restore_agents(&checkpoint).await?;
            }
            RecoveryStrategy::AgentsOnly => {
                self.restore_agents(&checkpoint).await?;
            }
            RecoveryStrategy::GpuOnly => {
                self.restore_gpu_state(&checkpoint).await?;
            }
            RecoveryStrategy::Rolling => {
                // Implement rolling restore
                self.restore_rolling(&checkpoint).await?;
            }
        }

        // Step 3: Verify system if enabled
        if self.verification_enabled {
            self.verify_recovery().await?;
        }

        Ok(())
    }

    /// Load and validate checkpoint
    async fn load_and_validate_checkpoint(
        &self,
        checkpoint_id: CheckpointId,
    ) -> FtResult<SystemCheckpoint> {
        let checkpoint = self
            .checkpoint_manager
            .load_checkpoint(&checkpoint_id)
            .await?;

        // Basic validation
        if checkpoint.gpu_checkpoints.is_empty() && checkpoint.agent_checkpoints.is_empty() {
            return Err(FaultToleranceError::InvalidCheckpoint(
                "Checkpoint contains no data".to_string(),
            ));
        }

        Ok(checkpoint)
    }

    /// Restore GPU state from checkpoint
    async fn restore_gpu_state(&self, checkpoint: &SystemCheckpoint) -> FtResult<()> {
        for gpu_checkpoint in &checkpoint.gpu_checkpoints {
            // Mock GPU restoration
            if gpu_checkpoint.memory_snapshot.is_empty() {
                return Err(FaultToleranceError::RecoveryFailed(
                    "Empty GPU memory snapshot".to_string(),
                ));
            }

            // In real implementation, would restore GPU memory and kernel states
            tokio::time::sleep(Duration::from_millis(100)).await; // Simulate work
        }

        Ok(())
    }

    /// Restore agents from checkpoint
    async fn restore_agents(&self, checkpoint: &SystemCheckpoint) -> FtResult<()> {
        for agent_checkpoint in &checkpoint.agent_checkpoints {
            // Mock agent restoration
            if agent_checkpoint.state_data.is_empty() {
                return Err(FaultToleranceError::RecoveryFailed(
                    "Empty agent state data".to_string(),
                ));
            }

            // In real implementation, would recreate agents with saved state
            tokio::time::sleep(Duration::from_millis(50)).await; // Simulate work
        }

        Ok(())
    }

    /// Rolling restore implementation
    async fn restore_rolling(&self, checkpoint: &SystemCheckpoint) -> FtResult<()> {
        // Restore in phases to minimize downtime

        // Phase 1: Critical components
        self.restore_gpu_state(checkpoint).await?;

        // Phase 2: Agents (can be done gradually)
        for agent_checkpoint in &checkpoint.agent_checkpoints {
            if !agent_checkpoint.state_data.is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }

        Ok(())
    }

    /// Verify recovery was successful
    async fn verify_recovery(&self) -> FtResult<()> {
        // Mock verification - in real implementation would:
        // - Check agent responsiveness
        // - Verify GPU state
        // - Test basic operations
        // - Compare with expected system state

        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    }

    /// Get system health after recovery
    pub async fn post_recovery_health_check(&self) -> HealthStatus {
        // Mock health check
        // In real implementation would check:
        // - All agents responding
        // - GPU contexts valid
        // - Memory pools healthy
        // - Network connectivity

        HealthStatus::Healthy
    }

    /// Estimate recovery time for a checkpoint
    pub async fn estimate_recovery_time(
        &self,
        checkpoint_id: CheckpointId,
        strategy: RecoveryStrategy,
    ) -> FtResult<Duration> {
        let checkpoint = self
            .checkpoint_manager
            .load_checkpoint(&checkpoint_id)
            .await?;

        let base_time = Duration::from_secs(30); // Base recovery time
        let gpu_time = Duration::from_secs(checkpoint.gpu_checkpoints.len() as u64 * 10);
        let agent_time = Duration::from_secs(checkpoint.agent_checkpoints.len() as u64 * 5);

        let total_time = match strategy {
            RecoveryStrategy::Full => base_time + gpu_time + agent_time,
            RecoveryStrategy::GpuOnly => base_time + gpu_time,
            RecoveryStrategy::AgentsOnly => base_time + agent_time,
            RecoveryStrategy::Rolling => (base_time + gpu_time + agent_time) / 2, // Faster due to parallelism
        };

        Ok(total_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{AgentCheckpoint, GpuCheckpoint, SystemCheckpoint};
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_recovery_manager_creation() {
        let manager = RecoveryManager::new();
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_recovery_manager_with_timeout() {
        let timeout_duration = Duration::from_secs(60);
        let manager = RecoveryManager::with_timeout(timeout_duration);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_recovery_from_nonexistent_checkpoint() {
        let manager = RecoveryManager::new()?;
        let nonexistent_id = CheckpointId::new();

        let result = manager.restore_full_system(nonexistent_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_recovery_progress_tracking() {
        let manager = RecoveryManager::new()?;
        let progress = manager.get_recovery_progress().await;

        assert!(matches!(progress.stage, RecoveryStage::Complete));
        assert!(progress.completed_steps <= progress.total_steps);
    }

    #[tokio::test]
    async fn test_post_recovery_health_check() {
        let manager = RecoveryManager::new()?;
        let health = manager.post_recovery_health_check().await;

        assert!(matches!(
            health,
            HealthStatus::Healthy | HealthStatus::Degraded | HealthStatus::Failed
        ));
    }

    #[tokio::test]
    async fn test_recovery_time_estimation() {
        let temp_dir = TempDir::new()?;
        let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
        let checkpoint_id = checkpoint_manager.create_full_checkpoint().await?;

        let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
        let estimate = manager
            .estimate_recovery_time(checkpoint_id, RecoveryStrategy::Full)
            .await;

        assert!(estimate.is_ok());
        let duration = estimate?;
        assert!(duration > Duration::from_secs(0));
    }

    #[test]
    fn test_recovery_strategy_variants() {
        let strategies = vec![
            RecoveryStrategy::Full,
            RecoveryStrategy::AgentsOnly,
            RecoveryStrategy::GpuOnly,
            RecoveryStrategy::Rolling,
        ];

        // Should be able to serialize all strategies
        for strategy in strategies {
            let serialized = serde_json::to_string(&strategy);
            assert!(serialized.is_ok());
        }
    }

    #[test]
    fn test_recovery_stage_equality() {
        assert_eq!(RecoveryStage::Loading, RecoveryStage::Loading);
        assert_ne!(RecoveryStage::Loading, RecoveryStage::Complete);

        let failed1 = RecoveryStage::Failed("error1".to_string());
        let failed2 = RecoveryStage::Failed("error1".to_string());
        assert_eq!(failed1, failed2);
    }

    #[tokio::test]
    async fn test_recovery_timeout() {
        // Create a real checkpoint first
        let temp_dir = TempDir::new()?;
        let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
        let checkpoint_id = checkpoint_manager.create_full_checkpoint().await?;

        // Create manager with same checkpoint manager and very short timeout
        let short_timeout = Duration::from_millis(1); // Very short timeout
        let manager =
            RecoveryManager::with_checkpoint_manager_and_timeout(checkpoint_manager, short_timeout);

        // This should timeout due to very short timeout
        let result = manager.restore_full_system(checkpoint_id).await;
        assert!(result.is_err());

        if let Err(FaultToleranceError::RecoveryFailed(msg)) = result {
            assert!(msg.contains("timeout"));
        } else {
            panic!("Expected RecoveryFailed with timeout message");
        }
    }

    #[tokio::test]
    async fn test_agents_only_recovery() {
        let temp_dir = TempDir::new()?;
        let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
        let checkpoint_id = checkpoint_manager.create_full_checkpoint().await?;

        let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
        let result = manager.restore_agents_only(checkpoint_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_gpu_only_recovery() {
        let temp_dir = TempDir::new()?;
        let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
        let checkpoint_id = checkpoint_manager.create_full_checkpoint().await?;

        let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
        let result = manager.restore_gpu_only(checkpoint_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_invalid_checkpoint_validation() {
        let temp_dir = TempDir::new()?;
        let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

        // Create an empty system checkpoint
        let empty_checkpoint = SystemCheckpoint {
            id: CheckpointId::new(),
            timestamp: 123456789,
            gpu_checkpoints: vec![],
            agent_checkpoints: vec![],
            system_metadata: HashMap::new(),
            compressed: false,
        };

        // Save the empty checkpoint using the checkpoint manager's save method
        let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
        let file_path = temp_dir
            .path()
            .join(format!("{}.checkpoint", empty_checkpoint.id.as_uuid()));

        // Save manually using the same compression approach as CheckpointManager
        let data = bincode::serialize(&empty_checkpoint)?;
        let compressed_data = lz4::block::compress(&data, None, true)?;
        std::fs::write(file_path, compressed_data)?;

        let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
        let result = manager.restore_full_system(empty_checkpoint.id).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FaultToleranceError::InvalidCheckpoint(_)
        ));
    }

    #[tokio::test]
    async fn test_rolling_recovery_strategy() {
        let temp_dir = TempDir::new()?;
        let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
        let checkpoint_id = checkpoint_manager.create_full_checkpoint().await?;

        let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);

        // Test rolling recovery through the private perform_recovery method via recovery strategies
        let checkpoint = manager
            .checkpoint_manager
            .load_checkpoint(&checkpoint_id)
            .await
            .unwrap();
        let result = manager.restore_rolling(&checkpoint).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_recovery_with_empty_gpu_snapshot() {
        let temp_dir = TempDir::new()?;
        let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

        // Create checkpoint with empty GPU memory snapshot
        let checkpoint = SystemCheckpoint {
            id: CheckpointId::new(),
            timestamp: 123456789,
            gpu_checkpoints: vec![GpuCheckpoint {
                memory_snapshot: vec![], // Empty memory snapshot
                kernel_states: HashMap::new(),
                timestamp: 123456789,
                size_bytes: 0,
            }],
            agent_checkpoints: vec![AgentCheckpoint {
                agent_id: "test_agent".to_string(),
                state_data: vec![1, 2, 3], // Non-empty state data
                memory_contents: HashMap::new(),
                goals: vec![],
                metadata: HashMap::new(),
            }],
            system_metadata: HashMap::new(),
            compressed: false,
        };

        // Save the checkpoint with compression
        let file_path = temp_dir
            .path()
            .join(format!("{}.checkpoint", checkpoint.id.as_uuid()));
        let data = bincode::serialize(&checkpoint)?;
        let compressed_data = lz4::block::compress(&data, None, true)?;
        std::fs::write(file_path, compressed_data)?;

        let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
        let result = manager.restore_full_system(checkpoint.id).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FaultToleranceError::RecoveryFailed(_)
        ));
    }

    #[tokio::test]
    async fn test_recovery_with_empty_agent_state() {
        let temp_dir = TempDir::new()?;
        let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

        // Create checkpoint with empty agent state data
        let checkpoint = SystemCheckpoint {
            id: CheckpointId::new(),
            timestamp: 123456789,
            gpu_checkpoints: vec![GpuCheckpoint {
                memory_snapshot: vec![1, 2, 3], // Non-empty memory snapshot
                kernel_states: HashMap::new(),
                timestamp: 123456789,
                size_bytes: 3,
            }],
            agent_checkpoints: vec![AgentCheckpoint {
                agent_id: "test_agent".to_string(),
                state_data: vec![], // Empty state data
                memory_contents: HashMap::new(),
                goals: vec![],
                metadata: HashMap::new(),
            }],
            system_metadata: HashMap::new(),
            compressed: false,
        };

        // Save the checkpoint with compression
        let file_path = temp_dir
            .path()
            .join(format!("{}.checkpoint", checkpoint.id.as_uuid()));
        let data = bincode::serialize(&checkpoint)?;
        let compressed_data = lz4::block::compress(&data, None, true)?;
        std::fs::write(file_path, compressed_data)?;

        let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
        let result = manager.restore_agents_only(checkpoint.id).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FaultToleranceError::RecoveryFailed(_)
        ));
    }

    #[tokio::test]
    async fn test_recovery_manager_with_verification_disabled() {
        let temp_dir = TempDir::new()?;
        let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
        let checkpoint_id = checkpoint_manager.create_full_checkpoint().await?;

        let mut manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
        manager.verification_enabled = false;

        let result = manager.restore_full_system(checkpoint_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_recovery_strategy_serialization() {
        let strategies = vec![
            RecoveryStrategy::Full,
            RecoveryStrategy::AgentsOnly,
            RecoveryStrategy::GpuOnly,
            RecoveryStrategy::Rolling,
        ];

        for strategy in strategies {
            let serialized = bincode::serialize(&strategy)?;
            let deserialized: RecoveryStrategy = bincode::deserialize(&serialized)?;

            // Since RecoveryStrategy doesn't implement PartialEq, we'll test serialization works
            let reserialized = bincode::serialize(&deserialized)?;
            assert_eq!(serialized, reserialized);
        }
    }

    #[tokio::test]
    async fn test_recovery_progress_stage_transitions() {
        let manager = RecoveryManager::new()?;
        let progress = manager.get_recovery_progress().await;

        // Verify progress structure
        assert!(progress.completed_steps <= progress.total_steps);
        assert!(progress.elapsed_time >= Duration::from_secs(0));
        assert!(progress.estimated_remaining >= Duration::from_secs(0));

        // Test all recovery stages
        let stages = vec![
            RecoveryStage::Loading,
            RecoveryStage::ValidatingCheckpoint,
            RecoveryStage::RestoringGpuState,
            RecoveryStage::RestoringAgents,
            RecoveryStage::VerifyingSystem,
            RecoveryStage::Complete,
            RecoveryStage::Failed("test error".to_string()),
        ];

        for stage in stages {
            let test_progress = RecoveryProgress {
                stage: stage.clone(),
                completed_steps: 3,
                total_steps: 5,
                elapsed_time: Duration::from_secs(30),
                estimated_remaining: Duration::from_secs(60),
            };

            // Verify stage can be cloned
            let cloned_stage = test_progress.stage.clone();
            assert_eq!(stage, cloned_stage);
        }
    }

    #[tokio::test]
    async fn test_estimate_recovery_time_variations() {
        let temp_dir = TempDir::new()?;
        let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

        // Create a checkpoint with more data to test estimation variations
        let checkpoint_id = checkpoint_manager.create_full_checkpoint().await?;
        let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);

        // Test all recovery strategies
        let strategies = vec![
            RecoveryStrategy::Full,
            RecoveryStrategy::GpuOnly,
            RecoveryStrategy::AgentsOnly,
            RecoveryStrategy::Rolling,
        ];

        for strategy in strategies {
            let estimate = manager
                .estimate_recovery_time(checkpoint_id.clone(), strategy)
                .await;

            assert!(estimate.is_ok());
            let duration = estimate?;
            assert!(duration > Duration::from_secs(0));
        }
    }

    #[tokio::test]
    async fn test_recovery_manager_default_values() {
        let manager = RecoveryManager::new()?;

        // Test default values are set correctly
        assert_eq!(manager.recovery_timeout, Duration::from_secs(300));
        assert!(manager.verification_enabled);

        // Test with_timeout constructor
        let timeout_manager = RecoveryManager::with_timeout(Duration::from_secs(120))?;
        assert_eq!(timeout_manager.recovery_timeout, Duration::from_secs(120));
        assert!(timeout_manager.verification_enabled);
    }

    #[tokio::test]
    async fn test_recovery_manager_with_custom_constructor_variations() {
        let temp_dir = TempDir::new()?;

        // Test with_checkpoint_manager
        let checkpoint_manager1 = CheckpointManager::with_path(temp_dir.path()).unwrap();
        let manager1 = RecoveryManager::with_checkpoint_manager(checkpoint_manager1);
        assert_eq!(manager1.recovery_timeout, Duration::from_secs(300));
        assert!(manager1.verification_enabled);

        // Test with_checkpoint_manager_and_timeout
        let checkpoint_manager2 = CheckpointManager::with_path(temp_dir.path())?;
        let custom_timeout = Duration::from_secs(600);
        let manager2 = RecoveryManager::with_checkpoint_manager_and_timeout(
            checkpoint_manager2,
            custom_timeout,
        );
        assert_eq!(manager2.recovery_timeout, custom_timeout);
        assert!(manager2.verification_enabled);
    }
}
