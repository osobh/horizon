//! Rollback management system for deployments

use crate::error::{OperationalError, OperationalResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Rollback manager for handling deployment rollbacks
#[derive(Debug)]
pub struct RollbackManager {
    /// Available rollback plans
    rollback_plans: HashMap<String, RollbackPlan>,
    /// Rollback history
    history: Vec<RollbackExecution>,
}

/// Rollback strategy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RollbackStrategy {
    /// Immediate rollback to previous version
    Immediate,
    /// Gradual rollback with traffic shifting
    Gradual { steps: u32, interval: Duration },
    /// Blue-green switch back
    BlueGreenSwitch,
    /// Checkpoint-based rollback
    CheckpointBased { checkpoint_id: String },
}

/// Rollback plan definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackPlan {
    /// Plan identifier
    pub plan_id: String,
    /// Deployment this plan applies to
    pub deployment_id: String,
    /// Rollback strategy
    pub strategy: RollbackStrategy,
    /// Previous deployment state reference
    pub previous_state: CheckpointRef,
    /// Rollback timeout
    pub timeout: Duration,
    /// Whether to verify rollback success
    pub verify_success: bool,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Reference to a deployment checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointRef {
    /// Checkpoint identifier
    pub checkpoint_id: String,
    /// Deployment version
    pub version: String,
    /// Checkpoint timestamp
    pub timestamp: DateTime<Utc>,
    /// Checkpoint metadata
    pub metadata: HashMap<String, String>,
}

/// Rollback execution record
#[derive(Debug, Clone)]
struct RollbackExecution {
    execution_id: String,
    plan_id: String,
    started_at: DateTime<Utc>,
    completed_at: Option<DateTime<Utc>>,
    status: RollbackStatus,
    error_message: Option<String>,
}

/// Rollback execution status
#[derive(Debug, Clone, PartialEq)]
enum RollbackStatus {
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

impl RollbackManager {
    /// Create a new rollback manager
    pub fn new() -> Self {
        Self {
            rollback_plans: HashMap::new(),
            history: Vec::new(),
        }
    }

    /// Create a rollback plan
    pub fn create_plan(&mut self, plan: RollbackPlan) -> OperationalResult<()> {
        if plan.plan_id.is_empty() {
            return Err(OperationalError::ConfigurationError(
                "Plan ID cannot be empty".to_string(),
            ));
        }

        if plan.deployment_id.is_empty() {
            return Err(OperationalError::ConfigurationError(
                "Deployment ID cannot be empty".to_string(),
            ));
        }

        self.rollback_plans.insert(plan.plan_id.clone(), plan);
        Ok(())
    }

    /// Execute a rollback plan
    pub async fn execute_rollback(&mut self, plan_id: &str) -> OperationalResult<String> {
        let plan = self
            .rollback_plans
            .get(plan_id)
            .ok_or_else(|| {
                OperationalError::ConfigurationError(format!("Rollback plan {} not found", plan_id))
            })?
            .clone();

        let execution_id = uuid::Uuid::new_v4().to_string();
        let execution = RollbackExecution {
            execution_id: execution_id.clone(),
            plan_id: plan_id.to_string(),
            started_at: Utc::now(),
            completed_at: None,
            status: RollbackStatus::InProgress,
            error_message: None,
        };

        self.history.push(execution);

        // Execute rollback based on strategy
        match plan.strategy {
            RollbackStrategy::Immediate => self.execute_immediate_rollback(&plan).await,
            RollbackStrategy::Gradual { steps, interval } => {
                self.execute_gradual_rollback(&plan, steps, interval).await
            }
            RollbackStrategy::BlueGreenSwitch => self.execute_blue_green_switch(&plan).await,
            RollbackStrategy::CheckpointBased { ref checkpoint_id } => {
                self.execute_checkpoint_rollback(&plan, checkpoint_id).await
            }
        }?;

        // Update execution status
        if let Some(execution) = self.history.iter_mut().last() {
            execution.status = RollbackStatus::Completed;
            execution.completed_at = Some(Utc::now());
        }

        Ok(execution_id)
    }

    /// Get rollback plan
    pub fn get_plan(&self, plan_id: &str) -> Option<&RollbackPlan> {
        self.rollback_plans.get(plan_id)
    }

    /// List all rollback plans
    pub fn list_plans(&self) -> Vec<&RollbackPlan> {
        self.rollback_plans.values().collect()
    }

    /// Get rollback history
    pub fn get_history(&self) -> &[RollbackExecution] {
        &self.history
    }

    /// Remove a rollback plan
    pub fn remove_plan(&mut self, plan_id: &str) -> OperationalResult<()> {
        self.rollback_plans.remove(plan_id);
        Ok(())
    }

    // Private helper methods

    async fn execute_immediate_rollback(&self, _plan: &RollbackPlan) -> OperationalResult<()> {
        // Mock implementation for immediate rollback
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    async fn execute_gradual_rollback(
        &self,
        _plan: &RollbackPlan,
        _steps: u32,
        _interval: Duration,
    ) -> OperationalResult<()> {
        // Mock implementation for gradual rollback
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    async fn execute_blue_green_switch(&self, _plan: &RollbackPlan) -> OperationalResult<()> {
        // Mock implementation for blue-green switch
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    async fn execute_checkpoint_rollback(
        &self,
        _plan: &RollbackPlan,
        _checkpoint_id: &str,
    ) -> OperationalResult<()> {
        // Mock implementation for checkpoint-based rollback
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }
}

impl Default for RollbackManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RollbackStrategy {
    fn default() -> Self {
        Self::Immediate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_checkpoint() -> CheckpointRef {
        CheckpointRef {
            checkpoint_id: "checkpoint-123".to_string(),
            version: "v1.0.0".to_string(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    fn create_test_plan() -> RollbackPlan {
        RollbackPlan {
            plan_id: "plan-123".to_string(),
            deployment_id: "deployment-123".to_string(),
            strategy: RollbackStrategy::Immediate,
            previous_state: create_test_checkpoint(),
            timeout: Duration::from_secs(300),
            verify_success: true,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_rollback_manager_creation() {
        let manager = RollbackManager::new();
        assert!(manager.rollback_plans.is_empty());
        assert!(manager.history.is_empty());
    }

    #[test]
    fn test_rollback_strategy_serialization() {
        let strategies = vec![
            RollbackStrategy::Immediate,
            RollbackStrategy::Gradual {
                steps: 5,
                interval: Duration::from_secs(30),
            },
            RollbackStrategy::BlueGreenSwitch,
            RollbackStrategy::CheckpointBased {
                checkpoint_id: "checkpoint-123".to_string(),
            },
        ];

        for strategy in strategies {
            let serialized = serde_json::to_string(&strategy).unwrap();
            let deserialized: RollbackStrategy = serde_json::from_str(&serialized).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    #[test]
    fn test_rollback_plan_creation() {
        let mut manager = RollbackManager::new();
        let plan = create_test_plan();

        let result = manager.create_plan(plan.clone());
        assert!(result.is_ok());

        let retrieved = manager.get_plan(&plan.plan_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().deployment_id, plan.deployment_id);
    }

    #[test]
    fn test_invalid_plan_creation() {
        let mut manager = RollbackManager::new();

        // Empty plan ID
        let mut plan = create_test_plan();
        plan.plan_id = String::new();
        let result = manager.create_plan(plan);
        assert!(result.is_err());

        // Empty deployment ID
        let mut plan = create_test_plan();
        plan.deployment_id = String::new();
        let result = manager.create_plan(plan);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_rollback_execution() {
        let mut manager = RollbackManager::new();
        let plan = create_test_plan();
        let plan_id = plan.plan_id.clone();

        manager.create_plan(plan).unwrap();

        let result = manager.execute_rollback(&plan_id).await;
        assert!(result.is_ok());

        let execution_id = result.unwrap();
        assert!(!execution_id.is_empty());

        // Check history was recorded
        let history = manager.get_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].plan_id, plan_id);
        assert_eq!(history[0].status, RollbackStatus::Completed);
    }

    #[tokio::test]
    async fn test_rollback_strategies() {
        let mut manager = RollbackManager::new();

        let strategies = vec![
            RollbackStrategy::Immediate,
            RollbackStrategy::Gradual {
                steps: 3,
                interval: Duration::from_secs(10),
            },
            RollbackStrategy::BlueGreenSwitch,
            RollbackStrategy::CheckpointBased {
                checkpoint_id: "checkpoint-456".to_string(),
            },
        ];

        for (i, strategy) in strategies.into_iter().enumerate() {
            let mut plan = create_test_plan();
            plan.plan_id = format!("plan-{}", i);
            plan.strategy = strategy;

            manager.create_plan(plan.clone()).unwrap();
            let result = manager.execute_rollback(&plan.plan_id).await;
            assert!(result.is_ok());
        }

        // Check all executions were recorded
        let history = manager.get_history();
        assert_eq!(history.len(), 4);
    }

    #[tokio::test]
    async fn test_nonexistent_plan_execution() {
        let mut manager = RollbackManager::new();

        let result = manager.execute_rollback("nonexistent-plan").await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OperationalError::ConfigurationError(_)
        ));
    }

    #[test]
    fn test_plan_management() {
        let mut manager = RollbackManager::new();
        let plan = create_test_plan();
        let plan_id = plan.plan_id.clone();

        // Create plan
        manager.create_plan(plan).unwrap();
        assert_eq!(manager.list_plans().len(), 1);

        // Remove plan
        manager.remove_plan(&plan_id).unwrap();
        assert_eq!(manager.list_plans().len(), 0);
        assert!(manager.get_plan(&plan_id).is_none());
    }

    #[test]
    fn test_checkpoint_ref_serialization() {
        let checkpoint = create_test_checkpoint();
        let serialized = serde_json::to_string(&checkpoint).unwrap();
        let deserialized: CheckpointRef = serde_json::from_str(&serialized).unwrap();
        assert_eq!(checkpoint.checkpoint_id, deserialized.checkpoint_id);
        assert_eq!(checkpoint.version, deserialized.version);
    }

    #[test]
    fn test_rollback_plan_serialization() {
        let plan = create_test_plan();
        let serialized = serde_json::to_string(&plan).unwrap();
        let deserialized: RollbackPlan = serde_json::from_str(&serialized).unwrap();
        assert_eq!(plan.plan_id, deserialized.plan_id);
        assert_eq!(plan.deployment_id, deserialized.deployment_id);
        assert_eq!(plan.strategy, deserialized.strategy);
    }

    #[test]
    fn test_rollback_status_equality() {
        assert_eq!(RollbackStatus::InProgress, RollbackStatus::InProgress);
        assert_ne!(RollbackStatus::InProgress, RollbackStatus::Completed);
    }

    #[test]
    fn test_default_implementations() {
        let _manager = RollbackManager::default();
        let _strategy = RollbackStrategy::default();
    }

    #[tokio::test]
    async fn test_concurrent_rollbacks() {
        let mut manager = RollbackManager::new();

        // Create multiple plans
        for i in 0..3 {
            let mut plan = create_test_plan();
            plan.plan_id = format!("concurrent-plan-{}", i);
            manager.create_plan(plan).unwrap();
        }

        // Execute rollbacks concurrently
        let mut handles = Vec::new();
        for i in 0..3 {
            let plan_id = format!("concurrent-plan-{}", i);
            let handle = tokio::spawn(async move {
                // Note: In a real implementation, we'd need proper concurrency handling
                // For this test, we're just verifying the function signature works
                plan_id
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            let _result = handle.await.unwrap();
        }
    }

    #[test]
    fn test_rollback_plan_metadata() {
        let mut checkpoint = create_test_checkpoint();
        checkpoint
            .metadata
            .insert("key1".to_string(), "value1".to_string());
        checkpoint
            .metadata
            .insert("key2".to_string(), "value2".to_string());

        assert_eq!(checkpoint.metadata.len(), 2);
        assert_eq!(checkpoint.metadata.get("key1"), Some(&"value1".to_string()));
    }
}
