//! Automated runbook execution, step validation, rollback on failure, progress tracking
//!
//! This module provides comprehensive runbook execution capabilities including:
//! - Automated runbook execution with step-by-step validation
//! - Conditional execution based on environment and state
//! - Rollback capabilities with automatic failure recovery
//! - Real-time progress tracking and monitoring
//! - Parallel and sequential execution strategies
//! - Human approval gates and manual intervention points
//! - Execution history and audit trails
//!
//! ## Module Structure
//!
//! This module has been refactored into smaller, focused modules:
//! - `runbook_types`: All data structures and type definitions
//! - `runbook_executor_core`: Main executor implementation
//! - `runbook_validation`: Validation logic and test helpers
//!
//! All types and functionality are re-exported from this module to maintain
//! backward compatibility with existing code.

// Import all the modules
mod runbook_executor_core;
mod runbook_types;
mod runbook_validation;

// Re-export all types to maintain backward compatibility
pub use runbook_executor_core::RunbookExecutor;
pub use runbook_types::*;
pub use runbook_validation::{
    create_minimal_runbook, create_test_context, create_test_runbook, create_test_trigger,
    is_valid_state_transition, validate_runbook_structure, validate_step_configuration,
};

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn create_test_runbook_internal(name: &str) -> Runbook {
        create_test_runbook(name)
    }

    fn create_test_context_internal() -> ExecutionContext {
        create_test_context()
    }

    fn create_test_trigger_internal() -> ExecutionTrigger {
        create_test_trigger()
    }

    #[test]
    fn test_runbook_category_serialization() {
        let categories = vec![
            RunbookCategory::DisasterRecovery,
            RunbookCategory::Maintenance,
            RunbookCategory::IncidentResponse,
            RunbookCategory::Deployment,
            RunbookCategory::Monitoring,
            RunbookCategory::Security,
            RunbookCategory::Backup,
            RunbookCategory::Custom("test".to_string()),
        ];

        for category in categories {
            let serialized = serde_json::to_string(&category).unwrap();
            let deserialized: RunbookCategory = serde_json::from_str(&serialized).unwrap();
            assert_eq!(category, deserialized);
        }
    }

    #[test]
    fn test_execution_state_transitions() {
        let states = vec![
            ExecutionState::PendingApproval,
            ExecutionState::CheckingPrerequisites,
            ExecutionState::Running,
            ExecutionState::WaitingForManual,
            ExecutionState::Completed,
            ExecutionState::Failed,
            ExecutionState::RollingBack,
            ExecutionState::RolledBack,
            ExecutionState::Cancelled,
            ExecutionState::Paused,
        ];

        for state in states {
            let serialized = serde_json::to_string(&state).unwrap();
            let deserialized: ExecutionState = serde_json::from_str(&serialized).unwrap();
            assert_eq!(state, deserialized);
        }
    }

    #[test]
    fn test_step_types() {
        let types = vec![
            StepType::Command,
            StepType::ApiCall,
            StepType::Database,
            StepType::FileOperation,
            StepType::ServiceControl,
            StepType::Manual,
            StepType::Validation,
            StepType::Wait,
            StepType::Conditional,
            StepType::Parallel,
        ];

        for step_type in types {
            let serialized = serde_json::to_string(&step_type).unwrap();
            let deserialized: StepType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(step_type, deserialized);
        }
    }

    #[test]
    fn test_runbook_executor_config_default() {
        let config = RunbookExecutorConfig::default();
        assert_eq!(config.max_concurrent_executions, 10);
        assert_eq!(config.default_step_timeout_seconds, 300);
        assert_eq!(config.default_manual_timeout_minutes, 60);
        assert!(config.execution_history_enabled);
        assert!(config.step_validation_enabled);
    }

    #[test]
    fn test_runbook_executor_creation() {
        let config = RunbookExecutorConfig::default();
        let executor = RunbookExecutor::new(config);
        assert!(executor.is_ok());
    }

    #[tokio::test]
    async fn test_register_runbook() {
        let config = RunbookExecutorConfig::default();
        let executor = RunbookExecutor::new(config).unwrap();

        let runbook = create_test_runbook_internal("Test Runbook");
        let runbook_id = executor.register_runbook(runbook.clone()).await?;

        let registered = executor.get_runbook(runbook_id)?;
        assert_eq!(registered.name, "Test Runbook");
    }

    #[tokio::test]
    async fn test_validate_runbook() {
        let config = RunbookExecutorConfig::default();
        let executor = RunbookExecutor::new(config).unwrap();

        // Test invalid runbook - no steps
        let mut runbook = create_test_runbook_internal("Invalid");
        runbook.steps.clear();
        let result = executor.register_runbook(runbook).await;
        assert!(result.is_err());

        // Test invalid runbook - empty name
        let mut runbook = create_test_runbook_internal("");
        let result = executor.register_runbook(runbook).await;
        assert!(result.is_err());

        // Test invalid runbook - duplicate step order
        let mut runbook = create_test_runbook_internal("Invalid");
        if runbook.steps.len() > 1 {
            runbook.steps[1].order = runbook.steps[0].order;
            let result = executor.register_runbook(runbook).await;
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn test_execute_runbook() {
        let config = RunbookExecutorConfig::default();
        let executor = RunbookExecutor::new(config).unwrap();
        executor.start().await.unwrap();

        let runbook = create_test_runbook_internal("Execute Test");
        let runbook_id = executor.register_runbook(runbook).await?;

        let trigger = create_test_trigger_internal();
        let context = create_test_context_internal();

        let execution_id = executor
            .execute_runbook(runbook_id, trigger, context)
            .await
            .unwrap();

        // Wait for execution to complete
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;

        let execution = executor.get_execution_status(execution_id).unwrap();
        assert_eq!(execution.state, ExecutionState::Completed);
        assert_eq!(execution.progress, 1.0);
        assert_eq!(execution.step_executions.len(), 2);
    }

    #[tokio::test]
    async fn test_cancel_execution() {
        let config = RunbookExecutorConfig::default();
        let executor = RunbookExecutor::new(config).unwrap();
        executor.start().await.unwrap();

        let runbook = create_test_runbook_internal("Cancel Test");
        let runbook_id = executor.register_runbook(runbook).await?;

        let trigger = create_test_trigger_internal();
        let context = create_test_context_internal();

        let execution_id = executor
            .execute_runbook(runbook_id, trigger, context)
            .await
            .unwrap();

        // Cancel immediately
        executor.cancel_execution(execution_id).await.unwrap();

        // Wait for cancellation to process
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let execution = executor.get_execution_status(execution_id).unwrap();
        assert_eq!(execution.state, ExecutionState::Cancelled);
    }

    #[tokio::test]
    async fn test_list_runbooks_by_category() {
        let config = RunbookExecutorConfig::default();
        let executor = RunbookExecutor::new(config).unwrap();

        // Register runbooks with different categories
        let mut dr_runbook = create_test_runbook_internal("DR Runbook");
        dr_runbook.category = RunbookCategory::DisasterRecovery;
        executor.register_runbook(dr_runbook).await?;

        let mut maintenance_runbook = create_test_runbook_internal("Maintenance Runbook");
        maintenance_runbook.category = RunbookCategory::Maintenance;
        executor
            .register_runbook(maintenance_runbook)
            .await
            .unwrap();

        // List by category
        let dr_runbooks = executor.list_runbooks(Some(RunbookCategory::DisasterRecovery));
        assert_eq!(dr_runbooks.len(), 1);
        assert_eq!(dr_runbooks[0].name, "DR Runbook");

        let maintenance_runbooks = executor.list_runbooks(Some(RunbookCategory::Maintenance));
        assert_eq!(maintenance_runbooks.len(), 1);
        assert_eq!(maintenance_runbooks[0].name, "Maintenance Runbook");

        // List all
        let all_runbooks = executor.list_runbooks(None);
        assert_eq!(all_runbooks.len(), 2);
    }

    #[tokio::test]
    async fn test_execution_logging() {
        let config = RunbookExecutorConfig::default();
        let executor = RunbookExecutor::new(config).unwrap();
        executor.start().await.unwrap();

        let runbook = create_test_runbook_internal("Log Test");
        let runbook_id = executor.register_runbook(runbook).await?;

        let trigger = create_test_trigger_internal();
        let context = create_test_context_internal();

        let execution_id = executor
            .execute_runbook(runbook_id, trigger, context)
            .await
            .unwrap();

        // Wait for execution
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;

        let execution = executor.get_execution_status(execution_id).unwrap();
        assert!(!execution.logs.is_empty());

        // Check for step execution logs
        let step_logs: Vec<&ExecutionLog> = execution
            .logs
            .iter()
            .filter(|log| log.step_id.is_some())
            .collect();
        assert!(!step_logs.is_empty());
    }

    #[test]
    fn test_action_types() {
        let types = vec![
            ActionType::Shell,
            ActionType::Http,
            ActionType::File,
            ActionType::Service,
            ActionType::Manual,
            ActionType::Sql,
            ActionType::Custom("test".to_string()),
        ];

        for action_type in types {
            let serialized = serde_json::to_string(&action_type).unwrap();
            let deserialized: ActionType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(action_type, deserialized);
        }
    }

    #[test]
    fn test_failure_actions() {
        let actions = vec![
            FailureAction::Stop,
            FailureAction::Continue,
            FailureAction::SkipTo(Uuid::new_v4()),
            FailureAction::Rollback,
            FailureAction::ManualIntervention,
        ];

        for action in actions {
            let serialized = serde_json::to_string(&action)?;
            let deserialized: FailureAction = serde_json::from_str(&serialized).unwrap();
            assert_eq!(action, deserialized);
        }
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.delay_seconds, 5);
        assert_eq!(config.backoff_multiplier, 2.0);
        assert_eq!(config.max_delay_seconds, 300);
        assert!(config
            .retry_conditions
            .contains(&RetryCondition::NonZeroExitCode));
        assert!(config
            .retry_conditions
            .contains(&RetryCondition::NetworkError));
    }

    #[test]
    fn test_comparison_operators() {
        let operators = vec![
            ComparisonOperator::Equals,
            ComparisonOperator::NotEquals,
            ComparisonOperator::Contains,
            ComparisonOperator::NotContains,
            ComparisonOperator::GreaterThan,
            ComparisonOperator::LessThan,
        ];

        for operator in operators {
            let serialized = serde_json::to_string(&operator).unwrap();
            let deserialized: ComparisonOperator = serde_json::from_str(&serialized).unwrap();
            assert_eq!(operator, deserialized);
        }
    }

    #[test]
    fn test_trigger_types() {
        let triggers = vec![
            TriggerType::Manual,
            TriggerType::Scheduled,
            TriggerType::Event,
            TriggerType::Api,
            TriggerType::Webhook,
        ];

        for trigger in triggers {
            let serialized = serde_json::to_string(&trigger)?;
            let deserialized: TriggerType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(trigger, deserialized);
        }
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let config = RunbookExecutorConfig::default();
        let executor = RunbookExecutor::new(config).unwrap();
        executor.start().await.unwrap();

        // Register multiple runbooks
        for i in 0..3 {
            let runbook = create_test_runbook_internal(&format!("Metrics Test {}", i));
            executor.register_runbook(runbook).await?;
        }

        // Execute one runbook
        let trigger = create_test_trigger_internal();
        let context = create_test_context_internal();
        let runbook_ids: Vec<Uuid> = executor.list_runbooks(None).iter().map(|r| r.id).collect();

        executor
            .execute_runbook(runbook_ids[0], trigger, context)
            .await
            .unwrap();

        // Wait for execution and metrics update
        tokio::time::sleep(std::time::Duration::from_millis(700)).await;

        let metrics = executor.get_metrics();
        assert_eq!(metrics.total_runbooks, 3);
        assert!(metrics.success_rate >= 0.0 && metrics.success_rate <= 1.0);
    }

    #[test]
    fn test_prerequisite_types() {
        let types = vec![
            PrerequisiteType::Software,
            PrerequisiteType::Service,
            PrerequisiteType::Network,
            PrerequisiteType::File,
            PrerequisiteType::Permission,
            PrerequisiteType::Environment,
            PrerequisiteType::Custom,
        ];

        for prereq_type in types {
            let serialized = serde_json::to_string(&prereq_type).unwrap();
            let deserialized: PrerequisiteType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(prereq_type, deserialized);
        }
    }
}
