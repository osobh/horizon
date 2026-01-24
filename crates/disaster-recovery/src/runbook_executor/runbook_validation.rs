//! Runbook validation utilities and helpers
//!
//! This module contains validation logic and helper functions used by
//! the runbook executor for testing and validation purposes.

use super::runbook_types::*;
use chrono::Utc;
use std::collections::HashMap;
use uuid::Uuid;

/// Create a test runbook for testing purposes
pub fn create_test_runbook(name: &str) -> Runbook {
    Runbook {
        id: Uuid::new_v4(),
        name: name.to_string(),
        description: format!("Test runbook: {}", name),
        version: "1.0.0".to_string(),
        author: "test@example.com".to_string(),
        category: RunbookCategory::DisasterRecovery,
        tags: vec!["test".to_string()],
        steps: vec![
            RunbookStep {
                id: Uuid::new_v4(),
                name: "Check Prerequisites".to_string(),
                description: "Verify system prerequisites".to_string(),
                step_type: StepType::Command,
                order: 1,
                action: StepAction {
                    action_type: ActionType::Shell,
                    command: Some("echo 'Checking prerequisites'".to_string()),
                    arguments: vec![],
                    working_directory: None,
                    http: None,
                    file_operation: None,
                    service: None,
                    manual: None,
                },
                expected_outcomes: vec![ExpectedOutcome {
                    outcome_type: OutcomeType::ExitCode,
                    expected_value: "0".to_string(),
                    operator: ComparisonOperator::Equals,
                    description: "Command should succeed".to_string(),
                }],
                validation_rules: vec![],
                dependencies: vec![],
                execution_mode: ExecutionMode::Sequential,
                timeout_seconds: 60,
                retry_config: RetryConfig::default(),
                on_failure: FailureAction::Stop,
                skippable: false,
                requires_approval: false,
                environment: HashMap::new(),
                condition: None,
            },
            RunbookStep {
                id: Uuid::new_v4(),
                name: "Execute Main Task".to_string(),
                description: "Execute the main recovery task".to_string(),
                step_type: StepType::Command,
                order: 2,
                action: StepAction {
                    action_type: ActionType::Shell,
                    command: Some("echo 'Executing main task'".to_string()),
                    arguments: vec![],
                    working_directory: None,
                    http: None,
                    file_operation: None,
                    service: None,
                    manual: None,
                },
                expected_outcomes: vec![],
                validation_rules: vec![],
                dependencies: vec![],
                execution_mode: ExecutionMode::Sequential,
                timeout_seconds: 300,
                retry_config: RetryConfig::default(),
                on_failure: FailureAction::Rollback,
                skippable: false,
                requires_approval: false,
                environment: HashMap::new(),
                condition: None,
            },
        ],
        rollback_steps: vec![],
        prerequisites: vec![],
        environment_requirements: HashMap::new(),
        timeout_minutes: 60,
        requires_approval: false,
        approval_roles: vec![],
        enabled: true,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

/// Create a test execution context
pub fn create_test_context() -> ExecutionContext {
    ExecutionContext {
        environment: HashMap::from([
            ("USER".to_string(), "test".to_string()),
            ("HOME".to_string(), "/home/test".to_string()),
        ]),
        working_directory: "/tmp".to_string(),
        user: "test".to_string(),
        host: "localhost".to_string(),
        parameters: HashMap::new(),
    }
}

/// Create a test execution trigger
pub fn create_test_trigger() -> ExecutionTrigger {
    ExecutionTrigger {
        trigger_type: TriggerType::Manual,
        triggered_by: "test_user".to_string(),
        reason: "Testing runbook execution".to_string(),
        parameters: HashMap::new(),
    }
}

/// Validate runbook structure and configuration
pub fn validate_runbook_structure(runbook: &Runbook) -> Result<(), String> {
    if runbook.steps.is_empty() {
        return Err("runbook must have at least one step".to_string());
    }

    if runbook.name.is_empty() {
        return Err("runbook name cannot be empty".to_string());
    }

    // Validate step ordering
    let mut orders: Vec<u32> = runbook.steps.iter().map(|s| s.order).collect();
    orders.sort();
    for (i, &order) in orders.iter().enumerate() {
        if i > 0 && order == orders[i - 1] {
            return Err("duplicate step order found".to_string());
        }
    }

    // Validate dependencies
    let step_ids: std::collections::HashSet<Uuid> = runbook.steps.iter().map(|s| s.id).collect();
    for step in &runbook.steps {
        for dep_id in &step.dependencies {
            if !step_ids.contains(dep_id) {
                return Err(format!("step dependency {} not found", dep_id));
            }
        }
    }

    Ok(())
}

/// Validate execution state transitions
pub fn is_valid_state_transition(from: ExecutionState, to: ExecutionState) -> bool {
    use ExecutionState::*;

    match (from, to) {
        // Valid transitions from PendingApproval
        (PendingApproval, CheckingPrerequisites) => true,
        (PendingApproval, Cancelled) => true,

        // Valid transitions from CheckingPrerequisites
        (CheckingPrerequisites, Running) => true,
        (CheckingPrerequisites, Failed) => true,
        (CheckingPrerequisites, Cancelled) => true,

        // Valid transitions from Running
        (Running, Completed) => true,
        (Running, Failed) => true,
        (Running, WaitingForManual) => true,
        (Running, RollingBack) => true,
        (Running, Paused) => true,
        (Running, Cancelled) => true,

        // Valid transitions from WaitingForManual
        (WaitingForManual, Running) => true,
        (WaitingForManual, Failed) => true,
        (WaitingForManual, Cancelled) => true,

        // Valid transitions from Paused
        (Paused, Running) => true,
        (Paused, Cancelled) => true,

        // Valid transitions from RollingBack
        (RollingBack, RolledBack) => true,
        (RollingBack, Failed) => true,

        // Same state is always valid
        (state, same_state) if state == same_state => true,

        // All other transitions are invalid
        _ => false,
    }
}

/// Validate step execution configuration
pub fn validate_step_configuration(step: &RunbookStep) -> Result<(), String> {
    // Validate timeout
    if step.timeout_seconds == 0 {
        return Err("step timeout must be greater than 0".to_string());
    }

    // Validate retry configuration
    if step.retry_config.max_attempts == 0 {
        return Err("retry max_attempts must be greater than 0".to_string());
    }

    if step.retry_config.delay_seconds == 0 {
        return Err("retry delay_seconds must be greater than 0".to_string());
    }

    // Validate action configuration
    match &step.action.action_type {
        ActionType::Shell => {
            if step.action.command.is_none() {
                return Err("shell action requires a command".to_string());
            }
        }
        ActionType::Http => {
            if step.action.http.is_none() {
                return Err("HTTP action requires HTTP configuration".to_string());
            }
        }
        ActionType::File => {
            if step.action.file_operation.is_none() {
                return Err("file action requires file operation configuration".to_string());
            }
        }
        ActionType::Service => {
            if step.action.service.is_none() {
                return Err("service action requires service configuration".to_string());
            }
        }
        ActionType::Manual => {
            if step.action.manual.is_none() {
                return Err("manual action requires manual configuration".to_string());
            }
        }
        ActionType::Sql | ActionType::Custom(_) => {
            // Custom validation can be added here
        }
    }

    Ok(())
}

/// Create a minimal valid runbook for testing
pub fn create_minimal_runbook() -> Runbook {
    Runbook {
        id: Uuid::new_v4(),
        name: "Minimal Test Runbook".to_string(),
        description: "A minimal runbook for testing".to_string(),
        version: "1.0.0".to_string(),
        author: "test@example.com".to_string(),
        category: RunbookCategory::DisasterRecovery,
        tags: vec![],
        steps: vec![RunbookStep {
            id: Uuid::new_v4(),
            name: "Simple Step".to_string(),
            description: "A simple test step".to_string(),
            step_type: StepType::Command,
            order: 1,
            action: StepAction {
                action_type: ActionType::Shell,
                command: Some("echo 'test'".to_string()),
                arguments: vec![],
                working_directory: None,
                http: None,
                file_operation: None,
                service: None,
                manual: None,
            },
            expected_outcomes: vec![],
            validation_rules: vec![],
            dependencies: vec![],
            execution_mode: ExecutionMode::Sequential,
            timeout_seconds: 60,
            retry_config: RetryConfig::default(),
            on_failure: FailureAction::Stop,
            skippable: false,
            requires_approval: false,
            environment: HashMap::new(),
            condition: None,
        }],
        rollback_steps: vec![],
        prerequisites: vec![],
        environment_requirements: HashMap::new(),
        timeout_minutes: 60,
        requires_approval: false,
        approval_roles: vec![],
        enabled: true,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_test_runbook() -> anyhow::Result<()> {
        let runbook = create_test_runbook("Test");
        assert_eq!(runbook.name, "Test");
        assert_eq!(runbook.steps.len(), 2);
        assert!(runbook.enabled);
        Ok(())
    }

    #[test]
    fn test_validate_runbook_structure() -> anyhow::Result<()> {
        let runbook = create_test_runbook("Valid");
        assert!(validate_runbook_structure(&runbook).is_ok());

        // Test empty steps
        let mut invalid_runbook = runbook.clone();
        invalid_runbook.steps.clear();
        assert!(validate_runbook_structure(&invalid_runbook).is_err());

        // Test empty name
        let mut invalid_runbook = runbook.clone();
        invalid_runbook.name.clear();
        assert!(validate_runbook_structure(&invalid_runbook).is_err());
        Ok(())
    }

    #[test]
    fn test_execution_state_transitions() -> anyhow::Result<()> {
        use ExecutionState::*;

        // Valid transitions
        assert!(is_valid_state_transition(
            PendingApproval,
            CheckingPrerequisites
        ));
        assert!(is_valid_state_transition(CheckingPrerequisites, Running));
        assert!(is_valid_state_transition(Running, Completed));
        assert!(is_valid_state_transition(Running, Paused));
        assert!(is_valid_state_transition(Paused, Running));

        // Invalid transitions
        assert!(!is_valid_state_transition(Completed, Running));
        assert!(!is_valid_state_transition(Failed, Running));
        assert!(!is_valid_state_transition(Cancelled, Running));
        Ok(())
    }

    #[test]
    fn test_step_configuration_validation() -> anyhow::Result<()> {
        let runbook = create_test_runbook("Test");
        let step = &runbook.steps[0];

        assert!(validate_step_configuration(step).is_ok());

        // Test invalid timeout
        let mut invalid_step = step.clone();
        invalid_step.timeout_seconds = 0;
        assert!(validate_step_configuration(&invalid_step).is_err());
        Ok(())
    }

    #[test]
    fn test_create_minimal_runbook() -> anyhow::Result<()> {
        let runbook = create_minimal_runbook();
        assert_eq!(runbook.name, "Minimal Test Runbook");
        assert_eq!(runbook.steps.len(), 1);
        assert!(validate_runbook_structure(&runbook).is_ok());
        Ok(())
    }

    #[test]
    fn test_test_helpers() -> anyhow::Result<()> {
        let context = create_test_context();
        assert_eq!(context.user, "test");
        assert_eq!(context.host, "localhost");

        let trigger = create_test_trigger();
        assert_eq!(trigger.trigger_type, TriggerType::Manual);
        assert_eq!(trigger.triggered_by, "test_user");
        Ok(())
    }
}
