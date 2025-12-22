//! Recovery procedures for emergency situations
//!
//! Provides recovery mechanisms for:
//! - System restoration after kill switch
//! - Agent recovery from suspension
//! - Resource cleanup
//! - State restoration
//! - Checkpoint recovery

use crate::{EmergencyError, EmergencyResult};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use tracing::{error, info, warn};

/// Recovery procedure types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecoveryType {
    /// Full system recovery
    SystemRecovery,
    /// Individual agent recovery
    AgentRecovery,
    /// Resource cleanup and reallocation
    ResourceRecovery,
    /// State restoration from checkpoint
    StateRecovery,
    /// Network reconnection
    NetworkRecovery,
    /// Data consistency check
    DataRecovery,
    /// Configuration reset
    ConfigurationRecovery,
}

/// Recovery states
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryState {
    /// Not started
    Pending,
    /// Idle (no recovery in progress)
    Idle,
    /// Currently in progress
    InProgress,
    /// Successfully completed
    Completed,
    /// Failed with errors
    Failed,
    /// Partially completed
    Partial,
    /// Running (for edge test compatibility)
    Running { procedure: String },
}

/// Recovery procedure definition
#[derive(Debug, Clone)]
pub struct RecoveryProcedure {
    pub id: String,
    pub recovery_type: RecoveryType,
    pub name: String,
    pub description: String,
    pub steps: Vec<RecoveryStep>,
    pub timeout: Duration,
    pub retry_count: u32,
    pub priority: u32,
}

/// Individual recovery step
#[derive(Debug, Clone)]
pub struct RecoveryStep {
    pub name: String,
    pub action: String,
    pub required: bool,
    pub timeout: Duration,
}

/// Recovery execution record
#[derive(Debug, Clone)]
pub struct RecoveryExecution {
    pub procedure_id: String,
    pub recovery_type: RecoveryType,
    pub state: RecoveryState,
    pub started_at: Instant,
    pub completed_at: Option<Instant>,
    pub error: Option<String>,
    pub steps_completed: usize,
    pub total_steps: usize,
    pub agent_id: Option<String>,
}

/// Recovery event for notifications
#[derive(Debug, Clone)]
pub struct RecoveryEvent {
    pub procedure_id: String,
    pub recovery_type: RecoveryType,
    pub state: RecoveryState,
    pub message: String,
    pub timestamp: Instant,
}

/// Recovery system manager
pub struct RecoverySystem {
    procedures: Arc<RwLock<Vec<RecoveryProcedure>>>,
    executions: Arc<DashMap<String, RecoveryExecution>>,
    event_sender: broadcast::Sender<RecoveryEvent>,
    max_concurrent_recoveries: usize,
    _global_timeout: Duration,
}

impl Default for RecoverySystem {
    fn default() -> Self {
        Self::new()
    }
}

impl RecoverySystem {
    /// Create new recovery system
    pub fn new() -> Self {
        let (event_sender, _) = broadcast::channel(1000);

        Self {
            procedures: Arc::new(RwLock::new(Self::default_procedures())),
            executions: Arc::new(DashMap::new()),
            event_sender,
            max_concurrent_recoveries: 5,
            _global_timeout: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Get default recovery procedures
    fn default_procedures() -> Vec<RecoveryProcedure> {
        vec![
            RecoveryProcedure {
                id: "system-recovery".to_string(),
                recovery_type: RecoveryType::SystemRecovery,
                name: "Full System Recovery".to_string(),
                description: "Recover entire system after emergency shutdown".to_string(),
                steps: vec![
                    RecoveryStep {
                        name: "Stop all agents".to_string(),
                        action: "agent.stop_all".to_string(),
                        required: true,
                        timeout: Duration::from_secs(30),
                    },
                    RecoveryStep {
                        name: "Clear resource locks".to_string(),
                        action: "resource.clear_locks".to_string(),
                        required: true,
                        timeout: Duration::from_secs(10),
                    },
                    RecoveryStep {
                        name: "Reset kill switches".to_string(),
                        action: "killswitch.reset_all".to_string(),
                        required: true,
                        timeout: Duration::from_secs(5),
                    },
                    RecoveryStep {
                        name: "Restore checkpoints".to_string(),
                        action: "checkpoint.restore_latest".to_string(),
                        required: false,
                        timeout: Duration::from_secs(60),
                    },
                    RecoveryStep {
                        name: "Restart core services".to_string(),
                        action: "service.restart_core".to_string(),
                        required: true,
                        timeout: Duration::from_secs(30),
                    },
                ],
                timeout: Duration::from_secs(180),
                retry_count: 3,
                priority: 100,
            },
            RecoveryProcedure {
                id: "agent-recovery".to_string(),
                recovery_type: RecoveryType::AgentRecovery,
                name: "Agent Recovery".to_string(),
                description: "Recover individual agent after suspension".to_string(),
                steps: vec![
                    RecoveryStep {
                        name: "Stop agent".to_string(),
                        action: "agent.stop".to_string(),
                        required: true,
                        timeout: Duration::from_secs(10),
                    },
                    RecoveryStep {
                        name: "Clear agent state".to_string(),
                        action: "agent.clear_state".to_string(),
                        required: true,
                        timeout: Duration::from_secs(5),
                    },
                    RecoveryStep {
                        name: "Restore agent checkpoint".to_string(),
                        action: "agent.restore_checkpoint".to_string(),
                        required: false,
                        timeout: Duration::from_secs(20),
                    },
                    RecoveryStep {
                        name: "Restart agent".to_string(),
                        action: "agent.restart".to_string(),
                        required: true,
                        timeout: Duration::from_secs(15),
                    },
                ],
                timeout: Duration::from_secs(60),
                retry_count: 2,
                priority: 50,
            },
        ]
    }

    /// Execute recovery procedure
    pub async fn execute_recovery(
        &self,
        procedure_id: &str,
        agent_id: Option<String>,
    ) -> EmergencyResult<String> {
        // Check concurrent recovery limit
        let active_count = self
            .executions
            .iter()
            .filter(|e| e.state == RecoveryState::InProgress)
            .count();

        if active_count >= self.max_concurrent_recoveries {
            return Err(EmergencyError::RecoveryFailed {
                procedure: procedure_id.to_string(),
                reason: "Maximum concurrent recoveries reached".to_string(),
            });
        }

        // Find procedure
        let procedures = self.procedures.read().await;
        let procedure = procedures
            .iter()
            .find(|p| p.id == procedure_id)
            .ok_or_else(|| EmergencyError::RecoveryFailed {
                procedure: procedure_id.to_string(),
                reason: "Procedure not found".to_string(),
            })?
            .clone();
        drop(procedures);

        // Create execution record
        let execution_id = format!("{}-{}", procedure_id, uuid::Uuid::new_v4());
        let execution = RecoveryExecution {
            procedure_id: procedure_id.to_string(),
            recovery_type: procedure.recovery_type,
            state: RecoveryState::InProgress,
            started_at: Instant::now(),
            completed_at: None,
            error: None,
            steps_completed: 0,
            total_steps: procedure.steps.len(),
            agent_id: agent_id.clone(),
        };

        self.executions.insert(execution_id.clone(), execution);

        // Send start event
        let _ = self.event_sender.send(RecoveryEvent {
            procedure_id: procedure_id.to_string(),
            recovery_type: procedure.recovery_type,
            state: RecoveryState::InProgress,
            message: format!("Starting recovery: {}", procedure.name),
            timestamp: Instant::now(),
        });

        // Execute steps (simulated for now)
        let execution_id_clone = execution_id.clone();
        let executions = self.executions.clone();
        let event_sender = self.event_sender.clone();
        let procedure_clone = procedure.clone();

        tokio::spawn(async move {
            let mut success = true;
            let mut steps_completed = 0;
            let mut error_msg = None;

            for (i, step) in procedure_clone.steps.iter().enumerate() {
                info!(
                    "Executing recovery step {}/{}: {}",
                    i + 1,
                    procedure_clone.steps.len(),
                    step.name
                );

                // Simulate step execution
                tokio::time::sleep(Duration::from_millis(100)).await;

                // Random failure simulation for testing
                if step.required && rand::random::<f32>() < 0.1 {
                    error!("Recovery step failed: {}", step.name);
                    if step.required {
                        success = false;
                        error_msg = Some(format!("Step '{}' failed", step.name));
                        break;
                    }
                }

                steps_completed += 1;

                // Update execution record
                if let Some(mut exec) = executions.get_mut(&execution_id_clone) {
                    exec.steps_completed = steps_completed;
                }
            }

            // Update final state
            let final_state = if success {
                RecoveryState::Completed
            } else if steps_completed > 0 {
                RecoveryState::Partial
            } else {
                RecoveryState::Failed
            };

            if let Some(mut exec) = executions.get_mut(&execution_id_clone) {
                exec.state = final_state.clone();
                exec.completed_at = Some(Instant::now());
                exec.error = error_msg;
            }

            // Send completion event
            let _ = event_sender.send(RecoveryEvent {
                procedure_id: procedure_clone.id,
                recovery_type: procedure_clone.recovery_type,
                state: final_state.clone(),
                message: match final_state {
                    RecoveryState::Completed => "Recovery completed successfully".to_string(),
                    RecoveryState::Partial => format!(
                        "Recovery partially completed ({}/{})",
                        steps_completed,
                        procedure_clone.steps.len()
                    ),
                    RecoveryState::Failed => "Recovery failed".to_string(),
                    _ => "Unknown state".to_string(),
                },
                timestamp: Instant::now(),
            });
        });

        Ok(execution_id)
    }

    /// Get recovery execution status
    pub fn get_execution_status(&self, execution_id: &str) -> Option<RecoveryExecution> {
        self.executions.get(execution_id).map(|e| e.clone())
    }

    /// Get all active recoveries
    pub fn get_active_recoveries(&self) -> Vec<RecoveryExecution> {
        self.executions
            .iter()
            .filter(|e| e.state == RecoveryState::InProgress)
            .map(|e| e.value().clone())
            .collect()
    }

    /// Get recovery history
    pub fn get_recovery_history(&self, limit: usize) -> Vec<RecoveryExecution> {
        let mut history: Vec<_> = self.executions.iter().map(|e| e.value().clone()).collect();

        history.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        history.truncate(limit);
        history
    }

    /// Cancel recovery
    pub async fn cancel_recovery(&self, execution_id: &str) -> EmergencyResult<()> {
        if let Some(mut exec) = self.executions.get_mut(execution_id) {
            if exec.state == RecoveryState::InProgress {
                exec.state = RecoveryState::Failed;
                exec.completed_at = Some(Instant::now());
                exec.error = Some("Cancelled by user".to_string());

                // Send cancellation event
                let _ = self.event_sender.send(RecoveryEvent {
                    procedure_id: exec.procedure_id.clone(),
                    recovery_type: exec.recovery_type,
                    state: RecoveryState::Failed,
                    message: "Recovery cancelled".to_string(),
                    timestamp: Instant::now(),
                });

                Ok(())
            } else {
                Err(EmergencyError::RecoveryFailed {
                    procedure: execution_id.to_string(),
                    reason: "Recovery not in progress".to_string(),
                })
            }
        } else {
            Err(EmergencyError::RecoveryFailed {
                procedure: execution_id.to_string(),
                reason: "Execution not found".to_string(),
            })
        }
    }

    /// Add custom recovery procedure
    pub async fn add_procedure(&self, procedure: RecoveryProcedure) -> EmergencyResult<()> {
        let mut procedures = self.procedures.write().await;

        // Check for duplicate ID
        if procedures.iter().any(|p| p.id == procedure.id) {
            return Err(EmergencyError::ConfigurationError(format!(
                "Procedure with ID '{}' already exists",
                procedure.id
            )));
        }

        procedures.push(procedure);

        // Sort by priority
        procedures.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(())
    }

    /// Remove recovery procedure
    pub async fn remove_procedure(&self, procedure_id: &str) -> EmergencyResult<()> {
        let mut procedures = self.procedures.write().await;
        let original_len = procedures.len();
        procedures.retain(|p| p.id != procedure_id);

        if procedures.len() == original_len {
            Err(EmergencyError::ConfigurationError(format!(
                "Procedure '{procedure_id}' not found"
            )))
        } else {
            Ok(())
        }
    }

    /// Get all procedures
    pub async fn get_procedures(&self) -> Vec<RecoveryProcedure> {
        self.procedures.read().await.clone()
    }

    /// Get procedures by type
    pub async fn get_procedures_by_type(
        &self,
        recovery_type: RecoveryType,
    ) -> Vec<RecoveryProcedure> {
        self.procedures
            .read()
            .await
            .iter()
            .filter(|p| p.recovery_type == recovery_type)
            .cloned()
            .collect()
    }

    /// Subscribe to recovery events
    pub fn subscribe(&self) -> broadcast::Receiver<RecoveryEvent> {
        self.event_sender.subscribe()
    }

    /// Clear completed executions
    pub fn clear_completed_executions(&self) {
        self.executions
            .retain(|_, e| e.state == RecoveryState::InProgress);
    }

    /// Emergency recovery (bypass all checks)
    pub async fn emergency_recovery(&self) -> EmergencyResult<()> {
        warn!("Initiating emergency recovery - bypassing all safety checks");

        // Execute all system recovery procedures
        let procedures = self.procedures.read().await;
        let system_procedures: Vec<_> = procedures
            .iter()
            .filter(|p| p.recovery_type == RecoveryType::SystemRecovery)
            .cloned()
            .collect();
        drop(procedures);

        for procedure in system_procedures {
            match self.execute_recovery(&procedure.id, None).await {
                Ok(exec_id) => {
                    info!(
                        "Started emergency recovery: {} ({})",
                        procedure.name, exec_id
                    );
                }
                Err(e) => {
                    error!(
                        "Failed to start emergency recovery {}: {}",
                        procedure.name, e
                    );
                }
            }
        }

        Ok(())
    }

    /// Execute a specific procedure
    pub async fn execute_procedure(&mut self, name: &str) -> EmergencyResult<()> {
        let procedures = self.procedures.read().await;
        let procedure = procedures.iter().find(|p| p.name == name).ok_or_else(|| {
            EmergencyError::RecoveryFailed {
                procedure: name.to_string(),
                reason: "Procedure not found".to_string(),
            }
        })?;

        let proc_id = procedure.id.clone();
        drop(procedures);

        self.execute_recovery(&proc_id, None).await?;
        Ok(())
    }

    /// Execute emergency procedure (high priority)
    pub async fn execute_emergency_procedure(&mut self, name: &str) -> EmergencyResult<()> {
        // Emergency procedures bypass normal limits
        self.execute_procedure(name).await
    }

    /// Get current recovery state
    pub fn state(&self) -> RecoveryState {
        // Find the most recent execution
        let mut latest_state = RecoveryState::Idle;
        let mut latest_time = None;

        for entry in self.executions.iter() {
            let exec = entry.value();
            if latest_time.is_none() || Some(exec.started_at) > latest_time {
                latest_time = Some(exec.started_at);
                latest_state = exec.state.clone();
            }
        }

        latest_state
    }

    /// Clone for async operations
    pub fn clone(&self) -> Self {
        Self {
            procedures: self.procedures.clone(),
            executions: self.executions.clone(),
            event_sender: self.event_sender.clone(),
            max_concurrent_recoveries: self.max_concurrent_recoveries,
            _global_timeout: self._global_timeout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_recovery_system_creation() {
        let recovery = RecoverySystem::new();
        let procedures = recovery.get_procedures().await;
        assert!(!procedures.is_empty());
        assert!(procedures
            .iter()
            .any(|p| p.recovery_type == RecoveryType::SystemRecovery));
        assert!(procedures
            .iter()
            .any(|p| p.recovery_type == RecoveryType::AgentRecovery));
    }

    #[tokio::test]
    async fn test_execute_recovery() {
        let recovery = RecoverySystem::new();
        let mut receiver = recovery.subscribe();

        let exec_id = recovery
            .execute_recovery("system-recovery", None)
            .await
            .unwrap();
        assert!(!exec_id.is_empty());

        // Check execution was created
        let status = recovery.get_execution_status(&exec_id).unwrap();
        assert_eq!(status.state, RecoveryState::InProgress);
        assert_eq!(status.recovery_type, RecoveryType::SystemRecovery);

        // Wait for start event
        let event = receiver.recv().await.unwrap();
        assert_eq!(event.state, RecoveryState::InProgress);
    }

    #[tokio::test]
    async fn test_recovery_with_agent_id() {
        let recovery = RecoverySystem::new();

        let exec_id = recovery
            .execute_recovery("agent-recovery", Some("agent-123".to_string()))
            .await
            .unwrap();

        let status = recovery.get_execution_status(&exec_id).unwrap();
        assert_eq!(status.agent_id, Some("agent-123".to_string()));
    }

    #[tokio::test]
    async fn test_concurrent_recovery_limit() {
        let mut recovery = RecoverySystem::new();
        recovery.max_concurrent_recoveries = 2;

        // Start two recoveries
        let _exec1 = recovery
            .execute_recovery("system-recovery", None)
            .await
            .unwrap();
        let _exec2 = recovery
            .execute_recovery("agent-recovery", Some("agent-1".to_string()))
            .await
            .unwrap();

        // Third should fail
        let result = recovery
            .execute_recovery("agent-recovery", Some("agent-2".to_string()))
            .await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            EmergencyError::RecoveryFailed { .. }
        ));
    }

    #[tokio::test]
    async fn test_get_active_recoveries() {
        let recovery = RecoverySystem::new();

        // Start multiple recoveries
        recovery
            .execute_recovery("system-recovery", None)
            .await
            .unwrap();
        recovery
            .execute_recovery("agent-recovery", Some("agent-1".to_string()))
            .await
            .unwrap();

        let active = recovery.get_active_recoveries();
        assert_eq!(active.len(), 2);
        assert!(active.iter().all(|e| e.state == RecoveryState::InProgress));
    }

    #[tokio::test]
    async fn test_recovery_history() {
        let recovery = RecoverySystem::new();

        // Create some recovery executions
        let _exec1 = recovery
            .execute_recovery("system-recovery", None)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        let _exec2 = recovery
            .execute_recovery("agent-recovery", Some("agent-1".to_string()))
            .await
            .unwrap();

        let history = recovery.get_recovery_history(10);
        assert!(history.len() >= 2);

        // Verify ordering (newest first)
        let exec2_time = history
            .iter()
            .find(|e| e.procedure_id == "agent-recovery")
            .unwrap()
            .started_at;
        let exec1_time = history
            .iter()
            .find(|e| e.procedure_id == "system-recovery")
            .unwrap()
            .started_at;
        assert!(exec2_time > exec1_time);
    }

    #[tokio::test]
    async fn test_cancel_recovery() {
        let recovery = RecoverySystem::new();
        let mut receiver = recovery.subscribe();

        let exec_id = recovery
            .execute_recovery("system-recovery", None)
            .await
            .unwrap();

        // Cancel the recovery
        recovery.cancel_recovery(&exec_id).await.unwrap();

        let status = recovery.get_execution_status(&exec_id).unwrap();
        assert_eq!(status.state, RecoveryState::Failed);
        assert_eq!(status.error, Some("Cancelled by user".to_string()));

        // Check cancellation event
        // Skip the start event
        let _ = receiver.recv().await;
        let event = receiver.recv().await.unwrap();
        assert_eq!(event.state, RecoveryState::Failed);
        assert!(event.message.contains("cancelled"));
    }

    #[tokio::test]
    async fn test_cancel_completed_recovery() {
        let recovery = RecoverySystem::new();

        // Create a fake completed execution
        let exec_id = "test-exec-123";
        recovery.executions.insert(
            exec_id.to_string(),
            RecoveryExecution {
                procedure_id: "test".to_string(),
                recovery_type: RecoveryType::SystemRecovery,
                state: RecoveryState::Completed,
                started_at: Instant::now(),
                completed_at: Some(Instant::now()),
                error: None,
                steps_completed: 5,
                total_steps: 5,
                agent_id: None,
            },
        );

        // Try to cancel completed recovery
        let result = recovery.cancel_recovery(exec_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_add_custom_procedure() {
        let recovery = RecoverySystem::new();

        let custom_procedure = RecoveryProcedure {
            id: "custom-recovery".to_string(),
            recovery_type: RecoveryType::DataRecovery,
            name: "Custom Data Recovery".to_string(),
            description: "Custom recovery procedure".to_string(),
            steps: vec![RecoveryStep {
                name: "Verify data".to_string(),
                action: "data.verify".to_string(),
                required: true,
                timeout: Duration::from_secs(30),
            }],
            timeout: Duration::from_secs(60),
            retry_count: 1,
            priority: 75,
        };

        recovery
            .add_procedure(custom_procedure.clone())
            .await
            .unwrap();

        let procedures = recovery.get_procedures().await;
        assert!(procedures.iter().any(|p| p.id == "custom-recovery"));
    }

    #[tokio::test]
    async fn test_add_duplicate_procedure() {
        let recovery = RecoverySystem::new();
        let procedures = recovery.get_procedures().await;
        let existing = procedures[0].clone();

        let result = recovery.add_procedure(existing).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            EmergencyError::ConfigurationError(_)
        ));
    }

    #[tokio::test]
    async fn test_remove_procedure() {
        let recovery = RecoverySystem::new();

        // Add a procedure
        let custom_procedure = RecoveryProcedure {
            id: "removable".to_string(),
            recovery_type: RecoveryType::ConfigurationRecovery,
            name: "Removable Procedure".to_string(),
            description: "This will be removed".to_string(),
            steps: vec![],
            timeout: Duration::from_secs(30),
            retry_count: 1,
            priority: 10,
        };

        recovery.add_procedure(custom_procedure).await.unwrap();

        // Remove it
        recovery.remove_procedure("removable").await.unwrap();

        let procedures = recovery.get_procedures().await;
        assert!(!procedures.iter().any(|p| p.id == "removable"));
    }

    #[tokio::test]
    async fn test_remove_nonexistent_procedure() {
        let recovery = RecoverySystem::new();
        let result = recovery.remove_procedure("does-not-exist").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_procedures_by_type() {
        let recovery = RecoverySystem::new();

        let system_procedures = recovery
            .get_procedures_by_type(RecoveryType::SystemRecovery)
            .await;
        assert!(!system_procedures.is_empty());
        assert!(system_procedures
            .iter()
            .all(|p| p.recovery_type == RecoveryType::SystemRecovery));

        let agent_procedures = recovery
            .get_procedures_by_type(RecoveryType::AgentRecovery)
            .await;
        assert!(!agent_procedures.is_empty());
        assert!(agent_procedures
            .iter()
            .all(|p| p.recovery_type == RecoveryType::AgentRecovery));
    }

    #[tokio::test]
    async fn test_clear_completed_executions() {
        let recovery = RecoverySystem::new();

        // Add some executions with different states
        recovery.executions.insert(
            "completed-1".to_string(),
            RecoveryExecution {
                procedure_id: "test".to_string(),
                recovery_type: RecoveryType::SystemRecovery,
                state: RecoveryState::Completed,
                started_at: Instant::now(),
                completed_at: Some(Instant::now()),
                error: None,
                steps_completed: 5,
                total_steps: 5,
                agent_id: None,
            },
        );

        recovery.executions.insert(
            "in-progress-1".to_string(),
            RecoveryExecution {
                procedure_id: "test".to_string(),
                recovery_type: RecoveryType::AgentRecovery,
                state: RecoveryState::InProgress,
                started_at: Instant::now(),
                completed_at: None,
                error: None,
                steps_completed: 2,
                total_steps: 5,
                agent_id: None,
            },
        );

        recovery.executions.insert(
            "failed-1".to_string(),
            RecoveryExecution {
                procedure_id: "test".to_string(),
                recovery_type: RecoveryType::DataRecovery,
                state: RecoveryState::Failed,
                started_at: Instant::now(),
                completed_at: Some(Instant::now()),
                error: Some("Test error".to_string()),
                steps_completed: 0,
                total_steps: 5,
                agent_id: None,
            },
        );

        // Clear completed
        recovery.clear_completed_executions();

        // Only in-progress should remain
        assert_eq!(recovery.executions.len(), 1);
        assert!(recovery.executions.contains_key("in-progress-1"));
    }

    #[tokio::test]
    async fn test_emergency_recovery() {
        let recovery = RecoverySystem::new();
        let mut receiver = recovery.subscribe();

        recovery.emergency_recovery().await.unwrap();

        // Should receive at least one start event
        let event = receiver.recv().await.unwrap();
        assert_eq!(event.recovery_type, RecoveryType::SystemRecovery);
        assert_eq!(event.state, RecoveryState::InProgress);
    }

    #[tokio::test]
    async fn test_procedure_priority_ordering() {
        let recovery = RecoverySystem::new();

        // Add procedures with different priorities
        let low_priority = RecoveryProcedure {
            id: "low-priority".to_string(),
            recovery_type: RecoveryType::ConfigurationRecovery,
            name: "Low Priority".to_string(),
            description: "Low priority procedure".to_string(),
            steps: vec![],
            timeout: Duration::from_secs(30),
            retry_count: 1,
            priority: 10,
        };

        let high_priority = RecoveryProcedure {
            id: "high-priority".to_string(),
            recovery_type: RecoveryType::ConfigurationRecovery,
            name: "High Priority".to_string(),
            description: "High priority procedure".to_string(),
            steps: vec![],
            timeout: Duration::from_secs(30),
            retry_count: 1,
            priority: 200,
        };

        recovery.add_procedure(low_priority).await.unwrap();
        recovery.add_procedure(high_priority).await.unwrap();

        let procedures = recovery.get_procedures().await;

        // Find positions
        let high_pos = procedures
            .iter()
            .position(|p| p.id == "high-priority")
            .unwrap();
        let low_pos = procedures
            .iter()
            .position(|p| p.id == "low-priority")
            .unwrap();

        // High priority should come before low priority
        assert!(high_pos < low_pos);
    }

    #[tokio::test]
    async fn test_recovery_type_serialization() {
        let recovery_type = RecoveryType::NetworkRecovery;
        let serialized = serde_json::to_string(&recovery_type).unwrap();
        let deserialized: RecoveryType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(recovery_type, deserialized);
    }

    #[tokio::test]
    async fn test_recovery_state_serialization() {
        let state = RecoveryState::Partial;
        let serialized = serde_json::to_string(&state).unwrap();
        let deserialized: RecoveryState = serde_json::from_str(&serialized).unwrap();
        assert_eq!(state, deserialized);
    }
}
