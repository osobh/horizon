//! Runbook executor core implementation
//!
//! This module contains the main RunbookExecutor implementation and
//! execution orchestration logic.

use super::runbook_types::*;
use crate::error::{DisasterRecoveryError, DisasterRecoveryResult};
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, Semaphore};
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Executor commands
#[derive(Debug)]
enum ExecutorCommand {
    /// Execute runbook
    ExecuteRunbook(Uuid, ExecutionTrigger, ExecutionContext),
    /// Cancel execution
    CancelExecution(Uuid),
    /// Approve step
    ApproveStep(Uuid, String, String),
    /// Reject step
    RejectStep(Uuid, String, String),
    /// Resume execution
    ResumeExecution(Uuid),
    /// Pause execution
    PauseExecution(Uuid),
    /// Manual step completed
    ManualStepCompleted(Uuid, Uuid, String),
    /// Cleanup history
    CleanupHistory,
    /// Update metrics
    UpdateMetrics,
}

/// Runbook executor
pub struct RunbookExecutor {
    /// Configuration
    config: Arc<RunbookExecutorConfig>,
    /// Runbooks
    runbooks: Arc<DashMap<Uuid, Runbook>>,
    /// Active executions
    executions: Arc<DashMap<Uuid, RunbookExecution>>,
    /// Execution history
    execution_history: Arc<RwLock<VecDeque<RunbookExecution>>>,
    /// Metrics
    metrics: Arc<RwLock<RunbookExecutorMetrics>>,
    /// Command channel
    command_tx: mpsc::Sender<ExecutorCommand>,
    /// Command receiver
    command_rx: Arc<Mutex<mpsc::Receiver<ExecutorCommand>>>,
    /// Execution semaphore
    execution_semaphore: Arc<Semaphore>,
    /// Shutdown flag
    shutdown: Arc<RwLock<bool>>,
}

impl RunbookExecutor {
    /// Create new runbook executor
    pub fn new(config: RunbookExecutorConfig) -> DisasterRecoveryResult<Self> {
        let (command_tx, command_rx) = mpsc::channel(10000);
        let execution_semaphore = Arc::new(Semaphore::new(config.max_concurrent_executions));

        Ok(Self {
            config: Arc::new(config),
            runbooks: Arc::new(DashMap::new()),
            executions: Arc::new(DashMap::new()),
            execution_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            metrics: Arc::new(RwLock::new(RunbookExecutorMetrics::default())),
            command_tx,
            command_rx: Arc::new(Mutex::new(command_rx)),
            execution_semaphore,
            shutdown: Arc::new(RwLock::new(false)),
        })
    }

    /// Start runbook executor
    pub async fn start(&self) -> DisasterRecoveryResult<()> {
        info!("Starting runbook executor");

        // Start background tasks
        self.start_command_processor().await?;
        self.start_cleanup_task().await?;
        self.start_metrics_updater().await?;

        Ok(())
    }

    /// Stop runbook executor
    pub async fn stop(&self) -> DisasterRecoveryResult<()> {
        info!("Stopping runbook executor");
        *self.shutdown.write() = true;
        Ok(())
    }

    /// Register runbook
    pub async fn register_runbook(&self, runbook: Runbook) -> DisasterRecoveryResult<Uuid> {
        // Validate runbook
        self.validate_runbook(&runbook)?;

        let runbook_id = runbook.id;
        self.runbooks.insert(runbook_id, runbook.clone());

        info!("Registered runbook: {} ({})", runbook.name, runbook_id);
        self.update_metrics_count().await;

        Ok(runbook_id)
    }

    /// Execute runbook
    pub async fn execute_runbook(
        &self,
        runbook_id: Uuid,
        trigger: ExecutionTrigger,
        context: ExecutionContext,
    ) -> DisasterRecoveryResult<Uuid> {
        // Check if runbook exists
        let runbook = self.runbooks.get(&runbook_id).ok_or_else(|| {
            DisasterRecoveryError::ResourceUnavailable {
                resource: "runbook".to_string(),
                reason: "runbook not found".to_string(),
            }
        })?;

        if !runbook.enabled {
            return Err(DisasterRecoveryError::RunbookFailed {
                runbook_id: runbook_id.to_string(),
                step: "validation".to_string(),
                reason: "runbook is disabled".to_string(),
            });
        }

        drop(runbook);

        // Create execution record
        let execution = RunbookExecution {
            id: Uuid::new_v4(),
            runbook_id,
            state: if self.runbooks.get(&runbook_id).unwrap().requires_approval {
                ExecutionState::PendingApproval
            } else {
                ExecutionState::CheckingPrerequisites
            },
            trigger,
            start_time: Utc::now(),
            end_time: None,
            current_step: None,
            step_executions: HashMap::new(),
            progress: 0.0,
            context,
            rollback_execution_id: None,
            approval_requests: Vec::new(),
            logs: VecDeque::new(),
        };

        let execution_id = execution.id;
        self.executions.insert(execution_id, execution);

        // Queue execution command
        self.command_tx
            .send(ExecutorCommand::ExecuteRunbook(
                runbook_id,
                self.executions.get(&execution_id).unwrap().trigger.clone(),
                self.executions.get(&execution_id).unwrap().context.clone(),
            ))
            .await
            .map_err(|_| DisasterRecoveryError::RunbookFailed {
                runbook_id: runbook_id.to_string(),
                step: "execution".to_string(),
                reason: "failed to queue execution".to_string(),
            })?;

        info!(
            "Initiated runbook execution: {} (execution: {})",
            runbook_id, execution_id
        );
        Ok(execution_id)
    }

    /// Cancel execution
    pub async fn cancel_execution(&self, execution_id: Uuid) -> DisasterRecoveryResult<()> {
        self.command_tx
            .send(ExecutorCommand::CancelExecution(execution_id))
            .await
            .map_err(|_| DisasterRecoveryError::NetworkError {
                details: "failed to queue cancellation".to_string(),
            })?;

        Ok(())
    }

    /// Approve step
    pub async fn approve_step(
        &self,
        execution_id: Uuid,
        step_id: Uuid,
        approver: String,
        reason: String,
    ) -> DisasterRecoveryResult<()> {
        self.command_tx
            .send(ExecutorCommand::ApproveStep(execution_id, approver, reason))
            .await
            .map_err(|_| DisasterRecoveryError::NetworkError {
                details: "failed to queue approval".to_string(),
            })?;

        Ok(())
    }

    /// Get execution status
    pub fn get_execution_status(&self, execution_id: Uuid) -> Option<RunbookExecution> {
        self.executions
            .get(&execution_id)
            .map(|entry| entry.value().clone())
    }

    /// List active executions
    pub fn list_active_executions(&self) -> Vec<RunbookExecution> {
        self.executions
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get execution history
    pub fn get_execution_history(&self, limit: Option<usize>) -> Vec<RunbookExecution> {
        let history = self.execution_history.read();
        let entries: Vec<RunbookExecution> = history
            .iter()
            .rev()
            .take(limit.unwrap_or(history.len()))
            .cloned()
            .collect();

        entries
    }

    /// Get runbook
    pub fn get_runbook(&self, runbook_id: Uuid) -> Option<Runbook> {
        self.runbooks
            .get(&runbook_id)
            .map(|entry| entry.value().clone())
    }

    /// List runbooks
    pub fn list_runbooks(&self, category: Option<RunbookCategory>) -> Vec<Runbook> {
        self.runbooks
            .iter()
            .filter(|entry| {
                if let Some(cat) = &category {
                    &entry.value().category == cat
                } else {
                    true
                }
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get metrics
    pub fn get_metrics(&self) -> RunbookExecutorMetrics {
        self.metrics.read().clone()
    }

    // Private helper methods

    async fn start_command_processor(&self) -> DisasterRecoveryResult<()> {
        let command_rx = Arc::clone(&self.command_rx);
        let shutdown = Arc::clone(&self.shutdown);
        let runbooks = Arc::clone(&self.runbooks);
        let executions = Arc::clone(&self.executions);
        let execution_history = Arc::clone(&self.execution_history);
        let execution_semaphore = Arc::clone(&self.execution_semaphore);

        tokio::spawn(async move {
            while !*shutdown.read() {
                let mut rx = command_rx.lock().await;
                if let Some(command) = rx.recv().await {
                    match command {
                        ExecutorCommand::ExecuteRunbook(runbook_id, trigger, context) => {
                            if let Ok(_permit) = execution_semaphore.try_acquire() {
                                if let Some(runbook) = runbooks.get(&runbook_id) {
                                    // Find the execution for this runbook
                                    let mut target_execution = None;
                                    for mut execution in executions.iter_mut() {
                                        if execution.runbook_id == runbook_id
                                            && execution.state
                                                == ExecutionState::CheckingPrerequisites
                                        {
                                            target_execution = Some(execution.id);
                                            execution.state = ExecutionState::Running;
                                            break;
                                        }
                                    }

                                    if let Some(execution_id) = target_execution {
                                        info!(
                                            "Executing runbook: {} ({})",
                                            runbook.name, execution_id
                                        );

                                        // Simulate step execution
                                        let steps = runbook.steps.clone();
                                        for (i, step) in steps.iter().enumerate() {
                                            if let Some(mut execution) =
                                                executions.get_mut(&execution_id)
                                            {
                                                execution.current_step = Some(step.id);
                                                execution.progress =
                                                    (i as f64 + 1.0) / steps.len() as f64;

                                                // Create step execution
                                                let step_execution = StepExecution {
                                                    step_id: step.id,
                                                    state: StepExecutionState::Running,
                                                    start_time: Utc::now(),
                                                    end_time: None,
                                                    duration_seconds: 0,
                                                    attempt_count: 1,
                                                    exit_code: Some(0),
                                                    output: format!("Executed step: {}", step.name),
                                                    error_output: String::new(),
                                                    validation_results: Vec::new(),
                                                    skip_reason: None,
                                                };

                                                execution
                                                    .step_executions
                                                    .insert(step.id, step_execution);

                                                // Add log entry
                                                execution.logs.push_back(ExecutionLog {
                                                    timestamp: Utc::now(),
                                                    level: LogLevel::Info,
                                                    message: format!(
                                                        "Executing step: {}",
                                                        step.name
                                                    ),
                                                    step_id: Some(step.id),
                                                    context: HashMap::new(),
                                                });
                                            }

                                            // Simulate step execution time
                                            tokio::time::sleep(std::time::Duration::from_millis(
                                                200,
                                            ))
                                            .await;

                                            // Update step as completed
                                            if let Some(mut execution) =
                                                executions.get_mut(&execution_id)
                                            {
                                                if let Some(mut step_exec) =
                                                    execution.step_executions.get_mut(&step.id)
                                                {
                                                    step_exec.state = StepExecutionState::Completed;
                                                    step_exec.end_time = Some(Utc::now());
                                                    step_exec.duration_seconds = 1;
                                                }
                                            }
                                        }

                                        // Complete execution
                                        if let Some(mut execution) =
                                            executions.get_mut(&execution_id)
                                        {
                                            execution.state = ExecutionState::Completed;
                                            execution.end_time = Some(Utc::now());
                                            execution.progress = 1.0;
                                            execution.current_step = None;

                                            execution.logs.push_back(ExecutionLog {
                                                timestamp: Utc::now(),
                                                level: LogLevel::Info,
                                                message: "Runbook execution completed successfully"
                                                    .to_string(),
                                                step_id: None,
                                                context: HashMap::new(),
                                            });
                                        }

                                        info!("Completed runbook execution: {}", execution_id);
                                    }
                                }
                            }
                        }
                        ExecutorCommand::CancelExecution(execution_id) => {
                            if let Some(mut execution) = executions.get_mut(&execution_id) {
                                execution.state = ExecutionState::Cancelled;
                                execution.end_time = Some(Utc::now());

                                execution.logs.push_back(ExecutionLog {
                                    timestamp: Utc::now(),
                                    level: LogLevel::Warning,
                                    message: "Execution cancelled by user".to_string(),
                                    step_id: None,
                                    context: HashMap::new(),
                                });

                                info!("Cancelled execution: {}", execution_id);
                            }
                        }
                        ExecutorCommand::ApproveStep(execution_id, approver, reason) => {
                            debug!("Processing step approval for execution: {}", execution_id);
                        }
                        ExecutorCommand::RejectStep(execution_id, rejector, reason) => {
                            debug!("Processing step rejection for execution: {}", execution_id);
                        }
                        ExecutorCommand::ResumeExecution(execution_id) => {
                            if let Some(mut execution) = executions.get_mut(&execution_id) {
                                if execution.state == ExecutionState::Paused {
                                    execution.state = ExecutionState::Running;
                                    info!("Resumed execution: {}", execution_id);
                                }
                            }
                        }
                        ExecutorCommand::PauseExecution(execution_id) => {
                            if let Some(mut execution) = executions.get_mut(&execution_id) {
                                if execution.state == ExecutionState::Running {
                                    execution.state = ExecutionState::Paused;
                                    info!("Paused execution: {}", execution_id);
                                }
                            }
                        }
                        ExecutorCommand::ManualStepCompleted(execution_id, step_id, result) => {
                            debug!(
                                "Manual step completed: {} in execution: {}",
                                step_id, execution_id
                            );
                        }
                        ExecutorCommand::CleanupHistory => {
                            let cutoff = Utc::now() - Duration::days(90);
                            let mut history = execution_history.write();
                            history.retain(|execution| execution.start_time > cutoff);
                            debug!("Cleaned up execution history");
                        }
                        ExecutorCommand::UpdateMetrics => {
                            debug!("Updating executor metrics");
                        }
                    }
                }
            }
        });

        Ok(())
    }

    async fn start_cleanup_task(&self) -> DisasterRecoveryResult<()> {
        let command_tx = self.command_tx.clone();
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut cleanup_interval = interval(std::time::Duration::from_secs(86400)); // Daily

            while !*shutdown.read() {
                cleanup_interval.tick().await;
                let _ = command_tx.send(ExecutorCommand::CleanupHistory).await;
            }
        });

        Ok(())
    }

    async fn start_metrics_updater(&self) -> DisasterRecoveryResult<()> {
        let runbooks = Arc::clone(&self.runbooks);
        let executions = Arc::clone(&self.executions);
        let metrics = Arc::clone(&self.metrics);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut update_interval = interval(std::time::Duration::from_secs(60));

            while !*shutdown.read() {
                update_interval.tick().await;

                let mut metrics_guard = metrics.write();

                metrics_guard.total_runbooks = runbooks.len();
                metrics_guard.active_executions = executions
                    .iter()
                    .filter(|entry| {
                        matches!(
                            entry.value().state,
                            ExecutionState::Running
                                | ExecutionState::PendingApproval
                                | ExecutionState::CheckingPrerequisites
                                | ExecutionState::WaitingForManual
                        )
                    })
                    .count();

                // Calculate success rate
                let completed_executions = executions
                    .iter()
                    .filter(|entry| {
                        matches!(
                            entry.value().state,
                            ExecutionState::Completed | ExecutionState::Failed
                        )
                    })
                    .count();

                let successful_executions = executions
                    .iter()
                    .filter(|entry| entry.value().state == ExecutionState::Completed)
                    .count();

                if completed_executions > 0 {
                    metrics_guard.success_rate =
                        successful_executions as f64 / completed_executions as f64;
                }

                metrics_guard.successful_executions = successful_executions as u64;
                metrics_guard.failed_executions = executions
                    .iter()
                    .filter(|entry| entry.value().state == ExecutionState::Failed)
                    .count() as u64;
            }
        });

        Ok(())
    }

    fn validate_runbook(&self, runbook: &Runbook) -> DisasterRecoveryResult<()> {
        if runbook.steps.is_empty() {
            return Err(DisasterRecoveryError::ConfigurationError {
                message: "runbook must have at least one step".to_string(),
            });
        }

        if runbook.name.is_empty() {
            return Err(DisasterRecoveryError::ConfigurationError {
                message: "runbook name cannot be empty".to_string(),
            });
        }

        // Validate step ordering
        let mut orders: Vec<u32> = runbook.steps.iter().map(|s| s.order).collect();
        orders.sort();
        for (i, &order) in orders.iter().enumerate() {
            if i > 0 && order == orders[i - 1] {
                return Err(DisasterRecoveryError::ConfigurationError {
                    message: "duplicate step order found".to_string(),
                });
            }
        }

        // Validate dependencies
        let step_ids: HashSet<Uuid> = runbook.steps.iter().map(|s| s.id).collect();
        for step in &runbook.steps {
            for dep_id in &step.dependencies {
                if !step_ids.contains(dep_id) {
                    return Err(DisasterRecoveryError::ConfigurationError {
                        message: format!("step dependency {} not found", dep_id),
                    });
                }
            }
        }

        Ok(())
    }

    async fn update_metrics_count(&self) {
        let mut metrics = self.metrics.write();
        metrics.total_runbooks = self.runbooks.len();
    }
}
