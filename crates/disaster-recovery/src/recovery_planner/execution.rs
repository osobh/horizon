//! Recovery plan execution and monitoring

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Recovery execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryExecution {
    /// Execution ID
    pub id: Uuid,
    /// Plan being executed
    pub plan_id: Uuid,
    /// Current execution state
    pub state: ExecutionState,
    /// Started timestamp
    pub started_at: DateTime<Utc>,
    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Current step being executed
    pub current_step_id: Option<Uuid>,
    /// Steps and their execution status
    pub step_executions: HashMap<Uuid, StepExecution>,
    /// Overall progress (0.0 to 1.0)
    pub progress: f64,
    /// Execution logs
    pub logs: Vec<ExecutionLog>,
    /// Error information if failed
    pub error_info: Option<String>,
    /// Resource allocations used
    pub resource_allocations: Vec<Uuid>,
}

/// Execution state enum
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionState {
    /// Execution is pending
    Pending,
    /// Currently running
    Running,
    /// Paused by user or system
    Paused,
    /// Completed successfully
    Completed,
    /// Failed with errors
    Failed,
    /// Cancelled by user
    Cancelled,
    /// Rolling back changes
    RollingBack,
    /// Rollback completed
    RolledBack,
}

/// Individual step execution tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepExecution {
    /// Step ID
    pub step_id: Uuid,
    /// Execution state
    pub state: ExecutionState,
    /// Started timestamp
    pub started_at: Option<DateTime<Utc>>,
    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Retry attempts made
    pub retry_count: u32,
    /// Step output/result
    pub output: Option<String>,
    /// Error details if failed
    pub error: Option<String>,
    /// Resource allocations for this step
    pub resource_allocations: Vec<Uuid>,
}

/// Execution log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLog {
    /// Log entry ID
    pub id: Uuid,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Log level
    pub level: LogLevel,
    /// Log message
    pub message: String,
    /// Related step ID if applicable
    pub step_id: Option<Uuid>,
    /// Additional context data
    pub context: HashMap<String, String>,
}

/// Log level enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    /// Debug information
    Debug,
    /// General information
    Info,
    /// Warning messages
    Warn,
    /// Error messages
    Error,
    /// Critical errors
    Critical,
}

impl RecoveryExecution {
    /// Create new recovery execution
    pub fn new(plan_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            plan_id,
            state: ExecutionState::Pending,
            started_at: Utc::now(),
            completed_at: None,
            current_step_id: None,
            step_executions: HashMap::new(),
            progress: 0.0,
            logs: Vec::new(),
            error_info: None,
            resource_allocations: Vec::new(),
        }
    }

    /// Add log entry
    pub fn add_log(&mut self, level: LogLevel, message: String, step_id: Option<Uuid>) {
        let log = ExecutionLog {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            level,
            message,
            step_id,
            context: HashMap::new(),
        };
        self.logs.push(log);
    }

    /// Update execution progress
    pub fn update_progress(&mut self) {
        if self.step_executions.is_empty() {
            self.progress = 0.0;
            return;
        }

        let completed_steps = self
            .step_executions
            .values()
            .filter(|step| step.state == ExecutionState::Completed)
            .count();

        self.progress = completed_steps as f64 / self.step_executions.len() as f64;
    }

    /// Check if execution is in terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            ExecutionState::Completed
                | ExecutionState::Failed
                | ExecutionState::Cancelled
                | ExecutionState::RolledBack
        )
    }

    /// Get execution duration
    pub fn duration(&self) -> chrono::Duration {
        let end_time = self.completed_at.unwrap_or_else(Utc::now);
        end_time - self.started_at
    }

    /// Get failed steps
    pub fn failed_steps(&self) -> Vec<&StepExecution> {
        self.step_executions
            .values()
            .filter(|step| step.state == ExecutionState::Failed)
            .collect()
    }
}

impl StepExecution {
    /// Create new step execution
    pub fn new(step_id: Uuid) -> Self {
        Self {
            step_id,
            state: ExecutionState::Pending,
            started_at: None,
            completed_at: None,
            retry_count: 0,
            output: None,
            error: None,
            resource_allocations: Vec::new(),
        }
    }

    /// Mark step as started
    pub fn start(&mut self) {
        self.state = ExecutionState::Running;
        self.started_at = Some(Utc::now());
    }

    /// Mark step as completed
    pub fn complete(&mut self, output: Option<String>) {
        self.state = ExecutionState::Completed;
        self.completed_at = Some(Utc::now());
        self.output = output;
    }

    /// Mark step as failed
    pub fn fail(&mut self, error: String) {
        self.state = ExecutionState::Failed;
        self.completed_at = Some(Utc::now());
        self.error = Some(error);
    }

    /// Get step duration
    pub fn duration(&self) -> Option<chrono::Duration> {
        if let (Some(start), Some(end)) = (self.started_at, self.completed_at) {
            Some(end - start)
        } else {
            None
        }
    }
}
