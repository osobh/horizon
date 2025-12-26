//! Runbook execution types and data structures
//!
//! This module contains all the type definitions for runbook execution including:
//! - Runbook definition structures
//! - Execution state and tracking types  
//! - Step configuration and action types
//! - Validation and condition types
//! - Configuration and metrics types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

/// Runbook definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Runbook {
    /// Runbook ID
    pub id: Uuid,
    /// Runbook name
    pub name: String,
    /// Description
    pub description: String,
    /// Version
    pub version: String,
    /// Author
    pub author: String,
    /// Runbook category
    pub category: RunbookCategory,
    /// Tags
    pub tags: Vec<String>,
    /// Execution steps
    pub steps: Vec<RunbookStep>,
    /// Rollback steps
    pub rollback_steps: Vec<RunbookStep>,
    /// Prerequisites
    pub prerequisites: Vec<Prerequisite>,
    /// Environment requirements
    pub environment_requirements: HashMap<String, String>,
    /// Execution timeout
    pub timeout_minutes: u64,
    /// Requires approval
    pub requires_approval: bool,
    /// Approval roles
    pub approval_roles: Vec<String>,
    /// Enabled
    pub enabled: bool,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Updated timestamp
    pub updated_at: DateTime<Utc>,
}

/// Runbook category
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RunbookCategory {
    /// Disaster recovery
    DisasterRecovery,
    /// System maintenance
    Maintenance,
    /// Incident response
    IncidentResponse,
    /// Deployment
    Deployment,
    /// Monitoring
    Monitoring,
    /// Security
    Security,
    /// Backup
    Backup,
    /// Custom category
    Custom(String),
}

/// Runbook step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunbookStep {
    /// Step ID
    pub id: Uuid,
    /// Step name
    pub name: String,
    /// Description
    pub description: String,
    /// Step type
    pub step_type: StepType,
    /// Execution order
    pub order: u32,
    /// Action to perform
    pub action: StepAction,
    /// Expected outcomes
    pub expected_outcomes: Vec<ExpectedOutcome>,
    /// Validation rules
    pub validation_rules: Vec<ValidationRule>,
    /// Dependencies (step IDs that must complete first)
    pub dependencies: Vec<Uuid>,
    /// Execution mode
    pub execution_mode: ExecutionMode,
    /// Timeout
    pub timeout_seconds: u64,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// On failure behavior
    pub on_failure: FailureAction,
    /// Can be skipped
    pub skippable: bool,
    /// Requires manual approval
    pub requires_approval: bool,
    /// Environment variables
    pub environment: HashMap<String, String>,
    /// Conditional execution
    pub condition: Option<ExecutionCondition>,
}

/// Step type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StepType {
    /// Execute command/script
    Command,
    /// HTTP API call
    ApiCall,
    /// Database query/update
    Database,
    /// File operation
    FileOperation,
    /// Service control
    ServiceControl,
    /// Manual task
    Manual,
    /// Validation/check
    Validation,
    /// Wait/delay
    Wait,
    /// Conditional branch
    Conditional,
    /// Parallel execution
    Parallel,
}

/// Step action definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepAction {
    /// Action type
    pub action_type: ActionType,
    /// Command or script
    pub command: Option<String>,
    /// Arguments
    pub arguments: Vec<String>,
    /// Working directory
    pub working_directory: Option<String>,
    /// HTTP details
    pub http: Option<HttpAction>,
    /// File operation details
    pub file_operation: Option<FileOperation>,
    /// Service details
    pub service: Option<ServiceAction>,
    /// Manual task details
    pub manual: Option<ManualAction>,
}

/// Action type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    /// Shell command
    Shell,
    /// HTTP request
    Http,
    /// File operation
    File,
    /// Service control
    Service,
    /// Manual intervention
    Manual,
    /// SQL query
    Sql,
    /// Custom action
    Custom(String),
}

/// HTTP action details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpAction {
    /// HTTP method
    pub method: String,
    /// URL
    pub url: String,
    /// Headers
    pub headers: HashMap<String, String>,
    /// Request body
    pub body: Option<String>,
    /// Expected status codes
    pub expected_status: Vec<u16>,
}

/// File operation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileOperation {
    /// Operation type
    pub operation: FileOperationType,
    /// Source path
    pub source: String,
    /// Destination path
    pub destination: Option<String>,
    /// File permissions
    pub permissions: Option<String>,
    /// Backup before operation
    pub backup: bool,
}

/// File operation type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileOperationType {
    /// Copy file
    Copy,
    /// Move file
    Move,
    /// Delete file
    Delete,
    /// Create directory
    CreateDir,
    /// Set permissions
    SetPermissions,
    /// Create symlink
    Symlink,
}

/// Service action details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceAction {
    /// Service name
    pub service_name: String,
    /// Action
    pub action: ServiceActionType,
    /// Timeout for action
    pub timeout_seconds: u64,
}

/// Service action type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceActionType {
    /// Start service
    Start,
    /// Stop service
    Stop,
    /// Restart service
    Restart,
    /// Reload configuration
    Reload,
    /// Check status
    Status,
    /// Enable service
    Enable,
    /// Disable service
    Disable,
}

/// Manual action details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualAction {
    /// Instructions for operator
    pub instructions: String,
    /// Expected confirmation
    pub confirmation_required: bool,
    /// Timeout for manual action
    pub timeout_minutes: u64,
    /// Contact information
    pub contact: Option<String>,
}

/// Expected outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutcome {
    /// Outcome type
    pub outcome_type: OutcomeType,
    /// Expected value
    pub expected_value: String,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Description
    pub description: String,
}

/// Outcome type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutcomeType {
    /// Exit code
    ExitCode,
    /// Output contains
    OutputContains,
    /// File exists
    FileExists,
    /// Service running
    ServiceRunning,
    /// HTTP status
    HttpStatus,
    /// Custom validation
    Custom(String),
}

/// Comparison operator
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Equal
    Equals,
    /// Not equal
    NotEquals,
    /// Contains
    Contains,
    /// Does not contain
    NotContains,
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
}

/// Validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule type
    pub rule_type: ValidationType,
    /// Parameters
    pub parameters: HashMap<String, String>,
    /// Validation command
    pub validation_command: Option<String>,
    /// Expected result
    pub expected_result: String,
}

/// Validation type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationType {
    /// Command exit code
    ExitCode,
    /// Output validation
    Output,
    /// File validation
    File,
    /// Network connectivity
    Network,
    /// Service health
    ServiceHealth,
    /// Custom validation
    Custom,
}

/// Execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Sequential execution
    Sequential,
    /// Parallel execution with other parallel steps
    Parallel,
    /// Background execution (fire and forget)
    Background,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum attempts
    pub max_attempts: u32,
    /// Delay between attempts (seconds)
    pub delay_seconds: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Maximum delay between retries
    pub max_delay_seconds: u64,
    /// Retry on specific conditions
    pub retry_conditions: Vec<RetryCondition>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            delay_seconds: 5,
            backoff_multiplier: 2.0,
            max_delay_seconds: 300,
            retry_conditions: vec![
                RetryCondition::NonZeroExitCode,
                RetryCondition::NetworkError,
            ],
        }
    }
}

/// Retry condition
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RetryCondition {
    /// Non-zero exit code
    NonZeroExitCode,
    /// Network error
    NetworkError,
    /// Timeout
    Timeout,
    /// Specific exit code
    ExitCode(i32),
    /// Output contains text
    OutputContains(String),
}

/// Failure action
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FailureAction {
    /// Stop execution
    Stop,
    /// Continue with next step
    Continue,
    /// Skip to specific step
    SkipTo(Uuid),
    /// Start rollback
    Rollback,
    /// Require manual intervention
    ManualIntervention,
}

/// Execution condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Left operand
    pub left_operand: String,
    /// Operator
    pub operator: ComparisonOperator,
    /// Right operand
    pub right_operand: String,
}

/// Condition type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConditionType {
    /// Environment variable
    Environment,
    /// System property
    System,
    /// Previous step result
    StepResult,
    /// Custom condition
    Custom,
}

/// Prerequisite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prerequisite {
    /// Prerequisite name
    pub name: String,
    /// Type
    pub prerequisite_type: PrerequisiteType,
    /// Description
    pub description: String,
    /// Validation command
    pub validation: Option<String>,
    /// Required
    pub required: bool,
}

/// Prerequisite type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrerequisiteType {
    /// Software installed
    Software,
    /// Service available
    Service,
    /// Network connectivity
    Network,
    /// File exists
    File,
    /// Permission
    Permission,
    /// Environment variable
    Environment,
    /// Custom prerequisite
    Custom,
}

/// Runbook execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunbookExecution {
    /// Execution ID
    pub id: Uuid,
    /// Runbook ID
    pub runbook_id: Uuid,
    /// Execution state
    pub state: ExecutionState,
    /// Trigger information
    pub trigger: ExecutionTrigger,
    /// Start timestamp
    pub start_time: DateTime<Utc>,
    /// End timestamp
    pub end_time: Option<DateTime<Utc>>,
    /// Current step
    pub current_step: Option<Uuid>,
    /// Step executions
    pub step_executions: HashMap<Uuid, StepExecution>,
    /// Overall progress (0.0-1.0)
    pub progress: f64,
    /// Execution context
    pub context: ExecutionContext,
    /// Rollback execution ID (if rolled back)
    pub rollback_execution_id: Option<Uuid>,
    /// Approval requests
    pub approval_requests: Vec<ApprovalRequest>,
    /// Execution logs
    pub logs: VecDeque<ExecutionLog>,
}

/// Execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionState {
    /// Waiting for approval
    PendingApproval,
    /// Prerequisites being checked
    CheckingPrerequisites,
    /// Running
    Running,
    /// Waiting for manual intervention
    WaitingForManual,
    /// Completed successfully
    Completed,
    /// Failed
    Failed,
    /// Rolling back
    RollingBack,
    /// Rolled back
    RolledBack,
    /// Cancelled
    Cancelled,
    /// Paused
    Paused,
}

/// Execution trigger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrigger {
    /// Trigger type
    pub trigger_type: TriggerType,
    /// Triggered by user
    pub triggered_by: String,
    /// Trigger reason
    pub reason: String,
    /// Trigger parameters
    pub parameters: HashMap<String, String>,
}

/// Trigger type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TriggerType {
    /// Manual execution
    Manual,
    /// Scheduled execution
    Scheduled,
    /// Event-driven execution
    Event,
    /// API trigger
    Api,
    /// Webhook trigger
    Webhook,
}

/// Step execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepExecution {
    /// Step ID
    pub step_id: Uuid,
    /// Execution state
    pub state: StepExecutionState,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: Option<DateTime<Utc>>,
    /// Duration in seconds
    pub duration_seconds: u64,
    /// Attempt count
    pub attempt_count: u32,
    /// Exit code
    pub exit_code: Option<i32>,
    /// Output
    pub output: String,
    /// Error output
    pub error_output: String,
    /// Validation results
    pub validation_results: Vec<ValidationResult>,
    /// Skip reason
    pub skip_reason: Option<String>,
}

/// Step execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StepExecutionState {
    /// Pending execution
    Pending,
    /// Currently running
    Running,
    /// Completed successfully
    Completed,
    /// Failed
    Failed,
    /// Skipped
    Skipped,
    /// Waiting for approval
    WaitingApproval,
    /// Waiting for manual action
    WaitingManual,
    /// Retrying
    Retrying,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Rule name
    pub rule_name: String,
    /// Passed validation
    pub passed: bool,
    /// Expected value
    pub expected: String,
    /// Actual value
    pub actual: String,
    /// Error message
    pub error_message: Option<String>,
}

/// Execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// Environment variables
    pub environment: HashMap<String, String>,
    /// Working directory
    pub working_directory: String,
    /// User context
    pub user: String,
    /// Execution host
    pub host: String,
    /// Additional parameters
    pub parameters: HashMap<String, String>,
}

/// Approval request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Request ID
    pub id: Uuid,
    /// Step ID requiring approval
    pub step_id: Uuid,
    /// Request message
    pub message: String,
    /// Required roles
    pub required_roles: Vec<String>,
    /// Request status
    pub status: ApprovalStatus,
    /// Requested at
    pub requested_at: DateTime<Utc>,
    /// Approved/rejected by
    pub decided_by: Option<String>,
    /// Decision timestamp
    pub decided_at: Option<DateTime<Utc>>,
    /// Decision reason
    pub decision_reason: Option<String>,
}

/// Approval status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApprovalStatus {
    /// Pending approval
    Pending,
    /// Approved
    Approved,
    /// Rejected
    Rejected,
    /// Expired
    Expired,
}

/// Execution log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLog {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Log level
    pub level: LogLevel,
    /// Message
    pub message: String,
    /// Step ID (if applicable)
    pub step_id: Option<Uuid>,
    /// Context
    pub context: HashMap<String, String>,
}

/// Log level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogLevel {
    /// Debug
    Debug,
    /// Info
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Fatal
    Fatal,
}

/// Runbook executor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunbookExecutorConfig {
    /// Maximum concurrent executions
    pub max_concurrent_executions: usize,
    /// Default timeout for steps (seconds)
    pub default_step_timeout_seconds: u64,
    /// Default timeout for manual steps (minutes)
    pub default_manual_timeout_minutes: u64,
    /// Enable execution history
    pub execution_history_enabled: bool,
    /// History retention days
    pub history_retention_days: u32,
    /// Log retention limit
    pub log_retention_limit: usize,
    /// Enable step validation
    pub step_validation_enabled: bool,
    /// Default working directory
    pub default_working_directory: String,
}

impl Default for RunbookExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_executions: 10,
            default_step_timeout_seconds: 300,
            default_manual_timeout_minutes: 60,
            execution_history_enabled: true,
            history_retention_days: 90,
            log_retention_limit: 10000,
            step_validation_enabled: true,
            default_working_directory: "/tmp".to_string(),
        }
    }
}

/// Runbook executor metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunbookExecutorMetrics {
    /// Total runbooks
    pub total_runbooks: usize,
    /// Active executions
    pub active_executions: usize,
    /// Total executions
    pub total_executions: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Average execution time (seconds)
    pub avg_execution_time_seconds: f64,
    /// Rollback executions
    pub rollback_executions: u64,
    /// Manual interventions
    pub manual_interventions: u64,
    /// Success rate
    pub success_rate: f64,
}
