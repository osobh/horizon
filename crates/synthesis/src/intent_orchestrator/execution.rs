//! Execution planning and management

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::{AgentId, BackoffStrategy, SuccessCriterion};

/// Execution record for tracking intent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    /// Execution ID
    pub id: String,
    /// Intent ID
    pub intent_id: String,
    /// Execution status
    pub status: ExecutionStatus,
    /// Action plan executed
    pub action_plan: ActionPlan,
    /// Start time
    pub started_at: DateTime<Utc>,
    /// End time
    pub ended_at: Option<DateTime<Utc>>,
    /// Execution duration
    pub duration: Option<Duration>,
    /// Execution result
    pub result: Option<ExecutionResult>,
    /// Execution metadata
    pub metadata: HashMap<String, String>,
}

/// Action plan for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPlan {
    /// Plan ID
    pub id: String,
    /// Action steps
    pub steps: Vec<ActionStep>,
    /// Resource requirements
    pub resources: ResourceRequirements,
    /// Retry policy
    pub retry_policy: RetryPolicy,
    /// Success criteria
    pub success_criteria: Vec<SuccessCriterion>,
    /// Plan metadata
    pub metadata: HashMap<String, String>,
}

/// Execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// Pending execution
    Pending,
    /// Currently running
    Running,
    /// Successfully completed
    Completed,
    /// Failed execution
    Failed,
    /// Cancelled by user
    Cancelled,
    /// Retrying after failure
    Retrying,
    /// Partially completed
    PartialSuccess,
    /// Timed out
    TimedOut,
}

/// Action step in execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionStep {
    /// Step ID
    pub id: String,
    /// Step name
    pub name: String,
    /// Action type
    pub action_type: ActionType,
    /// Step parameters
    pub parameters: HashMap<String, String>,
    /// Dependencies on other steps
    pub dependencies: Vec<String>,
    /// Timeout for step
    pub timeout: Duration,
    /// Agent responsible for execution
    pub agent_id: Option<AgentId>,
    /// Step status
    pub status: ExecutionStatus,
}

/// Action types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    /// Deploy service
    Deploy,
    /// Scale resources
    Scale,
    /// Configure settings
    Configure,
    /// Execute query
    Query,
    /// Monitor metrics
    Monitor,
    /// Optimize resources
    Optimize,
    /// Run diagnostic
    Diagnostic,
    /// Migrate data
    Migrate,
    /// Backup data
    Backup,
    /// Restore from backup
    Restore,
    /// Wait for condition
    Wait,
    /// Custom action
    Custom(String),
}

/// Retry policy for failed executions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Backoff strategy
    pub backoff_strategy: BackoffStrategy,
    /// Initial delay
    pub initial_delay: Duration,
    /// Maximum delay
    pub max_delay: Duration,
    /// Retry conditions
    pub retry_conditions: Vec<RetryCondition>,
}

/// Conditions for retry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryCondition {
    /// Error pattern to match
    pub error_pattern: String,
    /// Should retry on this error
    pub should_retry: bool,
    /// Optional custom delay for this error
    pub custom_delay: Option<Duration>,
}

/// Resource requirements for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU cores required
    pub cpu_cores: f32,
    /// Memory in GB
    pub memory_gb: f32,
    /// Storage in GB
    pub storage_gb: f32,
    /// Network bandwidth in Mbps
    pub bandwidth_mbps: Option<f32>,
    /// Estimated cost
    pub estimated_cost: Option<f32>,
}

/// Execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Success flag
    pub success: bool,
    /// Result data
    pub data: HashMap<String, serde_json::Value>,
    /// Error message if failed
    pub error: Option<String>,
    /// Metrics collected
    pub metrics: HashMap<String, f64>,
    /// Artifacts produced
    pub artifacts: Vec<String>,
}

impl ExecutionRecord {
    /// Create new execution record
    pub fn new(intent_id: String, action_plan: ActionPlan) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            intent_id,
            status: ExecutionStatus::Pending,
            action_plan,
            started_at: Utc::now(),
            ended_at: None,
            duration: None,
            result: None,
            metadata: HashMap::new(),
        }
    }

    /// Start execution
    pub fn start(&mut self) {
        self.status = ExecutionStatus::Running;
        self.started_at = Utc::now();
    }

    /// Complete execution
    pub fn complete(&mut self, result: ExecutionResult) {
        self.status = if result.success {
            ExecutionStatus::Completed
        } else {
            ExecutionStatus::Failed
        };
        self.result = Some(result);
        self.ended_at = Some(Utc::now());
        if let Some(ended) = self.ended_at {
            self.duration = Some(ended - self.started_at);
        }
    }

    /// Cancel execution
    pub fn cancel(&mut self) {
        self.status = ExecutionStatus::Cancelled;
        self.ended_at = Some(Utc::now());
    }

    /// Get execution duration in seconds
    pub fn duration_seconds(&self) -> Option<i64> {
        self.duration.map(|d| d.num_seconds())
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff_strategy: BackoffStrategy::Exponential,
            initial_delay: Duration::seconds(1),
            max_delay: Duration::seconds(60),
            retry_conditions: Vec::new(),
        }
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 1.0,
            memory_gb: 1.0,
            storage_gb: 10.0,
            bandwidth_mbps: None,
            estimated_cost: None,
        }
    }
}