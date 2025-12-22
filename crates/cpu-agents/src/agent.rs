//! CPU Agent trait and implementations
//!
//! Provides I/O focused agents with NO GPU dependencies

use crate::{CpuAgentError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Agent capabilities that define what tasks an agent can handle
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentCapability {
    /// File I/O operations
    FileIo,
    /// Network I/O operations
    NetworkIo,
    /// Database operations
    DatabaseIo,
    /// Workflow orchestration
    Orchestration,
    /// API management
    ApiManagement,
    /// Data transformation
    DataTransform,
    /// Monitoring and metrics
    Monitoring,
    // Note: GpuCompute explicitly excluded - CPU agents cannot use GPU
}

/// Agent status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    Created,
    Initializing,
    Ready,
    Busy,
    Error,
    Stopped,
}

/// Task types that CPU agents can execute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    /// Read file from filesystem
    FileRead(String),
    /// Write file to filesystem
    FileWrite(String),
    /// Delete file from filesystem
    FileDelete(String),
    /// Create directory
    CreateDirectory(String),
    /// List directory contents
    ListDirectory(String),
    /// Fetch data from network
    NetworkFetch(String),
    /// Send network request
    NetworkSend(String),
    /// Query database
    DatabaseQuery(String),
    /// Transform data
    Transform(String),
    /// Aggregate data
    Aggregate,
    /// Compute operation (CPU only)
    Compute(String),
    /// GPU compute request (forwarded to GPU agents)
    GpuCompute(String),
}

/// Individual task for agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: String,
    pub task_type: TaskType,
    pub priority: u8, // 1-10, 10 being highest
}

/// Result of task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskExecutionResult {
    Success(String),
    Failed(String),
}

/// Agent execution metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub total_task_duration_ms: f64,
    pub average_task_duration_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

/// Configuration for CPU agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuAgentConfig {
    pub id: String,
    pub capabilities: Vec<AgentCapability>,
    pub max_concurrent_tasks: usize,
    pub memory_limit_mb: usize,
}

impl Default for CpuAgentConfig {
    fn default() -> Self {
        Self {
            id: "cpu-agent-default".to_string(),
            capabilities: vec![AgentCapability::FileIo],
            max_concurrent_tasks: 10,
            memory_limit_mb: 100,
        }
    }
}

/// Main trait for CPU agents - NO GPU dependencies allowed
#[async_trait]
pub trait CpuAgent: Send + Sync {
    /// Get agent ID
    fn id(&self) -> &str;

    /// Get agent capabilities
    fn capabilities(&self) -> &[AgentCapability];

    /// Check if agent can handle a specific capability
    fn can_handle(&self, capability: &AgentCapability) -> bool;

    /// Get current agent status
    fn status(&self) -> AgentStatus;

    /// Initialize the agent
    async fn initialize(&mut self) -> Result<()>;

    /// Execute a single task
    async fn execute_task(&mut self, task: AgentTask) -> Result<TaskExecutionResult>;

    /// Submit task for async execution
    async fn submit_task(&self, task: AgentTask) -> Result<()>;

    /// Shutdown the agent
    async fn shutdown(&mut self) -> Result<()>;

    /// Get agent metrics
    fn get_metrics(&self) -> AgentMetrics;
}

/// Basic CPU agent implementation
pub struct BasicCpuAgent {
    config: CpuAgentConfig,
    status: AgentStatus,
    metrics: AgentMetrics,
    active_tasks: Vec<String>,
    start_time: Option<Instant>,
}

impl BasicCpuAgent {
    /// Create new basic CPU agent
    pub fn new(config: CpuAgentConfig) -> Self {
        Self {
            config,
            status: AgentStatus::Created,
            metrics: AgentMetrics::default(),
            active_tasks: Vec::new(),
            start_time: None,
        }
    }

    /// Check if agent is at capacity
    fn is_at_capacity(&self) -> bool {
        self.active_tasks.len() >= self.config.max_concurrent_tasks
    }

    /// Validate task before execution
    fn validate_task(&self, task: &AgentTask) -> Result<()> {
        // Check if we can handle this task type
        let required_capability = match &task.task_type {
            TaskType::FileRead(_)
            | TaskType::FileWrite(_)
            | TaskType::FileDelete(_)
            | TaskType::CreateDirectory(_)
            | TaskType::ListDirectory(_) => AgentCapability::FileIo,
            TaskType::NetworkFetch(_) | TaskType::NetworkSend(_) => AgentCapability::NetworkIo,
            TaskType::DatabaseQuery(_) => AgentCapability::DatabaseIo,
            TaskType::Transform(_) | TaskType::Aggregate => AgentCapability::DataTransform,
            TaskType::Compute(_) => {
                // CPU compute is always allowed
                return Ok(());
            }
            TaskType::GpuCompute(_) => {
                return Err(CpuAgentError::TaskError(
                    "CPU agents cannot execute GPU compute tasks".to_string(),
                ));
            }
        };

        if !self.can_handle(&required_capability) {
            return Err(CpuAgentError::TaskError(format!(
                "Agent does not have required capability: {:?}",
                required_capability
            )));
        }

        Ok(())
    }

    /// Execute task implementation
    async fn execute_task_impl(&mut self, task: &AgentTask) -> Result<String> {
        match &task.task_type {
            TaskType::FileRead(path) => match tokio::fs::read_to_string(path).await {
                Ok(content) => Ok(content),
                Err(e) => Err(CpuAgentError::IoError(format!(
                    "Failed to read file: {}",
                    e
                ))),
            },
            TaskType::FileWrite(path) => {
                // For demo, write a timestamp
                let content = format!("Written by {} at {}", self.id(), chrono::Utc::now());
                match tokio::fs::write(path, content).await {
                    Ok(_) => Ok("File written successfully".to_string()),
                    Err(e) => Err(CpuAgentError::IoError(format!(
                        "Failed to write file: {}",
                        e
                    ))),
                }
            }
            TaskType::FileDelete(path) => match tokio::fs::remove_file(path).await {
                Ok(_) => Ok("File deleted successfully".to_string()),
                Err(e) => Err(CpuAgentError::IoError(format!(
                    "Failed to delete file: {}",
                    e
                ))),
            },
            TaskType::CreateDirectory(path) => match tokio::fs::create_dir_all(path).await {
                Ok(_) => Ok("Directory created successfully".to_string()),
                Err(e) => Err(CpuAgentError::IoError(format!(
                    "Failed to create directory: {}",
                    e
                ))),
            },
            TaskType::ListDirectory(path) => match tokio::fs::read_dir(path).await {
                Ok(mut entries) => {
                    let mut files = Vec::new();
                    while let Some(entry) = entries.next_entry().await.map_err(|e| {
                        CpuAgentError::IoError(format!("Failed to read directory entry: {}", e))
                    })? {
                        files.push(entry.file_name().to_string_lossy().to_string());
                    }
                    Ok(serde_json::to_string(&files).unwrap())
                }
                Err(e) => Err(CpuAgentError::IoError(format!(
                    "Failed to list directory: {}",
                    e
                ))),
            },
            TaskType::NetworkFetch(url) => {
                // Simulate network fetch
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                Ok(format!("Fetched data from {}", url))
            }
            TaskType::NetworkSend(data) => {
                // Simulate network send
                tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;
                Ok(format!("Sent {} bytes", data.len()))
            }
            TaskType::DatabaseQuery(query) => {
                // Simulate database query
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                Ok(format!("Query result for: {}", query))
            }
            TaskType::Transform(operation) => {
                // Simulate data transformation
                tokio::time::sleep(tokio::time::Duration::from_millis(75)).await;
                Ok(format!("Transformed data using: {}", operation))
            }
            TaskType::Aggregate => {
                // Simulate data aggregation
                tokio::time::sleep(tokio::time::Duration::from_millis(120)).await;
                Ok("Aggregated data successfully".to_string())
            }
            TaskType::Compute(operation) => {
                // Simulate CPU computation
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
                Ok(format!("Computed: {}", operation))
            }
            TaskType::GpuCompute(_) => Err(CpuAgentError::TaskError(
                "CPU agents cannot execute GPU compute tasks".to_string(),
            )),
        }
    }

    /// Update metrics after task completion
    fn update_metrics(&mut self, duration_ms: f64, success: bool) {
        if success {
            self.metrics.tasks_completed += 1;
        } else {
            self.metrics.tasks_failed += 1;
        }

        self.metrics.total_task_duration_ms += duration_ms;
        let total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed;
        if total_tasks > 0 {
            self.metrics.average_task_duration_ms =
                self.metrics.total_task_duration_ms / total_tasks as f64;
        }

        // Simulate memory and CPU usage
        self.metrics.memory_usage_mb = 50.0 + (self.active_tasks.len() as f64 * 5.0);
        self.metrics.cpu_usage_percent = if self.active_tasks.is_empty() {
            5.0
        } else {
            20.0 + (self.active_tasks.len() as f64 * 10.0)
        };
    }
}

#[async_trait]
impl CpuAgent for BasicCpuAgent {
    fn id(&self) -> &str {
        &self.config.id
    }

    fn capabilities(&self) -> &[AgentCapability] {
        &self.config.capabilities
    }

    fn can_handle(&self, capability: &AgentCapability) -> bool {
        self.config.capabilities.contains(capability)
    }

    fn status(&self) -> AgentStatus {
        self.status
    }

    async fn initialize(&mut self) -> Result<()> {
        self.status = AgentStatus::Initializing;

        // Simulate initialization work
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        self.status = AgentStatus::Ready;
        self.start_time = Some(Instant::now());

        log::info!(
            "CPU Agent {} initialized with capabilities: {:?}",
            self.id(),
            self.capabilities()
        );

        Ok(())
    }

    async fn execute_task(&mut self, task: AgentTask) -> Result<TaskExecutionResult> {
        self.validate_task(&task)?;

        if self.status != AgentStatus::Ready {
            return Err(CpuAgentError::TaskError(
                "Agent is not ready to execute tasks".to_string(),
            ));
        }

        self.status = AgentStatus::Busy;
        self.active_tasks.push(task.id.clone());

        let start = Instant::now();
        let result = self.execute_task_impl(&task).await;
        let duration = start.elapsed();

        // Remove task from active list
        self.active_tasks.retain(|id| id != &task.id);

        if self.active_tasks.is_empty() {
            self.status = AgentStatus::Ready;
        }

        let duration_ms = duration.as_millis() as f64;
        let task_result = match result {
            Ok(output) => {
                self.update_metrics(duration_ms, true);
                TaskExecutionResult::Success(output)
            }
            Err(e) => {
                self.update_metrics(duration_ms, false);
                TaskExecutionResult::Failed(e.to_string())
            }
        };

        log::debug!(
            "Task {} completed in {:.2}ms: {:?}",
            task.id,
            duration_ms,
            task_result
        );

        Ok(task_result)
    }

    async fn submit_task(&self, task: AgentTask) -> Result<()> {
        if self.is_at_capacity() {
            return Err(CpuAgentError::TaskError(
                "Agent is at maximum capacity".to_string(),
            ));
        }

        self.validate_task(&task)?;

        // In a real implementation, this would add to a task queue
        // For now, just validate that we can accept the task
        log::debug!("Task {} submitted to agent {}", task.id, self.id());

        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        log::info!("Shutting down CPU Agent {}", self.id());

        // Wait for active tasks to complete
        while !self.active_tasks.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        self.status = AgentStatus::Stopped;

        log::info!(
            "CPU Agent {} shutdown complete. Metrics: {:?}",
            self.id(),
            self.metrics
        );

        Ok(())
    }

    fn get_metrics(&self) -> AgentMetrics {
        self.metrics.clone()
    }
}
