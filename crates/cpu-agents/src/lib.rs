//! CPU Agents Crate
//!
//! Provides CPU-based agents for I/O operations, orchestration, and external services.
//! This crate has NO GPU dependencies and communicates with GPU agents via message passing.

pub mod agent;
pub mod bridge;
pub mod io_manager;
pub mod orchestrator;

#[cfg(test)]
mod tests;

pub use agent::{
    AgentCapability, AgentMetrics, AgentStatus, AgentTask, CpuAgent, CpuAgentConfig,
    TaskExecutionResult, TaskType,
};
pub use bridge::{BridgeConfig, CpuGpuBridge, CpuGpuMessage, MessageType};
pub use io_manager::{IoConfig, IoManager, IoOperation, IoResult};
pub use orchestrator::{
    Orchestrator, OrchestratorConfig, TaskResult, Workflow, WorkflowResult, WorkflowTask,
};

/// CPU agent error types
#[derive(Debug, thiserror::Error)]
pub enum CpuAgentError {
    #[error("I/O operation failed: {0}")]
    IoError(String),

    #[error("Message passing failed: {0}")]
    MessageError(String),

    #[error("Task execution failed: {0}")]
    TaskError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Bridge communication failed: {0}")]
    BridgeError(String),
}

/// Result type for CPU agent operations
pub type Result<T> = std::result::Result<T, CpuAgentError>;
