use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum AgentError {
    #[error("Agent initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Agent not initialized")]
    NotInitialized,

    #[error("Agent already initialized")]
    AlreadyInitialized,

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Safety threshold exceeded: {0}")]
    SafetyThresholdExceeded(String),

    #[error("Approval required for action: {0}")]
    ApprovalRequired(String),

    #[error("Rollback failed: {0}")]
    RollbackFailed(String),

    #[error("Memory operation failed: {0}")]
    MemoryError(String),

    #[error("Health check failed: {0}")]
    HealthCheckFailed(String),

    #[error("Invalid autonomy level transition from {from:?} to {to:?}")]
    InvalidAutonomyTransition { from: String, to: String },

    #[error("Operation not allowed at autonomy level {level:?}")]
    OperationNotAllowed { level: String },

    #[error("Retry limit exceeded: {0}")]
    RetryLimitExceeded(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Agent error: {0}")]
    Other(String),
}

impl From<std::io::Error> for AgentError {
    #[cold]
    fn from(err: std::io::Error) -> Self {
        AgentError::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for AgentError {
    #[cold]
    fn from(err: serde_json::Error) -> Self {
        AgentError::SerializationError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, AgentError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AgentError::InitializationFailed("test".to_string());
        assert_eq!(err.to_string(), "Agent initialization failed: test");
    }

    #[test]
    fn test_invalid_autonomy_transition() {
        let err = AgentError::InvalidAutonomyTransition {
            from: "Low".to_string(),
            to: "Full".to_string(),
        };
        assert!(err.to_string().contains("Low"));
        assert!(err.to_string().contains("Full"));
    }

    #[test]
    fn test_operation_not_allowed() {
        let err = AgentError::OperationNotAllowed {
            level: "ReadOnly".to_string(),
        };
        assert!(err.to_string().contains("ReadOnly"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let agent_err: AgentError = io_err.into();
        assert!(matches!(agent_err, AgentError::IoError(_)));
    }

    #[test]
    fn test_serde_error_conversion() {
        let json_err = serde_json::from_str::<u32>("invalid").unwrap_err();
        let agent_err: AgentError = json_err.into();
        assert!(matches!(agent_err, AgentError::SerializationError(_)));
    }
}
