//! Fault tolerance error types

use thiserror::Error;

/// Fault tolerance system errors
#[derive(Error, Debug)]
pub enum FaultToleranceError {
    #[error("Checkpoint creation failed: {0}")]
    CheckpointFailed(String),

    #[error("Recovery failed: {0}")]
    RecoveryFailed(String),

    #[error("Coordination error: {0}")]
    CoordinationError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("GPU memory error: {0}")]
    GpuMemoryError(String),

    #[error("System health check failed: {0}")]
    HealthCheckFailed(String),

    #[error("Checkpoint not found: {0}")]
    CheckpointNotFound(String),

    #[error("Invalid checkpoint format: {0}")]
    InvalidCheckpoint(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    BincodeError(#[from] bincode::Error),
}

/// System health status
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Failed,
}

/// Result type for fault tolerance operations
pub type FtResult<T> = Result<T, FaultToleranceError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = FaultToleranceError::CheckpointFailed("test".to_string());
        assert!(error.to_string().contains("Checkpoint creation failed"));
    }

    #[test]
    fn test_health_status_equality() {
        assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
        assert_ne!(HealthStatus::Healthy, HealthStatus::Failed);
    }

    #[test]
    fn test_error_from_io() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let ft_error = FaultToleranceError::from(io_error);
        assert!(matches!(ft_error, FaultToleranceError::IoError(_)));
    }
}
