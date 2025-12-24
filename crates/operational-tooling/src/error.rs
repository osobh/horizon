//! Error types for operational tooling

use thiserror::Error;

/// Operational tooling errors
#[derive(Error, Debug)]
pub enum OperationalError {
    /// Deployment operation failed
    #[error("Deployment failed: {0}")]
    DeploymentFailed(String),

    /// Rollback operation failed
    #[error("Rollback failed: {0}")]
    RollbackFailed(String),

    /// Canary testing failed
    #[error("Canary test failed: {0}")]
    CanaryTestFailed(String),

    /// Monitoring setup failed
    #[error("Monitoring setup failed: {0}")]
    MonitoringFailed(String),

    /// Resource allocation failed
    #[error("Resource allocation failed: resource {resource}, reason: {reason}")]
    ResourceAllocationFailed { resource: String, reason: String },

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Network communication error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Agent operation failed
    #[error("Agent operation failed: {agent_id}, operation: {operation}, reason: {reason}")]
    AgentOperationFailed {
        agent_id: String,
        operation: String,
        reason: String,
    },

    /// GPU operation failed
    #[error("GPU operation failed: {device_id}, operation: {operation}")]
    GpuOperationFailed { device_id: u32, operation: String },

    /// Timeout error
    #[error("Operation timed out after {duration:?}")]
    Timeout { duration: std::time::Duration },

    /// Runtime error
    #[error("Runtime error: {0}")]
    RuntimeError(#[from] stratoswarm_runtime::RuntimeError),

    /// Agent core error
    #[error("Agent core error: {0}")]
    AgentCoreError(#[from] stratoswarm_agent_core::AgentError),

    /// Monitoring error
    #[error("Monitoring error: {0}")]
    MonitoringError(#[from] stratoswarm_monitoring::MonitoringError),

    /// CUDA error
    #[error("CUDA error: {0}")]
    CudaError(#[from] stratoswarm_cuda::CudaError),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Result type for operational operations
pub type OperationalResult<T> = Result<T, OperationalError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_error_display() {
        let error = OperationalError::DeploymentFailed("test deployment".to_string());
        assert!(error.to_string().contains("Deployment failed"));
        assert!(error.to_string().contains("test deployment"));
    }

    #[test]
    fn test_timeout_error() {
        let error = OperationalError::Timeout {
            duration: Duration::from_secs(30),
        };
        assert!(error.to_string().contains("timed out"));
        assert!(error.to_string().contains("30s"));
    }

    #[test]
    fn test_resource_allocation_error() {
        let error = OperationalError::ResourceAllocationFailed {
            resource: "GPU-0".to_string(),
            reason: "insufficient memory".to_string(),
        };
        assert!(error.to_string().contains("Resource allocation failed"));
        assert!(error.to_string().contains("GPU-0"));
        assert!(error.to_string().contains("insufficient memory"));
    }

    #[test]
    fn test_agent_operation_error() {
        let error = OperationalError::AgentOperationFailed {
            agent_id: "agent-123".to_string(),
            operation: "deploy".to_string(),
            reason: "validation failed".to_string(),
        };
        assert!(error.to_string().contains("Agent operation failed"));
        assert!(error.to_string().contains("agent-123"));
        assert!(error.to_string().contains("deploy"));
        assert!(error.to_string().contains("validation failed"));
    }

    #[test]
    fn test_gpu_operation_error() {
        let error = OperationalError::GpuOperationFailed {
            device_id: 0,
            operation: "memory allocation".to_string(),
        };
        assert!(error.to_string().contains("GPU operation failed"));
        assert!(error.to_string().contains("0"));
        assert!(error.to_string().contains("memory allocation"));
    }

    #[test]
    fn test_error_from_io() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let operational_error = OperationalError::from(io_error);
        assert!(matches!(operational_error, OperationalError::IoError(_)));
    }

    #[test]
    fn test_error_from_serde() {
        let json_error = serde_json::from_str::<i32>("invalid json").unwrap_err();
        let operational_error = OperationalError::from(json_error);
        assert!(matches!(
            operational_error,
            OperationalError::SerializationError(_)
        ));
    }

    #[test]
    fn test_result_type() {
        let success: OperationalResult<String> = Ok("success".to_string());
        assert!(success.is_ok());

        let failure: OperationalResult<String> =
            Err(OperationalError::DeploymentFailed("test".to_string()));
        assert!(failure.is_err());
    }

    #[test]
    fn test_all_error_variants() {
        let errors = vec![
            OperationalError::DeploymentFailed("deploy fail".to_string()),
            OperationalError::RollbackFailed("rollback fail".to_string()),
            OperationalError::CanaryTestFailed("canary fail".to_string()),
            OperationalError::MonitoringFailed("monitoring fail".to_string()),
            OperationalError::ConfigurationError("config error".to_string()),
            OperationalError::NetworkError("network error".to_string()),
        ];

        for error in errors {
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn test_error_debug_formatting() {
        let error = OperationalError::DeploymentFailed("debug test".to_string());
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("DeploymentFailed"));
        assert!(debug_str.contains("debug test"));
    }

    #[test]
    fn test_configuration_error() {
        let error = OperationalError::ConfigurationError(
            "invalid config: missing field 'timeout'".to_string(),
        );
        assert!(error.to_string().contains("Configuration error"));
        assert!(error.to_string().contains("missing field 'timeout'"));
    }

    #[test]
    fn test_network_error() {
        let error =
            OperationalError::NetworkError("connection refused to 127.0.0.1:8080".to_string());
        assert!(error.to_string().contains("Network error"));
        assert!(error.to_string().contains("connection refused"));
    }

    #[test]
    fn test_error_empty_strings() {
        let errors = vec![
            OperationalError::DeploymentFailed(String::new()),
            OperationalError::RollbackFailed(String::new()),
            OperationalError::CanaryTestFailed(String::new()),
            OperationalError::MonitoringFailed(String::new()),
            OperationalError::ConfigurationError(String::new()),
            OperationalError::NetworkError(String::new()),
            OperationalError::ResourceAllocationFailed {
                resource: String::new(),
                reason: String::new(),
            },
            OperationalError::AgentOperationFailed {
                agent_id: String::new(),
                operation: String::new(),
                reason: String::new(),
            },
            OperationalError::GpuOperationFailed {
                device_id: 0,
                operation: String::new(),
            },
        ];

        // All should handle empty strings gracefully
        for error in errors {
            let _ = error.to_string();
        }
    }

    #[test]
    fn test_error_unicode_strings() {
        let error = OperationalError::DeploymentFailed("部署失败 🚫".to_string());
        assert!(error.to_string().contains("部署失败 🚫"));

        let error = OperationalError::ConfigurationError("設定エラー: 無効な値".to_string());
        assert!(error.to_string().contains("設定エラー: 無効な値"));
    }

    #[test]
    fn test_error_very_long_strings() {
        let long_string = "x".repeat(1000);
        let error = OperationalError::DeploymentFailed(long_string.clone());
        assert!(error.to_string().contains(&long_string));
    }

    #[test]
    fn test_timeout_various_durations() {
        let durations = vec![
            Duration::from_secs(0),
            Duration::from_millis(100),
            Duration::from_secs(60),
            Duration::from_secs(3600),
            Duration::from_secs(86400),
        ];

        for duration in durations {
            let error = OperationalError::Timeout { duration };
            assert!(error.to_string().contains("timed out"));
        }
    }

    #[test]
    fn test_gpu_device_ids() {
        let device_ids = vec![0, 1, 7, u32::MAX];

        for device_id in device_ids {
            let error = OperationalError::GpuOperationFailed {
                device_id,
                operation: "test".to_string(),
            };
            assert!(error.to_string().contains(&device_id.to_string()));
        }
    }

    #[test]
    fn test_error_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<OperationalError>();
        assert_sync::<OperationalError>();
    }

    #[test]
    fn test_io_error_kinds() {
        use std::io::ErrorKind;

        let error_kinds = vec![
            ErrorKind::NotFound,
            ErrorKind::PermissionDenied,
            ErrorKind::ConnectionRefused,
            ErrorKind::BrokenPipe,
            ErrorKind::AlreadyExists,
            ErrorKind::InvalidData,
        ];

        for kind in error_kinds {
            let io_error = std::io::Error::new(kind, "test error");
            let op_error: OperationalError = io_error.into();
            assert!(matches!(op_error, OperationalError::IoError(_)));
        }
    }

    #[test]
    fn test_result_map_operations() {
        let result: OperationalResult<i32> = Ok(42);
        let mapped = result.map(|v| v * 2);
        assert_eq!(mapped.unwrap(), 84);

        let error_result: OperationalResult<i32> =
            Err(OperationalError::ConfigurationError("test".to_string()));
        let mapped_err = error_result.map(|v| v * 2);
        assert!(mapped_err.is_err());
    }

    #[test]
    fn test_error_chaining() {
        fn might_fail_io() -> Result<(), std::io::Error> {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "not found",
            ))
        }

        fn might_fail_operational() -> OperationalResult<()> {
            might_fail_io()?;
            Ok(())
        }

        let result = might_fail_operational();
        assert!(result.is_err());
        match result.unwrap_err() {
            OperationalError::IoError(e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
            }
            _ => panic!("Expected IoError variant"),
        }
    }
}
