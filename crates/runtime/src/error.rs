//! Runtime error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("Container not found: {id}")]
    ContainerNotFound { id: String },

    #[error("Container already exists: {id}")]
    ContainerAlreadyExists { id: String },

    #[error("Invalid container configuration: {reason}")]
    InvalidConfig { reason: String },

    #[error("Container startup failed: {reason}")]
    StartupFailed { reason: String },

    #[error("Container shutdown failed: {reason}")]
    ShutdownFailed { reason: String },

    // #[error("GPU context error: {0}")]
    // GpuContext(#[from] cust::error::CudaError),
    #[error("Memory allocation error: {0}")]
    Memory(#[from] stratoswarm_memory::MemoryError),

    #[error("Container state transition error: from {from} to {to}")]
    InvalidStateTransition { from: String, to: String },

    #[error("Resource limit exceeded: {resource} ({current}/{limit})")]
    ResourceLimitExceeded {
        resource: String,
        current: usize,
        limit: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_not_found_error() {
        let error = RuntimeError::ContainerNotFound {
            id: "container-123".to_string(),
        };
        assert_eq!(error.to_string(), "Container not found: container-123");
    }

    #[test]
    fn test_container_already_exists_error() {
        let error = RuntimeError::ContainerAlreadyExists {
            id: "container-456".to_string(),
        };
        assert_eq!(error.to_string(), "Container already exists: container-456");
    }

    #[test]
    fn test_invalid_config_error() {
        let error = RuntimeError::InvalidConfig {
            reason: "memory limit too low".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid container configuration: memory limit too low"
        );
    }

    #[test]
    fn test_startup_failed_error() {
        let error = RuntimeError::StartupFailed {
            reason: "GPU initialization failed".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Container startup failed: GPU initialization failed"
        );
    }

    #[test]
    fn test_shutdown_failed_error() {
        let error = RuntimeError::ShutdownFailed {
            reason: "process still running".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Container shutdown failed: process still running"
        );
    }

    #[test]
    fn test_invalid_state_transition_error() {
        let error = RuntimeError::InvalidStateTransition {
            from: "Running".to_string(),
            to: "Created".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Container state transition error: from Running to Created"
        );
    }

    #[test]
    fn test_resource_limit_exceeded_error() {
        let error = RuntimeError::ResourceLimitExceeded {
            resource: "GPU memory".to_string(),
            current: 2048,
            limit: 1024,
        };
        assert_eq!(
            error.to_string(),
            "Resource limit exceeded: GPU memory (2048/1024)"
        );
    }

    #[test]
    fn test_memory_error_conversion() {
        let mem_error = stratoswarm_memory::MemoryError::AllocationFailed {
            reason: "out of memory".to_string(),
        };
        let error: RuntimeError = mem_error.into();
        assert!(error.to_string().contains("Memory allocation error"));
    }

    #[test]
    fn test_error_debug_formatting() {
        let error = RuntimeError::ContainerNotFound {
            id: "debug-test".to_string(),
        };
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("ContainerNotFound"));
        assert!(debug_str.contains("debug-test"));
    }

    #[test]
    fn test_error_source_chain() {
        let mem_error = stratoswarm_memory::MemoryError::AllocationFailed {
            reason: "insufficient memory".to_string(),
        };
        let error = RuntimeError::Memory(mem_error);

        // Test that error source chain works
        let source = std::error::Error::source(&error);
        assert!(source.is_some());
    }

    #[test]
    fn test_error_empty_strings() {
        let errors = vec![
            RuntimeError::ContainerNotFound { id: String::new() },
            RuntimeError::ContainerAlreadyExists { id: String::new() },
            RuntimeError::InvalidConfig {
                reason: String::new(),
            },
            RuntimeError::StartupFailed {
                reason: String::new(),
            },
            RuntimeError::ShutdownFailed {
                reason: String::new(),
            },
        ];

        for error in errors {
            // Should handle empty strings gracefully
            let _ = error.to_string();
        }
    }

    #[test]
    fn test_error_unicode_strings() {
        let error = RuntimeError::ContainerNotFound {
            id: "容器-コンテナ-🚀".to_string(),
        };
        assert_eq!(error.to_string(), "Container not found: 容器-コンテナ-🚀");

        let error = RuntimeError::InvalidConfig {
            reason: "配置错误: 内存不足".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid container configuration: 配置错误: 内存不足"
        );
    }

    #[test]
    fn test_error_long_strings() {
        let long_id = "c".repeat(1000);
        let error = RuntimeError::ContainerNotFound {
            id: long_id.clone(),
        };
        assert!(error.to_string().contains(&long_id));

        let long_reason = "x".repeat(1000);
        let error = RuntimeError::InvalidConfig {
            reason: long_reason.clone(),
        };
        assert!(error.to_string().contains(&long_reason));
    }

    #[test]
    fn test_resource_limit_exceeded_variations() {
        let test_cases = vec![
            ("CPU cores", 16, 8),
            ("Memory", usize::MAX, usize::MAX - 1),
            ("GPU compute units", 0, 0),
            ("Network bandwidth", 1000000, 500000),
        ];

        for (resource, current, limit) in test_cases {
            let error = RuntimeError::ResourceLimitExceeded {
                resource: resource.to_string(),
                current,
                limit,
            };
            let error_str = error.to_string();
            assert!(error_str.contains(resource));
            assert!(error_str.contains(&current.to_string()));
            assert!(error_str.contains(&limit.to_string()));
        }
    }

    #[test]
    fn test_state_transition_variations() {
        let states = vec![
            ("Created", "Running"),
            ("Running", "Stopped"),
            ("Stopped", "Removed"),
            ("Running", "Failed"),
            ("Failed", "Removed"),
        ];

        for (from, to) in states {
            let error = RuntimeError::InvalidStateTransition {
                from: from.to_string(),
                to: to.to_string(),
            };
            let error_str = error.to_string();
            assert!(error_str.contains(from));
            assert!(error_str.contains(to));
        }
    }

    #[test]
    fn test_error_downcasting() {
        let error = RuntimeError::ContainerNotFound {
            id: "test-123".to_string(),
        };

        // Convert to Box<dyn Error>
        let boxed_error: Box<dyn std::error::Error> = Box::new(error);

        // Verify it can be downcast back
        assert!(boxed_error.downcast_ref::<RuntimeError>().is_some());
    }

    #[test]
    fn test_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RuntimeError>();
    }

    #[test]
    fn test_special_characters_in_errors() {
        let special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?";
        let error = RuntimeError::ContainerNotFound {
            id: special_chars.to_string(),
        };
        assert_eq!(
            error.to_string(),
            format!("Container not found: {}", special_chars)
        );

        let error = RuntimeError::InvalidConfig {
            reason: special_chars.to_string(),
        };
        assert_eq!(
            error.to_string(),
            format!("Invalid container configuration: {}", special_chars)
        );
    }

    #[test]
    fn test_resource_limit_boundary_values() {
        // Test with zero values
        let error = RuntimeError::ResourceLimitExceeded {
            resource: "connections".to_string(),
            current: 0,
            limit: 0,
        };
        assert_eq!(
            error.to_string(),
            "Resource limit exceeded: connections (0/0)"
        );

        // Test with max values
        let error = RuntimeError::ResourceLimitExceeded {
            resource: "bytes".to_string(),
            current: usize::MAX,
            limit: usize::MAX,
        };
        let error_str = error.to_string();
        assert!(error_str.contains("bytes"));
        assert!(error_str.contains(&usize::MAX.to_string()));
    }
}
