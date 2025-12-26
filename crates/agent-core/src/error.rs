//! Agent system error types

use thiserror::Error;

/// Agent system errors
#[derive(Debug, Error)]
pub enum AgentError {
    /// Agent not found
    #[error("Agent not found: {id}")]
    AgentNotFound {
        /// Agent ID that was not found
        id: String,
    },

    /// Goal not found
    #[error("Goal not found: {id}")]
    GoalNotFound {
        /// Goal ID that was not found
        id: String,
    },

    /// Agent already exists
    #[error("Agent already exists: {id}")]
    AgentAlreadyExists {
        /// Agent ID that already exists
        id: String,
    },

    /// Invalid agent state transition
    #[error("Invalid state transition from {from:?} to {to:?}")]
    InvalidStateTransition {
        /// Previous agent state
        from: crate::agent::AgentState,
        /// Target agent state
        to: crate::agent::AgentState,
    },

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {resource}")]
    ResourceLimitExceeded {
        /// Resource that exceeded the limit
        resource: String,
    },

    /// Communication failure
    #[error("Communication failure: {message}")]
    CommunicationFailure {
        /// Error message describing the failure
        message: String,
    },

    /// Memory operation failed
    #[error("Memory operation failed: {message}")]
    MemoryError {
        /// Error message describing the memory error
        message: String,
    },

    /// Scheduling error
    #[error("Scheduling error: {message}")]
    SchedulingError {
        /// Error message describing the scheduling error
        message: String,
    },

    /// Goal interpretation failed
    #[error("Goal interpretation failed: {message}")]
    GoalInterpretationError {
        /// Error message describing the interpretation error
        message: String,
    },

    /// Agent execution error
    #[error("Agent execution error: {message}")]
    ExecutionError {
        /// Error message describing the execution error
        message: String,
    },

    /// Timeout
    #[error("Operation timed out after {duration:?}")]
    Timeout {
        /// Duration after which the operation timed out
        duration: std::time::Duration,
    },

    /// Permission denied
    #[error("Permission denied: {action}")]
    PermissionDenied {
        /// Action that was denied
        action: String,
    },

    /// CUDA error
    #[error("CUDA error: {message}")]
    CudaError {
        /// Error message describing the CUDA error
        message: String,
    },

    /// Runtime error
    #[error("Runtime error: {message}")]
    RuntimeError {
        /// Error message describing the runtime error
        message: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Other error
    #[error("Agent error: {0}")]
    Other(String),
}

/// Result type for agent operations
pub type AgentResult<T> = Result<T, AgentError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentState;

    #[test]
    fn test_error_display() {
        let err = AgentError::AgentNotFound {
            id: "test-agent".to_string(),
        };
        assert_eq!(err.to_string(), "Agent not found: test-agent");

        let err = AgentError::ResourceLimitExceeded {
            resource: "memory".to_string(),
        };
        assert_eq!(err.to_string(), "Resource limit exceeded: memory");
    }

    #[test]
    fn test_all_error_variants() {
        // AgentNotFound
        let err = AgentError::AgentNotFound {
            id: "agent-123".to_string(),
        };
        assert_eq!(err.to_string(), "Agent not found: agent-123");

        // GoalNotFound
        let err = AgentError::GoalNotFound {
            id: "goal-456".to_string(),
        };
        assert_eq!(err.to_string(), "Goal not found: goal-456");

        // AgentAlreadyExists
        let err = AgentError::AgentAlreadyExists {
            id: "duplicate".to_string(),
        };
        assert_eq!(err.to_string(), "Agent already exists: duplicate");

        // InvalidStateTransition
        let err = AgentError::InvalidStateTransition {
            from: AgentState::Created,
            to: AgentState::Running,
        };
        assert!(err.to_string().contains("Invalid state transition"));

        // ResourceLimitExceeded
        let err = AgentError::ResourceLimitExceeded {
            resource: "GPU memory".to_string(),
        };
        assert_eq!(err.to_string(), "Resource limit exceeded: GPU memory");

        // CommunicationFailure
        let err = AgentError::CommunicationFailure {
            message: "Connection timeout".to_string(),
        };
        assert_eq!(err.to_string(), "Communication failure: Connection timeout");

        // MemoryError
        let err = AgentError::MemoryError {
            message: "Out of memory".to_string(),
        };
        assert_eq!(err.to_string(), "Memory operation failed: Out of memory");

        // SchedulingError
        let err = AgentError::SchedulingError {
            message: "No available slots".to_string(),
        };
        assert_eq!(err.to_string(), "Scheduling error: No available slots");

        // GoalInterpretationError
        let err = AgentError::GoalInterpretationError {
            message: "Invalid goal format".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Goal interpretation failed: Invalid goal format"
        );

        // ExecutionError
        let err = AgentError::ExecutionError {
            message: "Kernel launch failed".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Agent execution error: Kernel launch failed"
        );

        // Timeout
        let err = AgentError::Timeout {
            duration: std::time::Duration::from_secs(30),
        };
        assert!(err.to_string().contains("Operation timed out"));

        // PermissionDenied
        let err = AgentError::PermissionDenied {
            action: "access restricted resource".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Permission denied: access restricted resource"
        );

        // Other
        let err = AgentError::Other("Custom error message".to_string());
        assert_eq!(err.to_string(), "Agent error: Custom error message");
    }

    #[test]
    fn test_error_debug_format() {
        let err = AgentError::AgentNotFound {
            id: "debug-test".to_string(),
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("AgentNotFound"));
        assert!(debug_str.contains("debug-test"));
    }

    #[test]
    fn test_result_type() {
        fn test_function() -> AgentResult<i32> {
            Ok(42)
        }

        fn test_error_function() -> AgentResult<i32> {
            Err(AgentError::Other("test error".to_string()))
        }

        assert!(test_function().is_ok());
        assert_eq!(test_function().unwrap(), 42);

        assert!(test_error_function().is_err());
        assert!(matches!(
            test_error_function().unwrap_err(),
            AgentError::Other(_)
        ));
    }

    #[test]
    fn test_io_error_conversion() {
        use std::io;

        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let agent_err: AgentError = io_err.into();

        assert!(matches!(agent_err, AgentError::IoError(_)));
        assert!(agent_err.to_string().contains("IO error"));
    }

    #[test]
    fn test_error_chaining() {
        fn inner_function() -> AgentResult<()> {
            Err(AgentError::MemoryError {
                message: "allocation failed".to_string(),
            })
        }

        fn outer_function() -> AgentResult<()> {
            inner_function().map_err(|_| AgentError::ExecutionError {
                message: "failed due to memory error".to_string(),
            })
        }

        let result = outer_function();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AgentError::ExecutionError { .. }
        ));
    }

    #[test]
    fn test_error_pattern_matching() {
        let errors = vec![
            AgentError::AgentNotFound {
                id: "1".to_string(),
            },
            AgentError::GoalNotFound {
                id: "2".to_string(),
            },
            AgentError::Other("misc".to_string()),
        ];

        for err in errors {
            match err {
                AgentError::AgentNotFound { id } => assert_eq!(id, "1"),
                AgentError::GoalNotFound { id } => assert_eq!(id, "2"),
                AgentError::Other(msg) => assert_eq!(msg, "misc"),
                _ => panic!("Unexpected error variant"),
            }
        }
    }

    #[test]
    fn test_error_with_format_strings() {
        let agent_id = "agent-xyz";
        let err = AgentError::AgentNotFound {
            id: format!("{}-suffix", agent_id),
        };
        assert_eq!(err.to_string(), "Agent not found: agent-xyz-suffix");
    }

    #[test]
    fn test_error_size() {
        use std::mem::size_of;

        // Ensure error enum is reasonably sized
        assert!(size_of::<AgentError>() < 128); // Should be compact
    }

    #[test]
    fn test_multiple_error_contexts() {
        let resource_errors = vec![
            AgentError::ResourceLimitExceeded {
                resource: "CPU cores".to_string(),
            },
            AgentError::ResourceLimitExceeded {
                resource: "GPU memory".to_string(),
            },
            AgentError::ResourceLimitExceeded {
                resource: "Network bandwidth".to_string(),
            },
        ];

        for (i, err) in resource_errors.iter().enumerate() {
            assert!(err.to_string().contains("Resource limit exceeded"));
            match i {
                0 => assert!(err.to_string().contains("CPU cores")),
                1 => assert!(err.to_string().contains("GPU memory")),
                2 => assert!(err.to_string().contains("Network bandwidth")),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_timeout_duration_display() {
        let durations = vec![
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(60),
            std::time::Duration::from_millis(500),
        ];

        for duration in durations {
            let err = AgentError::Timeout { duration };
            let err_str = err.to_string();
            assert!(err_str.contains("Operation timed out"));
            assert!(err_str.contains(&format!("{:?}", duration)));
        }
    }

    #[test]
    fn test_state_transition_combinations() {
        let states = vec![
            AgentState::Created,
            AgentState::Initializing,
            AgentState::Running,
            AgentState::Suspended,
            AgentState::Stopped,
        ];

        // Test a few specific invalid transitions
        let invalid_transitions = vec![
            (AgentState::Created, AgentState::Suspended),
            (AgentState::Stopped, AgentState::Running),
        ];

        for (from, to) in invalid_transitions {
            let err = AgentError::InvalidStateTransition { from, to };
            let err_str = err.to_string();
            assert!(err_str.contains("Invalid state transition"));
            assert!(err_str.contains(&format!("{:?}", from)));
            assert!(err_str.contains(&format!("{:?}", to)));
        }
    }

    #[test]
    fn test_error_equality() {
        // Note: AgentError doesn't implement PartialEq, but we can test string representations
        let err1 = AgentError::AgentNotFound {
            id: "same".to_string(),
        };
        let err2 = AgentError::AgentNotFound {
            id: "same".to_string(),
        };

        assert_eq!(err1.to_string(), err2.to_string());
    }
}
