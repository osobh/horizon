//! Error types for emergency control operations

use thiserror::Error;

/// Result type for emergency control operations
pub type EmergencyResult<T> = Result<T, EmergencyError>;

/// Comprehensive error types for emergency control operations
#[derive(Error, Debug, Clone)]
pub enum EmergencyError {
    /// Kill switch activation failed
    #[error("Kill switch activation failed: {reason}")]
    KillSwitchFailed { reason: String },

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {resource}: {current} > {limit}")]
    ResourceLimitExceeded {
        resource: String,
        current: f64,
        limit: f64,
    },

    /// Safety violation detected
    #[error("Safety violation: {violation_type}: {details}")]
    SafetyViolation {
        violation_type: String,
        details: String,
    },

    /// Agent suspension failed
    #[error("Agent suspension failed: {agent_id}: {reason}")]
    AgentSuspensionFailed { agent_id: String, reason: String },

    /// Audit logging error
    #[error("Audit logging failed: {operation}: {reason}")]
    AuditLogError { operation: String, reason: String },

    /// Recovery procedure failed
    #[error("Recovery procedure failed: {procedure}: {reason}")]
    RecoveryFailed { procedure: String, reason: String },

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Monitoring error
    #[error("Monitoring error: {component}: {error}")]
    MonitoringError { component: String, error: String },

    /// Network isolation failed
    #[error("Network isolation failed: {target}: {reason}")]
    NetworkIsolationFailed { target: String, reason: String },

    /// Timeout error
    #[error("Operation timed out after {duration:?}")]
    Timeout { duration: std::time::Duration },
}

impl From<std::io::Error> for EmergencyError {
    fn from(err: std::io::Error) -> Self {
        EmergencyError::ConfigurationError(format!("IO error: {err}"))
    }
}

impl From<serde_json::Error> for EmergencyError {
    fn from(err: serde_json::Error) -> Self {
        EmergencyError::ConfigurationError(format!("JSON error: {err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_kill_switch_error() {
        let error = EmergencyError::KillSwitchFailed {
            reason: "Emergency shutdown triggered".to_string(),
        };
        assert!(error.to_string().contains("Kill switch activation failed"));
        assert!(error.to_string().contains("Emergency shutdown triggered"));
    }

    #[test]
    fn test_resource_limit_error() {
        let error = EmergencyError::ResourceLimitExceeded {
            resource: "GPU Memory".to_string(),
            current: 32768.0,
            limit: 16384.0,
        };
        assert!(error.to_string().contains("Resource limit exceeded"));
        assert!(error.to_string().contains("GPU Memory"));
        assert!(error.to_string().contains("32768"));
        assert!(error.to_string().contains("16384"));
    }

    #[test]
    fn test_safety_violation_error() {
        let error = EmergencyError::SafetyViolation {
            violation_type: "Memory Access".to_string(),
            details: "Out of bounds access detected".to_string(),
        };
        assert!(error.to_string().contains("Safety violation"));
        assert!(error.to_string().contains("Memory Access"));
        assert!(error.to_string().contains("Out of bounds access detected"));
    }

    #[test]
    fn test_agent_suspension_error() {
        let error = EmergencyError::AgentSuspensionFailed {
            agent_id: "agent-123".to_string(),
            reason: "Agent not responding".to_string(),
        };
        assert!(error.to_string().contains("Agent suspension failed"));
        assert!(error.to_string().contains("agent-123"));
        assert!(error.to_string().contains("Agent not responding"));
    }

    #[test]
    fn test_audit_log_error() {
        let error = EmergencyError::AuditLogError {
            operation: "write_event".to_string(),
            reason: "Disk full".to_string(),
        };
        assert!(error.to_string().contains("Audit logging failed"));
        assert!(error.to_string().contains("write_event"));
        assert!(error.to_string().contains("Disk full"));
    }

    #[test]
    fn test_recovery_failed_error() {
        let error = EmergencyError::RecoveryFailed {
            procedure: "restore_checkpoint".to_string(),
            reason: "Corrupt checkpoint data".to_string(),
        };
        assert!(error.to_string().contains("Recovery procedure failed"));
        assert!(error.to_string().contains("restore_checkpoint"));
        assert!(error.to_string().contains("Corrupt checkpoint data"));
    }

    #[test]
    fn test_configuration_error() {
        let error = EmergencyError::ConfigurationError("Missing kill switch config".to_string());
        assert!(error.to_string().contains("Configuration error"));
        assert!(error.to_string().contains("Missing kill switch config"));
    }

    #[test]
    fn test_monitoring_error() {
        let error = EmergencyError::MonitoringError {
            component: "resource_monitor".to_string(),
            error: "Metrics collection failed".to_string(),
        };
        assert!(error.to_string().contains("Monitoring error"));
        assert!(error.to_string().contains("resource_monitor"));
        assert!(error.to_string().contains("Metrics collection failed"));
    }

    #[test]
    fn test_network_isolation_error() {
        let error = EmergencyError::NetworkIsolationFailed {
            target: "agent-456".to_string(),
            reason: "Network interface busy".to_string(),
        };
        assert!(error.to_string().contains("Network isolation failed"));
        assert!(error.to_string().contains("agent-456"));
        assert!(error.to_string().contains("Network interface busy"));
    }

    #[test]
    fn test_timeout_error() {
        let error = EmergencyError::Timeout {
            duration: Duration::from_secs(30),
        };
        assert!(error.to_string().contains("Operation timed out"));
        assert!(error.to_string().contains("30s"));
    }

    #[test]
    fn test_error_from_io() {
        let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Access denied");
        let emergency_error = EmergencyError::from(io_error);
        assert!(matches!(
            emergency_error,
            EmergencyError::ConfigurationError(_)
        ));
        assert!(emergency_error.to_string().contains("IO error"));
        assert!(emergency_error.to_string().contains("Access denied"));
    }

    #[test]
    fn test_error_from_serde_json() {
        let json_error = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let emergency_error = EmergencyError::from(json_error);
        assert!(matches!(
            emergency_error,
            EmergencyError::ConfigurationError(_)
        ));
        assert!(emergency_error.to_string().contains("JSON error"));
    }

    #[test]
    fn test_error_cloning() {
        let error = EmergencyError::KillSwitchFailed {
            reason: "Test error".to_string(),
        };
        let cloned_error = error.clone();
        assert_eq!(error.to_string(), cloned_error.to_string());
    }

    #[test]
    fn test_result_type() {
        let success: EmergencyResult<String> = Ok("Success".to_string());
        assert!(success.is_ok());

        let failure: EmergencyResult<String> =
            Err(EmergencyError::ConfigurationError("Test error".to_string()));
        assert!(failure.is_err());
    }

    #[test]
    fn test_error_edge_cases_empty_strings() {
        let errors = vec![
            EmergencyError::KillSwitchFailed {
                reason: String::new(),
            },
            EmergencyError::SafetyViolation {
                violation_type: String::new(),
                details: String::new(),
            },
            EmergencyError::AgentSuspensionFailed {
                agent_id: String::new(),
                reason: String::new(),
            },
            EmergencyError::AuditLogError {
                operation: String::new(),
                reason: String::new(),
            },
            EmergencyError::RecoveryFailed {
                procedure: String::new(),
                reason: String::new(),
            },
            EmergencyError::ConfigurationError(String::new()),
            EmergencyError::MonitoringError {
                component: String::new(),
                error: String::new(),
            },
            EmergencyError::NetworkIsolationFailed {
                target: String::new(),
                reason: String::new(),
            },
        ];

        // All errors should handle empty strings gracefully
        for error in errors {
            let _ = error.to_string();
        }
    }

    #[test]
    fn test_error_unicode_strings() {
        let error = EmergencyError::SafetyViolation {
            violation_type: "メモリアクセス違反".to_string(),
            details: "境界外アクセスが検出されました".to_string(),
        };
        let error_str = error.to_string();
        assert!(error_str.contains("メモリアクセス違反"));
        assert!(error_str.contains("境界外アクセスが検出されました"));

        let error = EmergencyError::AgentSuspensionFailed {
            agent_id: "エージェント-123".to_string(),
            reason: "応答なし".to_string(),
        };
        let error_str = error.to_string();
        assert!(error_str.contains("エージェント-123"));
        assert!(error_str.contains("応答なし"));
    }

    #[test]
    fn test_error_very_long_strings() {
        let long_string = "x".repeat(1000);
        let error = EmergencyError::ConfigurationError(long_string.clone());
        let error_str = error.to_string();
        assert!(error_str.contains(&long_string));

        let error = EmergencyError::SafetyViolation {
            violation_type: long_string.clone(),
            details: long_string.clone(),
        };
        let error_str = error.to_string();
        assert!(error_str.contains(&long_string));
    }

    #[test]
    fn test_resource_limit_extreme_values() {
        let error = EmergencyError::ResourceLimitExceeded {
            resource: "Test".to_string(),
            current: f64::MAX,
            limit: f64::MIN,
        };
        let error_str = error.to_string();
        assert!(error_str.contains(&f64::MAX.to_string()));
        assert!(error_str.contains(&f64::MIN.to_string()));

        let error = EmergencyError::ResourceLimitExceeded {
            resource: "Test".to_string(),
            current: 0.0,
            limit: f64::INFINITY,
        };
        let error_str = error.to_string();
        assert!(error_str.contains("0"));
        assert!(error_str.contains("inf"));
    }

    #[test]
    fn test_timeout_duration_formatting() {
        let error = EmergencyError::Timeout {
            duration: Duration::from_millis(500),
        };
        let error_str = error.to_string();
        assert!(error_str.contains("500ms"));

        let error = EmergencyError::Timeout {
            duration: Duration::from_secs(60),
        };
        let error_str = error.to_string();
        assert!(error_str.contains("60s"));

        let error = EmergencyError::Timeout {
            duration: Duration::from_secs(3661), // 1 hour, 1 minute, 1 second
        };
        let error_str = error.to_string();
        assert!(error_str.contains("3661s"));
    }

    #[test]
    fn test_error_debug_formatting() {
        let errors = vec![
            EmergencyError::KillSwitchFailed {
                reason: "test".to_string(),
            },
            EmergencyError::ResourceLimitExceeded {
                resource: "test".to_string(),
                current: 100.0,
                limit: 50.0,
            },
            EmergencyError::SafetyViolation {
                violation_type: "test".to_string(),
                details: "test".to_string(),
            },
            EmergencyError::Timeout {
                duration: Duration::from_secs(10),
            },
        ];

        for error in errors {
            let debug_str = format!("{:?}", error);
            assert!(debug_str.contains("EmergencyError"));
        }
    }

    #[test]
    fn test_error_from_io_variants() {
        let io_errors = vec![
            std::io::Error::new(std::io::ErrorKind::NotFound, "File not found"),
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Access denied"),
            std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "Connection refused"),
            std::io::Error::new(std::io::ErrorKind::TimedOut, "Operation timed out"),
            std::io::Error::new(std::io::ErrorKind::Interrupted, "Operation interrupted"),
        ];

        for io_error in io_errors {
            let msg = io_error.to_string();
            let emergency_error = EmergencyError::from(io_error);
            match emergency_error {
                EmergencyError::ConfigurationError(err_msg) => {
                    assert!(err_msg.contains("IO error"));
                    assert!(err_msg.contains(&msg));
                }
                _ => panic!("Expected ConfigurationError"),
            }
        }
    }

    #[test]
    fn test_error_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<EmergencyError>();
        assert_sync::<EmergencyError>();
    }

    #[test]
    fn test_error_special_characters() {
        let special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?";
        let error = EmergencyError::SafetyViolation {
            violation_type: special_chars.to_string(),
            details: special_chars.to_string(),
        };
        let error_str = error.to_string();
        assert!(error_str.contains(special_chars));
    }

    #[test]
    fn test_error_matching_patterns() {
        let error = EmergencyError::KillSwitchFailed {
            reason: "test".to_string(),
        };

        match error {
            EmergencyError::KillSwitchFailed { ref reason } => {
                assert_eq!(reason, "test");
            }
            _ => panic!("Expected KillSwitchFailed"),
        }

        let error = EmergencyError::ResourceLimitExceeded {
            resource: "memory".to_string(),
            current: 100.0,
            limit: 50.0,
        };

        match error {
            EmergencyError::ResourceLimitExceeded {
                ref resource,
                current,
                limit,
            } => {
                assert_eq!(resource, "memory");
                assert_eq!(current, 100.0);
                assert_eq!(limit, 50.0);
            }
            _ => panic!("Expected ResourceLimitExceeded"),
        }
    }

    #[test]
    fn test_error_chain_conversion() {
        // Test that errors can be used in error chains
        fn process_config() -> EmergencyResult<()> {
            let json_result: Result<serde_json::Value, serde_json::Error> =
                serde_json::from_str("invalid");
            json_result?;
            Ok(())
        }

        let result = process_config();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            EmergencyError::ConfigurationError(_)
        ));
    }

    #[test]
    fn test_error_display_consistency() {
        // Ensure error messages are consistent across clones
        let original = EmergencyError::SafetyViolation {
            violation_type: "Memory".to_string(),
            details: "Out of bounds".to_string(),
        };
        let cloned = original.clone();

        assert_eq!(original.to_string(), cloned.to_string());
        assert_eq!(format!("{:?}", original), format!("{:?}", cloned));
    }
}
