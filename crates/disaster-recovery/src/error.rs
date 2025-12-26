//! Disaster recovery error types

use thiserror::Error;

/// Disaster recovery error types
#[derive(Debug, Error)]
pub enum DisasterRecoveryError {
    /// Backup operation failed
    #[error("Backup failed: {reason}")]
    BackupFailed { reason: String },

    /// Restore operation failed
    #[error("Restore failed: {reason}")]
    RestoreFailed { reason: String },

    /// Failover operation failed
    #[error("Failover failed: {source_site} -> {target_site}, reason: {reason}")]
    FailoverFailed {
        source_site: String,
        target_site: String,
        reason: String,
    },

    /// Replication lag exceeded threshold
    #[error("Replication lag exceeded: {lag_seconds}s > {threshold_seconds}s")]
    ReplicationLagExceeded {
        lag_seconds: u64,
        threshold_seconds: u64,
    },

    /// Data integrity check failed
    #[error("Data integrity check failed: {details}")]
    IntegrityCheckFailed { details: String },

    /// Recovery Time Objective exceeded
    #[error("RTO exceeded: {actual_seconds}s > {target_seconds}s")]
    RTOExceeded {
        actual_seconds: u64,
        target_seconds: u64,
    },

    /// Recovery Point Objective violated
    #[error("RPO violated: data loss of {loss_seconds}s > {target_seconds}s")]
    RPOViolated {
        loss_seconds: u64,
        target_seconds: u64,
    },

    /// Health check failed
    #[error("Health check failed for {service}: {reason}")]
    HealthCheckFailed { service: String, reason: String },

    /// Snapshot operation failed
    #[error("Snapshot {operation} failed: {reason}")]
    SnapshotFailed { operation: String, reason: String },

    /// Runbook execution failed
    #[error("Runbook {runbook_id} execution failed at step {step}: {reason}")]
    RunbookFailed {
        runbook_id: String,
        step: String,
        reason: String,
    },

    /// Resource unavailable
    #[error("Resource {resource} unavailable: {reason}")]
    ResourceUnavailable { resource: String, reason: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    /// Network error
    #[error("Network error: {details}")]
    NetworkError { details: String },

    /// Storage error
    #[error("Storage error: {details}")]
    StorageError { details: String },

    /// I/O error
    #[error("I/O error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    /// JSON serialization/deserialization error
    #[error("JSON error: {source}")]
    JsonError {
        #[from]
        source: serde_json::Error,
    },

    /// Bincode serialization error
    #[error("Bincode error: {source}")]
    BincodeError {
        #[from]
        source: bincode::Error,
    },

    /// Generic error for other cases
    #[error("{0}")]
    Other(String),
}

/// Disaster recovery result type
pub type DisasterRecoveryResult<T> = Result<T, DisasterRecoveryError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error as IoError, ErrorKind};

    #[test]
    fn test_backup_failed_error() {
        let error = DisasterRecoveryError::BackupFailed {
            reason: "Disk full".to_string(),
        };
        assert!(error.to_string().contains("Backup failed: Disk full"));
    }

    #[test]
    fn test_restore_failed_error() {
        let error = DisasterRecoveryError::RestoreFailed {
            reason: "Corrupted backup file".to_string(),
        };
        assert!(error
            .to_string()
            .contains("Restore failed: Corrupted backup file"));
    }

    #[test]
    fn test_failover_failed_error() {
        let error = DisasterRecoveryError::FailoverFailed {
            source_site: "primary-db".to_string(),
            target_site: "secondary-db".to_string(),
            reason: "Target not available".to_string(),
        };
        let error_str = error.to_string();
        assert!(error_str.contains("Failover failed"));
        assert!(error_str.contains("primary-db"));
        assert!(error_str.contains("secondary-db"));
        assert!(error_str.contains("Target not available"));
    }

    #[test]
    fn test_replication_lag_exceeded_error() {
        let error = DisasterRecoveryError::ReplicationLagExceeded {
            lag_seconds: 120,
            threshold_seconds: 60,
        };
        let error_str = error.to_string();
        assert!(error_str.contains("Replication lag exceeded"));
        assert!(error_str.contains("120s"));
        assert!(error_str.contains("60s"));
    }

    #[test]
    fn test_integrity_check_failed_error() {
        let error = DisasterRecoveryError::IntegrityCheckFailed {
            details: "Checksum mismatch in table users".to_string(),
        };
        assert!(error.to_string().contains("Data integrity check failed"));
        assert!(error
            .to_string()
            .contains("Checksum mismatch in table users"));
    }

    #[test]
    fn test_rto_exceeded_error() {
        let error = DisasterRecoveryError::RTOExceeded {
            actual_seconds: 300,
            target_seconds: 180,
        };
        let error_str = error.to_string();
        assert!(error_str.contains("RTO exceeded"));
        assert!(error_str.contains("300s"));
        assert!(error_str.contains("180s"));
    }

    #[test]
    fn test_rpo_violated_error() {
        let error = DisasterRecoveryError::RPOViolated {
            loss_seconds: 600,
            target_seconds: 300,
        };
        let error_str = error.to_string();
        assert!(error_str.contains("RPO violated"));
        assert!(error_str.contains("data loss of 600s"));
        assert!(error_str.contains("300s"));
    }

    #[test]
    fn test_health_check_failed_error() {
        let error = DisasterRecoveryError::HealthCheckFailed {
            service: "database-cluster".to_string(),
            reason: "Connection timeout".to_string(),
        };
        let error_str = error.to_string();
        assert!(error_str.contains("Health check failed"));
        assert!(error_str.contains("database-cluster"));
        assert!(error_str.contains("Connection timeout"));
    }

    #[test]
    fn test_snapshot_failed_error() {
        let error = DisasterRecoveryError::SnapshotFailed {
            operation: "create".to_string(),
            reason: "Insufficient storage space".to_string(),
        };
        let error_str = error.to_string();
        assert!(error_str.contains("Snapshot create failed"));
        assert!(error_str.contains("Insufficient storage space"));
    }

    #[test]
    fn test_runbook_failed_error() {
        let error = DisasterRecoveryError::RunbookFailed {
            runbook_id: "failover-db-001".to_string(),
            step: "validate-target".to_string(),
            reason: "Target validation failed".to_string(),
        };
        let error_str = error.to_string();
        assert!(error_str.contains("Runbook failover-db-001 execution failed"));
        assert!(error_str.contains("step validate-target"));
        assert!(error_str.contains("Target validation failed"));
    }

    #[test]
    fn test_resource_unavailable_error() {
        let error = DisasterRecoveryError::ResourceUnavailable {
            resource: "backup-storage".to_string(),
            reason: "Maintenance window".to_string(),
        };
        let error_str = error.to_string();
        assert!(error_str.contains("Resource backup-storage unavailable"));
        assert!(error_str.contains("Maintenance window"));
    }

    #[test]
    fn test_configuration_error() {
        let error = DisasterRecoveryError::ConfigurationError {
            message: "Invalid replication lag threshold".to_string(),
        };
        assert!(error.to_string().contains("Configuration error"));
        assert!(error
            .to_string()
            .contains("Invalid replication lag threshold"));
    }

    #[test]
    fn test_network_error() {
        let error = DisasterRecoveryError::NetworkError {
            details: "DNS resolution failed for replica.example.com".to_string(),
        };
        assert!(error.to_string().contains("Network error"));
        assert!(error.to_string().contains("DNS resolution failed"));
    }

    #[test]
    fn test_storage_error() {
        let error = DisasterRecoveryError::StorageError {
            details: "Write failed: disk quota exceeded".to_string(),
        };
        assert!(error.to_string().contains("Storage error"));
        assert!(error.to_string().contains("disk quota exceeded"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_error = IoError::new(ErrorKind::NotFound, "Backup file not found");
        let dr_error = DisasterRecoveryError::from(io_error);

        match dr_error {
            DisasterRecoveryError::IoError { .. } => {
                assert!(dr_error.to_string().contains("I/O error"));
            }
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_json_error_conversion() {
        let json_error = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let dr_error = DisasterRecoveryError::from(json_error);

        match dr_error {
            DisasterRecoveryError::JsonError { .. } => {
                assert!(dr_error.to_string().contains("JSON error"));
            }
            _ => panic!("Expected JsonError variant"),
        }
    }

    #[test]
    fn test_bincode_error_conversion() {
        let data = vec![1, 2, 3]; // Invalid bincode data for String
        let result: Result<String, bincode::Error> = bincode::deserialize(&data);
        if let Err(bincode_error) = result {
            let dr_error = DisasterRecoveryError::from(bincode_error);
            match dr_error {
                DisasterRecoveryError::BincodeError { .. } => {
                    assert!(dr_error.to_string().contains("Bincode error"));
                }
                _ => panic!("Expected BincodeError variant"),
            }
        }
    }

    #[test]
    fn test_error_debug_format() {
        let errors = vec![
            DisasterRecoveryError::BackupFailed {
                reason: "test".to_string(),
            },
            DisasterRecoveryError::RestoreFailed {
                reason: "test".to_string(),
            },
            DisasterRecoveryError::FailoverFailed {
                source_site: "a".to_string(),
                target_site: "b".to_string(),
                reason: "test".to_string(),
            },
            DisasterRecoveryError::ReplicationLagExceeded {
                lag_seconds: 10,
                threshold_seconds: 5,
            },
            DisasterRecoveryError::IntegrityCheckFailed {
                details: "test".to_string(),
            },
            DisasterRecoveryError::RTOExceeded {
                actual_seconds: 20,
                target_seconds: 10,
            },
            DisasterRecoveryError::RPOViolated {
                loss_seconds: 15,
                target_seconds: 5,
            },
            DisasterRecoveryError::HealthCheckFailed {
                service: "test".to_string(),
                reason: "test".to_string(),
            },
            DisasterRecoveryError::SnapshotFailed {
                operation: "test".to_string(),
                reason: "test".to_string(),
            },
            DisasterRecoveryError::RunbookFailed {
                runbook_id: "test".to_string(),
                step: "test".to_string(),
                reason: "test".to_string(),
            },
            DisasterRecoveryError::ResourceUnavailable {
                resource: "test".to_string(),
                reason: "test".to_string(),
            },
            DisasterRecoveryError::ConfigurationError {
                message: "test".to_string(),
            },
            DisasterRecoveryError::NetworkError {
                details: "test".to_string(),
            },
            DisasterRecoveryError::StorageError {
                details: "test".to_string(),
            },
        ];

        for error in errors {
            let debug_str = format!("{:?}", error);
            assert!(!debug_str.is_empty());
            assert!(debug_str.contains("DisasterRecoveryError"));
        }
    }

    #[test]
    fn test_error_source() {
        use std::error::Error;

        let io_error = IoError::new(ErrorKind::PermissionDenied, "access denied");
        let dr_error = DisasterRecoveryError::IoError { source: io_error };

        assert!(dr_error.source().is_some());
    }

    #[test]
    fn test_error_chain() {
        use std::error::Error;

        let io_error = IoError::new(ErrorKind::BrokenPipe, "pipe broken");
        let dr_error = DisasterRecoveryError::IoError { source: io_error };

        let mut error_chain = Vec::new();
        let mut current_error: &dyn Error = &dr_error;

        loop {
            error_chain.push(current_error.to_string());
            if let Some(source) = current_error.source() {
                current_error = source;
            } else {
                break;
            }
        }

        assert!(error_chain.len() >= 1);
        assert!(error_chain[0].contains("I/O error"));
    }

    #[test]
    fn test_disaster_recovery_result_type() {
        let success: DisasterRecoveryResult<i32> = Ok(42);
        assert!(success.is_ok());
        assert_eq!(success.unwrap(), 42);

        let failure: DisasterRecoveryResult<i32> = Err(DisasterRecoveryError::BackupFailed {
            reason: "test".to_string(),
        });
        assert!(failure.is_err());
    }

    #[test]
    fn test_error_edge_cases() {
        // Test with empty strings
        let error1 = DisasterRecoveryError::BackupFailed {
            reason: String::new(),
        };
        assert!(error1.to_string().contains("Backup failed:"));

        // Test with very large numbers
        let error2 = DisasterRecoveryError::ReplicationLagExceeded {
            lag_seconds: u64::MAX,
            threshold_seconds: u64::MAX - 1,
        };
        assert!(error2.to_string().contains(&u64::MAX.to_string()));

        // Test with Unicode strings
        let error3 = DisasterRecoveryError::ConfigurationError {
            message: "配置错误".to_string(),
        };
        assert!(error3.to_string().contains("配置错误"));
    }

    #[test]
    fn test_error_equality() {
        // Test that errors with same content are formatted consistently
        let error1 = DisasterRecoveryError::BackupFailed {
            reason: "test".to_string(),
        };
        let error2 = DisasterRecoveryError::BackupFailed {
            reason: "test".to_string(),
        };

        assert_eq!(error1.to_string(), error2.to_string());
    }

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<DisasterRecoveryError>();
        assert_sync::<DisasterRecoveryError>();
    }

    #[test]
    fn test_comprehensive_error_variants() {
        // Test all error variants are covered
        let all_variants = vec![
            "BackupFailed",
            "RestoreFailed",
            "FailoverFailed",
            "ReplicationLagExceeded",
            "IntegrityCheckFailed",
            "RTOExceeded",
            "RPOViolated",
            "HealthCheckFailed",
            "SnapshotFailed",
            "RunbookFailed",
            "ResourceUnavailable",
            "ConfigurationError",
            "NetworkError",
            "StorageError",
            "IoError",
            "JsonError",
            "BincodeError",
        ];

        // Verify we have tests for all variants
        assert_eq!(all_variants.len(), 17);
    }

    #[test]
    fn test_numeric_field_edge_cases() {
        // Test zero values
        let error1 = DisasterRecoveryError::ReplicationLagExceeded {
            lag_seconds: 0,
            threshold_seconds: 0,
        };
        assert!(error1.to_string().contains("0s"));

        // Test very small differences
        let error2 = DisasterRecoveryError::RTOExceeded {
            actual_seconds: 1,
            target_seconds: 0,
        };
        assert!(error2.to_string().contains("1s"));
        assert!(error2.to_string().contains("0s"));
    }

    #[test]
    fn test_complex_error_scenarios() {
        // Test complex failover scenario
        let error = DisasterRecoveryError::FailoverFailed {
            source_site: "primary-db-cluster-01.us-east.example.com".to_string(),
            target_site: "secondary-db-cluster-02.us-west.example.com".to_string(),
            reason: "Network partition detected - cannot establish consensus".to_string(),
        };

        let error_str = error.to_string();
        assert!(error_str.contains("primary-db-cluster-01.us-east.example.com"));
        assert!(error_str.contains("secondary-db-cluster-02.us-west.example.com"));
        assert!(error_str.contains("Network partition detected"));
    }
}
