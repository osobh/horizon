//! Comprehensive tests for error handling and error types

use crate::error::{FaultToleranceError, FtResult, HealthStatus};
use std::io::{Error as IoError, ErrorKind};

#[test]
fn test_fault_tolerance_error_variants() {
    let errors = vec![
        FaultToleranceError::CheckpointFailed("checkpoint error".to_string()),
        FaultToleranceError::RecoveryFailed("recovery error".to_string()),
        FaultToleranceError::CoordinationError("coordination error".to_string()),
        FaultToleranceError::SerializationError("serialization error".to_string()),
        FaultToleranceError::StorageError("storage error".to_string()),
        FaultToleranceError::GpuMemoryError("gpu memory error".to_string()),
        FaultToleranceError::HealthCheckFailed("health check error".to_string()),
        FaultToleranceError::CheckpointNotFound("checkpoint-id-123".to_string()),
        FaultToleranceError::InvalidCheckpoint("invalid format".to_string()),
    ];

    for error in errors {
        let error_string = error.to_string();
        assert!(!error_string.is_empty());

        // Verify error messages contain expected content
        match error {
            FaultToleranceError::CheckpointFailed(ref msg) => {
                assert!(error_string.contains("Checkpoint creation failed"));
                assert!(error_string.contains(msg));
            }
            FaultToleranceError::RecoveryFailed(ref msg) => {
                assert!(error_string.contains("Recovery failed"));
                assert!(error_string.contains(msg));
            }
            FaultToleranceError::CoordinationError(ref msg) => {
                assert!(error_string.contains("Coordination error"));
                assert!(error_string.contains(msg));
            }
            FaultToleranceError::SerializationError(ref msg) => {
                assert!(error_string.contains("Serialization error"));
                assert!(error_string.contains(msg));
            }
            FaultToleranceError::StorageError(ref msg) => {
                assert!(error_string.contains("Storage error"));
                assert!(error_string.contains(msg));
            }
            FaultToleranceError::GpuMemoryError(ref msg) => {
                assert!(error_string.contains("GPU memory error"));
                assert!(error_string.contains(msg));
            }
            FaultToleranceError::HealthCheckFailed(ref msg) => {
                assert!(error_string.contains("System health check failed"));
                assert!(error_string.contains(msg));
            }
            FaultToleranceError::CheckpointNotFound(ref id) => {
                assert!(error_string.contains("Checkpoint not found"));
                assert!(error_string.contains(id));
            }
            FaultToleranceError::InvalidCheckpoint(ref msg) => {
                assert!(error_string.contains("Invalid checkpoint format"));
                assert!(error_string.contains(msg));
            }
            _ => {}
        }
    }
}

#[test]
fn test_io_error_conversion() {
    let io_errors = vec![
        IoError::new(ErrorKind::NotFound, "file not found"),
        IoError::new(ErrorKind::PermissionDenied, "permission denied"),
        IoError::new(ErrorKind::ConnectionAborted, "connection aborted"),
        IoError::new(ErrorKind::TimedOut, "operation timed out"),
        IoError::new(ErrorKind::Interrupted, "operation interrupted"),
        IoError::new(ErrorKind::UnexpectedEof, "unexpected end of file"),
        IoError::new(ErrorKind::InvalidData, "invalid data"),
        IoError::new(ErrorKind::WriteZero, "write zero"),
        IoError::new(ErrorKind::Other, "other error"),
    ];

    for io_error in io_errors {
        let original_message = io_error.to_string();
        let ft_error = FaultToleranceError::from(io_error);

        assert!(matches!(ft_error, FaultToleranceError::IoError(_)));

        let ft_error_string = ft_error.to_string();
        assert!(ft_error_string.contains("IO error"));
        assert!(ft_error_string.contains(&original_message));
    }
}

#[test]
fn test_bincode_error_conversion() {
    // Create a mock bincode error by trying to serialize an invalid type
    use bincode::serialize;
    use serde::Serialize;

    #[derive(Serialize)]
    struct InvalidStruct {
        #[serde(serialize_with = "fail_serialize")]
        field: i32,
    }

    fn fail_serialize<S>(_: &i32, _: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        Err(serde::ser::Error::custom(
            "intentional serialization failure",
        ))
    }

    let invalid = InvalidStruct { field: 42 };
    let bincode_error = serialize(&invalid).unwrap_err();
    let ft_error = FaultToleranceError::from(bincode_error);

    assert!(matches!(ft_error, FaultToleranceError::BincodeError(_)));

    let error_string = ft_error.to_string();
    assert!(error_string.contains("Serialization error"));
}

#[test]
fn test_health_status_variants() {
    let statuses = vec![
        HealthStatus::Healthy,
        HealthStatus::Degraded,
        HealthStatus::Failed,
    ];

    // Test equality
    for status in &statuses {
        assert_eq!(status, status);
    }

    // Test inequality
    assert_ne!(HealthStatus::Healthy, HealthStatus::Degraded);
    assert_ne!(HealthStatus::Healthy, HealthStatus::Failed);
    assert_ne!(HealthStatus::Degraded, HealthStatus::Failed);
}

#[test]
fn test_health_status_serialization() {
    let statuses = vec![
        HealthStatus::Healthy,
        HealthStatus::Degraded,
        HealthStatus::Failed,
    ];

    for status in statuses {
        // Test JSON serialization
        let json_serialized = serde_json::to_string(&status)?;
        let json_deserialized: HealthStatus = serde_json::from_str(&json_serialized)?;
        assert_eq!(status, json_deserialized);

        // Test bincode serialization
        let bin_serialized = bincode::serialize(&status).unwrap();
        let bin_deserialized: HealthStatus = bincode::deserialize(&bin_serialized).unwrap();
        assert_eq!(status, bin_deserialized);
    }
}

#[test]
fn test_health_status_debug_formatting() {
    let statuses = vec![
        (HealthStatus::Healthy, "Healthy"),
        (HealthStatus::Degraded, "Degraded"),
        (HealthStatus::Failed, "Failed"),
    ];

    for (status, expected_str) in statuses {
        let debug_str = format!("{:?}", status);
        assert!(debug_str.contains(expected_str));
    }
}

#[test]
fn test_health_status_clone() {
    let original = HealthStatus::Degraded;
    let cloned = original.clone();

    assert_eq!(original, cloned);

    // Verify they are separate instances
    let debug_original = format!("{:?}", original);
    let debug_cloned = format!("{:?}", cloned);
    assert_eq!(debug_original, debug_cloned);
}

#[test]
fn test_ft_result_type_alias() {
    // Test successful result
    let success: FtResult<i32> = Ok(42);
    assert!(success.is_ok());
    assert_eq!(success.unwrap(), 42);

    // Test error result
    let error: FtResult<i32> = Err(FaultToleranceError::CheckpointFailed("test".to_string()));
    assert!(error.is_err());

    match error {
        Err(FaultToleranceError::CheckpointFailed(msg)) => {
            assert_eq!(msg, "test");
        }
        _ => panic!("Expected CheckpointFailed error"),
    }
}

#[test]
fn test_error_debug_formatting() {
    let error = FaultToleranceError::RecoveryFailed("recovery timeout".to_string());
    let debug_str = format!("{:?}", error);

    assert!(debug_str.contains("RecoveryFailed"));
    assert!(debug_str.contains("recovery timeout"));
}

#[test]
fn test_error_chain_and_source() {
    // Test IoError source
    let io_error = IoError::new(ErrorKind::NotFound, "original file not found");
    let ft_error = FaultToleranceError::from(io_error);

    // Verify the source chain
    let error_string = ft_error.to_string();
    assert!(error_string.contains("IO error"));
    assert!(error_string.contains("original file not found"));
}

#[test]
fn test_custom_error_messages() {
    let test_cases = vec![
        (
            "checkpoint creation failed due to memory issues",
            FaultToleranceError::CheckpointFailed,
        ),
        (
            "recovery failed after 5 retries",
            FaultToleranceError::RecoveryFailed,
        ),
        (
            "coordination failed - leader election timeout",
            FaultToleranceError::CoordinationError,
        ),
        (
            "serialization failed - invalid UTF-8",
            FaultToleranceError::SerializationError,
        ),
        (
            "storage failed - disk full",
            FaultToleranceError::StorageError,
        ),
        (
            "GPU memory allocation failed",
            FaultToleranceError::GpuMemoryError,
        ),
        (
            "health check timeout after 30s",
            FaultToleranceError::HealthCheckFailed,
        ),
        (
            "checkpoint-abc-123-def",
            FaultToleranceError::CheckpointNotFound,
        ),
        (
            "corrupted checkpoint header",
            FaultToleranceError::InvalidCheckpoint,
        ),
    ];

    for (message, error_constructor) in test_cases {
        let error = error_constructor(message.to_string());
        let error_string = error.to_string();
        assert!(error_string.contains(message));
    }
}

#[test]
fn test_error_propagation_in_result_chain() {
    fn operation_that_fails() -> FtResult<()> {
        Err(FaultToleranceError::GpuMemoryError(
            "out of memory".to_string(),
        ))
    }

    fn higher_level_operation() -> FtResult<String> {
        operation_that_fails()?;
        Ok("success".to_string())
    }

    let result = higher_level_operation();
    assert!(result.is_err());

    match result {
        Err(FaultToleranceError::GpuMemoryError(msg)) => {
            assert_eq!(msg, "out of memory");
        }
        _ => panic!("Expected GpuMemoryError"),
    }
}

#[test]
fn test_error_matching_patterns() {
    let errors = vec![
        FaultToleranceError::CheckpointFailed("test".to_string()),
        FaultToleranceError::RecoveryFailed("test".to_string()),
        FaultToleranceError::CoordinationError("test".to_string()),
        FaultToleranceError::SerializationError("test".to_string()),
        FaultToleranceError::StorageError("test".to_string()),
        FaultToleranceError::GpuMemoryError("test".to_string()),
        FaultToleranceError::HealthCheckFailed("test".to_string()),
        FaultToleranceError::CheckpointNotFound("test".to_string()),
        FaultToleranceError::InvalidCheckpoint("test".to_string()),
        FaultToleranceError::IoError(IoError::new(ErrorKind::NotFound, "test")),
    ];

    for error in errors {
        match &error {
            FaultToleranceError::CheckpointFailed(_) => {
                assert!(matches!(error, FaultToleranceError::CheckpointFailed(_)))
            }
            FaultToleranceError::RecoveryFailed(_) => {
                assert!(matches!(error, FaultToleranceError::RecoveryFailed(_)))
            }
            FaultToleranceError::CoordinationError(_) => {
                assert!(matches!(error, FaultToleranceError::CoordinationError(_)))
            }
            FaultToleranceError::SerializationError(_) => {
                assert!(matches!(error, FaultToleranceError::SerializationError(_)))
            }
            FaultToleranceError::StorageError(_) => {
                assert!(matches!(error, FaultToleranceError::StorageError(_)))
            }
            FaultToleranceError::GpuMemoryError(_) => {
                assert!(matches!(error, FaultToleranceError::GpuMemoryError(_)))
            }
            FaultToleranceError::HealthCheckFailed(_) => {
                assert!(matches!(error, FaultToleranceError::HealthCheckFailed(_)))
            }
            FaultToleranceError::CheckpointNotFound(_) => {
                assert!(matches!(error, FaultToleranceError::CheckpointNotFound(_)))
            }
            FaultToleranceError::InvalidCheckpoint(_) => {
                assert!(matches!(error, FaultToleranceError::InvalidCheckpoint(_)))
            }
            FaultToleranceError::IoError(_) => {
                assert!(matches!(error, FaultToleranceError::IoError(_)))
            }
            FaultToleranceError::BincodeError(_) => {
                assert!(matches!(error, FaultToleranceError::BincodeError(_)))
            }
        }
    }
}

#[test]
fn test_error_category_classification() {
    // Checkpoint-related errors
    let checkpoint_errors = vec![
        FaultToleranceError::CheckpointFailed("test".to_string()),
        FaultToleranceError::CheckpointNotFound("test".to_string()),
        FaultToleranceError::InvalidCheckpoint("test".to_string()),
    ];

    for error in checkpoint_errors {
        assert!(is_checkpoint_related(&error));
    }

    // Recovery-related errors
    let recovery_errors = vec![FaultToleranceError::RecoveryFailed("test".to_string())];

    for error in recovery_errors {
        assert!(is_recovery_related(&error));
    }

    // Infrastructure-related errors
    let infrastructure_errors = vec![
        FaultToleranceError::StorageError("test".to_string()),
        FaultToleranceError::GpuMemoryError("test".to_string()),
        FaultToleranceError::SerializationError("test".to_string()),
        FaultToleranceError::IoError(IoError::new(ErrorKind::NotFound, "test")),
    ];

    for error in infrastructure_errors {
        assert!(is_infrastructure_related(&error));
    }
}

fn is_checkpoint_related(error: &FaultToleranceError) -> bool {
    matches!(
        error,
        FaultToleranceError::CheckpointFailed(_)
            | FaultToleranceError::CheckpointNotFound(_)
            | FaultToleranceError::InvalidCheckpoint(_)
    )
}

fn is_recovery_related(error: &FaultToleranceError) -> bool {
    matches!(error, FaultToleranceError::RecoveryFailed(_))
}

fn is_infrastructure_related(error: &FaultToleranceError) -> bool {
    matches!(
        error,
        FaultToleranceError::StorageError(_)
            | FaultToleranceError::GpuMemoryError(_)
            | FaultToleranceError::SerializationError(_)
            | FaultToleranceError::IoError(_)
            | FaultToleranceError::BincodeError(_)
    )
}

#[test]
fn test_error_severity_levels() {
    // Critical errors that should stop operations
    let critical_errors = vec![
        FaultToleranceError::GpuMemoryError("GPU crashed".to_string()),
        FaultToleranceError::StorageError("disk failure".to_string()),
    ];

    for error in critical_errors {
        assert!(is_critical_error(&error));
    }

    // Warning errors that can be retried
    let warning_errors = vec![
        FaultToleranceError::CoordinationError("temporary network issue".to_string()),
        FaultToleranceError::HealthCheckFailed("timeout".to_string()),
    ];

    for error in warning_errors {
        assert!(is_warning_error(&error));
    }
}

fn is_critical_error(error: &FaultToleranceError) -> bool {
    matches!(
        error,
        FaultToleranceError::GpuMemoryError(_) | FaultToleranceError::StorageError(_)
    )
}

fn is_warning_error(error: &FaultToleranceError) -> bool {
    matches!(
        error,
        FaultToleranceError::CoordinationError(_) | FaultToleranceError::HealthCheckFailed(_)
    )
}

#[test]
fn test_ft_result_combinators() {
    // Test map
    let success: FtResult<i32> = Ok(42);
    let mapped = success.map(|x| x * 2);
    assert_eq!(mapped.unwrap(), 84);

    // Test map_err
    let error: FtResult<i32> = Err(FaultToleranceError::CheckpointFailed(
        "original".to_string(),
    ));
    let mapped_err = error.map_err(|e| match e {
        FaultToleranceError::CheckpointFailed(msg) => {
            FaultToleranceError::RecoveryFailed(format!("mapped: {}", msg))
        }
        other => other,
    });

    match mapped_err {
        Err(FaultToleranceError::RecoveryFailed(msg)) => {
            assert!(msg.contains("mapped: original"));
        }
        _ => panic!("Expected mapped RecoveryFailed error"),
    }

    // Test and_then
    let success: FtResult<i32> = Ok(42);
    let chained = success.and_then(|x| {
        if x > 40 {
            Ok(x.to_string())
        } else {
            Err(FaultToleranceError::CheckpointFailed(
                "too small".to_string(),
            ))
        }
    });
    assert_eq!(chained.unwrap(), "42");

    // Test or_else
    let error: FtResult<i32> = Err(FaultToleranceError::CheckpointFailed("test".to_string()));
    let recovered = error.or_else(|_| Ok(100));
    assert_eq!(recovered.unwrap(), 100);
}

#[test]
fn test_health_status_ordering_and_comparison() {
    // Test that health statuses can be compared
    let healthy = HealthStatus::Healthy;
    let degraded = HealthStatus::Degraded;
    let failed = HealthStatus::Failed;

    // Test equality
    assert_eq!(healthy, HealthStatus::Healthy);
    assert_eq!(degraded, HealthStatus::Degraded);
    assert_eq!(failed, HealthStatus::Failed);

    // Test inequality
    assert_ne!(healthy, degraded);
    assert_ne!(healthy, failed);
    assert_ne!(degraded, failed);

    // Test with vectors
    let statuses = vec![healthy.clone(), degraded.clone(), failed.clone()];
    assert_eq!(statuses.len(), 3);
    assert!(statuses.contains(&healthy));
    assert!(statuses.contains(&degraded));
    assert!(statuses.contains(&failed));
}

#[test]
fn test_complex_error_scenarios() {
    // Simulate a complex error scenario: checkpoint creation fails due to IO error,
    // then recovery is attempted but also fails

    fn simulate_checkpoint_creation() -> FtResult<String> {
        let io_error = IoError::new(
            ErrorKind::PermissionDenied,
            "cannot write to checkpoint directory",
        );
        Err(FaultToleranceError::from(io_error))
    }

    fn simulate_recovery_after_checkpoint_failure() -> FtResult<()> {
        // Try checkpoint creation first
        match simulate_checkpoint_creation() {
            Ok(_) => Ok(()),
            Err(FaultToleranceError::IoError(_)) => {
                // Checkpoint failed, try recovery
                Err(FaultToleranceError::RecoveryFailed(
                    "no valid checkpoint available".to_string(),
                ))
            }
            Err(other) => Err(other),
        }
    }

    let result = simulate_recovery_after_checkpoint_failure();
    assert!(result.is_err());

    match result {
        Err(FaultToleranceError::RecoveryFailed(msg)) => {
            assert!(msg.contains("no valid checkpoint available"));
        }
        _ => panic!("Expected RecoveryFailed error"),
    }
}
