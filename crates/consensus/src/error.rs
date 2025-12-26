//! Consensus protocol error types

use thiserror::Error;

/// Consensus protocol errors
#[derive(Error, Debug)]
pub enum ConsensusError {
    /// Validation failed
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// Network communication error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Consensus timeout
    #[error("Consensus timeout after {duration:?}")]
    Timeout { duration: std::time::Duration },

    /// Insufficient validators
    #[error("Insufficient validators: need {required}, have {available}")]
    InsufficientValidators { required: usize, available: usize },

    /// Invalid message format
    #[error("Invalid message: {0}")]
    InvalidMessage(String),

    /// Byzantine behavior detected
    #[error("Byzantine behavior detected from validator {validator_id}: {reason}")]
    ByzantineBehavior {
        validator_id: String,
        reason: String,
    },

    /// State synchronization error
    #[error("State sync error: {0}")]
    StateSyncError(String),

    /// GPU computation error
    #[error("GPU computation error: {0}")]
    GpuError(String),

    /// Leadership election failed
    #[error("Leader election failed: {0}")]
    LeaderElectionFailed(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// IO operation failed
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    /// Feature not yet implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// Result type for consensus operations
pub type ConsensusResult<T> = Result<T, ConsensusError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_error_display() {
        let error = ConsensusError::ValidationFailed("test validation".to_string());
        assert!(error.to_string().contains("Validation failed"));
    }

    #[test]
    fn test_timeout_error() {
        let error = ConsensusError::Timeout {
            duration: Duration::from_secs(30),
        };
        assert!(error.to_string().contains("Consensus timeout"));
    }

    #[test]
    fn test_insufficient_validators_error() {
        let error = ConsensusError::InsufficientValidators {
            required: 3,
            available: 1,
        };
        assert!(error.to_string().contains("need 3, have 1"));
    }

    #[test]
    fn test_byzantine_behavior_error() {
        let error = ConsensusError::ByzantineBehavior {
            validator_id: "validator123".to_string(),
            reason: "double voting".to_string(),
        };
        assert!(error.to_string().contains("validator123"));
        assert!(error.to_string().contains("double voting"));
    }

    #[test]
    fn test_error_from_io() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let consensus_error = ConsensusError::from(io_error);
        assert!(matches!(consensus_error, ConsensusError::IoError(_)));
    }

    #[test]
    fn test_network_error() {
        let error = ConsensusError::NetworkError("Connection lost".to_string());
        assert!(error.to_string().contains("Network error"));
        assert!(error.to_string().contains("Connection lost"));
    }

    #[test]
    fn test_invalid_message_error() {
        let error = ConsensusError::InvalidMessage("Malformed protocol buffer".to_string());
        assert!(error.to_string().contains("Invalid message"));
        assert!(error.to_string().contains("Malformed protocol buffer"));
    }

    #[test]
    fn test_state_sync_error() {
        let error = ConsensusError::StateSyncError("Inconsistent state".to_string());
        assert!(error.to_string().contains("State sync error"));
        assert!(error.to_string().contains("Inconsistent state"));
    }

    #[test]
    fn test_gpu_error() {
        let error = ConsensusError::GpuError("CUDA out of memory".to_string());
        assert!(error.to_string().contains("GPU computation error"));
        assert!(error.to_string().contains("CUDA out of memory"));
    }

    #[test]
    fn test_leader_election_failed_error() {
        let error = ConsensusError::LeaderElectionFailed("Split vote".to_string());
        assert!(error.to_string().contains("Leader election failed"));
        assert!(error.to_string().contains("Split vote"));
    }

    #[test]
    fn test_config_error() {
        let error = ConsensusError::ConfigError("Invalid timeout value".to_string());
        assert!(error.to_string().contains("Configuration error"));
        assert!(error.to_string().contains("Invalid timeout value"));
    }

    #[test]
    fn test_error_from_bincode() {
        // Create a mock bincode error scenario
        let data = vec![1, 2, 3]; // Invalid bincode data
        let result: Result<String, bincode::Error> = bincode::deserialize(&data);
        if let Err(bincode_error) = result {
            let consensus_error = ConsensusError::from(bincode_error);
            assert!(matches!(
                consensus_error,
                ConsensusError::SerializationError(_)
            ));
        }
    }

    #[test]
    fn test_error_debug_format() {
        let errors = vec![
            ConsensusError::ValidationFailed("test".to_string()),
            ConsensusError::NetworkError("test".to_string()),
            ConsensusError::Timeout {
                duration: Duration::from_secs(10),
            },
            ConsensusError::InsufficientValidators {
                required: 5,
                available: 2,
            },
            ConsensusError::InvalidMessage("test".to_string()),
            ConsensusError::ByzantineBehavior {
                validator_id: "test".to_string(),
                reason: "test".to_string(),
            },
            ConsensusError::StateSyncError("test".to_string()),
            ConsensusError::GpuError("test".to_string()),
            ConsensusError::LeaderElectionFailed("test".to_string()),
            ConsensusError::ConfigError("test".to_string()),
        ];

        for error in errors {
            let debug_str = format!("{:?}", error);
            assert!(!debug_str.is_empty());
            assert!(debug_str.contains("ConsensusError"));
        }
    }

    #[test]
    fn test_error_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<ConsensusError>();
        assert_sync::<ConsensusError>();
    }

    #[test]
    fn test_timeout_with_zero_duration() {
        let error = ConsensusError::Timeout {
            duration: Duration::from_secs(0),
        };
        assert!(error.to_string().contains("0s"));
    }

    #[test]
    fn test_insufficient_validators_edge_cases() {
        // Zero validators available
        let error1 = ConsensusError::InsufficientValidators {
            required: 3,
            available: 0,
        };
        assert!(error1.to_string().contains("need 3, have 0"));

        // Large numbers
        let error2 = ConsensusError::InsufficientValidators {
            required: 1000,
            available: 999,
        };
        assert!(error2.to_string().contains("need 1000, have 999"));
    }

    #[test]
    fn test_byzantine_behavior_empty_fields() {
        let error = ConsensusError::ByzantineBehavior {
            validator_id: String::new(),
            reason: String::new(),
        };
        let message = error.to_string();
        assert!(message.contains("Byzantine behavior detected"));
    }

    #[test]
    fn test_error_source() {
        use std::error::Error;

        let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let consensus_error = ConsensusError::IoError(io_error);

        assert!(consensus_error.source().is_some());
    }

    #[test]
    fn test_error_chain() {
        use std::error::Error;

        let io_error =
            std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "connection refused");
        let consensus_error = ConsensusError::IoError(io_error);

        let mut error_chain = Vec::new();
        let mut current_error: &dyn Error = &consensus_error;

        loop {
            error_chain.push(current_error.to_string());
            if let Some(source) = current_error.source() {
                current_error = source;
            } else {
                break;
            }
        }

        assert!(error_chain.len() >= 1);
        assert!(error_chain[0].contains("IO error"));
    }

    #[test]
    fn test_consensus_result_type() {
        let success: ConsensusResult<i32> = Ok(42);
        assert!(success.is_ok());
        assert_eq!(success.unwrap(), 42);

        let failure: ConsensusResult<i32> =
            Err(ConsensusError::ValidationFailed("test".to_string()));
        assert!(failure.is_err());
    }

    #[test]
    fn test_error_with_unicode() {
        let error = ConsensusError::NetworkError("网络错误".to_string());
        assert!(error.to_string().contains("网络错误"));
    }

    #[test]
    fn test_error_with_large_string() {
        let large_message = "x".repeat(10000);
        let error = ConsensusError::GpuError(large_message.clone());
        assert!(error.to_string().contains(&large_message));
    }
}
