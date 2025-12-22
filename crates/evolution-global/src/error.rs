//! Evolution global deployment error types

use thiserror::Error;

/// Evolution global deployment error types
#[derive(Debug, Error)]
pub enum EvolutionGlobalError {
    /// Evolution coordination failed
    #[error("Evolution coordination failed: {operation} - {reason}")]
    CoordinationFailed { operation: String, reason: String },

    /// AI safety compliance violation
    #[error("AI safety compliance violation: {safety_rule} in {region} - {details}")]
    SafetyViolation {
        safety_rule: String,
        region: String,
        details: String,
    },

    /// Secure multi-party protocol failed
    #[error("Secure multi-party protocol failed: {protocol} - {phase} - {reason}")]
    MultiPartyProtocolFailed {
        protocol: String,
        phase: String,
        reason: String,
    },

    /// Evolution intrusion detected
    #[error(
        "Evolution intrusion detected: {threat_type} from {source_location} - severity: {severity}"
    )]
    IntrusionDetected {
        threat_type: String,
        source_location: String,
        severity: String,
    },

    /// Consensus engine failure
    #[error("Consensus engine failure: {consensus_type} - {reason}")]
    ConsensusFailure {
        consensus_type: String,
        reason: String,
    },

    /// Cross-region synchronization failed
    #[error("Cross-region sync failed between {region1} and {region2} - {reason}")]
    CrossRegionSyncFailed {
        region1: String,
        region2: String,
        reason: String,
    },

    /// Evolution monitoring failed
    #[error("Evolution monitoring failed: {monitor_type} - {reason}")]
    MonitoringFailed {
        monitor_type: String,
        reason: String,
    },

    /// Evolution model validation failed
    #[error("Evolution model validation failed: {model_id} - {validation_error}")]
    ModelValidationFailed {
        model_id: String,
        validation_error: String,
    },

    /// Cryptographic operation failed
    #[error("Cryptographic operation failed: {operation} - {reason}")]
    CryptographicError { operation: String, reason: String },

    /// Regional compliance mismatch
    #[error("Regional compliance mismatch: {region} requires {required_compliance} but {current_compliance} provided")]
    ComplianceMismatch {
        region: String,
        required_compliance: String,
        current_compliance: String,
    },

    /// Evolution timeout
    #[error("Evolution timeout: operation {operation} exceeded {timeout_ms}ms")]
    EvolutionTimeout { operation: String, timeout_ms: u64 },

    /// Insufficient participants
    #[error("Insufficient participants for {protocol}: need {required}, have {available}")]
    InsufficientParticipants {
        protocol: String,
        required: usize,
        available: usize,
    },

    /// Configuration error
    #[error("Configuration error: {parameter} - {reason}")]
    ConfigurationError { parameter: String, reason: String },

    /// Network error
    #[error("Network error: {endpoint} - {details}")]
    NetworkError { endpoint: String, details: String },

    /// I/O error
    #[error("I/O error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    /// JSON error
    #[error("JSON error: {source}")]
    JsonError {
        #[from]
        source: serde_json::Error,
    },
}

/// Evolution global deployment result type
pub type EvolutionGlobalResult<T> = Result<T, EvolutionGlobalError>;
