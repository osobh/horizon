//! Multi-region deployment error types

use thiserror::Error;

/// Multi-region error types
#[derive(Debug, Error)]
pub enum MultiRegionError {
    /// Data sovereignty violation
    #[error("Data sovereignty violation: {message}")]
    SovereigntyViolation { message: String },

    /// Region not available
    #[error("Region {region} is not available")]
    RegionUnavailable { region: String },

    /// Cross-region replication failure
    #[error("Cross-region replication failed: {reason}")]
    ReplicationFailure { reason: String },

    /// Tunnel connection error
    #[error("Secure tunnel connection failed: {reason}")]
    TunnelError { reason: String },

    /// Load balancer configuration error
    #[error("Load balancer configuration error: {reason}")]
    LoadBalancerError { reason: String },

    /// Compliance mapping error
    #[error("Compliance mapping error: {reason}")]
    ComplianceMappingError { reason: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    /// Network error
    #[error("Network error: {source}")]
    NetworkError {
        #[from]
        source: reqwest::Error,
    },

    /// JSON serialization/deserialization error
    #[error("JSON error: {source}")]
    JsonError {
        #[from]
        source: serde_json::Error,
    },

    /// I/O error
    #[error("I/O error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    /// Anti-entropy repair failure
    #[error("Anti-entropy failure: {reason}")]
    AntiEntropyFailure { reason: String },
}

/// Multi-region result type
pub type MultiRegionResult<T> = Result<T, MultiRegionError>;
