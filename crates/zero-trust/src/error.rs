//! Zero-trust security error types

use thiserror::Error;

/// Zero-trust security error types
#[derive(Debug, Error)]
pub enum ZeroTrustError {
    /// Identity verification failed
    #[error("Identity verification failed: {message}")]
    IdentityVerificationFailed { message: String },

    /// Device trust validation failed
    #[error("Device trust validation failed: {reason}")]
    DeviceTrustFailed { reason: String },

    /// Network policy violation
    #[error("Network policy violation: {policy} - {reason}")]
    NetworkPolicyViolation { policy: String, reason: String },

    /// Behavioral anomaly detected
    #[error("Behavioral anomaly detected: {anomaly_type} - {details}")]
    BehavioralAnomaly {
        anomaly_type: String,
        details: String,
    },

    /// Risk score too high
    #[error("Risk score {score} exceeds threshold {threshold}")]
    RiskScoreTooHigh { score: f64, threshold: f64 },

    /// Session expired or invalid
    #[error("Session invalid: {reason}")]
    SessionInvalid { reason: String },

    /// Attestation failure
    #[error("Attestation failed: {component} - {reason}")]
    AttestationFailed { component: String, reason: String },

    /// Authentication token invalid
    #[error("Authentication token invalid: {reason}")]
    TokenInvalid { reason: String },

    /// Certificate validation failed
    #[error("Certificate validation failed: {reason}")]
    CertificateValidationFailed { reason: String },

    /// Cryptographic operation failed
    #[error("Cryptographic operation failed: {operation} - {reason}")]
    CryptographicError { operation: String, reason: String },

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

    /// JWT token error
    #[error("JWT error: {message}")]
    JwtError { message: String },

    /// I/O error
    #[error("I/O error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },
}

/// Zero-trust result type
pub type ZeroTrustResult<T> = Result<T, ZeroTrustError>;
