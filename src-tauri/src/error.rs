//! Error Types for Horizon
//!
//! This module provides typed error handling for the Horizon application.
//! All bridges and commands should use `HorizonError` instead of raw `String` errors.

use thiserror::Error;

/// The main error type for Horizon operations.
#[derive(Debug, Error)]
pub enum HorizonError {
    /// A resource was not found (e.g., policy, quota, job, session).
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Invalid configuration or input parameters.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Bridge operation failed.
    #[error("Bridge operation failed: {0}")]
    BridgeError(String),

    /// State lock was poisoned (should be rare with RwLock).
    #[error("State lock error")]
    StateLock,

    /// Failed to parse or serialize data.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Authentication or authorization failure.
    #[error("Authentication error: {0}")]
    AuthError(String),

    /// An operation timed out.
    #[error("Operation timed out")]
    Timeout,

    /// IO error occurred.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Convert HorizonError to String for Tauri command compatibility.
///
/// Tauri commands require errors to implement `Into<String>`.
impl From<HorizonError> for String {
    fn from(err: HorizonError) -> Self {
        err.to_string()
    }
}

/// Convert serde_json errors to HorizonError.
impl From<serde_json::Error> for HorizonError {
    fn from(err: serde_json::Error) -> Self {
        HorizonError::SerializationError(err.to_string())
    }
}
