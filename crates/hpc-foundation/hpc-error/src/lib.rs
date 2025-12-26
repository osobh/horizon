//! # HPC Platform Error Handling
//!
//! This crate provides a unified error type for the HPC Platform.
//! It uses `thiserror` for ergonomic error definitions and supports conversion
//! to `anyhow::Error` for application code.
//!
//! ## Features
//!
//! - **Comprehensive Error Variants**: Covers all major error categories in the platform
//! - **gRPC Integration**: Seamless conversion to/from `tonic::Status`
//! - **Error Categorization**: Helper methods to classify errors (retriable, client errors)
//! - **Type Safety**: Strong typing with thiserror-derived implementations
//! - **Context Chaining**: Works seamlessly with anyhow's context system
//!
//! ## Usage
//!
//! ```rust
//! use hpc_error::{HpcError, Result};
//!
//! fn operation() -> Result<String> {
//!     Err(HpcError::Database("connection failed".to_string()))
//! }
//!
//! // Convert to anyhow for application code
//! use anyhow::Context;
//!
//! fn app_code() -> anyhow::Result<()> {
//!     let value = operation()
//!         .context("failed to perform operation")?;
//!     Ok(())
//! }
//! ```

use thiserror::Error;

/// The main error type for the HPC Platform.
///
/// This enum covers all error categories that can occur across HPC services.
/// It implements `std::error::Error` via thiserror and can be converted to
/// `anyhow::Error` or `tonic::Status`.
#[derive(Error, Debug)]
pub enum HpcError {
    /// Configuration-related errors (invalid config, missing fields, etc.)
    #[error("configuration error: {0}")]
    Config(String),

    /// RPC/gRPC errors from service-to-service communication
    #[error("RPC error: {0}")]
    Rpc(#[source] Box<tonic::Status>),

    /// Database errors (connection failures, query errors, etc.)
    #[error("database error: {0}")]
    Database(String),

    /// IO errors (file operations, network IO, etc.)
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization errors (JSON, protobuf, etc.)
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Telemetry collection errors (metric gathering, export failures, etc.)
    #[error("telemetry error: {0}")]
    Telemetry(String),

    /// Policy engine errors (evaluation failures, syntax errors, etc.)
    #[error("policy error: {0}")]
    Policy(String),

    /// Cryptographic operation errors (signing, verification, hashing, etc.)
    #[error("cryptography error: {0}")]
    Crypto(String),

    /// Authentication/authorization errors
    #[error("authentication error: {0}")]
    Auth(String),

    /// Storage errors (Arrow/Parquet, object storage, etc.)
    #[error("storage error: {0}")]
    Storage(String),

    /// Network errors (connection failures, timeouts, etc.)
    #[error("network error: {0}")]
    Network(String),

    /// Timeout errors (operation deadlines exceeded)
    #[error("timeout: {0}")]
    Timeout(String),

    /// Resource not found errors
    #[error("{resource_type} not found: {resource_id}")]
    NotFound {
        resource_type: String,
        resource_id: String,
    },

    /// Resource already exists errors
    #[error("{resource_type} already exists: {resource_id}")]
    AlreadyExists {
        resource_type: String,
        resource_id: String,
    },

    /// Invalid input validation errors
    #[error("invalid input for field '{field}': {reason}")]
    InvalidInput { field: String, reason: String },

    /// Permission denied errors
    #[error("permission denied: cannot {action} on {resource}")]
    PermissionDenied { action: String, resource: String },

    /// Resource exhausted errors (quota limits, capacity, etc.)
    #[error("resource exhausted: {resource} (limit: {limit})")]
    ResourceExhausted { resource: String, limit: String },

    /// Scheduling errors (job placement, queue, etc.)
    #[error("scheduling error: {0}")]
    Scheduling(String),

    /// Cost attribution errors
    #[error("cost error: {0}")]
    Cost(String),

    /// Agent errors (autonomous agent failures)
    #[error("agent error: {0}")]
    Agent(String),

    /// GPU/compute resource errors
    #[error("GPU error: {0}")]
    Gpu(String),

    /// Consensus errors (BFT, distributed coordination)
    #[error("consensus error: {0}")]
    Consensus(String),

    /// Evolution engine errors
    #[error("evolution error: {0}")]
    Evolution(String),

    /// Internal errors (bugs, unexpected states, etc.)
    #[error("internal error: {0}")]
    Internal(String),

    /// Unknown/uncategorized errors
    #[error("unknown error: {0}")]
    Unknown(String),
}

/// Type alias for Results using HpcError
pub type Result<T> = std::result::Result<T, HpcError>;

// Conversion from tonic::Status
impl From<tonic::Status> for HpcError {
    fn from(status: tonic::Status) -> Self {
        HpcError::Rpc(Box::new(status))
    }
}

// Conversion from serde_json::Error
impl From<serde_json::Error> for HpcError {
    fn from(err: serde_json::Error) -> Self {
        HpcError::Serialization(err.to_string())
    }
}

// Conversion to tonic::Status for gRPC error responses
impl From<HpcError> for tonic::Status {
    fn from(err: HpcError) -> Self {
        match err {
            HpcError::Config(msg) => {
                tonic::Status::invalid_argument(format!("configuration error: {msg}"))
            }
            HpcError::Rpc(status) => *status,
            HpcError::Database(msg) => {
                tonic::Status::internal(format!("database error: {msg}"))
            }
            HpcError::Io(err) => tonic::Status::internal(format!("IO error: {err}")),
            HpcError::Serialization(msg) => {
                tonic::Status::invalid_argument(format!("serialization error: {msg}"))
            }
            HpcError::Telemetry(msg) => {
                tonic::Status::internal(format!("telemetry error: {msg}"))
            }
            HpcError::Policy(msg) => {
                tonic::Status::failed_precondition(format!("policy error: {msg}"))
            }
            HpcError::Crypto(msg) => {
                tonic::Status::internal(format!("cryptography error: {msg}"))
            }
            HpcError::Auth(msg) => {
                tonic::Status::unauthenticated(format!("authentication error: {msg}"))
            }
            HpcError::Storage(msg) => {
                tonic::Status::internal(format!("storage error: {msg}"))
            }
            HpcError::Network(msg) => {
                tonic::Status::unavailable(format!("network error: {msg}"))
            }
            HpcError::Timeout(msg) => tonic::Status::deadline_exceeded(format!("timeout: {msg}")),
            HpcError::NotFound {
                resource_type,
                resource_id,
            } => tonic::Status::not_found(format!("{resource_type} not found: {resource_id}")),
            HpcError::AlreadyExists {
                resource_type,
                resource_id,
            } => tonic::Status::already_exists(format!(
                "{resource_type} already exists: {resource_id}"
            )),
            HpcError::InvalidInput { field, reason } => {
                tonic::Status::invalid_argument(format!("invalid input for field '{field}': {reason}"))
            }
            HpcError::PermissionDenied { action, resource } => {
                tonic::Status::permission_denied(format!(
                    "permission denied: cannot {action} on {resource}"
                ))
            }
            HpcError::ResourceExhausted { resource, limit } => tonic::Status::resource_exhausted(
                format!("resource exhausted: {resource} (limit: {limit})"),
            ),
            HpcError::Scheduling(msg) => {
                tonic::Status::failed_precondition(format!("scheduling error: {msg}"))
            }
            HpcError::Cost(msg) => tonic::Status::internal(format!("cost error: {msg}")),
            HpcError::Agent(msg) => tonic::Status::internal(format!("agent error: {msg}")),
            HpcError::Gpu(msg) => tonic::Status::internal(format!("GPU error: {msg}")),
            HpcError::Consensus(msg) => {
                tonic::Status::internal(format!("consensus error: {msg}"))
            }
            HpcError::Evolution(msg) => {
                tonic::Status::internal(format!("evolution error: {msg}"))
            }
            HpcError::Internal(msg) => tonic::Status::internal(format!("internal error: {msg}")),
            HpcError::Unknown(msg) => tonic::Status::unknown(format!("unknown error: {msg}")),
        }
    }
}

// Optional feature: sqlx database errors
#[cfg(feature = "sqlx")]
impl From<sqlx::Error> for HpcError {
    fn from(err: sqlx::Error) -> Self {
        HpcError::Database(err.to_string())
    }
}

// Optional feature: CSV parsing errors
#[cfg(feature = "csv")]
impl From<csv::Error> for HpcError {
    fn from(err: csv::Error) -> Self {
        HpcError::Serialization(format!("CSV error: {err}"))
    }
}

// Optional feature: HTTP client errors
#[cfg(feature = "reqwest")]
impl From<reqwest::Error> for HpcError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            HpcError::Timeout(err.to_string())
        } else if err.is_connect() {
            HpcError::Network(format!("connection error: {err}"))
        } else {
            HpcError::Network(err.to_string())
        }
    }
}

// Optional feature: Axum HTTP response conversion
#[cfg(feature = "axum")]
impl axum::response::IntoResponse for HpcError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;
        use axum::Json;

        let (status, error_type) = match &self {
            HpcError::NotFound { .. } => (StatusCode::NOT_FOUND, "not_found"),
            HpcError::AlreadyExists { .. } => (StatusCode::CONFLICT, "already_exists"),
            HpcError::InvalidInput { .. } => (StatusCode::BAD_REQUEST, "invalid_input"),
            HpcError::Config(_) => (StatusCode::BAD_REQUEST, "config_error"),
            HpcError::Serialization(_) => (StatusCode::BAD_REQUEST, "serialization_error"),
            HpcError::Auth(_) => (StatusCode::UNAUTHORIZED, "auth_error"),
            HpcError::PermissionDenied { .. } => (StatusCode::FORBIDDEN, "permission_denied"),
            HpcError::Timeout(_) => (StatusCode::GATEWAY_TIMEOUT, "timeout"),
            HpcError::Network(_) => (StatusCode::BAD_GATEWAY, "network_error"),
            HpcError::ResourceExhausted { .. } => {
                (StatusCode::TOO_MANY_REQUESTS, "resource_exhausted")
            }
            HpcError::Database(_)
            | HpcError::Storage(_)
            | HpcError::Gpu(_)
            | HpcError::Consensus(_)
            | HpcError::Evolution(_)
            | HpcError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "unknown_error"),
        };

        let body = Json(serde_json::json!({
            "error": error_type,
            "message": self.to_string(),
        }));

        (status, body).into_response()
    }
}

impl HpcError {
    /// Determines if this error is retriable.
    ///
    /// Retriable errors are transient failures that may succeed on retry,
    /// such as network errors, timeouts, or resource exhaustion.
    #[must_use]
    pub fn is_retriable(&self) -> bool {
        match self {
            // Transient errors that may succeed on retry
            HpcError::Network(_)
            | HpcError::Timeout(_)
            | HpcError::ResourceExhausted { .. }
            | HpcError::Database(_)
            | HpcError::Storage(_)
            | HpcError::Consensus(_) => true,

            // gRPC status codes that are retriable
            HpcError::Rpc(status) => matches!(
                status.code(),
                tonic::Code::Unavailable
                    | tonic::Code::DeadlineExceeded
                    | tonic::Code::ResourceExhausted
                    | tonic::Code::Aborted
            ),

            // Permanent errors that won't succeed on retry
            _ => false,
        }
    }

    /// Determines if this error is a resource exhausted error.
    #[must_use]
    pub fn is_resource_exhausted(&self) -> bool {
        matches!(self, HpcError::ResourceExhausted { .. })
    }

    /// Determines if this error is a client error (4xx-equivalent).
    ///
    /// Client errors indicate that the request was invalid and should not
    /// be retried without modification.
    #[must_use]
    pub fn is_client_error(&self) -> bool {
        match self {
            // Client-side errors (invalid requests, not found, etc.)
            HpcError::Config(_)
            | HpcError::InvalidInput { .. }
            | HpcError::NotFound { .. }
            | HpcError::AlreadyExists { .. }
            | HpcError::PermissionDenied { .. }
            | HpcError::Auth(_)
            | HpcError::Serialization(_) => true,

            // Check gRPC status code
            HpcError::Rpc(status) => matches!(
                status.code(),
                tonic::Code::InvalidArgument
                    | tonic::Code::NotFound
                    | tonic::Code::AlreadyExists
                    | tonic::Code::PermissionDenied
                    | tonic::Code::Unauthenticated
                    | tonic::Code::FailedPrecondition
            ),

            // Server-side errors
            _ => false,
        }
    }

    // ==========================================
    // Convenience constructors
    // ==========================================

    /// Creates a not found error
    #[must_use]
    pub fn not_found(resource_type: impl Into<String>, resource_id: impl Into<String>) -> Self {
        HpcError::NotFound {
            resource_type: resource_type.into(),
            resource_id: resource_id.into(),
        }
    }

    /// Creates an invalid input error
    #[must_use]
    pub fn invalid_input(field: impl Into<String>, reason: impl Into<String>) -> Self {
        HpcError::InvalidInput {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// Creates a permission denied error
    #[must_use]
    pub fn permission_denied(action: impl Into<String>, resource: impl Into<String>) -> Self {
        HpcError::PermissionDenied {
            action: action.into(),
            resource: resource.into(),
        }
    }

    /// Creates a configuration error
    #[must_use]
    pub fn config(msg: impl Into<String>) -> Self {
        HpcError::Config(msg.into())
    }

    /// Creates a database error
    #[must_use]
    pub fn database(msg: impl Into<String>) -> Self {
        HpcError::Database(msg.into())
    }

    /// Creates a network error
    #[must_use]
    pub fn network(msg: impl Into<String>) -> Self {
        HpcError::Network(msg.into())
    }

    /// Creates a timeout error
    #[must_use]
    pub fn timeout(msg: impl Into<String>) -> Self {
        HpcError::Timeout(msg.into())
    }

    /// Creates an internal error
    #[must_use]
    pub fn internal(msg: impl Into<String>) -> Self {
        HpcError::Internal(msg.into())
    }

    /// Creates an agent error
    #[must_use]
    pub fn agent(msg: impl Into<String>) -> Self {
        HpcError::Agent(msg.into())
    }

    /// Creates a GPU error
    #[must_use]
    pub fn gpu(msg: impl Into<String>) -> Self {
        HpcError::Gpu(msg.into())
    }

    /// Creates a scheduling error
    #[must_use]
    pub fn scheduling(msg: impl Into<String>) -> Self {
        HpcError::Scheduling(msg.into())
    }

    /// Creates an already exists error
    #[must_use]
    pub fn already_exists(
        resource_type: impl Into<String>,
        resource_id: impl Into<String>,
    ) -> Self {
        HpcError::AlreadyExists {
            resource_type: resource_type.into(),
            resource_id: resource_id.into(),
        }
    }

    /// Creates a resource exhausted error
    #[must_use]
    pub fn resource_exhausted(resource: impl Into<String>, limit: impl Into<String>) -> Self {
        HpcError::ResourceExhausted {
            resource: resource.into(),
            limit: limit.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_implements_std_error() {
        let err = HpcError::Internal("test".to_string());
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<HpcError>();
        assert_sync::<HpcError>();
    }

    #[test]
    fn test_result_alias() {
        fn returns_result() -> Result<i32> {
            Ok(42)
        }
        assert!(returns_result().is_ok());
    }

    #[test]
    fn test_retriable_errors() {
        assert!(HpcError::Network("timeout".into()).is_retriable());
        assert!(HpcError::Timeout("deadline".into()).is_retriable());
        assert!(!HpcError::Auth("invalid".into()).is_retriable());
    }

    #[test]
    fn test_client_errors() {
        assert!(HpcError::not_found("job", "123").is_client_error());
        assert!(HpcError::invalid_input("name", "too long").is_client_error());
        assert!(!HpcError::Internal("bug".into()).is_client_error());
    }
}
