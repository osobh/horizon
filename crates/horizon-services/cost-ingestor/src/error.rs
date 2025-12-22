//! Error handling for the cost ingestor service.
//!
//! Uses `hpc_error::HpcError` as the unified error type with feature-gated
//! conversions for sqlx, csv, reqwest, and axum.

pub use hpc_error::{HpcError, Result};

/// Extension trait for domain-specific error construction
pub trait IngestorErrorExt {
    /// Creates an invalid provider error
    fn invalid_provider(provider: impl Into<String>) -> HpcError {
        HpcError::invalid_input("provider", format!("invalid provider: {}", provider.into()))
    }

    /// Creates an invalid billing data error
    fn invalid_billing_data(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("billing_data", reason)
    }

    /// Creates a parse error
    fn parse_error(reason: impl Into<String>) -> HpcError {
        HpcError::Serialization(format!("parse error: {}", reason.into()))
    }
}

impl IngestorErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_provider() {
        let err = HpcError::invalid_provider("aws-invalid");
        assert!(err.to_string().contains("invalid provider"));
    }

    #[test]
    fn test_invalid_billing_data() {
        let err = HpcError::invalid_billing_data("Missing amount");
        assert!(err.to_string().contains("Missing amount"));
    }

    #[test]
    fn test_parse_error() {
        let err = HpcError::parse_error("Invalid date format");
        assert!(err.to_string().contains("parse error"));
    }

    #[test]
    fn test_sqlx_conversion() {
        // sqlx errors convert to Database errors via #[from]
        // This is tested via the hpc-error crate
    }

    #[test]
    fn test_csv_conversion() {
        // csv errors convert to Serialization errors via #[from]
        // This is tested via the hpc-error crate
    }
}
