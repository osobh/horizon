//! Error handling for the governor service.
//!
//! Uses `hpc_error::HpcError` as the unified error type with feature-gated
//! conversions for sqlx and axum.

pub use hpc_error::{HpcError, Result};

/// Extension trait for governor-specific error construction
pub trait GovernorErrorExt {
    /// Creates a policy not found error
    fn policy_not_found(policy_id: impl Into<String>) -> HpcError {
        HpcError::not_found("policy", policy_id)
    }

    /// Creates a policy already exists error
    fn policy_already_exists(policy_id: impl Into<String>) -> HpcError {
        HpcError::already_exists("policy", policy_id)
    }

    /// Creates an invalid policy content error
    fn invalid_policy_content(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("policy_content", reason)
    }

    /// Creates a policy evaluation error
    fn evaluation_error(reason: impl Into<String>) -> HpcError {
        HpcError::internal(format!("Policy evaluation error: {}", reason.into()))
    }
}

impl GovernorErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_not_found() {
        let err = HpcError::policy_not_found("policy-123");
        assert!(err.to_string().contains("policy"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_policy_already_exists() {
        let err = HpcError::policy_already_exists("policy-456");
        assert!(err.to_string().contains("policy"));
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn test_invalid_policy_content() {
        let err = HpcError::invalid_policy_content("missing required field");
        assert!(err.to_string().contains("missing required field"));
    }

    #[test]
    fn test_evaluation_error() {
        let err = HpcError::evaluation_error("timeout during evaluation");
        assert!(err.to_string().contains("evaluation"));
    }
}
