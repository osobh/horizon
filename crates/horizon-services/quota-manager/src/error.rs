//! Error handling for the quota manager service.

pub use hpc_error::{HpcError, Result};

/// Extension trait for quota manager-specific error construction
pub trait QuotaErrorExt {
    /// Creates a quota not found error
    fn quota_not_found(id: impl Into<String>) -> HpcError {
        HpcError::not_found("quota", id)
    }

    /// Creates a quota already exists error
    fn quota_already_exists(id: impl Into<String>) -> HpcError {
        HpcError::already_exists("quota", id)
    }

    /// Creates a quota exceeded error
    fn quota_exceeded(reason: impl Into<String>) -> HpcError {
        HpcError::resource_exhausted("quota", reason)
    }

    /// Creates an invalid configuration error
    fn invalid_configuration(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("configuration", reason)
    }

    /// Creates an optimistic lock conflict error
    fn optimistic_lock_conflict() -> HpcError {
        HpcError::internal("Optimistic lock conflict: version mismatch")
    }

    /// Creates an invalid hierarchy error
    fn invalid_hierarchy(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("hierarchy", reason)
    }

    /// Creates an allocation not found error
    fn allocation_not_found(id: impl Into<String>) -> HpcError {
        HpcError::not_found("allocation", id)
    }
}

impl QuotaErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quota_not_found() {
        let err = HpcError::quota_not_found("test");
        assert!(err.to_string().contains("quota"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_quota_exceeded() {
        let err = HpcError::quota_exceeded("limit reached");
        assert!(err.to_string().contains("quota"));
    }
}
