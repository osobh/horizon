//! Error handling for the inventory service.

pub use hpc_error::{HpcError, Result};

/// Extension trait for inventory-specific error construction
pub trait InventoryErrorExt {
    /// Creates a validation error
    fn validation(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("validation", reason)
    }

    /// Creates an asset not found error
    fn asset_not_found(id: impl Into<String>) -> HpcError {
        HpcError::not_found("asset", id)
    }

    /// Creates a conflict error
    fn conflict(reason: impl Into<String>) -> HpcError {
        HpcError::conflict(reason)
    }

    /// Creates a bad request error
    fn bad_request(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("request", reason)
    }
}

impl InventoryErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error() {
        let err = HpcError::validation("invalid input");
        assert!(err.to_string().contains("validation"));
    }

    #[test]
    fn test_asset_not_found() {
        let err = HpcError::asset_not_found("asset-123");
        assert!(err.to_string().contains("asset"));
        assert!(err.to_string().contains("not found"));
    }
}
