//! Error handling for the vendor intelligence service.

pub use hpc_error::{HpcError, Result};

/// Extension trait for vendor-specific error construction
pub trait VendorErrorExt {
    /// Creates a vendor not found error
    fn vendor_not_found(id: impl Into<String>) -> HpcError {
        HpcError::not_found("vendor", id)
    }

    /// Creates an invalid request error
    fn invalid_request(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("request", reason)
    }
}

impl VendorErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vendor_not_found() {
        let err = HpcError::vendor_not_found("test");
        assert!(err.to_string().contains("vendor"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_invalid_request() {
        let err = HpcError::invalid_request("bad input");
        assert!(err.to_string().contains("request"));
    }
}
