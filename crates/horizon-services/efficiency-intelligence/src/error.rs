//! Error handling for the efficiency intelligence service.

pub use hpc_error::{HpcError, Result};

/// Extension trait for efficiency-specific error construction
pub trait EfficiencyErrorExt {
    /// Creates a detection not found error
    fn detection_not_found(id: impl Into<String>) -> HpcError {
        HpcError::not_found("detection", id)
    }

    /// Creates an invalid request error
    fn invalid_request(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("request", reason)
    }
}

impl EfficiencyErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_not_found() {
        let err = HpcError::detection_not_found("test");
        assert!(err.to_string().contains("detection"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_invalid_request() {
        let err = HpcError::invalid_request("bad input");
        assert!(err.to_string().contains("request"));
    }
}
