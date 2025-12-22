//! Error handling for the executive intelligence service.

pub use hpc_error::{HpcError, Result};

/// Extension trait for executive-specific error construction
pub trait ExecutiveErrorExt {
    /// Creates a report not found error
    fn report_not_found(id: impl Into<String>) -> HpcError {
        HpcError::not_found("report", id)
    }

    /// Creates an invalid request error
    fn invalid_request(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("request", reason)
    }
}

impl ExecutiveErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_not_found() {
        let err = HpcError::report_not_found("test");
        assert!(err.to_string().contains("report"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_invalid_request() {
        let err = HpcError::invalid_request("bad input");
        assert!(err.to_string().contains("request"));
    }
}
