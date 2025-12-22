//! Error handling for the initiative tracker service.

pub use hpc_error::{HpcError, Result};

/// Extension trait for initiative-specific error construction
pub trait InitiativeErrorExt {
    /// Creates an initiative not found error
    fn initiative_not_found(id: impl Into<String>) -> HpcError {
        HpcError::not_found("initiative", id)
    }

    /// Creates an invalid request error
    fn invalid_request(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("request", reason)
    }
}

impl InitiativeErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initiative_not_found() {
        let err = HpcError::initiative_not_found("test");
        assert!(err.to_string().contains("initiative"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_invalid_request() {
        let err = HpcError::invalid_request("bad input");
        assert!(err.to_string().contains("request"));
    }
}
