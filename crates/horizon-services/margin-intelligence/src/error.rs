//! Error handling for the margin intelligence service.

pub use hpc_error::{HpcError, Result};

/// Extension trait for margin-specific error construction
pub trait MarginErrorExt {
    /// Creates a profile not found error
    fn profile_not_found(id: impl Into<String>) -> HpcError {
        HpcError::not_found("customer_profile", id)
    }

    /// Creates a simulation not found error
    fn simulation_not_found(id: impl Into<String>) -> HpcError {
        HpcError::not_found("simulation", id)
    }

    /// Creates an invalid segment error
    fn invalid_segment(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("segment", reason)
    }

    /// Creates an invalid calculation error
    fn invalid_calculation(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("calculation", reason)
    }

    /// Creates an invalid request error
    fn invalid_request(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("request", reason)
    }
}

impl MarginErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_not_found() {
        let err = HpcError::profile_not_found("cust-123");
        assert!(err.to_string().contains("customer_profile"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_simulation_not_found() {
        let err = HpcError::simulation_not_found("sim-456");
        assert!(err.to_string().contains("simulation"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_invalid_segment() {
        let err = HpcError::invalid_segment("invalid");
        assert!(err.to_string().contains("segment"));
    }

    #[test]
    fn test_invalid_calculation() {
        let err = HpcError::invalid_calculation("division by zero");
        assert!(err.to_string().contains("calculation"));
    }
}
