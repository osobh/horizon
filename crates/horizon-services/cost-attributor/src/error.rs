//! Error handling for the cost attributor service.

pub use hpc_error::{HpcError, Result};

/// Extension trait for attributor-specific error construction
pub trait AttributorErrorExt {
    /// Creates an invalid attribution data error
    fn invalid_attribution_data(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("attribution_data", reason)
    }

    /// Creates an invalid pricing data error
    fn invalid_pricing_data(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("pricing_data", reason)
    }

    /// Creates a pricing not found error
    fn pricing_not_found(gpu_type: impl Into<String>) -> HpcError {
        HpcError::not_found("gpu_pricing", gpu_type)
    }

    /// Creates a job not found error
    fn job_not_found(id: impl Into<String>) -> HpcError {
        HpcError::not_found("job", id)
    }

    /// Creates an attribution not found error
    fn attribution_not_found(id: impl Into<String>) -> HpcError {
        HpcError::not_found("attribution", id)
    }

    /// Creates an accuracy threshold exceeded error
    fn accuracy_threshold_exceeded(threshold: rust_decimal::Decimal) -> HpcError {
        HpcError::invalid_input(
            "accuracy",
            format!("Accuracy threshold exceeded: {}%", threshold),
        )
    }

    /// Creates an invalid pricing model error
    fn invalid_pricing_model(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("pricing_model", reason)
    }

    /// Creates an invalid time range error
    fn invalid_time_range(start: impl Into<String>, end: impl Into<String>) -> HpcError {
        HpcError::invalid_input(
            "time_range",
            format!("Invalid time range: start={}, end={}", start.into(), end.into()),
        )
    }

    /// Creates a calculation error
    fn calculation_error(reason: impl Into<String>) -> HpcError {
        HpcError::internal(format!("Calculation error: {}", reason.into()))
    }
}

impl AttributorErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_attribution_data() {
        let err = HpcError::invalid_attribution_data("test error");
        assert!(err.to_string().contains("attribution_data"));
    }

    #[test]
    fn test_pricing_not_found() {
        let err = HpcError::pricing_not_found("A100");
        assert!(err.to_string().contains("gpu_pricing"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_accuracy_threshold() {
        use rust_decimal::Decimal;
        let err = HpcError::accuracy_threshold_exceeded(Decimal::new(752, 2));
        assert!(err.to_string().contains("7.52"));
    }

    #[test]
    fn test_invalid_time_range() {
        let err = HpcError::invalid_time_range("2024-01-01", "2023-12-31");
        assert!(err.to_string().contains("time_range"));
    }

    #[test]
    fn test_job_not_found() {
        let job_id = uuid::Uuid::new_v4().to_string();
        let err = HpcError::job_not_found(&job_id);
        assert!(err.to_string().contains("job"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_invalid_pricing_model() {
        let err = HpcError::invalid_pricing_model("invalid_model");
        assert!(err.to_string().contains("pricing_model"));
    }
}
