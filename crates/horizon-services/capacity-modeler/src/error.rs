//! Error handling for the capacity modeler service.

pub use hpc_error::{HpcError, Result};

/// Extension trait for capacity modeler-specific error construction
pub trait CapacityErrorExt {
    /// Creates a model not trained error
    fn model_not_trained() -> HpcError {
        HpcError::invalid_input("model", "Model not trained")
    }

    /// Creates a training failed error
    fn training_failed(reason: impl Into<String>) -> HpcError {
        HpcError::internal(format!("Training failed: {}", reason.into()))
    }

    /// Creates a forecast failed error
    fn forecast_failed(reason: impl Into<String>) -> HpcError {
        HpcError::internal(format!("Forecast failed: {}", reason.into()))
    }

    /// Creates an invalid parameters error
    fn invalid_parameters(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("parameters", reason)
    }

    /// Creates an insufficient data error
    fn insufficient_data(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("data", reason)
    }
}

impl CapacityErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_not_trained() {
        let err = HpcError::model_not_trained();
        assert!(err.to_string().contains("not trained"));
    }

    #[test]
    fn test_invalid_parameters() {
        let err = HpcError::invalid_parameters("bad input");
        assert!(err.to_string().contains("parameters"));
    }
}
