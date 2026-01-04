//! GPU runtime error types.

use thiserror::Error;

/// Result type for GPU runtime operations.
pub type Result<T> = std::result::Result<T, RuntimeError>;

/// GPU runtime errors.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// The requested backend is not available.
    #[error("Backend '{0}' is not available on this system")]
    BackendNotAvailable(String),

    /// No GPU backend is available.
    #[error("No GPU backend available. Consider using CPU fallback.")]
    NoGpuAvailable,

    /// Backend initialization failed.
    #[error("Failed to initialize {backend}: {message}")]
    InitializationFailed { backend: String, message: String },

    /// Feature not enabled.
    #[error(
        "Feature '{feature}' is not enabled. Enable it in Cargo.toml: features = [\"{feature}\"]"
    )]
    FeatureNotEnabled { feature: String },

    /// Operation not supported.
    #[error("Operation '{operation}' is not supported by backend '{backend}'")]
    OperationNotSupported { operation: String, backend: String },

    /// Metal-specific error.
    #[cfg(feature = "metal")]
    #[error("Metal error: {0}")]
    MetalError(#[from] stratoswarm_metal_core::error::MetalError),

    /// Generic I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

impl RuntimeError {
    /// Create a backend not available error.
    pub fn backend_not_available(backend: impl Into<String>) -> Self {
        RuntimeError::BackendNotAvailable(backend.into())
    }

    /// Create an initialization failed error.
    pub fn initialization_failed(backend: impl Into<String>, message: impl Into<String>) -> Self {
        RuntimeError::InitializationFailed {
            backend: backend.into(),
            message: message.into(),
        }
    }

    /// Create a feature not enabled error.
    pub fn feature_not_enabled(feature: impl Into<String>) -> Self {
        RuntimeError::FeatureNotEnabled {
            feature: feature.into(),
        }
    }

    /// Create an operation not supported error.
    pub fn operation_not_supported(
        operation: impl Into<String>,
        backend: impl Into<String>,
    ) -> Self {
        RuntimeError::OperationNotSupported {
            operation: operation.into(),
            backend: backend.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RuntimeError::backend_not_available("CUDA");
        assert!(err.to_string().contains("CUDA"));

        let err = RuntimeError::initialization_failed("Metal 3", "Device not found");
        assert!(err.to_string().contains("Metal 3"));
        assert!(err.to_string().contains("Device not found"));
    }

    #[test]
    fn test_feature_not_enabled() {
        let err = RuntimeError::feature_not_enabled("metal");
        assert!(err.to_string().contains("metal"));
        assert!(err.to_string().contains("Cargo.toml"));
    }
}
