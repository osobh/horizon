//! Error types for Metal operations.

use thiserror::Error;

/// Result type for Metal operations.
pub type Result<T> = std::result::Result<T, MetalError>;

/// Errors that can occur during Metal operations.
#[derive(Error, Debug)]
pub enum MetalError {
    /// No Metal device available on this system.
    #[error("No Metal device available")]
    NoDevice,

    /// Failed to create a Metal resource.
    #[error("Failed to create {resource}: {message}")]
    CreationFailed {
        /// The type of resource that failed to create.
        resource: &'static str,
        /// Error message from Metal.
        message: String,
    },

    /// Shader compilation failed.
    #[error("Shader compilation failed: {0}")]
    ShaderCompilationFailed(String),

    /// Function not found in shader library.
    #[error("Function '{0}' not found in shader library")]
    FunctionNotFound(String),

    /// Buffer size mismatch.
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected size in bytes.
        expected: usize,
        /// Actual size in bytes.
        actual: usize,
    },

    /// Invalid tensor descriptor.
    #[error("Invalid tensor descriptor: {0}")]
    InvalidTensorDescriptor(String),

    /// Command encoding failed.
    #[error("Command encoding failed: {0}")]
    EncodingFailed(String),

    /// Command execution failed.
    #[error("Command execution failed: {0}")]
    ExecutionFailed(String),

    /// Synchronization timeout.
    #[error("Synchronization timeout after {0}ms")]
    Timeout(u64),

    /// Feature not supported on this Metal version.
    #[error("Feature '{0}' requires Metal {1}")]
    UnsupportedFeature(&'static str, &'static str),

    /// Internal Metal error.
    #[error("Internal Metal error: {0}")]
    Internal(String),
}

impl MetalError {
    /// Create a creation failed error.
    pub fn creation_failed(resource: &'static str, message: impl Into<String>) -> Self {
        Self::CreationFailed {
            resource,
            message: message.into(),
        }
    }

    /// Create a shader compilation error.
    pub fn shader_error(message: impl Into<String>) -> Self {
        Self::ShaderCompilationFailed(message.into())
    }

    /// Create an internal error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}
