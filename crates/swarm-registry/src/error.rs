use thiserror::Error;

#[derive(Debug, Error)]
pub enum SwarmRegistryError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Invalid image format: {0}")]
    InvalidFormat(String),

    #[error("Image not found: {0}")]
    ImageNotFound(String),

    #[error("Build failed: {0}")]
    BuildFailed(String),

    #[error("Conversion failed: {0}")]
    ConversionFailed(String),

    #[error("Registry error: {0}")]
    Registry(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    #[error("Streaming error: {0}")]
    StreamingError(String),

    #[error("P2P error: {0}")]
    P2P(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, SwarmRegistryError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SwarmRegistryError::ImageNotFound("ubuntu:25.04".to_string());
        assert_eq!(err.to_string(), "Image not found: ubuntu:25.04");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = SwarmRegistryError::from(io_err);
        assert!(matches!(err, SwarmRegistryError::Io(_)));
    }
}
