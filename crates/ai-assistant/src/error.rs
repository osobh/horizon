//! Error types for the AI Assistant

use thiserror::Error;

pub type AssistantResult<T> = Result<T, AssistantError>;

#[derive(Debug, Error)]
pub enum AssistantError {
    #[error("Failed to parse natural language input: {0}")]
    ParseError(String),

    #[error("Failed to generate command: {0}")]
    CommandGenerationError(String),

    #[error("Query execution failed: {0}")]
    QueryError(String),

    #[error("Learning system error: {0}")]
    LearningError(String),

    #[error("Model not available: {0}")]
    ModelError(String),

    #[error("Confidence too low: {confidence} < {threshold}")]
    LowConfidence { confidence: f32, threshold: f32 },

    #[error("Template rendering failed: {0}")]
    TemplateError(String),

    #[error("Embedding generation failed: {0}")]
    EmbeddingError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AssistantError::ParseError("invalid syntax".to_string());
        assert_eq!(
            err.to_string(),
            "Failed to parse natural language input: invalid syntax"
        );

        let err = AssistantError::LowConfidence {
            confidence: 0.5,
            threshold: 0.8,
        };
        assert_eq!(err.to_string(), "Confidence too low: 0.5 < 0.8");
    }
}
