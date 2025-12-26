use thiserror::Error;

/// Error types for the policy engine
#[derive(Debug, Error)]
pub enum Error {
    /// Error parsing YAML policy document
    #[error("Failed to parse policy YAML: {0}")]
    ParseError(String),

    /// Error validating policy
    #[error("Policy validation failed: {0}")]
    ValidationError(String),

    /// Error evaluating policy
    #[error("Policy evaluation failed: {0}")]
    EvaluationError(String),

    /// Missing required field in policy
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid field value in policy
    #[error("Invalid field value for '{field}': {message}")]
    InvalidField { field: String, message: String },

    /// Pattern matching error
    #[error("Pattern matching error: {0}")]
    PatternError(String),

    /// Unsupported operator
    #[error("Unsupported operator: {0}")]
    UnsupportedOperator(String),

    /// Unsupported API version
    #[error("Unsupported API version: {0}")]
    UnsupportedVersion(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl From<serde_yaml::Error> for Error {
    fn from(err: serde_yaml::Error) -> Self {
        Error::ParseError(err.to_string())
    }
}

impl From<glob::PatternError> for Error {
    fn from(err: glob::PatternError) -> Self {
        Error::PatternError(err.to_string())
    }
}

impl From<regex::Error> for Error {
    fn from(err: regex::Error) -> Self {
        Error::PatternError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_parse_error() {
        let err = Error::ParseError("invalid yaml".to_string());
        assert_eq!(err.to_string(), "Failed to parse policy YAML: invalid yaml");
    }

    #[test]
    fn test_error_display_validation_error() {
        let err = Error::ValidationError("missing rules".to_string());
        assert_eq!(err.to_string(), "Policy validation failed: missing rules");
    }

    #[test]
    fn test_error_display_missing_field() {
        let err = Error::MissingField("metadata.name".to_string());
        assert_eq!(err.to_string(), "Missing required field: metadata.name");
    }

    #[test]
    fn test_error_display_invalid_field() {
        let err = Error::InvalidField {
            field: "spec.rules".to_string(),
            message: "must not be empty".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Invalid field value for 'spec.rules': must not be empty"
        );
    }
}
