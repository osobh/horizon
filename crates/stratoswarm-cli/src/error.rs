//! Error types for the CLI

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CliError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(#[from] stratoswarm_dsl::DslError),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Command error: {0}")]
    Command(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, CliError>;

impl From<serde_json::Error> for CliError {
    fn from(err: serde_json::Error) -> Self {
        CliError::Serialization(err.to_string())
    }
}

impl From<serde_yaml::Error> for CliError {
    fn from(err: serde_yaml::Error) -> Self {
        CliError::Serialization(err.to_string())
    }
}

impl From<rustyline::error::ReadlineError> for CliError {
    fn from(err: rustyline::error::ReadlineError) -> Self {
        match err {
            rustyline::error::ReadlineError::Io(io_err) => CliError::Io(io_err),
            other => CliError::Command(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CliError::NotFound("test.swarm".to_string());
        assert_eq!(err.to_string(), "Not found: test.swarm");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let cli_err: CliError = io_err.into();
        assert!(matches!(cli_err, CliError::Io(_)));
    }
}
