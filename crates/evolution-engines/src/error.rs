//! Error types for evolution engines

use thiserror::Error;

/// Evolution engine errors
#[derive(Error, Debug)]
pub enum EvolutionEngineError {
    /// Engine initialization failed
    #[error("Engine initialization failed: {message}")]
    InitializationFailed {
        /// Error message
        message: String,
    },

    /// Initialization error
    #[error("Initialization error: {message}")]
    InitializationError {
        /// Error message
        message: String,
    },

    /// Invalid configuration
    #[error("Invalid configuration: {message}")]
    InvalidConfiguration {
        /// Error message
        message: String,
    },

    /// Evolution step failed
    #[error("Evolution step failed: {message}")]
    EvolutionStepFailed {
        /// Error message
        message: String,
    },

    /// Agent synthesis error
    #[error("Agent synthesis failed: {0}")]
    SynthesisError(#[from] stratoswarm_synthesis::error::SynthesisError),

    /// Agent core error
    #[error("Agent core error: {0}")]
    AgentCoreError(#[from] stratoswarm_agent_core::error::AgentError),

    /// Fitness evaluation failed
    #[error("Fitness evaluation failed: {message}")]
    FitnessEvaluationFailed {
        /// Error message
        message: String,
    },

    /// Population management error
    #[error("Population error: {message}")]
    PopulationError {
        /// Error message
        message: String,
    },

    /// Convergence failure
    #[error("Failed to converge after {iterations} iterations")]
    ConvergenceFailed {
        /// Number of iterations attempted
        iterations: u32,
    },

    /// Evolution error (generic evolution process failure)
    #[error("Evolution error: {message}")]
    EvolutionError {
        /// Error message
        message: String,
    },

    /// Resource exhaustion
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted {
        /// Resource that was exhausted
        resource: String,
    },

    /// Communication error
    #[error("Communication error: {message}")]
    CommunicationError {
        /// Error message
        message: String,
    },

    /// Distributed computing error
    #[error("Distributed error: {0}")]
    DistributedError(String),

    /// Search error
    #[error("Search error: {message}")]
    SearchError {
        /// Error message
        message: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Lock poisoned error
    #[error("Lock poisoned: {message}")]
    LockPoisoned {
        /// Error message
        message: String,
    },

    /// System time error
    #[error("System time error: {message}")]
    SystemTimeError {
        /// Error message
        message: String,
    },

    /// Other error
    #[error("Evolution engine error: {0}")]
    Other(String),
}

impl<T> From<std::sync::PoisonError<T>> for EvolutionEngineError {
    #[cold]
    fn from(err: std::sync::PoisonError<T>) -> Self {
        EvolutionEngineError::LockPoisoned {
            message: err.to_string(),
        }
    }
}

impl From<std::time::SystemTimeError> for EvolutionEngineError {
    #[cold]
    fn from(err: std::time::SystemTimeError) -> Self {
        EvolutionEngineError::SystemTimeError {
            message: err.to_string(),
        }
    }
}

/// Result type for evolution engine operations
pub type EvolutionEngineResult<T> = Result<T, EvolutionEngineError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = EvolutionEngineError::InitializationFailed {
            message: "Test error".to_string(),
        };
        assert_eq!(err.to_string(), "Engine initialization failed: Test error");
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let engine_err: EvolutionEngineError = io_err.into();
        assert!(matches!(engine_err, EvolutionEngineError::IoError(_)));
    }
}
