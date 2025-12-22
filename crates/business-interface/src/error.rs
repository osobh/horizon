//! Error types for business interface operations

use thiserror::Error;

/// Result type for business interface operations
pub type BusinessResult<T> = Result<T, BusinessError>;

/// Comprehensive error types for business interface operations
#[derive(Error, Debug, Clone)]
pub enum BusinessError {
    /// Goal parsing failed
    #[error("Goal parsing failed: {message}")]
    GoalParsingFailed { message: String },

    /// Goal validation failed
    #[error("Goal validation failed: {reason}")]
    GoalValidationFailed { reason: String },

    /// Safety validation failed
    #[error("Safety validation failed: {reason}")]
    SafetyValidationFailed { reason: String },

    /// Safety check failed
    #[error("Safety check failed: {check}: {details}")]
    SafetyCheckFailed { check: String, details: String },

    /// Resource estimation failed
    #[error("Resource estimation failed: {resource}: {reason}")]
    ResourceEstimationFailed { resource: String, reason: String },

    /// LLM integration error
    #[error("LLM integration error: {service}: {error}")]
    LlmIntegrationError { service: String, error: String },

    /// Progress tracking error
    #[error("Progress tracking error: {goal_id}: {operation}")]
    ProgressTrackingError { goal_id: String, operation: String },

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Agent operation error
    #[error("Agent operation failed: {agent_id}: {operation}: {reason}")]
    AgentOperationError {
        agent_id: String,
        operation: String,
        reason: String,
    },

    /// Result explanation error
    #[error("Result explanation failed: {goal_id}: {reason}")]
    ResultExplanationError { goal_id: String, reason: String },

    /// Timeout error
    #[error("Operation timed out after {duration:?}")]
    Timeout { duration: std::time::Duration },
}

impl From<std::io::Error> for BusinessError {
    fn from(err: std::io::Error) -> Self {
        BusinessError::ConfigurationError(format!("IO error: {}", err))
    }
}

impl From<serde_json::Error> for BusinessError {
    fn from(err: serde_json::Error) -> Self {
        BusinessError::ConfigurationError(format!("JSON error: {}", err))
    }
}

impl From<async_openai::error::OpenAIError> for BusinessError {
    fn from(err: async_openai::error::OpenAIError) -> Self {
        BusinessError::LlmIntegrationError {
            service: "OpenAI".to_string(),
            error: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_goal_parsing_error() {
        let error = BusinessError::GoalParsingFailed {
            message: "Invalid syntax".to_string(),
        };
        assert!(error.to_string().contains("Goal parsing failed"));
        assert!(error.to_string().contains("Invalid syntax"));
    }

    #[test]
    fn test_goal_validation_error() {
        let error = BusinessError::GoalValidationFailed {
            reason: "Resource limits exceeded".to_string(),
        };
        assert!(error.to_string().contains("Goal validation failed"));
        assert!(error.to_string().contains("Resource limits exceeded"));
    }

    #[test]
    fn test_safety_check_error() {
        let error = BusinessError::SafetyCheckFailed {
            check: "Memory bounds".to_string(),
            details: "Exceeds 16GB limit".to_string(),
        };
        assert!(error.to_string().contains("Safety check failed"));
        assert!(error.to_string().contains("Memory bounds"));
        assert!(error.to_string().contains("Exceeds 16GB limit"));
    }

    #[test]
    fn test_resource_estimation_error() {
        let error = BusinessError::ResourceEstimationFailed {
            resource: "GPU memory".to_string(),
            reason: "Cannot estimate for this workload".to_string(),
        };
        assert!(error.to_string().contains("Resource estimation failed"));
        assert!(error.to_string().contains("GPU memory"));
    }

    #[test]
    fn test_llm_integration_error() {
        let error = BusinessError::LlmIntegrationError {
            service: "OpenAI".to_string(),
            error: "API rate limit exceeded".to_string(),
        };
        assert!(error.to_string().contains("LLM integration error"));
        assert!(error.to_string().contains("OpenAI"));
        assert!(error.to_string().contains("API rate limit exceeded"));
    }

    #[test]
    fn test_progress_tracking_error() {
        let error = BusinessError::ProgressTrackingError {
            goal_id: "goal-123".to_string(),
            operation: "status update".to_string(),
        };
        assert!(error.to_string().contains("Progress tracking error"));
        assert!(error.to_string().contains("goal-123"));
        assert!(error.to_string().contains("status update"));
    }

    #[test]
    fn test_configuration_error() {
        let error = BusinessError::ConfigurationError("Missing API key".to_string());
        assert!(error.to_string().contains("Configuration error"));
        assert!(error.to_string().contains("Missing API key"));
    }

    #[test]
    fn test_agent_operation_error() {
        let error = BusinessError::AgentOperationError {
            agent_id: "agent-456".to_string(),
            operation: "goal execution".to_string(),
            reason: "Insufficient resources".to_string(),
        };
        assert!(error.to_string().contains("Agent operation failed"));
        assert!(error.to_string().contains("agent-456"));
        assert!(error.to_string().contains("goal execution"));
        assert!(error.to_string().contains("Insufficient resources"));
    }

    #[test]
    fn test_result_explanation_error() {
        let error = BusinessError::ResultExplanationError {
            goal_id: "goal-789".to_string(),
            reason: "Complex result cannot be explained".to_string(),
        };
        assert!(error.to_string().contains("Result explanation failed"));
        assert!(error.to_string().contains("goal-789"));
        assert!(error
            .to_string()
            .contains("Complex result cannot be explained"));
    }

    #[test]
    fn test_timeout_error() {
        let error = BusinessError::Timeout {
            duration: Duration::from_secs(30),
        };
        assert!(error.to_string().contains("Operation timed out"));
        assert!(error.to_string().contains("30s"));
    }

    #[test]
    fn test_error_from_io() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let business_error = BusinessError::from(io_error);
        assert!(matches!(
            business_error,
            BusinessError::ConfigurationError(_)
        ));
        assert!(business_error.to_string().contains("IO error"));
        assert!(business_error.to_string().contains("File not found"));
    }

    #[test]
    fn test_error_from_serde_json() {
        let json_error = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let business_error = BusinessError::from(json_error);
        assert!(matches!(
            business_error,
            BusinessError::ConfigurationError(_)
        ));
        assert!(business_error.to_string().contains("JSON error"));
    }

    #[test]
    fn test_error_cloning() {
        let error = BusinessError::GoalParsingFailed {
            message: "Test error".to_string(),
        };
        let cloned_error = error.clone();
        assert_eq!(error.to_string(), cloned_error.to_string());
    }

    #[test]
    fn test_result_type() {
        let success: BusinessResult<String> = Ok("Success".to_string());
        assert!(success.is_ok());

        let failure: BusinessResult<String> =
            Err(BusinessError::ConfigurationError("Test error".to_string()));
        assert!(failure.is_err());
    }
}
