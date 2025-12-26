//! Error types for compliance framework

use thiserror::Error;

/// Result type for compliance operations
pub type ComplianceResult<T> = Result<T, ComplianceError>;

/// Compliance framework errors
#[derive(Debug, Error, Clone)]
pub enum ComplianceError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Data classification error
    #[error("Data classification error: {0}")]
    ClassificationError(String),

    /// Regulatory compliance violation
    #[error("Compliance violation: {regulation} - {violation}")]
    ComplianceViolation {
        /// Regulation violated (GDPR, HIPAA, etc.)
        regulation: String,
        /// Specific violation details
        violation: String,
    },

    /// Encryption error
    #[error("Encryption error: {0}")]
    EncryptionError(String),

    /// Audit log error
    #[error("Audit log error: {0}")]
    AuditLogError(String),

    /// Retention policy error
    #[error("Retention policy error: {0}")]
    RetentionError(String),

    /// AI safety violation
    #[error("AI safety violation: {0}")]
    AiSafetyViolation(String),

    /// Region restriction error
    #[error("Region restriction: data cannot be stored/processed in {region}")]
    RegionRestriction {
        /// Restricted region
        region: String,
    },

    /// Data sovereignty violation
    #[error("Data sovereignty violation: {0}")]
    DataSovereigntyViolation(String),

    /// Access control error
    #[error("Access control error: {0}")]
    AccessControlError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ComplianceError::ComplianceViolation {
            regulation: "GDPR".to_string(),
            violation: "Missing consent".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Compliance violation: GDPR - Missing consent"
        );
    }

    #[test]
    fn test_error_types() {
        let config_err = ComplianceError::ConfigurationError("Invalid config".to_string());
        assert!(matches!(config_err, ComplianceError::ConfigurationError(_)));

        let region_err = ComplianceError::RegionRestriction {
            region: "China".to_string(),
        };
        assert!(matches!(
            region_err,
            ComplianceError::RegionRestriction { .. }
        ));
    }
}
