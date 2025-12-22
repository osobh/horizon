//! Comprehensive tests for error types

use crate::error::*;

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_configuration_error() {
        let error = ComplianceError::ConfigurationError("Invalid setting".to_string());
        assert_eq!(error.to_string(), "Configuration error: Invalid setting");

        // Test cloning
        let cloned = error.clone();
        assert_eq!(error.to_string(), cloned.to_string());
    }

    #[test]
    fn test_classification_error() {
        let error = ComplianceError::ClassificationError("Unknown category".to_string());
        assert_eq!(
            error.to_string(),
            "Data classification error: Unknown category"
        );
    }

    #[test]
    fn test_compliance_violation() {
        let error = ComplianceError::ComplianceViolation {
            regulation: "GDPR".to_string(),
            violation: "Missing user consent".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Compliance violation: GDPR - Missing user consent"
        );

        // Test with different regulation
        let error2 = ComplianceError::ComplianceViolation {
            regulation: "HIPAA".to_string(),
            violation: "Unencrypted PHI transmission".to_string(),
        };
        assert_eq!(
            error2.to_string(),
            "Compliance violation: HIPAA - Unencrypted PHI transmission"
        );
    }

    #[test]
    fn test_encryption_error() {
        let error = ComplianceError::EncryptionError("Failed to encrypt data".to_string());
        assert_eq!(
            error.to_string(),
            "Encryption error: Failed to encrypt data"
        );
    }

    #[test]
    fn test_audit_log_error() {
        let error = ComplianceError::AuditLogError("Failed to write audit entry".to_string());
        assert_eq!(
            error.to_string(),
            "Audit log error: Failed to write audit entry"
        );
    }

    #[test]
    fn test_retention_error() {
        let error = ComplianceError::RetentionError("Invalid retention period".to_string());
        assert_eq!(
            error.to_string(),
            "Retention policy error: Invalid retention period"
        );
    }

    #[test]
    fn test_ai_safety_violation() {
        let error = ComplianceError::AiSafetyViolation("Model output contains bias".to_string());
        assert_eq!(
            error.to_string(),
            "AI safety violation: Model output contains bias"
        );
    }

    #[test]
    fn test_region_restriction() {
        let error = ComplianceError::RegionRestriction {
            region: "CN".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Region restriction: data cannot be stored/processed in CN"
        );

        // Test with different regions
        let error2 = ComplianceError::RegionRestriction {
            region: "RU".to_string(),
        };
        assert_eq!(
            error2.to_string(),
            "Region restriction: data cannot be stored/processed in RU"
        );
    }

    #[test]
    fn test_data_sovereignty_violation() {
        let error = ComplianceError::DataSovereigntyViolation(
            "Data must remain within EU borders".to_string(),
        );
        assert_eq!(
            error.to_string(),
            "Data sovereignty violation: Data must remain within EU borders"
        );
    }

    #[test]
    fn test_access_control_error() {
        let error = ComplianceError::AccessControlError("Unauthorized access attempt".to_string());
        assert_eq!(
            error.to_string(),
            "Access control error: Unauthorized access attempt"
        );
    }

    #[test]
    fn test_internal_error() {
        let error = ComplianceError::InternalError("Unexpected state".to_string());
        assert_eq!(error.to_string(), "Internal error: Unexpected state");
    }

    #[test]
    fn test_error_pattern_matching() {
        let errors = vec![
            ComplianceError::ConfigurationError("test".to_string()),
            ComplianceError::ClassificationError("test".to_string()),
            ComplianceError::ComplianceViolation {
                regulation: "GDPR".to_string(),
                violation: "test".to_string(),
            },
            ComplianceError::EncryptionError("test".to_string()),
            ComplianceError::AuditLogError("test".to_string()),
            ComplianceError::RetentionError("test".to_string()),
            ComplianceError::AiSafetyViolation("test".to_string()),
            ComplianceError::RegionRestriction {
                region: "test".to_string(),
            },
            ComplianceError::DataSovereigntyViolation("test".to_string()),
            ComplianceError::AccessControlError("test".to_string()),
            ComplianceError::InternalError("test".to_string()),
        ];

        // Test that all variants can be matched
        for error in errors {
            match error {
                ComplianceError::ConfigurationError(_) => {}
                ComplianceError::ClassificationError(_) => {}
                ComplianceError::ComplianceViolation { .. } => {}
                ComplianceError::EncryptionError(_) => {}
                ComplianceError::AuditLogError(_) => {}
                ComplianceError::RetentionError(_) => {}
                ComplianceError::AiSafetyViolation(_) => {}
                ComplianceError::RegionRestriction { .. } => {}
                ComplianceError::DataSovereigntyViolation(_) => {}
                ComplianceError::AccessControlError(_) => {}
                ComplianceError::InternalError(_) => {}
            }
        }
    }

    #[test]
    fn test_result_type() {
        // Test Ok variant
        let ok_result: ComplianceResult<String> = Ok("Success".to_string());
        assert!(ok_result.is_ok());
        assert_eq!(ok_result.unwrap(), "Success");

        // Test Err variant
        let err_result: ComplianceResult<String> =
            Err(ComplianceError::InternalError("Failed".to_string()));
        assert!(err_result.is_err());
        assert!(matches!(
            err_result.unwrap_err(),
            ComplianceError::InternalError(_)
        ));
    }

    #[test]
    fn test_error_chaining() {
        fn operation_that_fails() -> ComplianceResult<()> {
            Err(ComplianceError::EncryptionError(
                "Encryption key not found".to_string(),
            ))
        }

        fn higher_level_operation() -> ComplianceResult<()> {
            operation_that_fails()
                .map_err(|e| ComplianceError::InternalError(format!("Operation failed: {}", e)))
        }

        let result = higher_level_operation();
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(matches!(error, ComplianceError::InternalError(_)));
        assert!(error.to_string().contains("Encryption key not found"));
    }

    #[test]
    fn test_error_debug_impl() {
        let error = ComplianceError::ComplianceViolation {
            regulation: "SOC2".to_string(),
            violation: "Audit trail missing".to_string(),
        };

        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("ComplianceViolation"));
        assert!(debug_str.contains("SOC2"));
        assert!(debug_str.contains("Audit trail missing"));
    }

    #[test]
    fn test_error_serialization() {
        // Note: Error types typically don't implement Serialize/Deserialize
        // but we can test that they work properly with Result types

        fn process_data() -> ComplianceResult<String> {
            Err(ComplianceError::RegionRestriction {
                region: "CN".to_string(),
            })
        }

        let result = process_data();
        match result {
            Ok(_) => panic!("Expected error"),
            Err(e) => {
                assert_eq!(
                    e.to_string(),
                    "Region restriction: data cannot be stored/processed in CN"
                );
            }
        }
    }

    #[test]
    fn test_error_equality() {
        // Errors should be cloneable and comparable
        let error1 = ComplianceError::ConfigurationError("Test".to_string());
        let error2 = error1.clone();

        // Can't directly compare errors, but can compare their string representations
        assert_eq!(error1.to_string(), error2.to_string());
    }

    #[test]
    fn test_compliance_result_helpers() {
        // Test various Result helper methods
        let ok_result: ComplianceResult<i32> = Ok(42);
        assert_eq!(ok_result.clone().unwrap_or(0), 42);
        assert_eq!(ok_result.unwrap_or_default(), 42);

        let err_result: ComplianceResult<i32> =
            Err(ComplianceError::InternalError("Error".to_string()));
        assert_eq!(err_result.clone().unwrap_or(0), 0);
        assert_eq!(err_result.unwrap_or_default(), 0);
    }

    #[test]
    fn test_error_context() {
        // Test adding context to errors
        let base_error = ComplianceError::EncryptionError("Key expired".to_string());

        // Wrap with additional context
        let contextualized = ComplianceError::InternalError(format!(
            "Failed to process sensitive data: {}",
            base_error
        ));

        assert!(contextualized.to_string().contains("Key expired"));
        assert!(contextualized
            .to_string()
            .contains("Failed to process sensitive data"));
    }

    #[test]
    fn test_error_conversion() {
        // Test converting between error types
        fn convert_to_internal_error(error: ComplianceError) -> ComplianceError {
            match error {
                ComplianceError::InternalError(_) => error,
                _ => ComplianceError::InternalError(error.to_string()),
            }
        }

        let encryption_error = ComplianceError::EncryptionError("Test".to_string());
        let converted = convert_to_internal_error(encryption_error);
        assert!(matches!(converted, ComplianceError::InternalError(_)));

        let internal_error = ComplianceError::InternalError("Already internal".to_string());
        let converted2 = convert_to_internal_error(internal_error.clone());
        assert_eq!(internal_error.to_string(), converted2.to_string());
    }
}
