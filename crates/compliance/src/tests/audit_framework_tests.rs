//! Comprehensive tests for audit framework

use crate::{audit_framework::*, data_classification::*, error::*};
use chrono::{Duration, Utc};
use serde_json::json;

#[cfg(test)]
mod audit_framework_tests {
    use super::*;

    // Helper function to create test config
    fn test_config() -> ComplianceConfig {
        ComplianceConfig {
            regulations: vec![
                Regulation::GDPR,
                Regulation::HIPAA,
                Regulation::SOC2,
                Regulation::AISafety,
            ],
            default_retention_days: 365,
            audit_retention_days: 3650,
            encryption_by_default: true,
            allowed_regions: vec!["US".to_string(), "EU".to_string(), "UK".to_string()],
            ai_safety_enabled: true,
        }
    }

    #[test]
    fn test_compliance_config_default() {
        let config = ComplianceConfig::default();
        assert_eq!(config.regulations.len(), 4);
        assert_eq!(config.default_retention_days, 365 * 7);
        assert_eq!(config.audit_retention_days, 365 * 10);
        assert!(config.encryption_by_default);
        assert_eq!(config.allowed_regions, vec!["US", "EU"]);
        assert!(config.ai_safety_enabled);
    }

    #[test]
    fn test_compliance_config_serialization() {
        let config = test_config();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: ComplianceConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config.regulations.len(), deserialized.regulations.len());
        assert_eq!(config.allowed_regions, deserialized.allowed_regions);
    }

    #[tokio::test]
    async fn test_compliance_engine_with_empty_regions() {
        let mut config = test_config();
        config.allowed_regions.clear();
        let result = ComplianceEngine::new(config);
        assert!(result.is_err());
        if let Err(error) = result {
            match error {
                ComplianceError::ConfigurationError(msg) => {
                    assert_eq!(msg, "At least one region must be allowed");
                }
                _ => panic!("Expected ConfigurationError"),
            }
        }
    }

    #[tokio::test]
    async fn test_audit_all_operation_types() {
        let engine = ComplianceEngine::new(test_config()).unwrap();

        let operations = vec![
            AuditOperation::DataCreated,
            AuditOperation::DataAccessed,
            AuditOperation::DataModified,
            AuditOperation::DataDeleted,
            AuditOperation::DataExported,
            AuditOperation::ConsentGranted,
            AuditOperation::ConsentRevoked,
            AuditOperation::ComplianceCheck,
            AuditOperation::PolicyUpdate,
            AuditOperation::SecurityEvent,
        ];

        for op in operations {
            let result = engine
                .audit(
                    op.clone(),
                    "test_user".to_string(),
                    DataClassification::InternalData,
                    DataCategory::SystemLogs,
                    json!({"operation": format!("{:?}", op)}),
                )
                .await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_audit_entry_fields() {
        let engine = ComplianceEngine::new(test_config()).unwrap();

        let start_time = Utc::now();
        let audit_id = engine
            .audit(
                AuditOperation::DataCreated,
                "test_actor".to_string(),
                DataClassification::RestrictedData,
                DataCategory::PHI,
                json!({"patient_id": "12345", "record_type": "medical"}),
            )
            .await
            .unwrap();

        // Verify the audit entry was created (internals are private)
        // We can verify through the get_audit_entries method
        let entries = engine.get_audit_entries(start_time, Utc::now());
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry.id, audit_id);
        assert!(entry.timestamp >= start_time);
        assert!(entry.timestamp <= Utc::now());
        assert_eq!(entry.actor, "test_actor");
        assert_eq!(
            entry.data_classification,
            DataClassification::RestrictedData
        );
        assert_eq!(entry.data_category, DataCategory::PHI);
        assert_eq!(entry.regulation, Some(Regulation::HIPAA));
        assert!(entry.success);
        assert_eq!(entry.details["patient_id"], "12345");
    }

    #[tokio::test]
    async fn test_concurrent_audit_operations() {
        let engine = ComplianceEngine::new(test_config()).unwrap();

        let mut handles = vec![];

        for i in 0..100 {
            let engine_clone = engine.clone();
            let handle = tokio::spawn(async move {
                engine_clone
                    .audit(
                        AuditOperation::DataAccessed,
                        format!("user_{}", i),
                        DataClassification::InternalData,
                        DataCategory::BusinessData,
                        json!({"thread": i}),
                    )
                    .await
            });
            handles.push(handle);
        }

        let results: Vec<_> = futures::future::join_all(handles).await;
        for result in results {
            assert!(result.unwrap().is_ok());
        }

        // Verify 100 entries were created through the get_audit_entries method
        let entries = engine.get_audit_entries(
            Utc::now() - chrono::Duration::hours(1),
            Utc::now() + chrono::Duration::hours(1),
        );
        assert_eq!(entries.len(), 100);
    }

    #[tokio::test]
    async fn test_consent_lifecycle() {
        let engine = ComplianceEngine::new(test_config()).unwrap();
        let user_id = "test_user_123";

        // Initially no consent
        assert!(!engine.has_valid_consent(user_id, "marketing"));

        // Grant consent for multiple purposes
        engine
            .grant_consent(
                user_id.to_string(),
                vec![
                    "marketing".to_string(),
                    "analytics".to_string(),
                    "personalization".to_string(),
                ],
                Some(30), // 30 days
            )
            .await
            .unwrap();

        // Verify consent is active
        assert!(engine.has_valid_consent(user_id, "marketing"));
        assert!(engine.has_valid_consent(user_id, "analytics"));
        assert!(engine.has_valid_consent(user_id, "personalization"));
        assert!(!engine.has_valid_consent(user_id, "advertising")); // Not granted

        // Verify audit entry was created
        let entries = engine.get_audit_entries(
            Utc::now() - Duration::minutes(1),
            Utc::now() + Duration::minutes(1),
        );
        assert!(entries
            .iter()
            .any(|e| matches!(e.operation, AuditOperation::ConsentGranted)));

        // Revoke consent
        engine.revoke_consent(user_id.to_string()).await.unwrap();

        // Verify consent is revoked
        assert!(!engine.has_valid_consent(user_id, "marketing"));
        assert!(!engine.has_valid_consent(user_id, "analytics"));

        // Verify revocation was audited
        let entries = engine.get_audit_entries(
            Utc::now() - Duration::minutes(1),
            Utc::now() + Duration::minutes(1),
        );
        assert!(entries
            .iter()
            .any(|e| matches!(e.operation, AuditOperation::ConsentRevoked)));
    }

    #[tokio::test]
    async fn test_consent_expiration() {
        let engine = ComplianceEngine::new(test_config()).unwrap();
        let user_id = "expiring_user";

        // Grant consent with 0 days duration (immediate expiration)
        engine
            .grant_consent(
                user_id.to_string(),
                vec!["temporary_access".to_string()],
                Some(0),
            )
            .await
            .unwrap();

        // Consent should be expired
        // Note: This might be flaky due to timing, in production we'd mock time
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert!(!engine.has_valid_consent(user_id, "temporary_access"));
    }

    #[tokio::test]
    async fn test_consent_without_expiration() {
        let engine = ComplianceEngine::new(test_config()).unwrap();
        let user_id = "permanent_user";

        // Grant consent without expiration
        engine
            .grant_consent(
                user_id.to_string(),
                vec!["essential_services".to_string()],
                None,
            )
            .await
            .unwrap();

        // Consent should remain valid
        assert!(engine.has_valid_consent(user_id, "essential_services"));
    }

    #[tokio::test]
    async fn test_revoke_nonexistent_consent() {
        let engine = ComplianceEngine::new(test_config()).unwrap();

        let result = engine.revoke_consent("nonexistent_user".to_string()).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ComplianceError::AccessControlError(msg) => {
                assert_eq!(msg, "No consent record found");
            }
            _ => panic!("Expected AccessControlError"),
        }
    }

    #[tokio::test]
    async fn test_compliance_check_all_data_categories() {
        let engine = ComplianceEngine::new(test_config()).unwrap();

        let categories = vec![
            DataCategory::PII,
            DataCategory::PHI,
            DataCategory::Financial,
            DataCategory::ModelData,
            DataCategory::SystemLogs,
            DataCategory::EvolutionPatterns,
            DataCategory::BusinessData,
        ];

        for category in categories {
            let result = engine
                .check_compliance(&AuditOperation::DataAccessed, category, "US")
                .await;
            assert!(result.is_ok(), "Failed for category: {:?}", category);
        }
    }

    #[tokio::test]
    async fn test_compliance_check_region_restrictions() {
        let engine = ComplianceEngine::new(test_config()).unwrap();

        // Test allowed regions
        for region in &["US", "EU", "UK"] {
            let result = engine
                .check_compliance(
                    &AuditOperation::DataCreated,
                    DataCategory::BusinessData,
                    region,
                )
                .await;
            assert!(result.is_ok());
        }

        // Test disallowed regions
        for region in &["CN", "RU", "KP", "IR"] {
            let result = engine
                .check_compliance(
                    &AuditOperation::DataCreated,
                    DataCategory::BusinessData,
                    region,
                )
                .await;
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                ComplianceError::RegionRestriction { .. }
            ));
        }
    }

    #[tokio::test]
    async fn test_compliance_check_regulation_requirements() {
        let mut config = test_config();
        config.regulations = vec![Regulation::SOC2]; // Only SOC2 enabled
        let engine = ComplianceEngine::new(config).unwrap();

        // PHI requires HIPAA
        let result = engine
            .check_compliance(&AuditOperation::DataAccessed, DataCategory::PHI, "US")
            .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ComplianceError::ComplianceViolation {
                regulation,
                violation,
            } => {
                assert_eq!(regulation, "HIPAA");
                assert_eq!(violation, "HIPAA compliance not enabled");
            }
            _ => panic!("Expected ComplianceViolation"),
        }

        // PII requires GDPR
        let result = engine
            .check_compliance(&AuditOperation::DataAccessed, DataCategory::PII, "EU")
            .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ComplianceError::ComplianceViolation {
                regulation,
                violation,
            } => {
                assert_eq!(regulation, "GDPR");
                assert_eq!(violation, "GDPR compliance not enabled");
            }
            _ => panic!("Expected ComplianceViolation"),
        }
    }

    #[tokio::test]
    async fn test_audit_log_time_range_queries() {
        let engine = ComplianceEngine::new(test_config()).unwrap();

        let base_time = Utc::now();

        // Create entries at different times
        for i in 0..10 {
            engine
                .audit(
                    AuditOperation::DataAccessed,
                    format!("user_{}", i),
                    DataClassification::InternalData,
                    DataCategory::BusinessData,
                    json!({"index": i, "timestamp": base_time + Duration::minutes(i as i64)}),
                )
                .await
                .unwrap();
        }

        // Query different time ranges
        let all_entries = engine.get_audit_entries(
            base_time - Duration::hours(1),
            base_time + Duration::hours(1),
        );
        assert_eq!(all_entries.len(), 10);

        let recent_entries = engine.get_audit_entries(
            base_time + Duration::minutes(5),
            base_time + Duration::hours(1),
        );
        assert!(recent_entries.len() <= 5);

        let no_entries =
            engine.get_audit_entries(base_time - Duration::days(2), base_time - Duration::days(1));
        assert_eq!(no_entries.len(), 0);
    }

    #[tokio::test]
    async fn test_regulation_determination() {
        let engine = ComplianceEngine::new(test_config()).unwrap();

        // Test regulation mapping
        let test_cases = vec![
            (DataCategory::PHI, Some(Regulation::HIPAA)),
            (DataCategory::PII, Some(Regulation::GDPR)),
            (DataCategory::Financial, Some(Regulation::SOC2)),
            (DataCategory::ModelData, Some(Regulation::AISafety)),
            (DataCategory::EvolutionPatterns, Some(Regulation::AISafety)),
            (DataCategory::SystemLogs, None),
            (DataCategory::BusinessData, None),
        ];

        for (category, expected_regulation) in test_cases {
            let audit_id = engine
                .audit(
                    AuditOperation::DataCreated,
                    "test".to_string(),
                    match category {
                        DataCategory::PHI => DataClassification::RestrictedData,
                        _ => DataClassification::InternalData,
                    },
                    category,
                    json!({}),
                )
                .await
                .unwrap();

            // Verify through get_audit_entries
            let entries = engine.get_audit_entries(
                Utc::now() - chrono::Duration::minutes(1),
                Utc::now() + chrono::Duration::minutes(1),
            );
            let entry = entries.iter().find(|e| e.id == audit_id).unwrap();
            assert_eq!(
                entry.regulation, expected_regulation,
                "Failed for category: {:?}",
                category
            );
        }
    }

    #[tokio::test]
    async fn test_ai_safety_regulation_toggle() {
        let mut config = test_config();
        config.ai_safety_enabled = false;
        let engine = ComplianceEngine::new(config).unwrap();

        // With AI safety disabled, ModelData should not have AISafety regulation
        let audit_id = engine
            .audit(
                AuditOperation::DataCreated,
                "test".to_string(),
                DataClassification::InternalData,
                DataCategory::ModelData,
                json!({}),
            )
            .await
            .unwrap();

        // Verify through get_audit_entries
        let entries = engine.get_audit_entries(
            Utc::now() - chrono::Duration::minutes(1),
            Utc::now() + chrono::Duration::minutes(1),
        );
        let entry = entries.iter().find(|e| e.id == audit_id).unwrap();
        assert_eq!(entry.regulation, None);
    }

    #[tokio::test]
    async fn test_audit_entry_serialization() {
        let entry = AuditEntry {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            operation: AuditOperation::DataCreated,
            actor: "test_actor".to_string(),
            data_classification: DataClassification::ConfidentialData,
            data_category: DataCategory::Financial,
            regulation: Some(Regulation::SOC2),
            success: true,
            details: json!({"amount": 1000, "currency": "USD"}),
            data_hash: Some("abc123def456".to_string()),
        };

        let serialized = serde_json::to_string(&entry).unwrap();
        let deserialized: AuditEntry = serde_json::from_str(&serialized).unwrap();

        assert_eq!(entry.id, deserialized.id);
        assert_eq!(entry.actor, deserialized.actor);
        assert_eq!(entry.data_classification, deserialized.data_classification);
        assert_eq!(entry.data_category, deserialized.data_category);
        assert_eq!(entry.regulation, deserialized.regulation);
        assert_eq!(entry.success, deserialized.success);
        assert_eq!(entry.details, deserialized.details);
        assert_eq!(entry.data_hash, deserialized.data_hash);
    }
}

#[cfg(test)]
mod policy_engine_tests {
    use super::*;

    #[tokio::test]
    async fn test_gdpr_policy_validation() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();

        // Create an audit entry that should pass GDPR validation
        let result = engine
            .audit(
                AuditOperation::DataAccessed,
                "gdpr_user".to_string(),
                DataClassification::ConfidentialData,
                DataCategory::PII,
                json!({"purpose": "service_delivery"}),
            )
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_hipaa_policy_validation() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();

        // Test with proper classification (should pass)
        let result = engine
            .audit(
                AuditOperation::DataAccessed,
                "doctor".to_string(),
                DataClassification::RestrictedData,
                DataCategory::PHI,
                json!({"patient_id": "12345"}),
            )
            .await;

        assert!(result.is_ok());

        // Note: The current implementation always returns true for validators,
        // so we can't test failure cases without modifying the implementation
    }
}

#[cfg(test)]
mod compliance_engine_clone_tests {
    use super::*;

    #[test]
    fn test_compliance_engine_is_cloneable() {
        // This test ensures ComplianceEngine can be cloned for concurrent use
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();
        let _engine_clone = engine.clone();
        // If this compiles, the test passes
    }

    #[tokio::test]
    async fn test_shared_state_across_clones() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();
        let engine_clone = engine.clone();

        // Grant consent on original
        engine
            .grant_consent(
                "shared_user".to_string(),
                vec!["test_purpose".to_string()],
                None,
            )
            .await
            .unwrap();

        // Check consent on clone
        assert!(engine_clone.has_valid_consent("shared_user", "test_purpose"));
    }
}
