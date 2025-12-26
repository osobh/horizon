//! Integration tests for compliance framework

use crate::{
    ai_safety::*, audit_framework::*, data_classification::*, error::*, gdpr::*, hipaa::*, soc2::*,
};
use chrono::Utc;
use serde_json::json;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_gdpr_hipaa_integration() {
        // Test that GDPR and HIPAA compliance work together
        let config = ComplianceConfig {
            regulations: vec![Regulation::GDPR, Regulation::HIPAA],
            default_retention_days: 365 * 7,
            audit_retention_days: 365 * 10,
            encryption_by_default: true,
            allowed_regions: vec!["US".to_string(), "EU".to_string()],
            ai_safety_enabled: false,
        };

        let engine = ComplianceEngine::new(config).unwrap();
        let _gdpr_handler = GdprHandler::new();
        let mut hipaa_handler = HipaaHandler::new();

        // Test handling PII that is also PHI
        let user_id = "patient123";

        // Grant GDPR consent
        engine
            .grant_consent(
                user_id.to_string(),
                vec!["medical_treatment".to_string()],
                Some(365),
            )
            .await
            .unwrap();

        // Set up HIPAA access
        hipaa_handler
            .set_access_level("doctor456".to_string(), PhiAccessLevel::FullAccess)
            .unwrap();

        // Verify HIPAA access
        let access_allowed = hipaa_handler
            .check_phi_access("doctor456", PhiAccessType::Read)
            .unwrap();
        assert!(access_allowed);

        // Log the access for both GDPR and HIPAA
        let audit_id = engine
            .audit(
                AuditOperation::DataAccessed,
                "doctor456".to_string(),
                DataClassification::RestrictedData,
                DataCategory::PHI,
                json!({
                    "purpose": "treatment",
                    "data_type": "medical_records",
                    "gdpr_consent": true,
                    "hipaa_compliant": true
                }),
            )
            .await
            .unwrap();

        // Verify audit entry
        let entries = engine.get_audit_entries(
            Utc::now() - chrono::Duration::minutes(1),
            Utc::now() + chrono::Duration::minutes(1),
        );
        assert!(entries.iter().any(|e| e.id == audit_id));
    }

    #[tokio::test]
    async fn test_data_classification_with_compliance() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();
        let classifier = DataClassifier::new();

        // Classify different types of data and ensure compliance
        let test_cases = vec![
            (
                DataCategory::PII,
                DataClassification::ConfidentialData,
                "EU",
            ),
            (DataCategory::PHI, DataClassification::RestrictedData, "US"),
            (
                DataCategory::Financial,
                DataClassification::ConfidentialData,
                "US",
            ),
            (
                DataCategory::ModelData,
                DataClassification::InternalData,
                "EU",
            ),
        ];

        for (category, classification, region) in test_cases {
            // Classify the data
            let metadata = classifier
                .classify(category, classification, vec![region.to_string()])
                .unwrap();

            // Check compliance
            let compliance_result = engine
                .check_compliance(&AuditOperation::DataCreated, category, region)
                .await;
            assert!(compliance_result.is_ok());

            // Audit the operation
            let audit_result = engine
                .audit(
                    AuditOperation::DataCreated,
                    "system".to_string(),
                    classification,
                    category,
                    json!({
                        "region": region,
                        "encryption_required": metadata.encryption_required,
                        "audit_required": metadata.audit_required,
                    }),
                )
                .await;
            assert!(audit_result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_consent_and_erasure_workflow() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();
        let mut gdpr_handler = GdprHandler::new();
        let user_id = "test_user_workflow";

        // Grant consent
        engine
            .grant_consent(
                user_id.to_string(),
                vec!["marketing".to_string(), "analytics".to_string()],
                Some(90),
            )
            .await
            .unwrap();

        // Verify consent is active
        assert!(engine.has_valid_consent(user_id, "marketing"));
        assert!(engine.has_valid_consent(user_id, "analytics"));

        // Submit erasure request
        let request_id = gdpr_handler
            .submit_request(user_id.to_string(), GdprRight::Erasure)
            .unwrap();

        // Process erasure
        let erasure_report = gdpr_handler
            .process_erasure_request(&request_id)
            .await
            .unwrap();
        assert_eq!(erasure_report.subject_id, user_id);
        assert!(!erasure_report.data_categories_erased.is_empty());

        // Revoke consent as part of erasure
        engine.revoke_consent(user_id.to_string()).await.unwrap();

        // Verify consent is revoked
        assert!(!engine.has_valid_consent(user_id, "marketing"));
        assert!(!engine.has_valid_consent(user_id, "analytics"));
    }

    #[tokio::test]
    async fn test_multi_region_compliance() {
        let config = ComplianceConfig {
            regulations: vec![Regulation::GDPR, Regulation::SOC2],
            default_retention_days: 365 * 5,
            audit_retention_days: 365 * 7,
            encryption_by_default: true,
            allowed_regions: vec![
                "US".to_string(),
                "EU".to_string(),
                "UK".to_string(),
                "CA".to_string(),
            ],
            ai_safety_enabled: false,
        };

        let engine = ComplianceEngine::new(config).unwrap();
        let classifier = DataClassifier::new();

        // Test data that needs to be stored in multiple regions
        let multi_region_data = vec!["US".to_string(), "EU".to_string(), "UK".to_string()];

        // Financial data across regions
        let metadata = classifier
            .classify(
                DataCategory::Financial,
                DataClassification::ConfidentialData,
                multi_region_data.clone(),
            )
            .unwrap();

        // Check compliance for each region
        for region in &multi_region_data {
            let result = engine
                .check_compliance(
                    &AuditOperation::DataCreated,
                    DataCategory::Financial,
                    region,
                )
                .await;
            assert!(result.is_ok());
        }

        // Audit multi-region operation
        let audit_id = engine
            .audit(
                AuditOperation::DataCreated,
                "multi_region_system".to_string(),
                DataClassification::ConfidentialData,
                DataCategory::Financial,
                json!({
                    "regions": multi_region_data,
                    "encryption": metadata.encryption_required,
                    "retention_days": metadata.retention_period.map(|d| d.num_days()),
                }),
            )
            .await
            .unwrap();

        assert!(!audit_id.is_nil());
    }

    #[tokio::test]
    async fn test_soc2_compliance_assessment() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();
        let mut soc2_engine = Soc2Engine::new();

        // Update control status
        soc2_engine
            .update_control_status("CC6.1", ControlStatus::FullyImplemented)
            .unwrap();

        // Assess the control
        soc2_engine
            .assess_control(
                "CC6.1",
                ControlAssessmentResult::Effective,
                "External Auditor".to_string(),
                vec!["Access logs reviewed".to_string()],
                vec![],
            )
            .unwrap();

        // Audit the assessment
        engine
            .audit(
                AuditOperation::ComplianceCheck,
                "auditor".to_string(),
                DataClassification::ConfidentialData,
                DataCategory::SystemLogs,
                json!({
                    "control_id": "CC6.1",
                    "assessment_result": "Effective",
                    "framework": "SOC2",
                }),
            )
            .await
            .unwrap();

        // Get compliance status
        let status = soc2_engine.get_compliance_status();
        assert!(status.contains_key(&TrustServiceCriteria::Security));
    }

    #[tokio::test]
    async fn test_ai_safety_compliance() {
        let config = ComplianceConfig {
            regulations: vec![Regulation::AISafety],
            default_retention_days: 365 * 2,
            audit_retention_days: 365 * 5,
            encryption_by_default: true,
            allowed_regions: vec!["US".to_string()],
            ai_safety_enabled: true,
        };

        let engine = ComplianceEngine::new(config).unwrap();
        let mut ai_safety_engine = AISafetyEngine::new();
        let classifier = DataClassifier::new();

        // Add mutation boundary
        let boundary = MutationBoundary {
            id: "test_boundary".to_string(),
            component: "model".to_string(),
            allowed_mutations: vec![MutationType::ParameterAdjustment],
            max_mutation_rate: 0.1,
            rate_period_seconds: 3600,
            approval_required: false,
            rollback_capability: true,
            monitoring_level: MonitoringLevel::Enhanced,
        };

        ai_safety_engine
            .add_mutation_boundary("credit_model".to_string(), boundary)
            .unwrap();

        // Validate system safety
        let metadata = classifier
            .classify(
                DataCategory::ModelData,
                DataClassification::InternalData,
                vec!["US".to_string()],
            )
            .unwrap();

        let validation_result = ai_safety_engine
            .validate_system_safety("credit_model", AISystemType::DecisionSupport, &metadata)
            .await
            .unwrap();

        assert!(validation_result.safety_score > 0.0);
        assert!(!validation_result.bias_results.is_empty()); // Should detect bias in decision support

        // Audit AI operation
        engine
            .audit(
                AuditOperation::ComplianceCheck,
                "ai_system".to_string(),
                DataClassification::EvolutionData,
                DataCategory::ModelData,
                json!({
                    "model": "credit_model",
                    "safety_score": validation_result.safety_score,
                    "risk_level": format!("{:?}", validation_result.risk_level),
                    "bias_detected": !validation_result.bias_results.is_empty()
                }),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_data_portability_workflow() {
        let mut gdpr_handler = GdprHandler::new();
        let mut hipaa_handler = HipaaHandler::new();
        let user_id = "portable_user";

        // Submit portability request
        let request_id = gdpr_handler
            .submit_request(user_id.to_string(), GdprRight::Portability)
            .unwrap();

        // Process the request
        let portable_package = gdpr_handler
            .process_portability_request(&request_id)
            .await
            .unwrap();

        // Initialize encryption for secure transfer
        let key_id = hipaa_handler.initialize_encryption().unwrap();
        assert!(!key_id.is_empty());

        // Verify the portable package
        let package_json = serde_json::to_string(&portable_package).unwrap();
        let restored_package: PortableDataPackage = serde_json::from_str(&package_json).unwrap();

        assert_eq!(restored_package.subject_id, user_id);
        assert_eq!(restored_package.format, "JSON");
    }

    #[tokio::test]
    async fn test_compliance_violation_handling() {
        let config = ComplianceConfig {
            regulations: vec![Regulation::GDPR],
            default_retention_days: 365,
            audit_retention_days: 3650,
            encryption_by_default: true,
            allowed_regions: vec!["EU".to_string()],
            ai_safety_enabled: false,
        };

        let engine = ComplianceEngine::new(config).unwrap();

        // Try to process PHI without HIPAA enabled (should fail)
        let result = engine
            .check_compliance(&AuditOperation::DataAccessed, DataCategory::PHI, "EU")
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ComplianceError::ComplianceViolation {
                regulation,
                violation,
            } => {
                assert_eq!(regulation, "HIPAA");
                assert!(violation.contains("not enabled"));
            }
            _ => panic!("Expected ComplianceViolation"),
        }

        // Try to store data in non-allowed region
        let result = engine
            .check_compliance(
                &AuditOperation::DataCreated,
                DataCategory::PII,
                "US", // Not in allowed regions
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComplianceError::RegionRestriction { .. }
        ));
    }

    #[tokio::test]
    async fn test_retention_policy_enforcement() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();
        let classifier = DataClassifier::new();

        // Test different retention periods
        let test_cases = vec![
            (DataCategory::PII, 365 * 7),       // 7 years
            (DataCategory::PHI, 365 * 6),       // 6 years
            (DataCategory::Financial, 365 * 7), // 7 years
            (DataCategory::ModelData, 365 * 5), // 5 years
        ];

        for (category, expected_days) in test_cases {
            let metadata = classifier
                .classify(
                    category,
                    match category {
                        DataCategory::PHI => DataClassification::RestrictedData,
                        _ => DataClassification::ConfidentialData,
                    },
                    vec!["US".to_string()],
                )
                .unwrap();

            if let Some(retention) = metadata.retention_period {
                assert_eq!(
                    retention.num_days(),
                    expected_days,
                    "Wrong retention for {:?}",
                    category
                );
            }

            // Audit with retention info
            engine
                .audit(
                    AuditOperation::DataCreated,
                    "retention_test".to_string(),
                    metadata.classification,
                    category,
                    json!({
                        "retention_days": expected_days,
                        "auto_delete_enabled": true,
                    }),
                )
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn test_cross_regulation_data_handling() {
        let config = ComplianceConfig {
            regulations: vec![
                Regulation::GDPR,
                Regulation::HIPAA,
                Regulation::SOC2,
                Regulation::AISafety,
            ],
            default_retention_days: 365 * 5,
            audit_retention_days: 365 * 10,
            encryption_by_default: true,
            allowed_regions: vec!["US".to_string(), "EU".to_string()],
            ai_safety_enabled: true,
        };

        let engine = ComplianceEngine::new(config).unwrap();

        // Data that falls under multiple regulations
        // e.g., AI-generated medical recommendations (HIPAA + AI Safety)
        let audit_id = engine
            .audit(
                AuditOperation::DataCreated,
                "ai_medical_system".to_string(),
                DataClassification::RestrictedData,
                DataCategory::PHI,
                json!({
                    "data_type": "ai_medical_recommendation",
                    "regulations": ["HIPAA", "AISafety"],
                    "model_version": "med_ai_v3",
                    "encryption": "AES-256",
                    "patient_consent": true,
                    "bias_check": "passed",
                }),
            )
            .await
            .unwrap();

        // Verify the audit was created
        let entries = engine.get_audit_entries(
            Utc::now() - chrono::Duration::minutes(1),
            Utc::now() + chrono::Duration::minutes(1),
        );

        let entry = entries.iter().find(|e| e.id == audit_id).unwrap();
        assert_eq!(entry.regulation, Some(Regulation::HIPAA)); // Primary regulation
        assert_eq!(
            entry.data_classification,
            DataClassification::RestrictedData
        );
    }
}
