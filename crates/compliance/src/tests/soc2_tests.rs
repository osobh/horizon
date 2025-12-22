//! Comprehensive tests for SOC2 compliance

use crate::{data_classification::*, error::*, soc2::*};
use chrono::Utc;

#[cfg(test)]
mod soc2_engine_tests {
    use super::*;

    #[test]
    fn test_soc2_engine_creation() {
        let _engine = Soc2Engine::new();
        // Engine should be initialized with standard controls (internals are private)
    }

    #[test]
    fn test_soc2_engine_default() {
        let _engine1 = Soc2Engine::new();
        let _engine2 = Soc2Engine::default();

        // Both should create valid engines
    }

    #[test]
    fn test_trust_service_criteria_variants() {
        let criteria = vec![
            TrustServiceCriteria::Security,
            TrustServiceCriteria::Availability,
            TrustServiceCriteria::ProcessingIntegrity,
            TrustServiceCriteria::Confidentiality,
            TrustServiceCriteria::Privacy,
        ];

        // Test all variants are distinct
        for (i, c1) in criteria.iter().enumerate() {
            for (j, c2) in criteria.iter().enumerate() {
                if i == j {
                    assert_eq!(c1, c2);
                } else {
                    assert_ne!(c1, c2);
                }
            }
        }
    }

    #[test]
    fn test_control_category_variants() {
        let categories = vec![
            ControlCategory::AccessControl,
            ControlCategory::ChangeManagement,
            ControlCategory::SystemOperations,
            ControlCategory::RiskAssessment,
            ControlCategory::DataProtection,
            ControlCategory::IncidentResponse,
            ControlCategory::Monitoring,
            ControlCategory::VendorManagement,
        ];

        // All should be distinct
        for (i, cat1) in categories.iter().enumerate() {
            for (j, cat2) in categories.iter().enumerate() {
                if i == j {
                    assert_eq!(cat1, cat2);
                } else {
                    assert_ne!(cat1, cat2);
                }
            }
        }
    }

    #[test]
    fn test_control_status_ordering() {
        // Test logical progression of control statuses
        assert_ne!(
            ControlStatus::NotImplemented,
            ControlStatus::PartiallyImplemented
        );
        assert_ne!(
            ControlStatus::PartiallyImplemented,
            ControlStatus::FullyImplemented
        );
        assert_ne!(ControlStatus::Testing, ControlStatus::Failed);
    }

    #[test]
    fn test_add_control() {
        let mut engine = Soc2Engine::new();
        let custom_control = Soc2Control {
            id: "CUSTOM-001".to_string(),
            name: "Custom Access Control".to_string(),
            criteria: vec![
                TrustServiceCriteria::Security,
                TrustServiceCriteria::Confidentiality,
            ],
            category: ControlCategory::AccessControl,
            description: "Custom control for enhanced access management".to_string(),
            status: ControlStatus::NotImplemented,
            owner: "Security Team".to_string(),
            implementation_date: None,
            last_assessment: None,
            evidence_requirements: vec!["Access logs".to_string(), "User reviews".to_string()],
            is_automated: false,
        };

        engine.add_control(custom_control);
        // Control should be added successfully
        assert!(engine.get_control("CUSTOM-001").is_some());
    }

    #[test]
    fn test_get_control() {
        let engine = Soc2Engine::new();

        // Standard control should exist
        let control = engine.get_control("CC6.1");
        assert!(control.is_some());
        assert_eq!(control.unwrap().id, "CC6.1");

        // Non-existent control
        assert!(engine.get_control("INVALID").is_none());
    }

    #[test]
    fn test_get_controls_by_criteria() {
        let engine = Soc2Engine::new();

        // Security controls
        let security_controls = engine.get_controls_by_criteria(TrustServiceCriteria::Security);
        assert!(!security_controls.is_empty());
        assert!(security_controls
            .iter()
            .all(|c| c.criteria.contains(&TrustServiceCriteria::Security)));

        // Privacy controls
        let privacy_controls = engine.get_controls_by_criteria(TrustServiceCriteria::Privacy);
        assert!(!privacy_controls.is_empty());
        assert!(privacy_controls
            .iter()
            .all(|c| c.criteria.contains(&TrustServiceCriteria::Privacy)));
    }

    #[test]
    fn test_update_control_status() {
        let mut engine = Soc2Engine::new();

        // Update existing control
        let result = engine.update_control_status("CC6.1", ControlStatus::Testing);
        assert!(result.is_ok());

        let control = engine.get_control("CC6.1").unwrap();
        assert_eq!(control.status, ControlStatus::Testing);
        assert!(control.implementation_date.is_none()); // Not fully implemented yet

        // Update to fully implemented
        engine
            .update_control_status("CC6.1", ControlStatus::FullyImplemented)
            .unwrap();
        let control = engine.get_control("CC6.1").unwrap();
        assert_eq!(control.status, ControlStatus::FullyImplemented);
        assert!(control.implementation_date.is_some());
    }

    #[test]
    fn test_update_control_status_nonexistent() {
        let mut engine = Soc2Engine::new();

        let result = engine.update_control_status("INVALID", ControlStatus::FullyImplemented);
        assert!(result.is_err());
        match result.unwrap_err() {
            ComplianceError::ConfigurationError(msg) => {
                assert!(msg.contains("Control not found"));
            }
            _ => panic!("Expected ConfigurationError"),
        }
    }

    #[test]
    fn test_assess_control() {
        let mut engine = Soc2Engine::new();

        let result = engine.assess_control(
            "CC6.1",
            ControlAssessmentResult::Effective,
            "External Auditor".to_string(),
            vec![
                "Access logs reviewed".to_string(),
                "Policies documented".to_string(),
            ],
            vec![],
        );

        assert!(result.is_ok());
        // Assessment should be recorded successfully (internals are private)

        // Control should have last assessment date updated
        let control = engine.get_control("CC6.1").unwrap();
        assert!(control.last_assessment.is_some());
    }

    #[test]
    fn test_assess_control_with_deficiencies() {
        let mut engine = Soc2Engine::new();

        let result = engine.assess_control(
            "A1.1",
            ControlAssessmentResult::DeficiencyIdentified,
            "Internal Auditor".to_string(),
            vec!["Monitoring data reviewed".to_string()],
            vec!["Capacity thresholds not defined".to_string()],
        );

        assert!(result.is_ok());
        // Assessment with deficiencies should be recorded (internals are private)
    }

    #[test]
    fn test_assess_control_ineffective() {
        let mut engine = Soc2Engine::new();

        let result = engine.assess_control(
            "PI1.1",
            ControlAssessmentResult::Ineffective,
            "Risk Team".to_string(),
            vec![],
            vec!["No data validation controls found".to_string()],
        );

        assert!(result.is_ok());
        // Ineffective assessment should have recommendations (internals are private)
    }

    #[test]
    fn test_assess_nonexistent_control() {
        let mut engine = Soc2Engine::new();

        let result = engine.assess_control(
            "INVALID",
            ControlAssessmentResult::Effective,
            "Auditor".to_string(),
            vec![],
            vec![],
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            ComplianceError::ConfigurationError(msg) => {
                assert!(msg.contains("Control not found"));
            }
            _ => panic!("Expected ConfigurationError"),
        }
    }

    #[test]
    fn test_validate_data_handling_encryption_required() {
        let engine = Soc2Engine::new();

        // Restricted data without encryption
        let metadata = DataMetadata {
            classification: DataClassification::RestrictedData,
            category: DataCategory::Financial,
            retention_period: None,
            encryption_required: false, // This should fail
            allowed_regions: vec!["US".to_string()],
            audit_required: true,
            owner: None,
            created_at: Utc::now(),
        };

        let result = engine.validate_data_handling(&metadata, "transfer");
        assert!(result.is_err());
        match result.unwrap_err() {
            ComplianceError::ComplianceViolation {
                regulation,
                violation,
            } => {
                assert_eq!(regulation, "SOC2");
                assert!(violation.contains("encryption"));
            }
            _ => panic!("Expected ComplianceViolation"),
        }
    }

    #[test]
    fn test_validate_data_handling_pii_audit_required() {
        let engine = Soc2Engine::new();

        // PII without audit trail
        let metadata = DataMetadata {
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: None,
            encryption_required: true,
            allowed_regions: vec!["US".to_string()],
            audit_required: false, // This should fail
            owner: None,
            created_at: Utc::now(),
        };

        let result = engine.validate_data_handling(&metadata, "process");
        assert!(result.is_err());
        match result.unwrap_err() {
            ComplianceError::ComplianceViolation {
                regulation,
                violation,
            } => {
                assert_eq!(regulation, "SOC2");
                assert!(violation.contains("audit trail"));
            }
            _ => panic!("Expected ComplianceViolation"),
        }
    }

    #[test]
    fn test_validate_data_handling_success() {
        let engine = Soc2Engine::new();
        let classifier = DataClassifier::new();

        // Properly configured confidential data
        let metadata = classifier
            .classify(
                DataCategory::Financial,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        let result = engine.validate_data_handling(&metadata, "update");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_compliance_status_not_implemented() {
        let engine = Soc2Engine::new();

        // All controls start as NotImplemented
        let status = engine.get_compliance_status();

        for criteria in &[
            TrustServiceCriteria::Security,
            TrustServiceCriteria::Availability,
            TrustServiceCriteria::ProcessingIntegrity,
            TrustServiceCriteria::Confidentiality,
            TrustServiceCriteria::Privacy,
        ] {
            assert_eq!(status[criteria], ComplianceStatus::NonCompliant);
        }
    }

    #[test]
    fn test_get_compliance_status_fully_implemented() {
        let mut engine = Soc2Engine::new();

        // Implement all security controls
        let security_controls: Vec<String> = engine
            .get_controls_by_criteria(TrustServiceCriteria::Security)
            .iter()
            .map(|c| c.id.clone())
            .collect();

        for control_id in security_controls {
            engine
                .update_control_status(&control_id, ControlStatus::FullyImplemented)
                .unwrap();
        }

        let status = engine.get_compliance_status();
        assert_eq!(
            status[&TrustServiceCriteria::Security],
            ComplianceStatus::Compliant
        );

        // Other criteria should still be non-compliant
        assert_eq!(
            status[&TrustServiceCriteria::Availability],
            ComplianceStatus::NonCompliant
        );
    }

    #[test]
    fn test_get_compliance_status_partial() {
        let mut engine = Soc2Engine::new();

        // Add multiple controls for a criteria
        for i in 1..=10 {
            engine.add_control(Soc2Control {
                id: format!("TEST-{}", i),
                name: format!("Test Control {}", i),
                criteria: vec![TrustServiceCriteria::Availability],
                category: ControlCategory::SystemOperations,
                description: "Test".to_string(),
                status: if i <= 6 {
                    ControlStatus::FullyImplemented
                } else {
                    ControlStatus::NotImplemented
                },
                owner: "Test".to_string(),
                implementation_date: None,
                last_assessment: None,
                evidence_requirements: vec![],
                is_automated: false,
            });
        }

        let status = engine.get_compliance_status();
        // 60% implemented + 1 original control = ~54% overall
        assert_eq!(
            status[&TrustServiceCriteria::Availability],
            ComplianceStatus::PartiallyCompliant
        );
    }

    #[test]
    fn test_control_serialization() {
        let control = Soc2Control {
            id: "TEST-001".to_string(),
            name: "Test Control".to_string(),
            criteria: vec![
                TrustServiceCriteria::Security,
                TrustServiceCriteria::Privacy,
            ],
            category: ControlCategory::DataProtection,
            description: "Test description".to_string(),
            status: ControlStatus::Testing,
            owner: "Test Owner".to_string(),
            implementation_date: Some(Utc::now()),
            last_assessment: Some(Utc::now()),
            evidence_requirements: vec!["Evidence 1".to_string()],
            is_automated: true,
        };

        let serialized = serde_json::to_string(&control).unwrap();
        let deserialized: Soc2Control = serde_json::from_str(&serialized).unwrap();

        assert_eq!(control.id, deserialized.id);
        assert_eq!(control.name, deserialized.name);
        assert_eq!(control.criteria, deserialized.criteria);
        assert_eq!(control.category, deserialized.category);
        assert_eq!(control.status, deserialized.status);
        assert_eq!(control.is_automated, deserialized.is_automated);
    }

    #[test]
    fn test_assessment_result_serialization() {
        let assessment = AssessmentResult {
            control_id: "CC6.1".to_string(),
            result: ControlAssessmentResult::DeficiencyIdentified,
            assessment_date: Utc::now(),
            assessor: "External Auditor".to_string(),
            evidence: vec!["Log1".to_string(), "Doc2".to_string()],
            findings: vec!["Finding1".to_string()],
            recommendations: vec!["Fix this".to_string()],
        };

        let serialized = serde_json::to_string(&assessment).unwrap();
        let deserialized: AssessmentResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(assessment.control_id, deserialized.control_id);
        assert_eq!(assessment.result, deserialized.result);
        assert_eq!(assessment.assessor, deserialized.assessor);
        assert_eq!(assessment.evidence, deserialized.evidence);
        assert_eq!(assessment.findings, deserialized.findings);
        assert_eq!(assessment.recommendations, deserialized.recommendations);
    }

    #[test]
    fn test_multiple_assessments_per_control() {
        let mut engine = Soc2Engine::new();

        // First assessment
        engine
            .assess_control(
                "CC6.1",
                ControlAssessmentResult::Ineffective,
                "Auditor1".to_string(),
                vec![],
                vec!["Major issues found".to_string()],
            )
            .unwrap();

        // Second assessment after remediation
        engine
            .assess_control(
                "CC6.1",
                ControlAssessmentResult::Effective,
                "Auditor2".to_string(),
                vec!["Remediation evidence".to_string()],
                vec![],
            )
            .unwrap();

        // Multiple assessments should be recorded (internals are private)
    }

    #[test]
    fn test_control_evidence_requirements() {
        let engine = Soc2Engine::new();

        let control = engine.get_control("CC6.1").unwrap();
        assert!(!control.evidence_requirements.is_empty());
        assert!(control
            .evidence_requirements
            .contains(&"Access control policies".to_string()));

        let privacy_control = engine.get_control("P1.1").unwrap();
        assert!(privacy_control
            .evidence_requirements
            .contains(&"Privacy notices".to_string()));
    }

    #[test]
    fn test_control_automation_flag() {
        let engine = Soc2Engine::new();

        // Access control is manual
        let manual_control = engine.get_control("CC6.1").unwrap();
        assert!(!manual_control.is_automated);

        // Availability monitoring is automated
        let automated_control = engine.get_control("A1.1").unwrap();
        assert!(automated_control.is_automated);

        // Processing integrity is automated
        let pi_control = engine.get_control("PI1.1").unwrap();
        assert!(pi_control.is_automated);
    }
}

#[cfg(test)]
mod compliance_status_tests {
    use super::*;

    #[test]
    fn test_compliance_status_variants() {
        let statuses = vec![
            ComplianceStatus::Compliant,
            ComplianceStatus::SubstantiallyCompliant,
            ComplianceStatus::PartiallyCompliant,
            ComplianceStatus::NonCompliant,
            ComplianceStatus::NotAssessed,
        ];

        // All should be distinct
        for (i, s1) in statuses.iter().enumerate() {
            for (j, s2) in statuses.iter().enumerate() {
                if i == j {
                    assert_eq!(s1, s2);
                } else {
                    assert_ne!(s1, s2);
                }
            }
        }
    }

    #[test]
    fn test_compliance_status_thresholds() {
        let mut engine = Soc2Engine::new();

        // Add 20 test controls
        for i in 1..=20 {
            engine.add_control(Soc2Control {
                id: format!("TEST-{:02}", i),
                name: format!("Test Control {}", i),
                criteria: vec![TrustServiceCriteria::Security],
                category: ControlCategory::AccessControl,
                description: "Test".to_string(),
                status: ControlStatus::NotImplemented,
                owner: "Test".to_string(),
                implementation_date: None,
                last_assessment: None,
                evidence_requirements: vec![],
                is_automated: false,
            });
        }

        // Test different implementation percentages
        let test_cases = vec![
            (20, ComplianceStatus::Compliant),              // 100%
            (19, ComplianceStatus::SubstantiallyCompliant), // 95% of test controls, but less overall
            (18, ComplianceStatus::SubstantiallyCompliant), // 90%
            (16, ComplianceStatus::PartiallyCompliant),     // 80% of test controls, less overall
            (14, ComplianceStatus::PartiallyCompliant),     // 70%
            (10, ComplianceStatus::NonCompliant), // 50% of test controls, but less overall
            (8, ComplianceStatus::NonCompliant),  // 40%
        ];

        for (implemented_count, expected_status) in test_cases {
            // Reset all to not implemented
            for i in 1..=20 {
                engine
                    .update_control_status(&format!("TEST-{:02}", i), ControlStatus::NotImplemented)
                    .unwrap();
            }

            // Implement specified number
            for i in 1..=implemented_count {
                engine
                    .update_control_status(
                        &format!("TEST-{:02}", i),
                        ControlStatus::FullyImplemented,
                    )
                    .unwrap();
            }

            let status = engine.get_compliance_status();
            assert_eq!(
                status[&TrustServiceCriteria::Security],
                expected_status,
                "Failed for {} implemented controls",
                implemented_count
            );
        }
    }
}
