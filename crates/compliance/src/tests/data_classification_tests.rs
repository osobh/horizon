//! Comprehensive tests for data classification

use crate::{data_classification::*, error::*};
use chrono::Duration;

#[cfg(test)]
mod data_classification_tests {
    use super::*;

    #[test]
    fn test_data_classification_ordering() {
        // Test that classification levels have proper ordering
        assert!((DataClassification::PublicData as u8) < (DataClassification::InternalData as u8));
        assert!(
            (DataClassification::InternalData as u8) < (DataClassification::ConfidentialData as u8)
        );
        assert!(
            (DataClassification::ConfidentialData as u8)
                < (DataClassification::RestrictedData as u8)
        );
    }

    #[test]
    fn test_data_classification_equality() {
        assert_eq!(
            DataClassification::PublicData,
            DataClassification::PublicData
        );
        assert_ne!(
            DataClassification::PublicData,
            DataClassification::InternalData
        );
        assert_ne!(
            DataClassification::ConfidentialData,
            DataClassification::RestrictedData
        );
    }

    #[test]
    fn test_data_category_variants() {
        let categories = vec![
            DataCategory::PII,
            DataCategory::PHI,
            DataCategory::Financial,
            DataCategory::ModelData,
            DataCategory::SystemLogs,
            DataCategory::EvolutionPatterns,
            DataCategory::BusinessData,
        ];

        // Ensure all categories are distinct
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
    fn test_data_metadata_creation() {
        let metadata = DataMetadata {
            classification: DataClassification::ConfidentialData,
            category: DataCategory::Financial,
            retention_period: Some(Duration::days(2555)), // 7 years
            encryption_required: true,
            allowed_regions: vec!["US".to_string(), "EU".to_string()],
            audit_required: true,
            owner: Some("finance_team".to_string()),
            created_at: chrono::Utc::now(),
        };

        assert_eq!(
            metadata.classification,
            DataClassification::ConfidentialData
        );
        assert_eq!(metadata.category, DataCategory::Financial);
        assert!(metadata.encryption_required);
        assert!(metadata.audit_required);
        assert_eq!(metadata.allowed_regions.len(), 2);
        assert_eq!(metadata.owner, Some("finance_team".to_string()));
    }

    #[test]
    fn test_data_metadata_serialization() {
        let metadata = DataMetadata {
            classification: DataClassification::RestrictedData,
            category: DataCategory::PHI,
            retention_period: Some(Duration::days(2190)), // 6 years
            encryption_required: true,
            allowed_regions: vec!["US".to_string()],
            audit_required: true,
            owner: Some("medical_records".to_string()),
            created_at: chrono::Utc::now(),
        };

        let serialized = serde_json::to_string(&metadata).unwrap();
        let deserialized: DataMetadata = serde_json::from_str(&serialized).unwrap();

        assert_eq!(metadata.classification, deserialized.classification);
        assert_eq!(metadata.category, deserialized.category);
        assert_eq!(metadata.retention_period, deserialized.retention_period);
        assert_eq!(
            metadata.encryption_required,
            deserialized.encryption_required
        );
        assert_eq!(metadata.allowed_regions, deserialized.allowed_regions);
        assert_eq!(metadata.owner, deserialized.owner);
    }

    #[test]
    fn test_data_classifier_default() {
        let _classifier = DataClassifier::default();
        let _classifier2 = DataClassifier::new();

        // Both classifiers should be created with same default rules (internals are private)
    }

    #[test]
    fn test_classify_pii_minimum_classification() {
        let classifier = DataClassifier::new();

        // PII with sufficient classification
        let result = classifier.classify(
            DataCategory::PII,
            DataClassification::ConfidentialData,
            vec!["US".to_string()],
        );
        assert!(result.is_ok());

        let result = classifier.classify(
            DataCategory::PII,
            DataClassification::RestrictedData,
            vec!["US".to_string()],
        );
        assert!(result.is_ok());

        // PII with insufficient classification
        let result = classifier.classify(
            DataCategory::PII,
            DataClassification::PublicData,
            vec!["US".to_string()],
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            ComplianceError::ClassificationError(msg) => {
                assert!(msg.contains("requires at least"));
                assert!(msg.contains("ConfidentialData"));
            }
            _ => panic!("Expected ClassificationError"),
        }

        let result = classifier.classify(
            DataCategory::PII,
            DataClassification::InternalData,
            vec!["US".to_string()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_classify_phi_strict_requirements() {
        let classifier = DataClassifier::new();

        // PHI requires RestrictedData classification
        let result = classifier.classify(
            DataCategory::PHI,
            DataClassification::RestrictedData,
            vec!["US".to_string()],
        );
        assert!(result.is_ok());
        let metadata = result.unwrap();
        assert!(metadata.encryption_required);
        assert!(metadata.audit_required);
        assert_eq!(metadata.retention_period, Some(Duration::days(365 * 6)));

        // PHI with lower classification should fail
        for classification in &[
            DataClassification::PublicData,
            DataClassification::InternalData,
            DataClassification::ConfidentialData,
        ] {
            let result =
                classifier.classify(DataCategory::PHI, *classification, vec!["US".to_string()]);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_classify_phi_region_restrictions() {
        let classifier = DataClassifier::new();

        // Allowed regions
        for region in &["US", "EU", "UK", "CA", "AU"] {
            let result = classifier.classify(
                DataCategory::PHI,
                DataClassification::RestrictedData,
                vec![region.to_string()],
            );
            assert!(result.is_ok(), "Failed for region: {}", region);
        }

        // Restricted regions for PHI
        for region in &["CN", "RU"] {
            let result = classifier.classify(
                DataCategory::PHI,
                DataClassification::RestrictedData,
                vec![region.to_string()],
            );
            assert!(result.is_err());
            match result.unwrap_err() {
                ComplianceError::RegionRestriction { region: r } => {
                    assert_eq!(r, *region);
                }
                _ => panic!("Expected RegionRestriction"),
            }
        }
    }

    #[test]
    fn test_classify_financial_data() {
        let classifier = DataClassifier::new();

        // Financial data requirements
        let result = classifier.classify(
            DataCategory::Financial,
            DataClassification::ConfidentialData,
            vec!["US".to_string(), "EU".to_string()],
        );
        assert!(result.is_ok());
        let metadata = result.unwrap();
        assert!(metadata.encryption_required);
        assert!(metadata.audit_required);
        assert_eq!(metadata.retention_period, Some(Duration::days(365 * 7)));
        assert_eq!(metadata.allowed_regions, vec!["US", "EU"]);

        // Financial with RestrictedData should also work
        let result = classifier.classify(
            DataCategory::Financial,
            DataClassification::RestrictedData,
            vec!["US".to_string()],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_classify_model_data() {
        let classifier = DataClassifier::new();

        // Model data has lower requirements
        let result = classifier.classify(
            DataCategory::ModelData,
            DataClassification::InternalData,
            vec!["US".to_string()],
        );
        assert!(result.is_ok());
        let metadata = result.unwrap();
        assert!(!metadata.encryption_required); // Not required for model data
        assert!(!metadata.audit_required); // Only required for Confidential/Restricted
        assert_eq!(metadata.retention_period, Some(Duration::days(365 * 5)));

        // Can use higher classification too
        let result = classifier.classify(
            DataCategory::ModelData,
            DataClassification::ConfidentialData,
            vec!["US".to_string()],
        );
        assert!(result.is_ok());
        let metadata = result.unwrap();
        assert!(metadata.audit_required); // Required due to classification level
    }

    #[test]
    fn test_classify_system_logs() {
        let classifier = DataClassifier::new();

        // System logs (no specific rule, should use defaults)
        let result = classifier.classify(
            DataCategory::SystemLogs,
            DataClassification::InternalData,
            vec!["US".to_string()],
        );
        assert!(result.is_err()); // No rule defined
        match result.unwrap_err() {
            ComplianceError::ClassificationError(msg) => {
                assert!(msg.contains("No rule for category"));
            }
            _ => panic!("Expected ClassificationError"),
        }
    }

    #[test]
    fn test_classify_evolution_data() {
        let classifier = DataClassifier::new();

        // Evolution data classification
        let result = classifier.classify(
            DataCategory::ModelData,
            DataClassification::EvolutionData,
            vec!["US".to_string(), "EU".to_string()],
        );
        assert!(result.is_ok());
        let metadata = result.unwrap();
        assert_eq!(metadata.classification, DataClassification::EvolutionData);
        assert!(!metadata.audit_required); // EvolutionData doesn't automatically require audit
    }

    #[test]
    fn test_classify_multiple_regions() {
        let classifier = DataClassifier::new();

        // Test with multiple regions
        let regions = vec![
            "US".to_string(),
            "EU".to_string(),
            "UK".to_string(),
            "CA".to_string(),
        ];

        let result = classifier.classify(
            DataCategory::PII,
            DataClassification::ConfidentialData,
            regions.clone(),
        );
        assert!(result.is_ok());
        let metadata = result.unwrap();
        assert_eq!(metadata.allowed_regions, regions);
    }

    #[test]
    fn test_classify_with_mixed_regions() {
        let classifier = DataClassifier::new();

        // Mix of allowed and restricted regions for PHI
        let result = classifier.classify(
            DataCategory::PHI,
            DataClassification::RestrictedData,
            vec!["US".to_string(), "CN".to_string()], // CN is restricted for PHI
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            ComplianceError::RegionRestriction { region } => {
                assert_eq!(region, "CN");
            }
            _ => panic!("Expected RegionRestriction"),
        }
    }

    #[test]
    fn test_classification_level_helper() {
        // Classification levels are tested through public API behavior
        // PublicData < InternalData < ConfidentialData < RestrictedData
        // EvolutionData is at ConfidentialData level
        // The actual numeric levels are internal implementation details

        // Test through the ordering implementation
        assert!((DataClassification::PublicData as u8) < (DataClassification::InternalData as u8));
        assert!(
            (DataClassification::InternalData as u8) < (DataClassification::ConfidentialData as u8)
        );
        assert!(
            (DataClassification::ConfidentialData as u8)
                < (DataClassification::RestrictedData as u8)
        );
    }

    #[test]
    fn test_audit_requirement_logic() {
        let classifier = DataClassifier::new();

        // Audit required for Confidential and Restricted
        let result = classifier.classify(
            DataCategory::Financial,
            DataClassification::ConfidentialData,
            vec!["US".to_string()],
        );
        assert!(result.unwrap().audit_required);

        let result = classifier.classify(
            DataCategory::Financial,
            DataClassification::RestrictedData,
            vec!["US".to_string()],
        );
        assert!(result.unwrap().audit_required);

        // Audit not required for lower classifications
        let result = classifier.classify(
            DataCategory::ModelData,
            DataClassification::InternalData,
            vec!["US".to_string()],
        );
        assert!(!result.unwrap().audit_required);
    }

    #[test]
    fn test_empty_regions() {
        let classifier = DataClassifier::new();

        // Empty regions should be allowed (no restriction)
        let result = classifier.classify(
            DataCategory::PII,
            DataClassification::ConfidentialData,
            vec![],
        );
        assert!(result.is_ok());
        let metadata = result.unwrap();
        assert!(metadata.allowed_regions.is_empty());
    }

    #[test]
    fn test_metadata_owner_field() {
        let classifier = DataClassifier::new();

        let result = classifier.classify(
            DataCategory::Financial,
            DataClassification::ConfidentialData,
            vec!["US".to_string()],
        );
        assert!(result.is_ok());
        let metadata = result.unwrap();
        assert_eq!(metadata.owner, None); // Owner not set by classifier
    }

    #[test]
    fn test_metadata_timestamp() {
        let classifier = DataClassifier::new();

        let before = chrono::Utc::now();
        let result = classifier.classify(
            DataCategory::Financial,
            DataClassification::ConfidentialData,
            vec!["US".to_string()],
        );
        let after = chrono::Utc::now();

        assert!(result.is_ok());
        let metadata = result.unwrap();
        assert!(metadata.created_at >= before);
        assert!(metadata.created_at <= after);
    }

    #[test]
    fn test_all_data_categories_serialization() {
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
            let serialized = serde_json::to_string(&category).unwrap();
            let deserialized: DataCategory = serde_json::from_str(&serialized).unwrap();
            assert_eq!(category, deserialized);
        }
    }

    #[test]
    fn test_all_data_classifications_serialization() {
        let classifications = vec![
            DataClassification::PublicData,
            DataClassification::InternalData,
            DataClassification::ConfidentialData,
            DataClassification::RestrictedData,
            DataClassification::EvolutionData,
        ];

        for classification in classifications {
            let serialized = serde_json::to_string(&classification).unwrap();
            let deserialized: DataClassification = serde_json::from_str(&serialized).unwrap();
            assert_eq!(classification, deserialized);
        }
    }
}
