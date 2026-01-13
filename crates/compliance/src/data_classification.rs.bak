//! Data classification for compliance

use crate::error::{ComplianceError, ComplianceResult};
use chrono::Duration;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Data classification levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataClassification {
    /// Public data - no restrictions
    PublicData,
    /// Internal data - organization use only
    InternalData,
    /// Confidential data - restricted access
    ConfidentialData,
    /// Restricted data - highest security
    RestrictedData,
    /// Evolution data - AI system generated
    EvolutionData,
}

/// Data categories for compliance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataCategory {
    /// Personally Identifiable Information
    PII,
    /// Protected Health Information (HIPAA)
    PHI,
    /// Financial data
    Financial,
    /// AI/ML model data
    ModelData,
    /// System logs
    SystemLogs,
    /// Evolution patterns
    EvolutionPatterns,
    /// General business data
    BusinessData,
}

/// Data classification metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMetadata {
    /// Classification level
    pub classification: DataClassification,
    /// Data category
    pub category: DataCategory,
    /// Retention period
    pub retention_period: Option<Duration>,
    /// Encryption required
    pub encryption_required: bool,
    /// Allowed regions for storage
    pub allowed_regions: Vec<String>,
    /// Audit trail required
    pub audit_required: bool,
    /// Data owner
    pub owner: Option<String>,
    /// Created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Additional tags for metadata
    #[serde(default)]
    pub tags: std::collections::HashMap<String, String>,
}

/// Data classifier
pub struct DataClassifier {
    rules: HashMap<DataCategory, ClassificationRule>,
}

/// Classification rule
#[derive(Debug, Clone)]
struct ClassificationRule {
    min_classification: DataClassification,
    encryption_required: bool,
    default_retention: Option<Duration>,
    restricted_regions: Vec<String>,
}

impl DataClassifier {
    /// Create new data classifier
    pub fn new() -> Self {
        let mut rules = HashMap::new();

        // PII requires at least confidential classification
        rules.insert(
            DataCategory::PII,
            ClassificationRule {
                min_classification: DataClassification::ConfidentialData,
                encryption_required: true,
                default_retention: Some(Duration::days(365 * 7)), // 7 years
                restricted_regions: vec![],
            },
        );

        // PHI requires restricted classification (HIPAA)
        rules.insert(
            DataCategory::PHI,
            ClassificationRule {
                min_classification: DataClassification::RestrictedData,
                encryption_required: true,
                default_retention: Some(Duration::days(365 * 6)), // 6 years
                restricted_regions: vec!["CN".to_string(), "RU".to_string()],
            },
        );

        // Financial data
        rules.insert(
            DataCategory::Financial,
            ClassificationRule {
                min_classification: DataClassification::ConfidentialData,
                encryption_required: true,
                default_retention: Some(Duration::days(365 * 7)), // 7 years
                restricted_regions: vec![],
            },
        );

        // Model data
        rules.insert(
            DataCategory::ModelData,
            ClassificationRule {
                min_classification: DataClassification::InternalData,
                encryption_required: false,
                default_retention: Some(Duration::days(365 * 5)), // 5 years
                restricted_regions: vec![],
            },
        );

        Self { rules }
    }

    /// Classify data
    pub fn classify(
        &self,
        category: DataCategory,
        requested_classification: DataClassification,
        regions: Vec<String>,
    ) -> ComplianceResult<DataMetadata> {
        let rule = self.rules.get(&category).ok_or_else(|| {
            ComplianceError::ClassificationError(format!("No rule for category: {category:?}"))
        })?;

        // Check minimum classification requirement
        if self.classification_level(requested_classification)
            < self.classification_level(rule.min_classification)
        {
            return Err(ComplianceError::ClassificationError(format!(
                "Category {:?} requires at least {:?} classification",
                category, rule.min_classification
            )));
        }

        // Check region restrictions
        for region in &regions {
            if rule.restricted_regions.contains(region) {
                return Err(ComplianceError::RegionRestriction {
                    region: region.clone(),
                });
            }
        }

        Ok(DataMetadata {
            classification: requested_classification,
            category,
            retention_period: rule.default_retention,
            encryption_required: rule.encryption_required,
            allowed_regions: regions,
            audit_required: matches!(
                requested_classification,
                DataClassification::ConfidentialData | DataClassification::RestrictedData
            ),
            owner: None,
            created_at: chrono::Utc::now(),
            tags: std::collections::HashMap::new(),
        })
    }

    /// Get classification level numeric value for comparison
    fn classification_level(&self, classification: DataClassification) -> u8 {
        match classification {
            DataClassification::PublicData => 0,
            DataClassification::InternalData => 1,
            DataClassification::ConfidentialData => 2,
            DataClassification::RestrictedData => 3,
            DataClassification::EvolutionData => 2, // Same as confidential
        }
    }
}

impl Default for DataClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_classification_levels() {
        let classifier = DataClassifier::new();

        // Test classification level ordering
        assert!(
            classifier.classification_level(DataClassification::PublicData)
                < classifier.classification_level(DataClassification::InternalData)
        );
        assert!(
            classifier.classification_level(DataClassification::InternalData)
                < classifier.classification_level(DataClassification::ConfidentialData)
        );
        assert!(
            classifier.classification_level(DataClassification::ConfidentialData)
                < classifier.classification_level(DataClassification::RestrictedData)
        );
    }

    #[test]
    fn test_pii_classification() {
        let classifier = DataClassifier::new();

        // PII with proper classification should succeed
        let result = classifier.classify(
            DataCategory::PII,
            DataClassification::ConfidentialData,
            vec!["US".to_string(), "EU".to_string()],
        );
        assert!(result.is_ok());
        let metadata = result?;
        assert_eq!(
            metadata.classification,
            DataClassification::ConfidentialData
        );
        assert!(metadata.encryption_required);
        assert!(metadata.audit_required);

        // PII with insufficient classification should fail
        let result = classifier.classify(
            DataCategory::PII,
            DataClassification::PublicData,
            vec!["US".to_string()],
        );
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComplianceError::ClassificationError(_)
        ));
    }

    #[test]
    fn test_phi_classification_with_region_restriction() {
        let classifier = DataClassifier::new();

        // PHI in allowed region should succeed
        let result = classifier.classify(
            DataCategory::PHI,
            DataClassification::RestrictedData,
            vec!["US".to_string()],
        );
        assert!(result.is_ok());

        // PHI in restricted region should fail
        let result = classifier.classify(
            DataCategory::PHI,
            DataClassification::RestrictedData,
            vec!["CN".to_string()],
        );
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComplianceError::RegionRestriction { .. }
        ));
    }

    #[test]
    fn test_financial_data_classification() {
        let classifier = DataClassifier::new();

        let result = classifier.classify(
            DataCategory::Financial,
            DataClassification::ConfidentialData,
            vec!["US".to_string(), "UK".to_string()],
        );
        assert!(result.is_ok());
        let metadata = result?;
        assert!(metadata.encryption_required);
        assert_eq!(metadata.allowed_regions.len(), 2);
    }

    #[test]
    fn test_retention_periods() {
        let classifier = DataClassifier::new();

        // PII should have 7 year retention
        let pii_result = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();
        assert_eq!(pii_result.retention_period, Some(Duration::days(365 * 7)));

        // PHI should have 6 year retention
        let phi_result = classifier
            .classify(
                DataCategory::PHI,
                DataClassification::RestrictedData,
                vec!["US".to_string()],
            )
            .unwrap();
        assert_eq!(phi_result.retention_period, Some(Duration::days(365 * 6)));
    }

    #[test]
    fn test_evolution_data_classification() {
        let classifier = DataClassifier::new();

        // Evolution data classification
        let result = classifier.classify(
            DataCategory::ModelData,
            DataClassification::EvolutionData,
            vec!["US".to_string()],
        );
        assert!(result.is_ok());
        let metadata = result?;
        assert_eq!(metadata.classification, DataClassification::EvolutionData);
    }
}
