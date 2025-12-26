//! Data sovereignty enforcement

use crate::error::{MultiRegionError, MultiRegionResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Data sovereignty manager
pub struct DataSovereignty {
    rules: HashMap<String, SovereigntyRule>,
    violations: Vec<SovereigntyViolation>,
}

/// Sovereignty rule for data classification and jurisdiction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereigntyRule {
    /// Rule identifier
    pub id: String,
    /// Data classification level
    pub data_classification: String,
    /// Allowed jurisdictions
    pub allowed_jurisdictions: Vec<String>,
    /// Forbidden jurisdictions
    pub forbidden_jurisdictions: Vec<String>,
    /// Required encryption level
    pub encryption_level: EncryptionLevel,
    /// Data residency requirements
    pub residency_requirements: ResidencyRequirements,
    /// Compliance frameworks
    pub compliance_frameworks: Vec<String>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Encryption level requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EncryptionLevel {
    /// No encryption required
    None,
    /// Standard encryption (AES-256)
    Standard,
    /// Strong encryption (AES-256 + additional protection)
    Strong,
    /// Military-grade encryption
    MilitaryGrade,
}

/// Data residency requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyRequirements {
    /// Must remain in origin jurisdiction
    pub must_remain_in_origin: bool,
    /// Allowed transit jurisdictions
    pub allowed_transit: Vec<String>,
    /// Maximum distance from origin (km)
    pub max_distance_km: Option<u32>,
    /// Backup storage requirements
    pub backup_requirements: BackupRequirements,
}

/// Backup storage requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRequirements {
    /// Cross-border backup allowed
    pub cross_border_allowed: bool,
    /// Required backup jurisdictions
    pub required_jurisdictions: Vec<String>,
    /// Minimum backup copies
    pub min_copies: u32,
}

/// Data sovereignty violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereigntyViolation {
    /// Violation ID
    pub id: String,
    /// Rule that was violated
    pub rule_id: String,
    /// Data identifier
    pub data_id: String,
    /// Source jurisdiction
    pub source_jurisdiction: String,
    /// Target jurisdiction
    pub target_jurisdiction: String,
    /// Violation type
    pub violation_type: ViolationType,
    /// Violation timestamp
    pub timestamp: DateTime<Utc>,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Types of sovereignty violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Data moved to forbidden jurisdiction
    ForbiddenJurisdiction,
    /// Insufficient encryption level
    InsufficientEncryption,
    /// Residency requirement violation
    ResidencyViolation,
    /// Compliance framework violation
    ComplianceViolation,
    /// Cross-border backup violation
    BackupViolation,
}

/// Data placement request
#[derive(Debug, Clone)]
pub struct DataPlacementRequest {
    /// Data identifier
    pub data_id: String,
    /// Data classification
    pub classification: String,
    /// Source jurisdiction
    pub source_jurisdiction: String,
    /// Target jurisdiction
    pub target_jurisdiction: String,
    /// Operation type (store, move, backup)
    pub operation: DataOperation,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Data operation types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataOperation {
    /// Store data in target jurisdiction
    Store,
    /// Move data from source to target
    Move,
    /// Create backup in target jurisdiction
    Backup,
    /// Temporary processing in target jurisdiction
    Process,
}

impl DataSovereignty {
    /// Create new data sovereignty manager
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            violations: Vec::new(),
        }
    }

    /// Add sovereignty rule
    pub fn add_rule(&mut self, rule: SovereigntyRule) {
        self.rules.insert(rule.id.clone(), rule);
    }

    /// Remove sovereignty rule
    pub fn remove_rule(&mut self, rule_id: &str) -> Option<SovereigntyRule> {
        self.rules.remove(rule_id)
    }

    /// Get sovereignty rule
    pub fn get_rule(&self, rule_id: &str) -> Option<&SovereigntyRule> {
        self.rules.get(rule_id)
    }

    /// Validate data placement request
    pub fn validate_placement(&mut self, request: &DataPlacementRequest) -> MultiRegionResult<()> {
        for rule in self.rules.values() {
            if rule.data_classification == request.classification {
                if let Err(violation) = self.check_rule_compliance(rule, request) {
                    self.violations.push(violation.clone());
                    return Err(MultiRegionError::SovereigntyViolation {
                        message: format!(
                            "Sovereignty violation for data {} in rule {}: {:?}",
                            request.data_id, rule.id, violation.violation_type
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Check if placement complies with specific rule
    fn check_rule_compliance(
        &self,
        rule: &SovereigntyRule,
        request: &DataPlacementRequest,
    ) -> Result<(), SovereigntyViolation> {
        // Check forbidden jurisdictions
        if rule
            .forbidden_jurisdictions
            .contains(&request.target_jurisdiction)
        {
            return Err(SovereigntyViolation {
                id: uuid::Uuid::new_v4().to_string(),
                rule_id: rule.id.clone(),
                data_id: request.data_id.clone(),
                source_jurisdiction: request.source_jurisdiction.clone(),
                target_jurisdiction: request.target_jurisdiction.clone(),
                violation_type: ViolationType::ForbiddenJurisdiction,
                timestamp: Utc::now(),
                context: request.metadata.clone(),
            });
        }

        // Check allowed jurisdictions (if specified)
        if !rule.allowed_jurisdictions.is_empty()
            && !rule
                .allowed_jurisdictions
                .contains(&request.target_jurisdiction)
        {
            return Err(SovereigntyViolation {
                id: uuid::Uuid::new_v4().to_string(),
                rule_id: rule.id.clone(),
                data_id: request.data_id.clone(),
                source_jurisdiction: request.source_jurisdiction.clone(),
                target_jurisdiction: request.target_jurisdiction.clone(),
                violation_type: ViolationType::ForbiddenJurisdiction,
                timestamp: Utc::now(),
                context: request.metadata.clone(),
            });
        }

        // Check residency requirements
        if rule.residency_requirements.must_remain_in_origin
            && request.source_jurisdiction != request.target_jurisdiction
            && request.operation != DataOperation::Process
        {
            return Err(SovereigntyViolation {
                id: uuid::Uuid::new_v4().to_string(),
                rule_id: rule.id.clone(),
                data_id: request.data_id.clone(),
                source_jurisdiction: request.source_jurisdiction.clone(),
                target_jurisdiction: request.target_jurisdiction.clone(),
                violation_type: ViolationType::ResidencyViolation,
                timestamp: Utc::now(),
                context: request.metadata.clone(),
            });
        }

        // Check backup requirements for backup operations
        if request.operation == DataOperation::Backup {
            if !rule
                .residency_requirements
                .backup_requirements
                .cross_border_allowed
                && request.source_jurisdiction != request.target_jurisdiction
            {
                return Err(SovereigntyViolation {
                    id: uuid::Uuid::new_v4().to_string(),
                    rule_id: rule.id.clone(),
                    data_id: request.data_id.clone(),
                    source_jurisdiction: request.source_jurisdiction.clone(),
                    target_jurisdiction: request.target_jurisdiction.clone(),
                    violation_type: ViolationType::BackupViolation,
                    timestamp: Utc::now(),
                    context: request.metadata.clone(),
                });
            }
        }

        Ok(())
    }

    /// Get all violations
    pub fn get_violations(&self) -> &[SovereigntyViolation] {
        &self.violations
    }

    /// Get violations for specific data
    pub fn get_data_violations(&self, data_id: &str) -> Vec<&SovereigntyViolation> {
        self.violations
            .iter()
            .filter(|v| v.data_id == data_id)
            .collect()
    }

    /// Clear violations
    pub fn clear_violations(&mut self) {
        self.violations.clear();
    }

    /// Get supported jurisdictions for data classification
    pub fn get_supported_jurisdictions(&self, classification: &str) -> Vec<String> {
        let mut jurisdictions = Vec::new();
        for rule in self.rules.values() {
            if rule.data_classification == classification {
                jurisdictions.extend(rule.allowed_jurisdictions.clone());
            }
        }
        jurisdictions.sort();
        jurisdictions.dedup();
        jurisdictions
    }

    /// Check if jurisdiction combination is allowed
    pub fn is_jurisdiction_allowed(
        &self,
        classification: &str,
        source: &str,
        target: &str,
        operation: &DataOperation,
    ) -> bool {
        let request = DataPlacementRequest {
            data_id: "test".to_string(),
            classification: classification.to_string(),
            source_jurisdiction: source.to_string(),
            target_jurisdiction: target.to_string(),
            operation: operation.clone(),
            metadata: HashMap::new(),
        };

        // Create a temporary copy to avoid mutable borrow issues
        let mut temp_sovereignty = DataSovereignty {
            rules: self.rules.clone(),
            violations: Vec::new(),
        };

        temp_sovereignty.validate_placement(&request).is_ok()
    }
}

impl Default for DataSovereignty {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_rule() -> SovereigntyRule {
        SovereigntyRule {
            id: "test-rule".to_string(),
            data_classification: "PII".to_string(),
            allowed_jurisdictions: vec!["US".to_string(), "EU".to_string()],
            forbidden_jurisdictions: vec!["CN".to_string(), "RU".to_string()],
            encryption_level: EncryptionLevel::Strong,
            residency_requirements: ResidencyRequirements {
                must_remain_in_origin: false,
                allowed_transit: vec!["UK".to_string()],
                max_distance_km: Some(10000),
                backup_requirements: BackupRequirements {
                    cross_border_allowed: true,
                    required_jurisdictions: vec!["US".to_string()],
                    min_copies: 2,
                },
            },
            compliance_frameworks: vec!["GDPR".to_string(), "CCPA".to_string()],
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_sovereignty_rule_creation() {
        let rule = create_test_rule();
        assert_eq!(rule.id, "test-rule");
        assert_eq!(rule.data_classification, "PII");
        assert!(rule.allowed_jurisdictions.contains(&"US".to_string()));
        assert!(rule.forbidden_jurisdictions.contains(&"CN".to_string()));
    }

    #[test]
    fn test_data_sovereignty_manager_creation() {
        let sovereignty = DataSovereignty::new();
        assert_eq!(sovereignty.rules.len(), 0);
        assert_eq!(sovereignty.violations.len(), 0);
    }

    #[test]
    fn test_add_and_get_rule() {
        let mut sovereignty = DataSovereignty::new();
        let rule = create_test_rule();

        sovereignty.add_rule(rule.clone());
        assert_eq!(sovereignty.rules.len(), 1);

        let retrieved = sovereignty.get_rule("test-rule");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test-rule");
    }

    #[test]
    fn test_forbidden_jurisdiction_violation() {
        let mut sovereignty = DataSovereignty::new();
        sovereignty.add_rule(create_test_rule());

        let request = DataPlacementRequest {
            data_id: "test-data".to_string(),
            classification: "PII".to_string(),
            source_jurisdiction: "US".to_string(),
            target_jurisdiction: "CN".to_string(), // Forbidden
            operation: DataOperation::Store,
            metadata: HashMap::new(),
        };

        let result = sovereignty.validate_placement(&request);
        assert!(result.is_err());
        assert_eq!(sovereignty.violations.len(), 1);

        let violation = &sovereignty.violations[0];
        assert!(matches!(
            violation.violation_type,
            ViolationType::ForbiddenJurisdiction
        ));
    }

    #[test]
    fn test_allowed_jurisdiction_success() {
        let mut sovereignty = DataSovereignty::new();
        sovereignty.add_rule(create_test_rule());

        let request = DataPlacementRequest {
            data_id: "test-data".to_string(),
            classification: "PII".to_string(),
            source_jurisdiction: "US".to_string(),
            target_jurisdiction: "EU".to_string(), // Allowed
            operation: DataOperation::Store,
            metadata: HashMap::new(),
        };

        let result = sovereignty.validate_placement(&request);
        assert!(result.is_ok());
        assert_eq!(sovereignty.violations.len(), 0);
    }

    #[test]
    fn test_residency_requirements() {
        let mut sovereignty = DataSovereignty::new();
        let mut rule = create_test_rule();
        rule.residency_requirements.must_remain_in_origin = true;
        sovereignty.add_rule(rule);

        let request = DataPlacementRequest {
            data_id: "test-data".to_string(),
            classification: "PII".to_string(),
            source_jurisdiction: "US".to_string(),
            target_jurisdiction: "EU".to_string(),
            operation: DataOperation::Move, // Not processing, so should violate
            metadata: HashMap::new(),
        };

        let result = sovereignty.validate_placement(&request);
        assert!(result.is_err());
        assert_eq!(sovereignty.violations.len(), 1);

        let violation = &sovereignty.violations[0];
        assert!(matches!(
            violation.violation_type,
            ViolationType::ResidencyViolation
        ));
    }

    #[test]
    fn test_processing_exception_for_residency() {
        let mut sovereignty = DataSovereignty::new();
        let mut rule = create_test_rule();
        rule.residency_requirements.must_remain_in_origin = true;
        sovereignty.add_rule(rule);

        let request = DataPlacementRequest {
            data_id: "test-data".to_string(),
            classification: "PII".to_string(),
            source_jurisdiction: "US".to_string(),
            target_jurisdiction: "EU".to_string(),
            operation: DataOperation::Process, // Processing should be allowed
            metadata: HashMap::new(),
        };

        let result = sovereignty.validate_placement(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backup_cross_border_restriction() {
        let mut sovereignty = DataSovereignty::new();
        let mut rule = create_test_rule();
        rule.residency_requirements
            .backup_requirements
            .cross_border_allowed = false;
        sovereignty.add_rule(rule);

        let request = DataPlacementRequest {
            data_id: "test-data".to_string(),
            classification: "PII".to_string(),
            source_jurisdiction: "US".to_string(),
            target_jurisdiction: "EU".to_string(),
            operation: DataOperation::Backup,
            metadata: HashMap::new(),
        };

        let result = sovereignty.validate_placement(&request);
        assert!(result.is_err());
        assert_eq!(sovereignty.violations.len(), 1);

        let violation = &sovereignty.violations[0];
        assert!(matches!(
            violation.violation_type,
            ViolationType::BackupViolation
        ));
    }

    #[test]
    fn test_get_supported_jurisdictions() {
        let mut sovereignty = DataSovereignty::new();
        sovereignty.add_rule(create_test_rule());

        let jurisdictions = sovereignty.get_supported_jurisdictions("PII");
        assert!(jurisdictions.contains(&"US".to_string()));
        assert!(jurisdictions.contains(&"EU".to_string()));
        assert!(!jurisdictions.contains(&"CN".to_string()));
    }

    #[test]
    fn test_is_jurisdiction_allowed() {
        let mut sovereignty = DataSovereignty::new();
        sovereignty.add_rule(create_test_rule());

        // Test allowed combination
        assert!(sovereignty.is_jurisdiction_allowed("PII", "US", "EU", &DataOperation::Store));

        // Test forbidden combination
        assert!(!sovereignty.is_jurisdiction_allowed("PII", "US", "CN", &DataOperation::Store));
    }

    #[test]
    fn test_remove_rule() {
        let mut sovereignty = DataSovereignty::new();
        sovereignty.add_rule(create_test_rule());
        assert_eq!(sovereignty.rules.len(), 1);

        let removed = sovereignty.remove_rule("test-rule");
        assert!(removed.is_some());
        assert_eq!(sovereignty.rules.len(), 0);

        let not_found = sovereignty.remove_rule("non-existent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_clear_violations() {
        let mut sovereignty = DataSovereignty::new();
        sovereignty.add_rule(create_test_rule());

        // Create a violation
        let request = DataPlacementRequest {
            data_id: "test-data".to_string(),
            classification: "PII".to_string(),
            source_jurisdiction: "US".to_string(),
            target_jurisdiction: "CN".to_string(),
            operation: DataOperation::Store,
            metadata: HashMap::new(),
        };

        let _ = sovereignty.validate_placement(&request);
        assert_eq!(sovereignty.violations.len(), 1);

        sovereignty.clear_violations();
        assert_eq!(sovereignty.violations.len(), 0);
    }

    #[test]
    fn test_encryption_level_ordering() {
        assert!(EncryptionLevel::None < EncryptionLevel::Standard);
        assert!(EncryptionLevel::Standard < EncryptionLevel::Strong);
        assert!(EncryptionLevel::Strong < EncryptionLevel::MilitaryGrade);
    }

    #[test]
    fn test_sovereignty_rule_serialization() {
        let rule = create_test_rule();
        let json = serde_json::to_string(&rule);
        assert!(json.is_ok());

        let deserialized: Result<SovereigntyRule, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }
}
