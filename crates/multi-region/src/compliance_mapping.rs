//! Compliance requirements mapping to region restrictions and data sovereignty rules
//!
//! This module provides comprehensive compliance framework mapping:
//! - GDPR, CCPA, HIPAA, SOX, PCI-DSS compliance rules
//! - Data classification and handling requirements
//! - Cross-border transfer restrictions
//! - Audit trail and reporting requirements
//! - Real-time compliance validation

use crate::error::MultiRegionResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Compliance mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMappingConfig {
    /// Enabled compliance frameworks
    pub enabled_frameworks: Vec<ComplianceFramework>,
    /// Default data classification
    pub default_classification: DataClassification,
    /// Regional compliance overrides
    pub regional_overrides: HashMap<String, RegionalCompliance>,
    /// Audit configuration
    pub audit_config: AuditConfig,
    /// Cross-border transfer policy
    pub transfer_policy: CrossBorderPolicy,
    /// Encryption requirements
    pub encryption_requirements: EncryptionRequirements,
}

/// Supported compliance frameworks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceFramework {
    /// General Data Protection Regulation (EU)
    GDPR,
    /// California Consumer Privacy Act (US)
    CCPA,
    /// Health Insurance Portability and Accountability Act (US)
    HIPAA,
    /// Sarbanes-Oxley Act (US)
    SOX,
    /// Payment Card Industry Data Security Standard
    PCIDSS,
    /// Federal Information Security Management Act (US)
    FISMA,
    /// Personal Information Protection and Electronic Documents Act (Canada)
    PIPEDA,
    /// Lei Geral de Proteção de Dados (Brazil)
    LGPD,
    /// Data Protection Act (UK)
    DPA,
    /// Personal Data Protection Act (Singapore)
    PDPA,
}

/// Data classification levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DataClassification {
    /// Public data - no restrictions
    Public,
    /// Internal data - organization only
    Internal,
    /// Confidential data - restricted access
    Confidential,
    /// Personally Identifiable Information
    PII,
    /// Protected Health Information
    PHI,
    /// Payment Card Information
    PCI,
    /// Financial data
    Financial,
    /// Classified/Secret data
    Classified,
}

/// Regional compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalCompliance {
    /// Region identifier
    pub region_id: String,
    /// Applicable frameworks
    pub frameworks: Vec<ComplianceFramework>,
    /// Data residency requirements
    pub residency_requirements: DataResidencyRequirements,
    /// Transfer restrictions
    pub transfer_restrictions: TransferRestrictions,
    /// Local regulations
    pub local_regulations: Vec<LocalRegulation>,
}

/// Data residency requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataResidencyRequirements {
    /// Data must remain in country
    pub must_remain_in_country: bool,
    /// Data must remain in jurisdiction
    pub must_remain_in_jurisdiction: bool,
    /// Allowed processing locations
    pub allowed_processing_locations: Vec<String>,
    /// Backup storage requirements
    pub backup_requirements: BackupStorageRequirements,
    /// Transit restrictions
    pub transit_restrictions: TransitRestrictions,
}

/// Backup storage requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupStorageRequirements {
    /// Cross-border backup allowed
    pub cross_border_allowed: bool,
    /// Required backup jurisdictions
    pub required_jurisdictions: Vec<String>,
    /// Encryption requirements for backups
    pub encryption_required: bool,
    /// Maximum backup retention period (days)
    pub max_retention_days: Option<u32>,
}

/// Transit restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitRestrictions {
    /// Allowed transit countries
    pub allowed_transit_countries: Vec<String>,
    /// Forbidden transit countries
    pub forbidden_transit_countries: Vec<String>,
    /// Encryption required in transit
    pub encryption_in_transit_required: bool,
    /// Maximum transit time (hours)
    pub max_transit_time_hours: Option<u32>,
}

/// Cross-border transfer restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferRestrictions {
    /// Transfers allowed to
    pub allowed_destinations: Vec<String>,
    /// Transfers forbidden to
    pub forbidden_destinations: Vec<String>,
    /// Adequacy decisions required
    pub adequacy_decisions_required: bool,
    /// Standard contractual clauses required
    pub scc_required: bool,
    /// Binding corporate rules required
    pub bcr_required: bool,
    /// Consent required for transfers
    pub consent_required: bool,
}

/// Local regulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalRegulation {
    /// Regulation name
    pub name: String,
    /// Regulation description
    pub description: String,
    /// Applicable data types
    pub applicable_data_types: Vec<DataClassification>,
    /// Requirements
    pub requirements: Vec<ComplianceRequirement>,
}

/// Compliance requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRequirement {
    /// Requirement ID
    pub id: String,
    /// Requirement description
    pub description: String,
    /// Requirement type
    pub requirement_type: RequirementType,
    /// Severity level
    pub severity: SeverityLevel,
    /// Applicable data classifications
    pub applicable_classifications: Vec<DataClassification>,
    /// Implementation deadline
    pub deadline: Option<DateTime<Utc>>,
}

/// Requirement types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequirementType {
    /// Data encryption requirement
    Encryption,
    /// Access control requirement
    AccessControl,
    /// Audit logging requirement
    AuditLogging,
    /// Data retention requirement
    DataRetention,
    /// Data deletion requirement
    DataDeletion,
    /// Consent management requirement
    ConsentManagement,
    /// Breach notification requirement
    BreachNotification,
    /// Data portability requirement
    DataPortability,
}

/// Severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SeverityLevel {
    /// Low severity - guidance
    Low,
    /// Medium severity - recommended
    Medium,
    /// High severity - required
    High,
    /// Critical severity - mandatory
    Critical,
}

/// Cross-border transfer policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossBorderPolicy {
    /// Default policy
    pub default_policy: TransferPolicy,
    /// Framework-specific policies
    pub framework_policies: HashMap<ComplianceFramework, TransferPolicy>,
    /// Data-specific policies
    pub data_policies: HashMap<DataClassification, TransferPolicy>,
}

/// Transfer policy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferPolicy {
    /// Allow all transfers
    Allow,
    /// Block all transfers
    Block,
    /// Require explicit approval
    RequireApproval,
    /// Allow with conditions
    ConditionalAllow { conditions: Vec<String> },
}

/// Encryption requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionRequirements {
    /// Framework-specific encryption requirements
    pub framework_requirements: HashMap<ComplianceFramework, EncryptionSpec>,
    /// Data-specific encryption requirements
    pub data_requirements: HashMap<DataClassification, EncryptionSpec>,
    /// Regional encryption requirements
    pub regional_requirements: HashMap<String, EncryptionSpec>,
}

/// Encryption specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionSpec {
    /// Minimum encryption algorithm
    pub algorithm: String,
    /// Minimum key size (bits)
    pub min_key_size: u32,
    /// Key management requirements
    pub key_management: KeyManagementRequirements,
    /// Encryption at rest required
    pub at_rest_required: bool,
    /// Encryption in transit required
    pub in_transit_required: bool,
    /// Encryption in processing required
    pub in_processing_required: bool,
}

/// Key management requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementRequirements {
    /// Hardware security module required
    pub hsm_required: bool,
    /// Key rotation interval (days)
    pub rotation_interval_days: u32,
    /// Key escrow required
    pub escrow_required: bool,
    /// Multi-person control required
    pub multi_person_control: bool,
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Log all data access
    pub log_data_access: bool,
    /// Log all data modifications
    pub log_data_modifications: bool,
    /// Log cross-border transfers
    pub log_cross_border_transfers: bool,
    /// Audit retention period (days)
    pub retention_days: u32,
    /// Audit log encryption required
    pub encrypt_logs: bool,
    /// Real-time monitoring
    pub real_time_monitoring: bool,
}

/// Compliance validation result
#[derive(Debug, Clone)]
pub struct ComplianceValidationResult {
    /// Validation passed
    pub is_compliant: bool,
    /// Applicable frameworks
    pub applicable_frameworks: Vec<ComplianceFramework>,
    /// Validation errors
    pub violations: Vec<ComplianceViolation>,
    /// Warnings
    pub warnings: Vec<ComplianceWarning>,
    /// Required actions
    pub required_actions: Vec<ComplianceAction>,
}

/// Compliance violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    /// Violation ID
    pub id: String,
    /// Compliance framework
    pub framework: ComplianceFramework,
    /// Violation type
    pub violation_type: ViolationType,
    /// Severity level
    pub severity: SeverityLevel,
    /// Description
    pub description: String,
    /// Data classification involved
    pub data_classification: DataClassification,
    /// Source region
    pub source_region: String,
    /// Target region
    pub target_region: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Remediation actions
    pub remediation_actions: Vec<String>,
}

/// Violation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Unauthorized cross-border transfer
    UnauthorizedTransfer,
    /// Inadequate encryption
    InadequateEncryption,
    /// Missing consent
    MissingConsent,
    /// Data retention violation
    RetentionViolation,
    /// Access control violation
    AccessControlViolation,
    /// Audit logging violation
    AuditLoggingViolation,
    /// Breach notification violation
    BreachNotificationViolation,
}

/// Compliance warning
#[derive(Debug, Clone)]
pub struct ComplianceWarning {
    /// Warning message
    pub message: String,
    /// Recommendation
    pub recommendation: String,
    /// Framework
    pub framework: ComplianceFramework,
}

/// Compliance action
#[derive(Debug, Clone)]
pub struct ComplianceAction {
    /// Action description
    pub description: String,
    /// Action type
    pub action_type: ActionType,
    /// Priority
    pub priority: SeverityLevel,
    /// Deadline
    pub deadline: Option<DateTime<Utc>>,
}

/// Action types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionType {
    /// Implement encryption
    ImplementEncryption,
    /// Obtain consent
    ObtainConsent,
    /// Update access controls
    UpdateAccessControls,
    /// Enable audit logging
    EnableAuditLogging,
    /// Configure data retention
    ConfigureDataRetention,
    /// Setup breach notification
    SetupBreachNotification,
}

/// Compliance mapping manager
pub struct ComplianceMappingManager {
    config: ComplianceMappingConfig,
    framework_rules: HashMap<ComplianceFramework, Vec<ComplianceRule>>,
    violation_history: Vec<ComplianceViolation>,
}

/// Compliance rule
#[derive(Debug, Clone)]
pub struct ComplianceRule {
    /// Rule ID
    pub id: String,
    /// Framework
    pub framework: ComplianceFramework,
    /// Applicable data classifications
    pub data_classifications: Vec<DataClassification>,
    /// Applicable regions
    pub regions: Vec<String>,
    /// Rule validator
    pub validator: RuleValidator,
}

/// Rule validator function type
pub type RuleValidator = fn(&ComplianceContext) -> MultiRegionResult<bool>;

/// Compliance context for validation
#[derive(Debug, Clone)]
pub struct ComplianceContext {
    /// Data classification
    pub data_classification: DataClassification,
    /// Source region
    pub source_region: String,
    /// Target region
    pub target_region: String,
    /// Operation type
    pub operation: DataOperation,
    /// Encryption status
    pub encryption_enabled: bool,
    /// Consent status
    pub consent_obtained: bool,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Data operation types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataOperation {
    /// Store data
    Store,
    /// Retrieve data
    Retrieve,
    /// Transfer data
    Transfer,
    /// Process data
    Process,
    /// Delete data
    Delete,
    /// Backup data
    Backup,
}

impl ComplianceMappingManager {
    /// Create new compliance mapping manager
    pub fn new(config: ComplianceMappingConfig) -> Self {
        let mut manager = Self {
            config,
            framework_rules: HashMap::new(),
            violation_history: Vec::new(),
        };

        // Initialize framework rules
        manager.initialize_framework_rules();
        manager
    }

    /// Initialize compliance framework rules
    fn initialize_framework_rules(&mut self) {
        // GDPR rules
        self.add_gdpr_rules();

        // CCPA rules
        self.add_ccpa_rules();

        // HIPAA rules
        self.add_hipaa_rules();

        // PCI-DSS rules
        self.add_pci_dss_rules();

        // Add other frameworks...
    }

    /// Add GDPR-specific rules
    fn add_gdpr_rules(&mut self) {
        let gdpr_rules = vec![
            ComplianceRule {
                id: "gdpr_pii_transfer".to_string(),
                framework: ComplianceFramework::GDPR,
                data_classifications: vec![DataClassification::PII],
                regions: vec!["EU".to_string()],
                validator: Self::validate_gdpr_pii_transfer,
            },
            ComplianceRule {
                id: "gdpr_encryption".to_string(),
                framework: ComplianceFramework::GDPR,
                data_classifications: vec![
                    DataClassification::PII,
                    DataClassification::Confidential,
                ],
                regions: vec!["EU".to_string()],
                validator: Self::validate_gdpr_encryption,
            },
            ComplianceRule {
                id: "gdpr_consent".to_string(),
                framework: ComplianceFramework::GDPR,
                data_classifications: vec![DataClassification::PII],
                regions: vec!["EU".to_string()],
                validator: Self::validate_gdpr_consent,
            },
        ];

        self.framework_rules
            .insert(ComplianceFramework::GDPR, gdpr_rules);
    }

    /// Add CCPA-specific rules
    fn add_ccpa_rules(&mut self) {
        let ccpa_rules = vec![ComplianceRule {
            id: "ccpa_pii_rights".to_string(),
            framework: ComplianceFramework::CCPA,
            data_classifications: vec![DataClassification::PII],
            regions: vec!["US".to_string(), "CA".to_string()],
            validator: Self::validate_ccpa_pii_rights,
        }];

        self.framework_rules
            .insert(ComplianceFramework::CCPA, ccpa_rules);
    }

    /// Add HIPAA-specific rules
    fn add_hipaa_rules(&mut self) {
        let hipaa_rules = vec![ComplianceRule {
            id: "hipaa_phi_encryption".to_string(),
            framework: ComplianceFramework::HIPAA,
            data_classifications: vec![DataClassification::PHI],
            regions: vec!["US".to_string()],
            validator: Self::validate_hipaa_phi_encryption,
        }];

        self.framework_rules
            .insert(ComplianceFramework::HIPAA, hipaa_rules);
    }

    /// Add PCI-DSS-specific rules
    fn add_pci_dss_rules(&mut self) {
        let pci_rules = vec![ComplianceRule {
            id: "pci_card_data_encryption".to_string(),
            framework: ComplianceFramework::PCIDSS,
            data_classifications: vec![DataClassification::PCI],
            regions: vec![], // Global
            validator: Self::validate_pci_card_data_encryption,
        }];

        self.framework_rules
            .insert(ComplianceFramework::PCIDSS, pci_rules);
    }

    /// Validate compliance for data operation
    pub fn validate_compliance(
        &mut self,
        context: &ComplianceContext,
    ) -> MultiRegionResult<ComplianceValidationResult> {
        let mut violations = Vec::new();
        let warnings = Vec::new();
        let mut required_actions = Vec::new();
        let mut applicable_frameworks = Vec::new();

        // Check each enabled framework
        for framework in &self.config.enabled_frameworks {
            applicable_frameworks.push(*framework);

            if let Some(rules) = self.framework_rules.get(framework) {
                for rule in rules {
                    // Check if rule applies to this context
                    if self.rule_applies_to_context(rule, context) {
                        match (rule.validator)(context) {
                            Ok(true) => {
                                // Rule passed
                            }
                            Ok(false) => {
                                // Rule failed - create violation
                                let violation = self.create_violation(rule, context)?;
                                violations.push(violation);
                            }
                            Err(e) => {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }

        // Generate required actions based on violations
        for violation in &violations {
            required_actions.extend(self.generate_remediation_actions(violation));
        }

        // Store violations in history
        self.violation_history.extend(violations.clone());

        let is_compliant = violations.is_empty();

        Ok(ComplianceValidationResult {
            is_compliant,
            applicable_frameworks,
            violations,
            warnings,
            required_actions,
        })
    }

    /// Check if rule applies to context
    fn rule_applies_to_context(&self, rule: &ComplianceRule, context: &ComplianceContext) -> bool {
        // Check data classification
        if !rule.data_classifications.is_empty()
            && !rule
                .data_classifications
                .contains(&context.data_classification)
        {
            return false;
        }

        // Check regions
        if !rule.regions.is_empty() {
            let source_matches = rule.regions.contains(&context.source_region);
            let target_matches = rule.regions.contains(&context.target_region);
            if !source_matches && !target_matches {
                return false;
            }
        }

        true
    }

    /// Create violation from failed rule
    fn create_violation(
        &self,
        rule: &ComplianceRule,
        context: &ComplianceContext,
    ) -> MultiRegionResult<ComplianceViolation> {
        let violation_type = match rule.id.as_str() {
            id if id.contains("transfer") => ViolationType::UnauthorizedTransfer,
            id if id.contains("encryption") => ViolationType::InadequateEncryption,
            id if id.contains("consent") => ViolationType::MissingConsent,
            _ => ViolationType::AccessControlViolation,
        };

        Ok(ComplianceViolation {
            id: uuid::Uuid::new_v4().to_string(),
            framework: rule.framework,
            violation_type,
            severity: SeverityLevel::High,
            description: format!("Compliance rule {} violated", rule.id),
            data_classification: context.data_classification,
            source_region: context.source_region.clone(),
            target_region: context.target_region.clone(),
            timestamp: Utc::now(),
            remediation_actions: vec!["Review compliance requirements".to_string()],
        })
    }

    /// Generate remediation actions for violation
    fn generate_remediation_actions(
        &self,
        violation: &ComplianceViolation,
    ) -> Vec<ComplianceAction> {
        let mut actions = Vec::new();

        match violation.violation_type {
            ViolationType::InadequateEncryption => {
                actions.push(ComplianceAction {
                    description: "Implement required encryption standards".to_string(),
                    action_type: ActionType::ImplementEncryption,
                    priority: violation.severity,
                    deadline: Some(Utc::now() + chrono::Duration::days(30)),
                });
            }
            ViolationType::MissingConsent => {
                actions.push(ComplianceAction {
                    description: "Obtain required user consent".to_string(),
                    action_type: ActionType::ObtainConsent,
                    priority: violation.severity,
                    deadline: Some(Utc::now() + chrono::Duration::days(7)),
                });
            }
            ViolationType::UnauthorizedTransfer => {
                actions.push(ComplianceAction {
                    description: "Review and approve cross-border transfer".to_string(),
                    action_type: ActionType::UpdateAccessControls,
                    priority: violation.severity,
                    deadline: Some(Utc::now() + chrono::Duration::days(1)),
                });
            }
            _ => {
                actions.push(ComplianceAction {
                    description: "Review compliance requirements".to_string(),
                    action_type: ActionType::UpdateAccessControls,
                    priority: violation.severity,
                    deadline: Some(Utc::now() + chrono::Duration::days(14)),
                });
            }
        }

        actions
    }

    /// Get compliance status for region
    pub fn get_region_compliance_status(&self, region_id: &str) -> Option<&RegionalCompliance> {
        self.config.regional_overrides.get(region_id)
    }

    /// Check if transfer is allowed between regions
    pub fn is_transfer_allowed(
        &self,
        data_classification: DataClassification,
        source_region: &str,
        target_region: &str,
    ) -> MultiRegionResult<bool> {
        let context = ComplianceContext {
            data_classification,
            source_region: source_region.to_string(),
            target_region: target_region.to_string(),
            operation: DataOperation::Transfer,
            encryption_enabled: true, // Assume encryption by default
            consent_obtained: false,  // Will be validated by rules
            metadata: HashMap::new(),
        };

        let mut temp_manager = Self::new(self.config.clone());
        let validation_result = temp_manager.validate_compliance(&context)?;
        Ok(validation_result.is_compliant)
    }

    /// Get required encryption specifications
    pub fn get_encryption_requirements(
        &self,
        framework: ComplianceFramework,
        data_classification: DataClassification,
    ) -> Option<&EncryptionSpec> {
        // Check framework-specific requirements first
        if let Some(spec) = self
            .config
            .encryption_requirements
            .framework_requirements
            .get(&framework)
        {
            return Some(spec);
        }

        // Fall back to data-specific requirements
        self.config
            .encryption_requirements
            .data_requirements
            .get(&data_classification)
    }

    /// Get violation history
    pub fn get_violation_history(&self) -> &[ComplianceViolation] {
        &self.violation_history
    }

    /// Clear violation history
    pub fn clear_violation_history(&mut self) {
        self.violation_history.clear();
    }

    // Validator functions for different compliance rules

    /// Validate GDPR PII transfer rules
    fn validate_gdpr_pii_transfer(context: &ComplianceContext) -> MultiRegionResult<bool> {
        // GDPR requires adequacy decision or appropriate safeguards for transfers outside EU
        if context.source_region == "EU" && context.target_region != "EU" {
            // Check if target region has adequacy decision (simplified)
            let adequate_regions = vec!["UK", "Switzerland", "Canada", "Japan"];
            if !adequate_regions.contains(&context.target_region.as_str()) {
                return Ok(false); // Would need SCCs or other safeguards
            }
        }
        Ok(true)
    }

    /// Validate GDPR encryption requirements
    fn validate_gdpr_encryption(context: &ComplianceContext) -> MultiRegionResult<bool> {
        // GDPR recommends encryption for personal data
        Ok(context.encryption_enabled)
    }

    /// Validate GDPR consent requirements
    fn validate_gdpr_consent(context: &ComplianceContext) -> MultiRegionResult<bool> {
        // For PII transfers, consent should be obtained
        if context.operation == DataOperation::Transfer {
            Ok(context.consent_obtained)
        } else {
            Ok(true) // Consent not required for all operations
        }
    }

    /// Validate CCPA PII rights
    fn validate_ccpa_pii_rights(_context: &ComplianceContext) -> MultiRegionResult<bool> {
        // CCPA focuses on consumer rights - simplified validation
        Ok(true)
    }

    /// Validate HIPAA PHI encryption
    fn validate_hipaa_phi_encryption(context: &ComplianceContext) -> MultiRegionResult<bool> {
        // HIPAA requires encryption for PHI
        if context.data_classification == DataClassification::PHI {
            Ok(context.encryption_enabled)
        } else {
            Ok(true)
        }
    }

    /// Validate PCI card data encryption
    fn validate_pci_card_data_encryption(context: &ComplianceContext) -> MultiRegionResult<bool> {
        // PCI-DSS requires encryption for cardholder data
        if context.data_classification == DataClassification::PCI {
            Ok(context.encryption_enabled)
        } else {
            Ok(true)
        }
    }
}

impl Default for ComplianceMappingConfig {
    fn default() -> Self {
        Self {
            enabled_frameworks: vec![
                ComplianceFramework::GDPR,
                ComplianceFramework::CCPA,
                ComplianceFramework::HIPAA,
            ],
            default_classification: DataClassification::Internal,
            regional_overrides: HashMap::new(),
            audit_config: AuditConfig {
                enabled: true,
                log_data_access: true,
                log_data_modifications: true,
                log_cross_border_transfers: true,
                retention_days: 2555, // 7 years
                encrypt_logs: true,
                real_time_monitoring: true,
            },
            transfer_policy: CrossBorderPolicy {
                default_policy: TransferPolicy::RequireApproval,
                framework_policies: HashMap::new(),
                data_policies: HashMap::new(),
            },
            encryption_requirements: EncryptionRequirements {
                framework_requirements: HashMap::new(),
                data_requirements: HashMap::new(),
                regional_requirements: HashMap::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> ComplianceMappingConfig {
        ComplianceMappingConfig {
            enabled_frameworks: vec![ComplianceFramework::GDPR, ComplianceFramework::HIPAA],
            default_classification: DataClassification::Internal,
            regional_overrides: HashMap::new(),
            audit_config: AuditConfig {
                enabled: true,
                log_data_access: true,
                log_data_modifications: true,
                log_cross_border_transfers: true,
                retention_days: 365,
                encrypt_logs: true,
                real_time_monitoring: false,
            },
            transfer_policy: CrossBorderPolicy {
                default_policy: TransferPolicy::RequireApproval,
                framework_policies: HashMap::new(),
                data_policies: HashMap::new(),
            },
            encryption_requirements: EncryptionRequirements {
                framework_requirements: HashMap::new(),
                data_requirements: HashMap::new(),
                regional_requirements: HashMap::new(),
            },
        }
    }

    fn create_test_context() -> ComplianceContext {
        ComplianceContext {
            data_classification: DataClassification::PII,
            source_region: "EU".to_string(),
            target_region: "US".to_string(),
            operation: DataOperation::Transfer,
            encryption_enabled: true,
            consent_obtained: false,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_compliance_manager_creation() {
        let config = create_test_config();
        let manager = ComplianceMappingManager::new(config);
        assert_eq!(manager.framework_rules.len(), 4); // GDPR, CCPA, HIPAA, PCI-DSS rules added
    }

    #[test]
    fn test_data_classification_ordering() {
        assert!(DataClassification::Public < DataClassification::Internal);
        assert!(DataClassification::Internal < DataClassification::Confidential);
        assert!(DataClassification::Confidential < DataClassification::PII);
        assert!(DataClassification::PII < DataClassification::PHI);
        assert!(DataClassification::PHI < DataClassification::Classified);
    }

    #[test]
    fn test_severity_level_ordering() {
        assert!(SeverityLevel::Low < SeverityLevel::Medium);
        assert!(SeverityLevel::Medium < SeverityLevel::High);
        assert!(SeverityLevel::High < SeverityLevel::Critical);
    }

    #[test]
    fn test_compliance_validation_gdpr_transfer() {
        let config = create_test_config();
        let mut manager = ComplianceMappingManager::new(config);
        let context = create_test_context();

        let result = manager.validate_compliance(&context).unwrap();
        assert!(!result.is_compliant); // Should fail due to missing consent
        assert!(!result.violations.is_empty());
        assert!(result
            .applicable_frameworks
            .contains(&ComplianceFramework::GDPR));
    }

    #[test]
    fn test_compliance_validation_with_consent() {
        let config = create_test_config();
        let mut manager = ComplianceMappingManager::new(config);
        let mut context = create_test_context();
        context.consent_obtained = true;
        context.target_region = "UK".to_string(); // Adequate region

        let result = manager.validate_compliance(&context).unwrap();
        assert!(result.is_compliant);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_hipaa_phi_validation() {
        let config = create_test_config();
        let mut manager = ComplianceMappingManager::new(config);
        let mut context = create_test_context();
        context.data_classification = DataClassification::PHI;
        context.source_region = "US".to_string();
        context.encryption_enabled = false;

        let result = manager.validate_compliance(&context).unwrap();
        assert!(!result.is_compliant); // Should fail due to missing encryption for PHI
    }

    #[test]
    fn test_transfer_allowed_validation() {
        let config = create_test_config();
        let manager = ComplianceMappingManager::new(config);

        let allowed = manager
            .is_transfer_allowed(DataClassification::Public, "US", "EU")
            .unwrap();
        assert!(allowed); // Public data should be transferable

        let restricted = manager
            .is_transfer_allowed(
                DataClassification::PII,
                "EU",
                "RU", // Non-adequate region
            )
            .unwrap();
        assert!(!restricted); // PII transfer to non-adequate region should be restricted
    }

    #[test]
    fn test_violation_history_management() {
        let config = create_test_config();
        let mut manager = ComplianceMappingManager::new(config);
        let context = create_test_context();

        // Initial validation creates violations
        let _ = manager.validate_compliance(&context).unwrap();
        assert!(!manager.get_violation_history().is_empty());

        // Clear history
        manager.clear_violation_history();
        assert!(manager.get_violation_history().is_empty());
    }

    #[test]
    fn test_encryption_requirements() {
        let mut config = create_test_config();

        // Add encryption requirements
        let encryption_spec = EncryptionSpec {
            algorithm: "AES-256".to_string(),
            min_key_size: 256,
            key_management: KeyManagementRequirements {
                hsm_required: true,
                rotation_interval_days: 90,
                escrow_required: false,
                multi_person_control: true,
            },
            at_rest_required: true,
            in_transit_required: true,
            in_processing_required: false,
        };

        config
            .encryption_requirements
            .framework_requirements
            .insert(ComplianceFramework::GDPR, encryption_spec.clone());

        let manager = ComplianceMappingManager::new(config);
        let retrieved_spec =
            manager.get_encryption_requirements(ComplianceFramework::GDPR, DataClassification::PII);

        assert!(retrieved_spec.is_some());
        assert_eq!(retrieved_spec.unwrap().algorithm, "AES-256");
        assert_eq!(retrieved_spec.unwrap().min_key_size, 256);
    }

    #[test]
    fn test_regional_compliance_override() {
        let mut config = create_test_config();

        let regional_compliance = RegionalCompliance {
            region_id: "APAC".to_string(),
            frameworks: vec![ComplianceFramework::PDPA],
            residency_requirements: DataResidencyRequirements {
                must_remain_in_country: true,
                must_remain_in_jurisdiction: true,
                allowed_processing_locations: vec!["SG".to_string()],
                backup_requirements: BackupStorageRequirements {
                    cross_border_allowed: false,
                    required_jurisdictions: vec!["SG".to_string()],
                    encryption_required: true,
                    max_retention_days: Some(2555),
                },
                transit_restrictions: TransitRestrictions {
                    allowed_transit_countries: vec!["MY".to_string()],
                    forbidden_transit_countries: vec!["CN".to_string()],
                    encryption_in_transit_required: true,
                    max_transit_time_hours: Some(24),
                },
            },
            transfer_restrictions: TransferRestrictions {
                allowed_destinations: vec!["MY".to_string()],
                forbidden_destinations: vec!["CN".to_string()],
                adequacy_decisions_required: true,
                scc_required: false,
                bcr_required: false,
                consent_required: true,
            },
            local_regulations: vec![],
        };

        config
            .regional_overrides
            .insert("APAC".to_string(), regional_compliance);
        let manager = ComplianceMappingManager::new(config);

        let status = manager.get_region_compliance_status("APAC");
        assert!(status.is_some());
        assert_eq!(status.unwrap().region_id, "APAC");
        assert!(status
            .unwrap()
            .frameworks
            .contains(&ComplianceFramework::PDPA));
    }

    #[test]
    fn test_compliance_framework_serialization() {
        let frameworks = vec![
            ComplianceFramework::GDPR,
            ComplianceFramework::CCPA,
            ComplianceFramework::HIPAA,
            ComplianceFramework::SOX,
            ComplianceFramework::PCIDSS,
        ];

        for framework in frameworks {
            let json = serde_json::to_string(&framework).unwrap();
            let deserialized: ComplianceFramework = serde_json::from_str(&json).unwrap();
            assert_eq!(framework, deserialized);
        }
    }

    #[test]
    fn test_transfer_policy_logic() {
        let policies = vec![
            TransferPolicy::Allow,
            TransferPolicy::Block,
            TransferPolicy::RequireApproval,
            TransferPolicy::ConditionalAllow {
                conditions: vec!["encryption".to_string(), "consent".to_string()],
            },
        ];

        for policy in policies {
            let json = serde_json::to_string(&policy).unwrap();
            let deserialized: TransferPolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(policy, deserialized);
        }
    }

    #[test]
    fn test_violation_remediation_actions() {
        let violation = ComplianceViolation {
            id: "test-violation".to_string(),
            framework: ComplianceFramework::GDPR,
            violation_type: ViolationType::InadequateEncryption,
            severity: SeverityLevel::High,
            description: "Test violation".to_string(),
            data_classification: DataClassification::PII,
            source_region: "EU".to_string(),
            target_region: "US".to_string(),
            timestamp: Utc::now(),
            remediation_actions: vec!["Implement encryption".to_string()],
        };

        let config = create_test_config();
        let manager = ComplianceMappingManager::new(config);
        let actions = manager.generate_remediation_actions(&violation);

        assert!(!actions.is_empty());
        assert_eq!(actions[0].action_type, ActionType::ImplementEncryption);
        assert_eq!(actions[0].priority, SeverityLevel::High);
    }

    #[test]
    fn test_data_operation_types() {
        let operations = vec![
            DataOperation::Store,
            DataOperation::Retrieve,
            DataOperation::Transfer,
            DataOperation::Process,
            DataOperation::Delete,
            DataOperation::Backup,
        ];

        // Test that all operations are distinct
        for (i, op1) in operations.iter().enumerate() {
            for (j, op2) in operations.iter().enumerate() {
                if i != j {
                    assert_ne!(op1, op2);
                }
            }
        }
    }

    #[test]
    fn test_compliance_config_default() {
        let config = ComplianceMappingConfig::default();
        assert!(config
            .enabled_frameworks
            .contains(&ComplianceFramework::GDPR));
        assert!(config
            .enabled_frameworks
            .contains(&ComplianceFramework::CCPA));
        assert!(config
            .enabled_frameworks
            .contains(&ComplianceFramework::HIPAA));
        assert_eq!(config.default_classification, DataClassification::Internal);
        assert!(config.audit_config.enabled);
        assert_eq!(
            config.transfer_policy.default_policy,
            TransferPolicy::RequireApproval
        );
    }
}
