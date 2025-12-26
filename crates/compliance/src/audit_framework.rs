//! Compliance audit framework

use crate::data_classification::{DataCategory, DataClassification};
use crate::error::{ComplianceError, ComplianceResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Compliance engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Enabled regulations
    pub regulations: Vec<Regulation>,
    /// Default data retention days
    pub default_retention_days: u32,
    /// Audit log retention days
    pub audit_retention_days: u32,
    /// Encryption required by default
    pub encryption_by_default: bool,
    /// Allowed storage regions
    pub allowed_regions: Vec<String>,
    /// AI safety enabled
    pub ai_safety_enabled: bool,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            regulations: vec![
                Regulation::GDPR,
                Regulation::HIPAA,
                Regulation::SOC2,
                Regulation::AISafety,
            ],
            default_retention_days: 365 * 7, // 7 years
            audit_retention_days: 365 * 10,  // 10 years
            encryption_by_default: true,
            allowed_regions: vec!["US".to_string(), "EU".to_string()],
            ai_safety_enabled: true,
        }
    }
}

/// Supported regulations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Regulation {
    /// General Data Protection Regulation
    GDPR,
    /// Health Insurance Portability and Accountability Act
    HIPAA,
    /// Service Organization Control 2
    SOC2,
    /// AI Safety and Ethics
    AISafety,
}

/// Compliance audit entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique ID
    pub id: Uuid,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Operation type
    pub operation: AuditOperation,
    /// Actor (user/system)
    pub actor: String,
    /// Data classification
    pub data_classification: DataClassification,
    /// Data category
    pub data_category: DataCategory,
    /// Regulation context
    pub regulation: Option<Regulation>,
    /// Success/failure
    pub success: bool,
    /// Details
    pub details: serde_json::Value,
    /// Data hash for integrity
    pub data_hash: Option<String>,
}

/// Audit operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditOperation {
    /// Data creation
    DataCreated,
    /// Data access
    DataAccessed,
    /// Data modification
    DataModified,
    /// Data deletion
    DataDeleted,
    /// Data export
    DataExported,
    /// Consent granted
    ConsentGranted,
    /// Consent revoked
    ConsentRevoked,
    /// Compliance check
    ComplianceCheck,
    /// Policy update
    PolicyUpdate,
    /// Security event
    SecurityEvent,
}

/// Compliance engine
#[derive(Clone)]
pub struct ComplianceEngine {
    config: Arc<RwLock<ComplianceConfig>>,
    audit_log: Arc<DashMap<Uuid, AuditEntry>>,
    consent_registry: Arc<DashMap<String, ConsentRecord>>,
    policy_engine: Arc<PolicyEngine>,
}

/// Consent record
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConsentRecord {
    user_id: String,
    purposes: Vec<String>,
    granted_at: DateTime<Utc>,
    expires_at: Option<DateTime<Utc>>,
    withdrawn: bool,
}

/// Policy engine for compliance rules
#[derive(Clone)]
struct PolicyEngine {
    rules: DashMap<Regulation, Vec<ComplianceRule>>,
}

/// Compliance rule
#[derive(Clone)]
struct ComplianceRule {
    id: String,
    description: String,
    validator: Arc<dyn Fn(&AuditEntry) -> bool + Send + Sync>,
}

impl std::fmt::Debug for ComplianceRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComplianceRule")
            .field("id", &self.id)
            .field("description", &self.description)
            .field("validator", &"<function>")
            .finish()
    }
}

impl ComplianceEngine {
    /// Create new compliance engine
    pub fn new(config: ComplianceConfig) -> ComplianceResult<Self> {
        if config.allowed_regions.is_empty() {
            return Err(ComplianceError::ConfigurationError(
                "At least one region must be allowed".to_string(),
            ));
        }

        let policy_engine = PolicyEngine::new();

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            audit_log: Arc::new(DashMap::new()),
            consent_registry: Arc::new(DashMap::new()),
            policy_engine: Arc::new(policy_engine),
        })
    }

    /// Record audit entry
    pub async fn audit(
        &self,
        operation: AuditOperation,
        actor: String,
        data_classification: DataClassification,
        data_category: DataCategory,
        details: serde_json::Value,
    ) -> ComplianceResult<Uuid> {
        let entry = AuditEntry {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            operation,
            actor,
            data_classification,
            data_category,
            regulation: self.determine_regulation(&data_category),
            success: true,
            details,
            data_hash: None,
        };

        // Validate against policies
        self.validate_policies(&entry)?;

        // Store audit entry
        let id = entry.id;
        self.audit_log.insert(id, entry);

        Ok(id)
    }

    /// Check compliance for operation
    pub async fn check_compliance(
        &self,
        _operation: &AuditOperation,
        data_category: DataCategory,
        region: &str,
    ) -> ComplianceResult<()> {
        let config = self.config.read();

        // Check region restrictions
        if !config.allowed_regions.contains(&region.to_string()) {
            return Err(ComplianceError::RegionRestriction {
                region: region.to_string(),
            });
        }

        // Check specific regulations
        match data_category {
            DataCategory::PHI => {
                if !config.regulations.contains(&Regulation::HIPAA) {
                    return Err(ComplianceError::ComplianceViolation {
                        regulation: "HIPAA".to_string(),
                        violation: "HIPAA compliance not enabled".to_string(),
                    });
                }
            }
            DataCategory::PII => {
                if !config.regulations.contains(&Regulation::GDPR) {
                    return Err(ComplianceError::ComplianceViolation {
                        regulation: "GDPR".to_string(),
                        violation: "GDPR compliance not enabled".to_string(),
                    });
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Grant consent
    pub async fn grant_consent(
        &self,
        user_id: String,
        purposes: Vec<String>,
        duration_days: Option<u32>,
    ) -> ComplianceResult<()> {
        let expires_at = duration_days.map(|days| Utc::now() + chrono::Duration::days(days as i64));

        let record = ConsentRecord {
            user_id: user_id.clone(),
            purposes: purposes.clone(),
            granted_at: Utc::now(),
            expires_at,
            withdrawn: false,
        };

        self.consent_registry.insert(user_id.clone(), record);

        // Audit the consent
        self.audit(
            AuditOperation::ConsentGranted,
            user_id,
            DataClassification::ConfidentialData,
            DataCategory::PII,
            serde_json::json!({
                "purposes": purposes,
                "duration_days": duration_days,
            }),
        )
        .await?;

        Ok(())
    }

    /// Revoke consent
    pub async fn revoke_consent(&self, user_id: String) -> ComplianceResult<()> {
        if let Some(mut record) = self.consent_registry.get_mut(&user_id) {
            record.withdrawn = true;
        } else {
            return Err(ComplianceError::AccessControlError(
                "No consent record found".to_string(),
            ));
        }

        // Audit the revocation
        self.audit(
            AuditOperation::ConsentRevoked,
            user_id,
            DataClassification::ConfidentialData,
            DataCategory::PII,
            serde_json::json!({}),
        )
        .await?;

        Ok(())
    }

    /// Check if consent is valid
    pub fn has_valid_consent(&self, user_id: &str, purpose: &str) -> bool {
        if let Some(record) = self.consent_registry.get(user_id) {
            if record.withdrawn {
                return false;
            }

            if let Some(expires_at) = record.expires_at {
                if expires_at < Utc::now() {
                    return false;
                }
            }

            record.purposes.contains(&purpose.to_string())
        } else {
            false
        }
    }

    /// Get audit entries for a time range
    pub fn get_audit_entries(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<AuditEntry> {
        self.audit_log
            .iter()
            .filter(|entry| entry.timestamp >= start && entry.timestamp <= end)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Determine regulation based on data category
    fn determine_regulation(&self, category: &DataCategory) -> Option<Regulation> {
        match category {
            DataCategory::PHI => Some(Regulation::HIPAA),
            DataCategory::PII => Some(Regulation::GDPR),
            DataCategory::Financial => Some(Regulation::SOC2),
            DataCategory::ModelData | DataCategory::EvolutionPatterns => {
                if self.config.read().ai_safety_enabled {
                    Some(Regulation::AISafety)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Validate against compliance policies
    fn validate_policies(&self, entry: &AuditEntry) -> ComplianceResult<()> {
        if let Some(regulation) = entry.regulation {
            if let Some(rules) = self.policy_engine.rules.get(&regulation) {
                for rule in rules.iter() {
                    if !(rule.validator)(entry) {
                        return Err(ComplianceError::ComplianceViolation {
                            regulation: format!("{regulation:?}"),
                            violation: rule.description.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

impl PolicyEngine {
    fn new() -> Self {
        let mut engine = Self {
            rules: DashMap::new(),
        };

        // Add default GDPR rules
        engine.add_gdpr_rules();

        // Add default HIPAA rules
        engine.add_hipaa_rules();

        engine
    }

    fn add_gdpr_rules(&mut self) {
        let mut gdpr_rules = vec![];

        // Rule: PII must have consent
        gdpr_rules.push(ComplianceRule {
            id: "gdpr-consent".to_string(),
            description: "PII operations require valid consent".to_string(),
            validator: Arc::new(|_entry| {
                // In real implementation, check consent registry
                true
            }),
        });

        self.rules.insert(Regulation::GDPR, gdpr_rules);
    }

    fn add_hipaa_rules(&mut self) {
        let mut hipaa_rules = vec![];

        // Rule: PHI must be encrypted
        hipaa_rules.push(ComplianceRule {
            id: "hipaa-encryption".to_string(),
            description: "PHI must be encrypted".to_string(),
            validator: Arc::new(|entry| {
                matches!(
                    entry.data_classification,
                    DataClassification::RestrictedData | DataClassification::ConfidentialData
                )
            }),
        });

        self.rules.insert(Regulation::HIPAA, hipaa_rules);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compliance_engine_creation() {
        let config = ComplianceConfig::default();
        let engine = ComplianceEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_compliance_engine_invalid_config() {
        let mut config = ComplianceConfig::default();
        config.allowed_regions.clear();
        let engine = ComplianceEngine::new(config);
        assert!(engine.is_err());
    }

    #[tokio::test]
    async fn test_audit_entry_creation() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();

        let result = engine
            .audit(
                AuditOperation::DataCreated,
                "user123".to_string(),
                DataClassification::InternalData,
                DataCategory::BusinessData,
                serde_json::json!({"file": "report.pdf"}),
            )
            .await;

        assert!(result.is_ok());
        let audit_id = result?;
        assert!(engine.audit_log.contains_key(&audit_id));
    }

    #[tokio::test]
    async fn test_region_compliance_check() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();

        // Allowed region should pass
        let result = engine
            .check_compliance(
                &AuditOperation::DataAccessed,
                DataCategory::BusinessData,
                "US",
            )
            .await;
        assert!(result.is_ok());

        // Disallowed region should fail
        let result = engine
            .check_compliance(
                &AuditOperation::DataAccessed,
                DataCategory::BusinessData,
                "CN",
            )
            .await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComplianceError::RegionRestriction { .. }
        ));
    }

    #[tokio::test]
    async fn test_consent_management() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();

        // Grant consent
        let result = engine
            .grant_consent(
                "user123".to_string(),
                vec!["marketing".to_string(), "analytics".to_string()],
                Some(365),
            )
            .await;
        assert!(result.is_ok());

        // Check consent
        assert!(engine.has_valid_consent("user123", "marketing"));
        assert!(engine.has_valid_consent("user123", "analytics"));
        assert!(!engine.has_valid_consent("user123", "advertising"));

        // Revoke consent
        let result = engine.revoke_consent("user123".to_string()).await;
        assert!(result.is_ok());

        // Check consent after revocation
        assert!(!engine.has_valid_consent("user123", "marketing"));
    }

    #[tokio::test]
    async fn test_phi_compliance_check() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();

        // PHI should require HIPAA
        let result = engine
            .check_compliance(&AuditOperation::DataAccessed, DataCategory::PHI, "US")
            .await;
        assert!(result.is_ok());

        // Disable HIPAA and check again
        let mut config = ComplianceConfig::default();
        config.regulations.retain(|r| *r != Regulation::HIPAA);
        let engine = ComplianceEngine::new(config)?;

        let result = engine
            .check_compliance(&AuditOperation::DataAccessed, DataCategory::PHI, "US")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_audit_log_retrieval() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();

        // Create multiple audit entries
        for i in 0..5 {
            engine
                .audit(
                    AuditOperation::DataAccessed,
                    format!("user{}", i),
                    DataClassification::InternalData,
                    DataCategory::BusinessData,
                    serde_json::json!({"index": i}),
                )
                .await
                .unwrap();
        }

        // Retrieve entries
        let start = Utc::now() - chrono::Duration::hours(1);
        let end = Utc::now() + chrono::Duration::hours(1);
        let entries = engine.get_audit_entries(start, end);

        assert_eq!(entries.len(), 5);
    }

    #[tokio::test]
    async fn test_data_classification_audit() {
        let engine = ComplianceEngine::new(ComplianceConfig::default()).unwrap();

        // Audit restricted data access
        let result = engine
            .audit(
                AuditOperation::DataAccessed,
                "admin".to_string(),
                DataClassification::RestrictedData,
                DataCategory::PHI,
                serde_json::json!({"patient_id": "12345"}),
            )
            .await;

        assert!(result.is_ok());

        // Verify the audit entry
        let audit_id = result?;
        let entry = engine.audit_log.get(&audit_id)?;
        assert_eq!(
            entry.data_classification,
            DataClassification::RestrictedData
        );
        assert_eq!(entry.data_category, DataCategory::PHI);
        assert_eq!(entry.regulation, Some(Regulation::HIPAA));
    }
}
