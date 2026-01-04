//! Compliance-aware data handling

use crate::error::{GlobalKnowledgeGraphError, GlobalKnowledgeGraphResult};
use crate::graph_manager::{Edge, Node};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

/// Compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Enable compliance checking
    pub enable_compliance: bool,
    /// Default data classification
    pub default_classification: DataClassification,
    /// Regional compliance rules
    pub regional_rules: HashMap<String, RegionalCompliance>,
    /// Global retention policy in days
    pub global_retention_days: u32,
    /// Enable audit logging
    pub enable_audit_log: bool,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        let mut regional_rules = HashMap::new();

        // GDPR for EU
        regional_rules.insert(
            "eu-west-1".to_string(),
            RegionalCompliance {
                regulation: "GDPR".to_string(),
                allowed_classifications: vec![
                    DataClassification::Public,
                    DataClassification::Internal,
                    DataClassification::Confidential,
                ],
                pii_handling: PiiHandling::Pseudonymize,
                retention_days: 365,
                cross_border_allowed: false,
                required_encryption: true,
            },
        );

        // CCPA for US West
        regional_rules.insert(
            "us-west-2".to_string(),
            RegionalCompliance {
                regulation: "CCPA".to_string(),
                allowed_classifications: vec![
                    DataClassification::Public,
                    DataClassification::Internal,
                    DataClassification::Confidential,
                    DataClassification::Restricted,
                ],
                pii_handling: PiiHandling::Encrypt,
                retention_days: 730,
                cross_border_allowed: true,
                required_encryption: true,
            },
        );

        // General US compliance
        regional_rules.insert(
            "us-east-1".to_string(),
            RegionalCompliance {
                regulation: "SOC2".to_string(),
                allowed_classifications: vec![
                    DataClassification::Public,
                    DataClassification::Internal,
                    DataClassification::Confidential,
                    DataClassification::Restricted,
                ],
                pii_handling: PiiHandling::Encrypt,
                retention_days: 730,
                cross_border_allowed: true,
                required_encryption: false,
            },
        );

        // PDPA for Asia Pacific
        regional_rules.insert(
            "ap-southeast-1".to_string(),
            RegionalCompliance {
                regulation: "PDPA".to_string(),
                allowed_classifications: vec![
                    DataClassification::Public,
                    DataClassification::Internal,
                    DataClassification::Confidential,
                ],
                pii_handling: PiiHandling::Mask,
                retention_days: 540,
                cross_border_allowed: false,
                required_encryption: true,
            },
        );

        Self {
            enable_compliance: true,
            default_classification: DataClassification::Internal,
            regional_rules,
            global_retention_days: 730,
            enable_audit_log: true,
        }
    }
}

/// Data classification levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DataClassification {
    /// Publicly available data
    Public,
    /// Internal use only
    Internal,
    /// Confidential data
    Confidential,
    /// Restricted access
    Restricted,
}

/// PII handling strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PiiHandling {
    /// No special handling
    None,
    /// Mask PII fields
    Mask,
    /// Encrypt PII fields
    Encrypt,
    /// Pseudonymize PII
    Pseudonymize,
    /// Remove PII
    Remove,
}

/// Regional compliance rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalCompliance {
    /// Regulation name
    pub regulation: String,
    /// Allowed data classifications
    pub allowed_classifications: Vec<DataClassification>,
    /// PII handling requirement
    pub pii_handling: PiiHandling,
    /// Data retention in days
    pub retention_days: u32,
    /// Cross-border data transfer allowed
    pub cross_border_allowed: bool,
    /// Encryption required
    pub required_encryption: bool,
}

/// Compliance violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    /// Violation ID
    pub id: String,
    /// Violation type
    pub violation_type: ViolationType,
    /// Affected data ID
    pub data_id: String,
    /// Source region
    pub source_region: String,
    /// Target region (if applicable)
    pub target_region: Option<String>,
    /// Details
    pub details: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Violation type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ViolationType {
    /// Unauthorized cross-border transfer
    CrossBorderTransfer,
    /// Invalid data classification
    InvalidClassification,
    /// PII mishandling
    PiiMishandling,
    /// Retention policy violation
    RetentionViolation,
    /// Missing encryption
    MissingEncryption,
    /// Access control violation
    AccessViolation,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Entry ID
    pub id: String,
    /// Action performed
    pub action: AuditAction,
    /// User or service
    pub actor: String,
    /// Affected resource
    pub resource_id: String,
    /// Resource type
    pub resource_type: String,
    /// Region
    pub region: String,
    /// Success status
    pub success: bool,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Audit action types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuditAction {
    /// Data access
    Access,
    /// Data creation
    Create,
    /// Data update
    Update,
    /// Data deletion
    Delete,
    /// Data export
    Export,
    /// Data import
    Import,
    /// Compliance check
    ComplianceCheck,
}

/// Data residency requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataResidency {
    /// Data origin region
    pub origin_region: String,
    /// Allowed regions
    pub allowed_regions: HashSet<String>,
    /// Prohibited regions
    pub prohibited_regions: HashSet<String>,
}

/// Compliance checker trait
#[async_trait]
pub trait ComplianceChecker: Send + Sync {
    /// Check if data can be stored in region
    async fn can_store_in_region(
        &self,
        data: &Node,
        region: &str,
    ) -> GlobalKnowledgeGraphResult<bool>;

    /// Check if data can be transferred between regions
    async fn can_transfer(
        &self,
        data: &Node,
        from_region: &str,
        to_region: &str,
    ) -> GlobalKnowledgeGraphResult<bool>;

    /// Get required PII handling
    async fn get_pii_handling(&self, region: &str) -> PiiHandling;
}

/// Compliance handler for regional regulations
pub struct ComplianceHandler {
    config: Arc<ComplianceConfig>,
    violations: Arc<DashMap<String, ComplianceViolation>>,
    audit_log: Arc<DashMap<String, Vec<AuditEntry>>>,
    data_residency: Arc<DashMap<String, DataResidency>>,
    pii_fields: Arc<RwLock<HashSet<String>>>,
}

impl ComplianceHandler {
    /// Create new compliance handler
    pub fn new(config: ComplianceConfig) -> Self {
        let mut pii_fields = HashSet::new();
        // Common PII fields
        pii_fields.insert("email".to_string());
        pii_fields.insert("phone".to_string());
        pii_fields.insert("ssn".to_string());
        pii_fields.insert("credit_card".to_string());
        pii_fields.insert("passport".to_string());
        pii_fields.insert("driver_license".to_string());
        pii_fields.insert("date_of_birth".to_string());
        pii_fields.insert("full_name".to_string());
        pii_fields.insert("address".to_string());

        Self {
            config: Arc::new(config),
            violations: Arc::new(DashMap::new()),
            audit_log: Arc::new(DashMap::new()),
            data_residency: Arc::new(DashMap::new()),
            pii_fields: Arc::new(RwLock::new(pii_fields)),
        }
    }

    /// Register PII field
    pub fn register_pii_field(&self, field: String) {
        self.pii_fields.write().insert(field);
    }

    /// Check node compliance
    pub async fn check_node_compliance(
        &self,
        node: &Node,
        target_region: &str,
    ) -> GlobalKnowledgeGraphResult<()> {
        if !self.config.enable_compliance {
            return Ok(());
        }

        // Check if region has compliance rules
        let rules = self
            .config
            .regional_rules
            .get(target_region)
            .ok_or_else(|| GlobalKnowledgeGraphError::ComplianceViolation {
                regulation: "Unknown".to_string(),
                region: target_region.to_string(),
                details: "No compliance rules defined for region".to_string(),
            })?;

        // Check data classification
        let classification = self.get_node_classification(node);
        if !rules.allowed_classifications.contains(&classification) {
            return Err(GlobalKnowledgeGraphError::ComplianceViolation {
                regulation: rules.regulation.clone(),
                region: target_region.to_string(),
                details: format!("Data classification {:?} not allowed", classification),
            });
        }

        // Check data residency
        if node.region != target_region {
            self.check_cross_border_transfer(&node.region, target_region)?;
        }

        // Check PII handling
        if self.contains_pii(node) {
            self.validate_pii_handling(node, &rules.pii_handling)?;
        }

        // Log compliance check
        if self.config.enable_audit_log {
            self.log_audit_entry(AuditEntry {
                id: Uuid::new_v4().to_string(),
                action: AuditAction::ComplianceCheck,
                actor: "system".to_string(),
                resource_id: node.id.clone(),
                resource_type: "node".to_string(),
                region: target_region.to_string(),
                success: true,
                metadata: HashMap::new(),
                timestamp: chrono::Utc::now(),
            })
            .await;
        }

        Ok(())
    }

    /// Check edge compliance
    pub async fn check_edge_compliance(
        &self,
        _edge: &Edge,
        source_region: &str,
        target_region: &str,
    ) -> GlobalKnowledgeGraphResult<()> {
        if !self.config.enable_compliance {
            return Ok(());
        }

        // Check if cross-region edge is allowed
        if source_region != target_region {
            self.check_cross_border_transfer(source_region, target_region)?;
        }

        Ok(())
    }

    /// Check cross-border data transfer
    fn check_cross_border_transfer(
        &self,
        from_region: &str,
        to_region: &str,
    ) -> GlobalKnowledgeGraphResult<()> {
        let from_rules = self.config.regional_rules.get(from_region).ok_or_else(|| {
            GlobalKnowledgeGraphError::ComplianceViolation {
                regulation: "Unknown".to_string(),
                region: from_region.to_string(),
                details: "No compliance rules defined for source region".to_string(),
            }
        })?;

        let to_rules = self.config.regional_rules.get(to_region).ok_or_else(|| {
            GlobalKnowledgeGraphError::ComplianceViolation {
                regulation: "Unknown".to_string(),
                region: to_region.to_string(),
                details: "No compliance rules defined for target region".to_string(),
            }
        })?;

        if !from_rules.cross_border_allowed || !to_rules.cross_border_allowed {
            let violation = ComplianceViolation {
                id: Uuid::new_v4().to_string(),
                violation_type: ViolationType::CrossBorderTransfer,
                data_id: "".to_string(),
                source_region: from_region.to_string(),
                target_region: Some(to_region.to_string()),
                details: "Cross-border transfer not allowed".to_string(),
                timestamp: chrono::Utc::now(),
            };

            self.violations
                .insert(violation.id.clone(), violation.clone());

            return Err(GlobalKnowledgeGraphError::DataSovereigntyViolation {
                origin_region: from_region.to_string(),
                target_region: to_region.to_string(),
            });
        }

        Ok(())
    }

    /// Get node data classification
    fn get_node_classification(&self, node: &Node) -> DataClassification {
        node.properties
            .get("classification")
            .and_then(|v| v.as_str())
            .and_then(|s| match s {
                "public" => Some(DataClassification::Public),
                "internal" => Some(DataClassification::Internal),
                "confidential" => Some(DataClassification::Confidential),
                "restricted" => Some(DataClassification::Restricted),
                _ => None,
            })
            .unwrap_or(self.config.default_classification.clone())
    }

    /// Check if node contains PII
    fn contains_pii(&self, node: &Node) -> bool {
        let pii_fields = self.pii_fields.read();
        node.properties.keys().any(|key| pii_fields.contains(key))
    }

    /// Validate PII handling
    fn validate_pii_handling(
        &self,
        node: &Node,
        required_handling: &PiiHandling,
    ) -> GlobalKnowledgeGraphResult<()> {
        match required_handling {
            PiiHandling::None => Ok(()),
            PiiHandling::Encrypt => {
                // Check if PII fields are encrypted
                let pii_fields = self.pii_fields.read();
                for (key, value) in &node.properties {
                    if pii_fields.contains(key) && !self.is_encrypted(value) {
                        return Err(GlobalKnowledgeGraphError::ComplianceViolation {
                            regulation: "PII Protection".to_string(),
                            region: node.region.clone(),
                            details: format!("PII field '{}' not encrypted", key),
                        });
                    }
                }
                Ok(())
            }
            _ => Ok(()), // Other handling types would have specific implementations
        }
    }

    /// Check if value appears to be encrypted
    fn is_encrypted(&self, value: &serde_json::Value) -> bool {
        if let Some(str_value) = value.as_str() {
            // Simple heuristic: encrypted values are typically base64 with certain length
            str_value.len() > 32
                && str_value
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '+' || c == '/' || c == '=')
        } else {
            false
        }
    }

    /// Apply PII handling to node
    pub async fn apply_pii_handling(
        &self,
        mut node: Node,
        region: &str,
    ) -> GlobalKnowledgeGraphResult<Node> {
        if let Some(rules) = self.config.regional_rules.get(region) {
            match rules.pii_handling {
                PiiHandling::Mask => {
                    let pii_fields = self.pii_fields.read();
                    for (key, value) in node.properties.iter_mut() {
                        if pii_fields.contains(key) {
                            *value = serde_json::Value::String("***MASKED***".to_string());
                        }
                    }
                }
                PiiHandling::Remove => {
                    let pii_fields = self.pii_fields.read();
                    node.properties.retain(|key, _| !pii_fields.contains(key));
                }
                PiiHandling::Pseudonymize => {
                    let pii_fields = self.pii_fields.read();
                    for (key, value) in node.properties.iter_mut() {
                        if pii_fields.contains(key) {
                            *value =
                                serde_json::Value::String(format!("PSEUDO_{}", Uuid::new_v4()));
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(node)
    }

    /// Set data residency requirements
    pub fn set_data_residency(&self, data_id: String, residency: DataResidency) {
        self.data_residency.insert(data_id, residency);
    }

    /// Check data residency
    pub fn check_data_residency(
        &self,
        data_id: &str,
        target_region: &str,
    ) -> GlobalKnowledgeGraphResult<()> {
        if let Some(residency) = self.data_residency.get(data_id) {
            if residency.prohibited_regions.contains(target_region) {
                return Err(GlobalKnowledgeGraphError::DataSovereigntyViolation {
                    origin_region: residency.origin_region.clone(),
                    target_region: target_region.to_string(),
                });
            }

            if !residency.allowed_regions.is_empty()
                && !residency.allowed_regions.contains(target_region)
            {
                return Err(GlobalKnowledgeGraphError::DataSovereigntyViolation {
                    origin_region: residency.origin_region.clone(),
                    target_region: target_region.to_string(),
                });
            }
        }
        Ok(())
    }

    /// Log audit entry
    pub async fn log_audit_entry(&self, entry: AuditEntry) {
        if self.config.enable_audit_log {
            self.audit_log
                .entry(entry.region.clone())
                .or_insert_with(Vec::new)
                .push(entry);
        }
    }

    /// Get audit log for region
    pub fn get_audit_log(&self, region: &str, limit: usize) -> Vec<AuditEntry> {
        self.audit_log
            .get(region)
            .map(|entries| entries.iter().rev().take(limit).cloned().collect())
            .unwrap_or_default()
    }

    /// Get compliance violations
    pub fn get_violations(&self, limit: usize) -> Vec<ComplianceViolation> {
        self.violations
            .iter()
            .take(limit)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Check retention compliance
    pub async fn check_retention_compliance(
        &self,
        node: &Node,
        region: &str,
    ) -> GlobalKnowledgeGraphResult<()> {
        if let Some(rules) = self.config.regional_rules.get(region) {
            let age_days = (chrono::Utc::now() - node.created_at).num_days() as u32;

            if age_days > rules.retention_days {
                let violation = ComplianceViolation {
                    id: Uuid::new_v4().to_string(),
                    violation_type: ViolationType::RetentionViolation,
                    data_id: node.id.clone(),
                    source_region: region.to_string(),
                    target_region: None,
                    details: format!(
                        "Data exceeds retention period of {} days",
                        rules.retention_days
                    ),
                    timestamp: chrono::Utc::now(),
                };

                self.violations.insert(violation.id.clone(), violation);

                return Err(GlobalKnowledgeGraphError::ComplianceViolation {
                    regulation: rules.regulation.clone(),
                    region: region.to_string(),
                    details: "Data retention period exceeded".to_string(),
                });
            }
        }
        Ok(())
    }

    /// Filter nodes by compliance
    pub async fn filter_compliant_nodes(&self, nodes: Vec<Node>, region: &str) -> Vec<Node> {
        let mut compliant_nodes = Vec::new();

        for node in nodes {
            if self.check_node_compliance(&node, region).await.is_ok() {
                compliant_nodes.push(node);
            }
        }

        compliant_nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compliance_handler_creation() {
        let config = ComplianceConfig::default();
        let handler = ComplianceHandler::new(config);
        assert_eq!(handler.violations.len(), 0);
    }

    #[tokio::test]
    async fn test_node_compliance_check() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let mut properties = HashMap::new();
        properties.insert(
            "classification".to_string(),
            serde_json::Value::String("public".to_string()),
        );

        let node = Node {
            id: "test-node".to_string(),
            node_type: "entity".to_string(),
            properties,
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let result = handler.check_node_compliance(&node, "us-east-1").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_invalid_classification() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let mut properties = HashMap::new();
        properties.insert(
            "classification".to_string(),
            serde_json::Value::String("restricted".to_string()),
        );

        let node = Node {
            id: "test-node".to_string(),
            node_type: "entity".to_string(),
            properties,
            region: "eu-west-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let result = handler.check_node_compliance(&node, "eu-west-1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cross_border_transfer() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let result = handler.check_cross_border_transfer("eu-west-1", "us-east-1");
        assert!(result.is_err()); // EU doesn't allow cross-border transfers
    }

    #[tokio::test]
    async fn test_cross_border_transfer_allowed() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let result = handler.check_cross_border_transfer("us-east-1", "us-west-2");
        assert!(result.is_ok()); // US regions allow cross-border transfers
    }

    #[tokio::test]
    async fn test_pii_detection() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let mut properties = HashMap::new();
        properties.insert(
            "email".to_string(),
            serde_json::Value::String("test@example.com".to_string()),
        );

        let node = Node {
            id: "test-node".to_string(),
            node_type: "entity".to_string(),
            properties,
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        assert!(handler.contains_pii(&node));
    }

    #[tokio::test]
    async fn test_register_pii_field() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        handler.register_pii_field("custom_pii".to_string());

        let mut properties = HashMap::new();
        properties.insert(
            "custom_pii".to_string(),
            serde_json::Value::String("sensitive".to_string()),
        );

        let node = Node {
            id: "test-node".to_string(),
            node_type: "entity".to_string(),
            properties,
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        assert!(handler.contains_pii(&node));
    }

    #[tokio::test]
    async fn test_apply_pii_masking() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let mut properties = HashMap::new();
        properties.insert(
            "email".to_string(),
            serde_json::Value::String("test@example.com".to_string()),
        );
        properties.insert(
            "name".to_string(),
            serde_json::Value::String("John Doe".to_string()),
        );

        let node = Node {
            id: "test-node".to_string(),
            node_type: "entity".to_string(),
            properties,
            region: "ap-southeast-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let masked = handler
            .apply_pii_handling(node, "ap-southeast-1")
            .await
            .unwrap();
        assert_eq!(masked.properties.get("email").unwrap(), "***MASKED***");
    }

    #[tokio::test]
    async fn test_apply_pii_removal() {
        let mut config = ComplianceConfig::default();
        config.regional_rules.get_mut("us-east-1")?.pii_handling = PiiHandling::Remove;

        let handler = ComplianceHandler::new(config);

        let mut properties = HashMap::new();
        properties.insert(
            "email".to_string(),
            serde_json::Value::String("test@example.com".to_string()),
        );
        properties.insert(
            "data".to_string(),
            serde_json::Value::String("non-pii".to_string()),
        );

        let node = Node {
            id: "test-node".to_string(),
            node_type: "entity".to_string(),
            properties,
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let processed = handler.apply_pii_handling(node, "us-east-1").await.unwrap();
        assert!(!processed.properties.contains_key("email"));
        assert!(processed.properties.contains_key("data"));
    }

    #[tokio::test]
    async fn test_apply_pii_pseudonymization() {
        let mut config = ComplianceConfig::default();
        config.regional_rules.get_mut("us-east-1")?.pii_handling = PiiHandling::Pseudonymize;

        let handler = ComplianceHandler::new(config);

        let mut properties = HashMap::new();
        properties.insert(
            "email".to_string(),
            serde_json::Value::String("test@example.com".to_string()),
        );

        let node = Node {
            id: "test-node".to_string(),
            node_type: "entity".to_string(),
            properties,
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let processed = handler.apply_pii_handling(node, "us-east-1").await.unwrap();
        let email_value = processed.properties.get("email").unwrap().as_str().unwrap();
        assert!(email_value.starts_with("PSEUDO_"));
    }

    #[tokio::test]
    async fn test_data_residency() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let mut allowed_regions = HashSet::new();
        allowed_regions.insert("us-east-1".to_string());
        allowed_regions.insert("us-west-2".to_string());

        let residency = DataResidency {
            origin_region: "us-east-1".to_string(),
            allowed_regions,
            prohibited_regions: HashSet::new(),
        };

        handler.set_data_residency("data-123".to_string(), residency);

        let result1 = handler.check_data_residency("data-123", "us-west-2");
        assert!(result1.is_ok());

        let result2 = handler.check_data_residency("data-123", "eu-west-1");
        assert!(result2.is_err());
    }

    #[tokio::test]
    async fn test_prohibited_regions() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let mut prohibited_regions = HashSet::new();
        prohibited_regions.insert("eu-west-1".to_string());

        let residency = DataResidency {
            origin_region: "us-east-1".to_string(),
            allowed_regions: HashSet::new(),
            prohibited_regions,
        };

        handler.set_data_residency("data-456".to_string(), residency);

        let result = handler.check_data_residency("data-456", "eu-west-1");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_audit_logging() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let entry = AuditEntry {
            id: Uuid::new_v4().to_string(),
            action: AuditAction::Access,
            actor: "user123".to_string(),
            resource_id: "node-123".to_string(),
            resource_type: "node".to_string(),
            region: "us-east-1".to_string(),
            success: true,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        };

        handler.log_audit_entry(entry).await;

        let log = handler.get_audit_log("us-east-1", 10);
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].actor, "user123");
    }

    #[tokio::test]
    async fn test_audit_log_disabled() {
        let mut config = ComplianceConfig::default();
        config.enable_audit_log = false;

        let handler = ComplianceHandler::new(config);

        let entry = AuditEntry {
            id: Uuid::new_v4().to_string(),
            action: AuditAction::Access,
            actor: "user123".to_string(),
            resource_id: "node-123".to_string(),
            resource_type: "node".to_string(),
            region: "us-east-1".to_string(),
            success: true,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        };

        handler.log_audit_entry(entry).await;

        let log = handler.get_audit_log("us-east-1", 10);
        assert_eq!(log.len(), 0);
    }

    #[tokio::test]
    async fn test_get_violations() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let violation = ComplianceViolation {
            id: "v1".to_string(),
            violation_type: ViolationType::CrossBorderTransfer,
            data_id: "data-123".to_string(),
            source_region: "eu-west-1".to_string(),
            target_region: Some("us-east-1".to_string()),
            details: "Test violation".to_string(),
            timestamp: chrono::Utc::now(),
        };

        handler.violations.insert(violation.id.clone(), violation);

        let violations = handler.get_violations(10);
        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].violation_type,
            ViolationType::CrossBorderTransfer
        );
    }

    #[tokio::test]
    async fn test_retention_compliance() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let node = Node {
            id: "old-node".to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "eu-west-1".to_string(),
            created_at: chrono::Utc::now() - chrono::Duration::days(400),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let result = handler.check_retention_compliance(&node, "eu-west-1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retention_compliance_ok() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let node = Node {
            id: "new-node".to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "eu-west-1".to_string(),
            created_at: chrono::Utc::now() - chrono::Duration::days(100),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let result = handler.check_retention_compliance(&node, "eu-west-1").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_filter_compliant_nodes() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let mut nodes = Vec::new();

        // Compliant node
        let mut properties1 = HashMap::new();
        properties1.insert(
            "classification".to_string(),
            serde_json::Value::String("public".to_string()),
        );
        nodes.push(Node {
            id: "node1".to_string(),
            node_type: "entity".to_string(),
            properties: properties1,
            region: "eu-west-1".to_string(), // Same region for compliance
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        });

        // Non-compliant node (restricted in EU)
        let mut properties2 = HashMap::new();
        properties2.insert(
            "classification".to_string(),
            serde_json::Value::String("restricted".to_string()),
        );
        nodes.push(Node {
            id: "node2".to_string(),
            node_type: "entity".to_string(),
            properties: properties2,
            region: "eu-west-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        });

        let compliant = handler.filter_compliant_nodes(nodes, "eu-west-1").await;
        assert_eq!(compliant.len(), 1);
        assert_eq!(compliant[0].id, "node1");
    }

    #[tokio::test]
    async fn test_compliance_disabled() {
        let mut config = ComplianceConfig::default();
        config.enable_compliance = false;

        let handler = ComplianceHandler::new(config);

        // Even with invalid classification, should pass when compliance is disabled
        let mut properties = HashMap::new();
        properties.insert(
            "classification".to_string(),
            serde_json::Value::String("restricted".to_string()),
        );

        let node = Node {
            id: "test-node".to_string(),
            node_type: "entity".to_string(),
            properties,
            region: "eu-west-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let result = handler.check_node_compliance(&node, "eu-west-1").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_encryption_validation() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        // Test encrypted value detection
        let encrypted =
            serde_json::Value::String("dGVzdEBlbmNyeXB0ZWQuY29tYmFzZTY0ZW5jb2RlZA==".to_string());
        assert!(handler.is_encrypted(&encrypted));

        // Test unencrypted value
        let plain = serde_json::Value::String("test@example.com".to_string());
        assert!(!handler.is_encrypted(&plain));
    }

    #[tokio::test]
    async fn test_edge_compliance() {
        let handler = ComplianceHandler::new(ComplianceConfig::default());

        let edge = Edge {
            id: "edge1".to_string(),
            source: "node1".to_string(),
            target: "node2".to_string(),
            edge_type: "relates_to".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        };

        // Same region - should pass
        let result1 = handler
            .check_edge_compliance(&edge, "us-east-1", "us-east-1")
            .await;
        assert!(result1.is_ok());

        // Cross-region where not allowed - should fail
        let result2 = handler
            .check_edge_compliance(&edge, "eu-west-1", "us-east-1")
            .await;
        assert!(result2.is_err());
    }
}
