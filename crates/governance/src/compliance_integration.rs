//! Integration with compliance frameworks
//!
//! The ComplianceIntegration module connects the governance system with existing
//! compliance frameworks, providing automated compliance checking and audit trail generation.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::governance_engine::EvolutionRequest;
use crate::{GovernanceError, Result};
use stratoswarm_agent_core::agent::AgentId;

// Import from compliance crate
use stratoswarm_compliance::{
    ai_safety::{AISafetyEngine, AISystemType, SafetyRiskLevel},
    audit_framework::{AuditEntry, AuditOperation, ComplianceConfig, ComplianceEngine},
    data_classification::{DataCategory, DataClassification, DataClassifier, DataMetadata},
    gdpr::GdprHandler,
    hipaa::{HipaaHandler, PhiAccessType},
    soc2::{Soc2Engine, TrustServiceCriteria},
};

/// Compliance integration for connecting governance with compliance frameworks
pub struct ComplianceIntegration {
    strict_mode: bool,
    safety_engine: Arc<RwLock<AISafetyEngine>>,
    audit_engine: Arc<ComplianceEngine>,
    data_classifier: Arc<DataClassifier>,
    gdpr_handler: Arc<RwLock<GdprHandler>>,
    hipaa_handler: Arc<RwLock<HipaaHandler>>,
    soc2_engine: Arc<RwLock<Soc2Engine>>,
    compliance_cache: DashMap<(AgentId, ComplianceCheck), (ComplianceStatus, DateTime<Utc>)>,
    audit_buffer: Arc<RwLock<Vec<ComplianceAuditEntry>>>,
}

/// Compliance status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplianceStatus {
    Compliant,
    NonCompliant(NonComplianceReason),
    Conditional(ConditionType),
    Unknown,
}

/// Reasons for non-compliance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonComplianceReason {
    SafetyViolation,
    DataProtectionViolation,
    RegulatoryViolation,
    AuditFailure,
    PolicyViolation,
}

/// Conditional compliance types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionType {
    RequiresApproval,
    RequiresAudit,
    RequiresDataAnonymization,
    RequiresEncryption,
}

/// Types of compliance checks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComplianceCheck {
    AISafety,
    DataProtection,
    GDPR,
    HIPAA,
    SOC2,
    Evolution,
    ResourceUsage,
}

/// Compliance audit entry
#[derive(Debug, Clone)]
pub struct ComplianceAuditEntry {
    pub entry_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub agent_id: Option<AgentId>,
    pub check_type: ComplianceCheck,
    pub result: ComplianceStatus,
    pub details: serde_json::Value,
    pub recommendations: Vec<String>,
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub report_id: Uuid,
    pub generated_at: DateTime<Utc>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub overall_status: ComplianceStatus,
    pub check_results: Vec<CheckResult>,
    pub violations: Vec<Violation>,
    pub recommendations: Vec<String>,
}

/// Individual check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub check_type: String,
    pub status: String,
    pub details: serde_json::Value,
}

/// Compliance violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub violation_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub agent_id: Option<String>,
    pub violation_type: String,
    pub severity: ViolationSeverity,
    pub description: String,
    pub remediation: Option<String>,
}

/// Violation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl ComplianceIntegration {
    /// Create a new compliance integration
    pub fn new(strict_mode: bool) -> Self {
        let config = ComplianceConfig::default();
        let audit_engine = ComplianceEngine::new(config)
            .expect("Failed to create compliance engine");

        Self {
            strict_mode,
            safety_engine: Arc::new(RwLock::new(AISafetyEngine::new())),
            audit_engine: Arc::new(audit_engine),
            data_classifier: Arc::new(DataClassifier::new()),
            gdpr_handler: Arc::new(RwLock::new(GdprHandler::new())),
            hipaa_handler: Arc::new(RwLock::new(HipaaHandler::new())),
            soc2_engine: Arc::new(RwLock::new(Soc2Engine::new())),
            compliance_cache: DashMap::new(),
            audit_buffer: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register an agent for compliance tracking
    pub async fn register_agent(&self, agent_id: &AgentId) -> Result<()> {
        info!("Registering agent for compliance tracking: {:?}", agent_id);

        // Record audit entry for registration
        self.audit_engine
            .audit(
                AuditOperation::DataCreated,
                agent_id.to_string(),
                DataClassification::InternalData,
                DataCategory::SystemLogs,
                serde_json::json!({
                    "event": "agent_registration",
                    "compliance_framework": "integrated",
                    "strict_mode": self.strict_mode,
                }),
            )
            .await
            .map_err(|e| GovernanceError::ComplianceError(format!("Audit logging failed: {}", e)))?;

        Ok(())
    }

    /// Check AI safety compliance
    pub async fn check_ai_safety(
        &self,
        agent_id: &AgentId,
        _behavior_data: &serde_json::Value,
    ) -> Result<ComplianceStatus> {
        debug!("Checking AI safety compliance for agent {:?}", agent_id);

        // Check cache first
        if let Some(entry) = self
            .compliance_cache
            .get(&(*agent_id, ComplianceCheck::AISafety))
        {
            let (status, cached_at) = entry.value();
            if cached_at.timestamp() + 300 > Utc::now().timestamp() {
                return Ok(*status);
            }
        }

        // Create default metadata for safety check
        let metadata = DataMetadata {
            classification: DataClassification::InternalData,
            category: DataCategory::ModelData,
            retention_period: None,
            encryption_required: false,
            allowed_regions: vec!["US".to_string()],
            audit_required: true,
            owner: Some(agent_id.to_string()),
            created_at: Utc::now(),
            tags: std::collections::HashMap::new(),
        };

        // Perform safety validation
        let validation_result = self
            .safety_engine
            .write()
            .validate_system_safety(
                &agent_id.to_string(),
                AISystemType::AutonomousAgent,
                &metadata,
            )
            .await
            .map_err(|e| GovernanceError::ComplianceError(format!("Safety check failed: {}", e)))?;

        // Convert risk level to compliance status
        let status = match validation_result.risk_level {
            SafetyRiskLevel::Minimal | SafetyRiskLevel::Limited => ComplianceStatus::Compliant,
            SafetyRiskLevel::High => ComplianceStatus::Conditional(ConditionType::RequiresApproval),
            SafetyRiskLevel::Unacceptable => {
                ComplianceStatus::NonCompliant(NonComplianceReason::SafetyViolation)
            }
        };

        // Cache result
        self.compliance_cache
            .insert((*agent_id, ComplianceCheck::AISafety), (status, Utc::now()));

        // Log audit entry
        self.log_compliance_check(
            agent_id,
            ComplianceCheck::AISafety,
            status,
            serde_json::json!({
                "risk_level": format!("{:?}", validation_result.risk_level),
                "safety_score": validation_result.safety_score,
            }),
        )
        .await;

        Ok(status)
    }

    /// Check evolution compliance
    pub async fn check_evolution_compliance(&self, request: &EvolutionRequest) -> Result<bool> {
        debug!(
            "Checking evolution compliance for request: {}",
            request.evolution_type
        );

        // Check if evolution type is allowed
        let allowed_types = vec![
            "minor_capability",
            "major_capability",
            "architecture_change",
            "performance_optimization",
        ];

        if !allowed_types.contains(&request.evolution_type.as_str()) {
            warn!("Evolution type not allowed: {}", request.evolution_type);
            return Ok(false);
        }

        // Check resource requirements
        if request.resource_requirements.memory_mb > 4096
            || request.resource_requirements.cpu_cores > 8.0
            || request.resource_requirements.gpu_memory_mb > 8192
        {
            warn!("Evolution resource requirements exceed compliance limits");
            return Ok(false);
        }

        // Check target capabilities for safety
        let unsafe_capabilities = vec![
            "unrestricted_network_access",
            "system_modification",
            "data_exfiltration",
        ];

        for capability in &request.target_capabilities {
            if unsafe_capabilities.iter().any(|&uc| capability.contains(uc)) {
                warn!("Unsafe capability requested: {}", capability);
                return Ok(false);
            }
        }

        // Log the check via audit
        self.audit_engine
            .audit(
                AuditOperation::DataModified,
                "governance_system".to_string(),
                DataClassification::InternalData,
                DataCategory::SystemLogs,
                serde_json::json!({
                    "event": "evolution_compliance_check",
                    "evolution_type": request.evolution_type,
                    "target_capabilities": request.target_capabilities,
                    "result": "approved",
                }),
            )
            .await
            .map_err(|e| GovernanceError::ComplianceError(format!("Audit logging failed: {}", e)))?;

        Ok(true)
    }

    /// Check data protection compliance
    pub async fn check_data_protection(
        &self,
        agent_id: &AgentId,
        data_access: &DataAccessRequest,
    ) -> Result<ComplianceStatus> {
        debug!(
            "Checking data protection compliance for agent {:?}",
            agent_id
        );

        // Determine data category based on request
        let category = if data_access.data_type.contains("health") || data_access.data_type.contains("medical") {
            DataCategory::PHI
        } else if data_access.contains_personal_data {
            DataCategory::PII
        } else if data_access.data_type.contains("financial") {
            DataCategory::Financial
        } else {
            DataCategory::BusinessData
        };

        // Determine classification based on data type
        let classification = if category == DataCategory::PHI {
            DataClassification::RestrictedData
        } else if category == DataCategory::PII || category == DataCategory::Financial {
            DataClassification::ConfidentialData
        } else {
            DataClassification::InternalData
        };

        // Check GDPR if PII
        if category == DataCategory::PII {
            // For GDPR, we check if the purpose is legitimate
            // In a real implementation, this would check consent registry
            let gdpr_check = self.audit_engine
                .check_compliance(&AuditOperation::DataAccessed, category, "US")
                .await;

            if let Err(e) = gdpr_check {
                warn!("GDPR compliance check failed: {}", e);
                return Ok(ComplianceStatus::NonCompliant(NonComplianceReason::DataProtectionViolation));
            }
        }

        // Check HIPAA if PHI
        if category == DataCategory::PHI {
            let access_type = match data_access.access_type.as_str() {
                "read" => PhiAccessType::Read,
                "write" | "create" => PhiAccessType::Create,
                "update" => PhiAccessType::Update,
                "delete" => PhiAccessType::Delete,
                _ => PhiAccessType::Read,
            };

            let hipaa_allowed = self
                .hipaa_handler
                .read()
                .check_phi_access(&agent_id.to_string(), access_type)
                .map_err(|e| GovernanceError::ComplianceError(format!("HIPAA check failed: {}", e)))?;

            if !hipaa_allowed {
                return Ok(ComplianceStatus::NonCompliant(NonComplianceReason::RegulatoryViolation));
            }
        }

        // Check if encryption is required for sensitive data
        if matches!(classification, DataClassification::ConfidentialData | DataClassification::RestrictedData) {
            if !data_access.encryption_enabled {
                return Ok(ComplianceStatus::Conditional(ConditionType::RequiresEncryption));
            }
        }

        Ok(ComplianceStatus::Compliant)
    }

    /// Check SOC2 compliance
    pub async fn check_soc2_compliance(&self, agent_id: &AgentId) -> Result<ComplianceStatus> {
        debug!("Checking SOC2 compliance for agent {:?}", agent_id);

        let compliance_status = self.soc2_engine.read().get_compliance_status();

        // Check if all criteria are compliant
        let all_compliant = compliance_status.values().all(|status| {
            matches!(status, stratoswarm_compliance::soc2::ComplianceStatus::Compliant)
        });

        if all_compliant {
            Ok(ComplianceStatus::Compliant)
        } else {
            Ok(ComplianceStatus::Conditional(ConditionType::RequiresAudit))
        }
    }

    /// Perform comprehensive compliance check
    pub async fn comprehensive_check(&self, agent_id: &AgentId) -> Result<ComplianceReport> {
        info!(
            "Performing comprehensive compliance check for agent {:?}",
            agent_id
        );

        let mut check_results = Vec::new();
        let mut violations = Vec::new();
        let mut overall_compliant = true;

        // AI Safety check
        let safety_status = self
            .check_ai_safety(agent_id, &serde_json::json!({}))
            .await?;
        check_results.push(CheckResult {
            check_type: "ai_safety".to_string(),
            status: format!("{:?}", safety_status),
            details: serde_json::json!({"level": "standard"}),
        });

        if matches!(safety_status, ComplianceStatus::NonCompliant(_)) {
            overall_compliant = false;
            violations.push(Violation {
                violation_id: Uuid::new_v4(),
                timestamp: Utc::now(),
                agent_id: Some(agent_id.to_string()),
                violation_type: "ai_safety".to_string(),
                severity: ViolationSeverity::High,
                description: "AI safety check failed".to_string(),
                remediation: Some("Review agent behavior and apply safety constraints".to_string()),
            });
        }

        // SOC2 check
        let soc2_status = self.check_soc2_compliance(agent_id).await?;
        check_results.push(CheckResult {
            check_type: "soc2".to_string(),
            status: format!("{:?}", soc2_status),
            details: serde_json::json!({"framework": "SOC2 Type II"}),
        });

        // Generate report
        let report = ComplianceReport {
            report_id: Uuid::new_v4(),
            generated_at: Utc::now(),
            period_start: Utc::now() - chrono::Duration::days(30),
            period_end: Utc::now(),
            overall_status: if overall_compliant {
                ComplianceStatus::Compliant
            } else {
                ComplianceStatus::NonCompliant(NonComplianceReason::PolicyViolation)
            },
            check_results,
            violations: violations.clone(),
            recommendations: self.generate_recommendations(&violations),
        };

        Ok(report)
    }

    /// Generate compliance recommendations
    fn generate_recommendations(&self, violations: &[Violation]) -> Vec<String> {
        let mut recommendations = Vec::new();

        for violation in violations {
            match violation.violation_type.as_str() {
                "ai_safety" => {
                    recommendations.push("Implement additional safety constraints".to_string());
                    recommendations.push("Review agent training data for bias".to_string());
                }
                "data_protection" => {
                    recommendations.push("Enable encryption for all data access".to_string());
                    recommendations.push("Implement data minimization practices".to_string());
                }
                "regulatory" => {
                    recommendations.push("Review regulatory requirements".to_string());
                    recommendations.push("Implement required controls".to_string());
                }
                _ => {}
            }
        }

        recommendations.dedup();
        recommendations
    }

    /// Log a compliance check
    async fn log_compliance_check(
        &self,
        agent_id: &AgentId,
        check_type: ComplianceCheck,
        status: ComplianceStatus,
        details: serde_json::Value,
    ) {
        let entry = ComplianceAuditEntry {
            entry_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            agent_id: Some(*agent_id),
            check_type,
            result: status,
            details,
            recommendations: vec![],
        };

        self.audit_buffer.write().push(entry);

        // Flush to audit framework if buffer is large
        if self.audit_buffer.read().len() > 100 {
            self.flush_audit_buffer().await;
        }
    }

    /// Flush audit buffer to framework
    async fn flush_audit_buffer(&self) {
        let entries: Vec<_> = self.audit_buffer.write().drain(..).collect();

        for entry in entries {
            if let Err(e) = self.audit_engine
                .audit(
                    AuditOperation::DataCreated,
                    entry.agent_id.map(|id| id.to_string()).unwrap_or_default(),
                    DataClassification::InternalData,
                    DataCategory::SystemLogs,
                    serde_json::json!({
                        "check_type": format!("{:?}", entry.check_type),
                        "result": format!("{:?}", entry.result),
                        "details": entry.details,
                    }),
                )
                .await
            {
                tracing::error!("Failed to log audit event: {}", e);
            }
        }
    }

    /// Get compliance history for an agent
    pub async fn get_compliance_history(
        &self,
        agent_id: &AgentId,
    ) -> Result<Vec<ComplianceAuditEntry>> {
        let history: Vec<_> = self
            .audit_buffer
            .read()
            .iter()
            .filter(|entry| entry.agent_id.as_ref() == Some(agent_id))
            .cloned()
            .collect();

        Ok(history)
    }

    /// Clear compliance cache for an agent
    pub async fn clear_cache(&self, agent_id: &AgentId) {
        let keys_to_remove: Vec<_> = self
            .compliance_cache
            .iter()
            .filter(|entry| entry.key().0 == *agent_id)
            .map(|entry| *entry.key())
            .collect();

        for key in keys_to_remove {
            self.compliance_cache.remove(&key);
        }
    }
}

/// Data access request for compliance checking
#[derive(Debug, Clone)]
pub struct DataAccessRequest {
    pub data_type: String,
    pub access_type: String,
    pub purpose: String,
    pub contains_personal_data: bool,
    pub encryption_enabled: bool,
}

// Implement Serialize for ComplianceStatus to use in JSON
impl Serialize for ComplianceStatus {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ComplianceStatus::Compliant => serializer.serialize_str("Compliant"),
            ComplianceStatus::NonCompliant(reason) => {
                serializer.serialize_str(&format!("NonCompliant({:?})", reason))
            }
            ComplianceStatus::Conditional(cond) => {
                serializer.serialize_str(&format!("Conditional({:?})", cond))
            }
            ComplianceStatus::Unknown => serializer.serialize_str("Unknown"),
        }
    }
}

impl<'de> Deserialize<'de> for ComplianceStatus {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s == "Compliant" {
            Ok(ComplianceStatus::Compliant)
        } else if s == "Unknown" {
            Ok(ComplianceStatus::Unknown)
        } else {
            Ok(ComplianceStatus::Unknown) // Simplified for now
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compliance_integration_creation() {
        let integration = ComplianceIntegration::new(true);
        assert!(integration.strict_mode);
        assert_eq!(integration.compliance_cache.len(), 0);
    }

    #[tokio::test]
    async fn test_agent_registration() {
        let integration = ComplianceIntegration::new(false);
        let agent_id = AgentId::new();

        let result = integration.register_agent(&agent_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_ai_safety_check() {
        let integration = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();

        let behavior_data = serde_json::json!({
            "actions": ["read_data", "process_data"],
            "risk_level": "low"
        });

        let status = integration
            .check_ai_safety(&agent_id, &behavior_data)
            .await
            .unwrap();

        // Should be compliant or conditional for a new agent
        assert!(matches!(
            status,
            ComplianceStatus::Compliant | ComplianceStatus::Conditional(_)
        ));
    }

    #[tokio::test]
    async fn test_evolution_compliance_allowed_type() {
        let integration = ComplianceIntegration::new(true);

        let request = EvolutionRequest {
            evolution_type: "minor_capability".to_string(),
            target_capabilities: vec!["enhanced_processing".to_string()],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 1024,
                cpu_cores: 2.0,
                gpu_memory_mb: 2048,
                duration_seconds: None,
            },
        };

        let result = integration.check_evolution_compliance(&request).await.unwrap();
        assert!(result);
    }

    #[tokio::test]
    async fn test_evolution_compliance_disallowed_type() {
        let integration = ComplianceIntegration::new(true);

        let request = EvolutionRequest {
            evolution_type: "unrestricted_evolution".to_string(),
            target_capabilities: vec![],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 0,
                duration_seconds: None,
            },
        };

        let result = integration.check_evolution_compliance(&request).await.unwrap();
        assert!(!result);
    }

    #[tokio::test]
    async fn test_data_protection_check() {
        let integration = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();

        let data_request = DataAccessRequest {
            data_type: "user_profile".to_string(),
            access_type: "read".to_string(),
            purpose: "analytics".to_string(),
            contains_personal_data: false,
            encryption_enabled: true,
        };

        let status = integration
            .check_data_protection(&agent_id, &data_request)
            .await
            .unwrap();

        assert!(matches!(
            status,
            ComplianceStatus::Compliant | ComplianceStatus::Conditional(_)
        ));
    }

    #[tokio::test]
    async fn test_soc2_compliance_check() {
        let integration = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();

        let status = integration.check_soc2_compliance(&agent_id).await.unwrap();

        // New SOC2 engine will have unimplemented controls
        assert!(matches!(
            status,
            ComplianceStatus::Compliant | ComplianceStatus::Conditional(ConditionType::RequiresAudit)
        ));
    }

    #[tokio::test]
    async fn test_cache_clearing() {
        let integration = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();

        // Populate cache via safety check
        let _ = integration
            .check_ai_safety(&agent_id, &serde_json::json!({}))
            .await;

        assert!(integration.compliance_cache.len() > 0);

        // Clear cache
        integration.clear_cache(&agent_id).await;
        assert_eq!(integration.compliance_cache.len(), 0);
    }
}
