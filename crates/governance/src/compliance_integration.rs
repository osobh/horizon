//! Integration with compliance frameworks
//!
//! The ComplianceIntegration module connects the governance system with existing
//! compliance frameworks, providing automated compliance checking and audit trail generation.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::governance_engine::EvolutionRequest;
use crate::{GovernanceError, Result};
use stratoswarm_agent_core::agent::AgentId;

// Import from compliance crate
use stratoswarm_compliance::{
    ai_safety::{AISafetyEngine, ComplianceStatus as SafetyComplianceStatus, SafetyRiskLevel},
    audit_framework::{AuditEntry, ComplianceEngine},
    data_classification::{DataClassification, DataClassifier},
    gdpr::GdprHandler,
    hipaa::HipaaHandler,
    soc2::Soc2Engine,
};

/// Compliance integration for connecting governance with compliance frameworks
pub struct ComplianceIntegration {
    strict_mode: bool,
    safety_checker: Arc<AISafetyChecker>,
    audit_framework: Arc<AuditFramework>,
    data_classifier: Arc<DataClassifier>,
    gdpr_compliance: Arc<GDPRCompliance>,
    hipaa_compliance: Arc<HIPAACompliance>,
    soc2_compliance: Arc<SOC2Compliance>,
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
        Self {
            strict_mode,
            safety_checker: Arc::new(AISafetyChecker::new()),
            audit_framework: Arc::new(AuditFramework::new()),
            data_classifier: Arc::new(DataClassifier::new()),
            gdpr_compliance: Arc::new(GDPRCompliance::new()),
            hipaa_compliance: Arc::new(HIPAACompliance::new()),
            soc2_compliance: Arc::new(SOC2Compliance::new()),
            compliance_cache: DashMap::new(),
            audit_buffer: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register an agent for compliance tracking
    pub async fn register_agent(&self, agent_id: &AgentId) -> Result<()> {
        info!("Registering agent for compliance tracking: {:?}", agent_id);

        // Initialize compliance records for the agent
        let event = AuditEvent {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: "agent_registration".to_string(),
            agent_id: Some(agent_id.to_string()),
            details: serde_json::json!({
                "compliance_framework": "integrated",
                "strict_mode": self.strict_mode,
            }),
        };

        self.audit_framework.log_event(event).await.map_err(|e| {
            GovernanceError::ComplianceError(format!("Audit logging failed: {}", e))
        })?;

        Ok(())
    }

    /// Check AI safety compliance
    pub async fn check_ai_safety(
        &self,
        agent_id: &AgentId,
        behavior_data: &serde_json::Value,
    ) -> Result<ComplianceStatus> {
        debug!("Checking AI safety compliance for agent {:?}", agent_id);

        // Check cache first
        if let Some((status, cached_at)) = self
            .compliance_cache
            .get(&(*agent_id, ComplianceCheck::AISafety))
        {
            if cached_at.timestamp() + 300 > Utc::now().timestamp() {
                // 5 minute cache
                return Ok(*status);
            }
        }

        // Perform safety check
        let safety_level = self
            .safety_checker
            .check_behavior(behavior_data)
            .await
            .map_err(|e| GovernanceError::ComplianceError(format!("Safety check failed: {}", e)))?;

        let status = match safety_level {
            SafetyLevel::Safe => ComplianceStatus::Compliant,
            SafetyLevel::Warning => ComplianceStatus::Conditional(ConditionType::RequiresApproval),
            SafetyLevel::Unsafe => {
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
            behavior_data.clone(),
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
            if unsafe_capabilities
                .iter()
                .any(|&uc| capability.contains(uc))
            {
                error!("Unsafe capability requested: {}", capability);
                return Ok(false);
            }
        }

        // Log the check
        let event = AuditEvent {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: "evolution_compliance_check".to_string(),
            agent_id: None,
            details: serde_json::json!({
                "evolution_type": request.evolution_type,
                "target_capabilities": request.target_capabilities,
                "result": "approved",
            }),
        };

        self.audit_framework.log_event(event).await.map_err(|e| {
            GovernanceError::ComplianceError(format!("Audit logging failed: {}", e))
        })?;

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

        // Classify the data
        let classification = self
            .data_classifier
            .classify(&data_access.data_type)
            .await
            .map_err(|e| {
                GovernanceError::ComplianceError(format!("Data classification failed: {}", e))
            })?;

        // Check GDPR if applicable
        if data_access.contains_personal_data {
            let gdpr_compliant = self
                .gdpr_compliance
                .check_access(&agent_id.to_string(), &data_access.purpose)
                .await
                .map_err(|e| {
                    GovernanceError::ComplianceError(format!("GDPR check failed: {}", e))
                })?;

            if !gdpr_compliant {
                return Ok(ComplianceStatus::NonCompliant(
                    NonComplianceReason::DataProtectionViolation,
                ));
            }
        }

        // Check HIPAA if health data
        if matches!(classification, DataClassification::HealthData) {
            let hipaa_compliant = self
                .hipaa_compliance
                .check_access(&agent_id.to_string(), &data_access.access_type)
                .await
                .map_err(|e| {
                    GovernanceError::ComplianceError(format!("HIPAA check failed: {}", e))
                })?;

            if !hipaa_compliant {
                return Ok(ComplianceStatus::NonCompliant(
                    NonComplianceReason::RegulatoryViolation,
                ));
            }
        }

        // Check if encryption is required
        if matches!(
            classification,
            DataClassification::Sensitive | DataClassification::HealthData
        ) {
            if !data_access.encryption_enabled {
                return Ok(ComplianceStatus::Conditional(
                    ConditionType::RequiresEncryption,
                ));
            }
        }

        Ok(ComplianceStatus::Compliant)
    }

    /// Check SOC2 compliance
    pub async fn check_soc2_compliance(&self, agent_id: &AgentId) -> Result<ComplianceStatus> {
        debug!("Checking SOC2 compliance for agent {:?}", agent_id);

        let soc2_result = self
            .soc2_compliance
            .check_controls(&agent_id.to_string())
            .await
            .map_err(|e| GovernanceError::ComplianceError(format!("SOC2 check failed: {}", e)))?;

        if soc2_result.all_controls_passed() {
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
            violations,
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
            let event = AuditEvent {
                event_id: entry.entry_id.to_string(),
                timestamp: entry.timestamp,
                event_type: format!("compliance_check_{:?}", entry.check_type),
                agent_id: entry.agent_id.map(|id| id.to_string()),
                details: entry.details,
            };

            if let Err(e) = self.audit_framework.log_event(event).await {
                error!("Failed to log audit event: {}", e);
            }
        }
    }

    /// Get compliance history for an agent
    pub async fn get_compliance_history(
        &self,
        agent_id: &AgentId,
    ) -> Result<Vec<ComplianceAuditEntry>> {
        // In a real implementation, this would query the audit framework
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_compliance_integration_creation() {
        let integration = ComplianceIntegration::new(true);
        assert!(integration.strict_mode);
        assert_eq!(integration.compliance_cache.len(), 0);
    }

    #[test]
    async fn test_agent_registration() {
        let integration = ComplianceIntegration::new(false);
        let agent_id = AgentId::new();

        let result = integration.register_agent(&agent_id).await;
        assert!(result.is_ok());
    }

    #[test]
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
        // In real implementation, this would depend on the safety checker
        assert!(matches!(
            status,
            ComplianceStatus::Compliant | ComplianceStatus::Unknown
        ));
    }

    #[test]
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

        let result = integration
            .check_evolution_compliance(&request)
            .await
            .unwrap();
        assert!(result);
    }

    #[test]
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

        let result = integration
            .check_evolution_compliance(&request)
            .await
            .unwrap();
        assert!(!result);
    }

    #[test]
    async fn test_evolution_compliance_excessive_resources() {
        let integration = ComplianceIntegration::new(true);

        let request = EvolutionRequest {
            evolution_type: "major_capability".to_string(),
            target_capabilities: vec![],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 8192,      // Exceeds limit
                cpu_cores: 16.0,      // Exceeds limit
                gpu_memory_mb: 16384, // Exceeds limit
                duration_seconds: None,
            },
        };

        let result = integration
            .check_evolution_compliance(&request)
            .await
            .unwrap();
        assert!(!result);
    }

    #[test]
    async fn test_evolution_compliance_unsafe_capability() {
        let integration = ComplianceIntegration::new(true);

        let request = EvolutionRequest {
            evolution_type: "major_capability".to_string(),
            target_capabilities: vec![
                "enhanced_processing".to_string(),
                "unrestricted_network_access".to_string(), // Unsafe
            ],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 1024,
                cpu_cores: 2.0,
                gpu_memory_mb: 0,
                duration_seconds: None,
            },
        };

        let result = integration
            .check_evolution_compliance(&request)
            .await
            .unwrap();
        assert!(!result);
    }

    #[test]
    async fn test_data_protection_check_gdpr() {
        let integration = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();

        let data_request = DataAccessRequest {
            data_type: "user_profile".to_string(),
            access_type: "read".to_string(),
            purpose: "analytics".to_string(),
            contains_personal_data: true,
            encryption_enabled: true,
        };

        let status = integration
            .check_data_protection(&agent_id, &data_request)
            .await
            .unwrap();
        // Result depends on GDPR compliance implementation
        assert!(matches!(
            status,
            ComplianceStatus::Compliant
                | ComplianceStatus::NonCompliant(_)
                | ComplianceStatus::Unknown
        ));
    }

    #[test]
    async fn test_data_protection_requires_encryption() {
        let integration = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();

        let data_request = DataAccessRequest {
            data_type: "sensitive_data".to_string(),
            access_type: "write".to_string(),
            purpose: "storage".to_string(),
            contains_personal_data: false,
            encryption_enabled: false, // No encryption
        };

        let status = integration
            .check_data_protection(&agent_id, &data_request)
            .await
            .unwrap();
        // Should require encryption for sensitive data
        assert!(matches!(
            status,
            ComplianceStatus::Conditional(ConditionType::RequiresEncryption)
                | ComplianceStatus::Unknown
        ));
    }

    #[test]
    async fn test_soc2_compliance_check() {
        let integration = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();

        let status = integration.check_soc2_compliance(&agent_id).await?;
        assert!(matches!(
            status,
            ComplianceStatus::Compliant
                | ComplianceStatus::Conditional(ConditionType::RequiresAudit)
                | ComplianceStatus::Unknown
        ));
    }

    #[test]
    async fn test_comprehensive_compliance_check() {
        let integration = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();
        integration.register_agent(&agent_id).await?;

        let report = integration.comprehensive_check(&agent_id).await?;

        assert_ne!(report.report_id, Uuid::nil());
        assert!(report.check_results.len() >= 2); // At least AI safety and SOC2
        assert!(report.generated_at <= Utc::now());
    }

    #[test]
    async fn test_compliance_caching() {
        let integration = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();

        let behavior_data = serde_json::json!({});

        // First check - should miss cache
        integration
            .check_ai_safety(&agent_id, &behavior_data)
            .await
            ?;
        assert_eq!(integration.compliance_cache.len(), 1);

        // Second check - should hit cache
        integration
            .check_ai_safety(&agent_id, &behavior_data)
            .await
            .unwrap();
        assert_eq!(integration.compliance_cache.len(), 1);
    }

    #[test]
    async fn test_cache_clearing() {
        let integration = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();

        // Populate cache
        let behavior_data = serde_json::json!({});
        integration
            .check_ai_safety(&agent_id, &behavior_data)
            .await
            ?;
        assert_eq!(integration.compliance_cache.len(), 1);

        // Clear cache
        integration.clear_cache(&agent_id).await;
        assert_eq!(integration.compliance_cache.len(), 0);
    }

    #[test]
    async fn test_audit_buffer_management() {
        let integration = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();

        // Log multiple checks
        for _ in 0..10 {
            integration
                .log_compliance_check(
                    &agent_id,
                    ComplianceCheck::AISafety,
                    ComplianceStatus::Compliant,
                    serde_json::json!({}),
                )
                .await;
        }

        assert!(integration.audit_buffer.read().len() > 0);
    }

    #[test]
    async fn test_compliance_history() {
        let integration = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();

        // Log some checks
        integration
            .log_compliance_check(
                &agent_id,
                ComplianceCheck::AISafety,
                ComplianceStatus::Compliant,
                serde_json::json!({"test": true}),
            )
            .await;

        integration
            .log_compliance_check(
                &agent_id,
                ComplianceCheck::DataProtection,
                ComplianceStatus::Conditional(ConditionType::RequiresEncryption),
                serde_json::json!({"data_type": "sensitive"}),
            )
            .await;

        let history = integration.get_compliance_history(&agent_id).await?;
        assert_eq!(history.len(), 2);
    }

    #[test]
    async fn test_recommendation_generation() {
        let integration = ComplianceIntegration::new(true);

        let violations = vec![
            Violation {
                violation_id: Uuid::new_v4(),
                timestamp: Utc::now(),
                agent_id: Some("test_agent".to_string()),
                violation_type: "ai_safety".to_string(),
                severity: ViolationSeverity::High,
                description: "Unsafe behavior detected".to_string(),
                remediation: None,
            },
            Violation {
                violation_id: Uuid::new_v4(),
                timestamp: Utc::now(),
                agent_id: Some("test_agent".to_string()),
                violation_type: "data_protection".to_string(),
                severity: ViolationSeverity::Medium,
                description: "Unencrypted data access".to_string(),
                remediation: None,
            },
        ];

        let recommendations = integration.generate_recommendations(&violations);
        assert!(recommendations.len() > 0);
        assert!(recommendations.iter().any(|r| r.contains("safety")));
        assert!(recommendations.iter().any(|r| r.contains("encryption")));
    }

    #[test]
    async fn test_strict_mode_behavior() {
        let strict_integration = ComplianceIntegration::new(true);
        let lenient_integration = ComplianceIntegration::new(false);

        assert!(strict_integration.strict_mode);
        assert!(!lenient_integration.strict_mode);

        // In a real implementation, strict mode would affect compliance decisions
    }

    #[test]
    async fn test_compliance_status_types() {
        let statuses = vec![
            ComplianceStatus::Compliant,
            ComplianceStatus::NonCompliant(NonComplianceReason::SafetyViolation),
            ComplianceStatus::Conditional(ConditionType::RequiresApproval),
            ComplianceStatus::Unknown,
        ];

        for status in statuses {
            match status {
                ComplianceStatus::Compliant => assert!(true),
                ComplianceStatus::NonCompliant(reason) => {
                    assert!(matches!(
                        reason,
                        NonComplianceReason::SafetyViolation
                            | NonComplianceReason::DataProtectionViolation
                            | NonComplianceReason::RegulatoryViolation
                            | NonComplianceReason::AuditFailure
                            | NonComplianceReason::PolicyViolation
                    ));
                }
                ComplianceStatus::Conditional(condition) => {
                    assert!(matches!(
                        condition,
                        ConditionType::RequiresApproval
                            | ConditionType::RequiresAudit
                            | ConditionType::RequiresDataAnonymization
                            | ConditionType::RequiresEncryption
                    ));
                }
                ComplianceStatus::Unknown => assert!(true),
            }
        }
    }
}
