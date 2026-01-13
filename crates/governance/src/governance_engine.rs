//! Main governance coordination engine
//!
//! The GovernanceEngine is the central coordinator for all governance activities
//! in the ExoRust system. It manages policies, permissions, lifecycle decisions,
//! and cross-agent coordination.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::compliance_integration::ComplianceIntegration;
use crate::coordination_manager::{CoordinationManager, CoordinationRequest};
use crate::lifecycle_governance::{LifecycleDecision, LifecycleGovernor};
use crate::monitoring_governance::GovernanceMonitor;
use crate::permission_system::{Permission, PermissionSystem};
use crate::policy_manager::{Policy, PolicyManager, PolicyType};
use crate::{GovernanceError, Result};

use stratoswarm_agent_core::agent::AgentId;
use stratoswarm_emergency_controls::kill_switch::KillSwitchSystem;

/// Configuration for the governance engine
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GovernanceConfig {
    /// Maximum number of agents allowed in the system
    pub max_agents: usize,
    /// Enable strict compliance mode
    pub strict_compliance: bool,
    /// Enable emergency override capabilities
    pub emergency_override_enabled: bool,
    /// Default resource quotas for new agents
    pub default_resource_quota: ResourceQuota,
    /// Audit retention period in days
    pub audit_retention_days: u32,
    /// Policy evaluation timeout in milliseconds
    pub policy_evaluation_timeout_ms: u64,
}

impl Default for GovernanceConfig {
    fn default() -> Self {
        Self {
            max_agents: 1000,
            strict_compliance: true,
            emergency_override_enabled: true,
            default_resource_quota: ResourceQuota::default(),
            audit_retention_days: 90,
            policy_evaluation_timeout_ms: 5000,
        }
    }
}

/// Resource quota for agents
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceQuota {
    pub max_memory_mb: u64,
    pub max_cpu_cores: f32,
    pub max_gpu_memory_mb: u64,
    pub max_network_bandwidth_mbps: u64,
    pub max_storage_gb: u64,
}

impl Default for ResourceQuota {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024,
            max_cpu_cores: 2.0,
            max_gpu_memory_mb: 2048,
            max_network_bandwidth_mbps: 100,
            max_storage_gb: 10,
        }
    }
}

/// Main governance engine that coordinates all governance activities
pub struct GovernanceEngine {
    config: Arc<RwLock<GovernanceConfig>>,
    policy_manager: Arc<PolicyManager>,
    permission_system: Arc<PermissionSystem>,
    lifecycle_governor: Arc<LifecycleGovernor>,
    coordination_manager: Arc<CoordinationManager>,
    compliance_integration: Arc<ComplianceIntegration>,
    monitor: Arc<GovernanceMonitor>,
    kill_switch: Arc<KillSwitchSystem>,
    active_agents: DashMap<AgentId, AgentGovernanceState>,
    audit_log: Arc<RwLock<Vec<AuditEntry>>>,
}

/// Governance state for an individual agent
#[derive(Debug, Clone)]
pub struct AgentGovernanceState {
    pub agent_id: AgentId,
    pub created_at: DateTime<Utc>,
    pub resource_quota: ResourceQuota,
    pub active_policies: Vec<Uuid>,
    pub permissions: Vec<Permission>,
    pub compliance_status: bool,
    pub lifecycle_phase: LifecyclePhase,
    pub violations: Vec<PolicyViolation>,
}

/// Lifecycle phases for agents
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LifecyclePhase {
    Initializing,
    Active,
    Evolving,
    Suspended,
    Terminating,
    Terminated,
}

/// Policy violation record
#[derive(Debug, Clone)]
pub struct PolicyViolation {
    pub timestamp: DateTime<Utc>,
    pub policy_id: Uuid,
    pub violation_type: String,
    pub severity: ViolationSeverity,
    pub details: String,
}

/// Severity levels for policy violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Audit log entry
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    pub agent_id: Option<AgentId>,
    pub action: String,
    pub result: String,
    pub details: serde_json::Value,
}

impl GovernanceEngine {
    /// Create a new governance engine
    pub async fn new(config: GovernanceConfig) -> Result<Self> {
        info!("Initializing governance engine");

        let policy_manager = Arc::new(PolicyManager::new());
        let permission_system = Arc::new(PermissionSystem::new());
        let lifecycle_governor = Arc::new(LifecycleGovernor::new(
            config.default_resource_quota.clone(),
        ));
        let coordination_manager = Arc::new(CoordinationManager::new());
        let compliance_integration = Arc::new(ComplianceIntegration::new(config.strict_compliance));
        let monitor = Arc::new(GovernanceMonitor::new());
        let kill_switch = Arc::new(KillSwitchSystem::new(Default::default()));

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            policy_manager,
            permission_system,
            lifecycle_governor,
            coordination_manager,
            compliance_integration,
            monitor,
            kill_switch,
            active_agents: DashMap::new(),
            audit_log: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Register a new agent with the governance system
    pub async fn register_agent(&self, agent_id: AgentId) -> Result<()> {
        info!("Registering agent: {:?}", agent_id);

        // Check if we're at capacity
        if self.active_agents.len() >= self.config.read().max_agents {
            return Err(GovernanceError::ResourceLimitExceeded(
                "Maximum agent capacity reached".to_string(),
            ));
        }

        // Create initial governance state
        let state = AgentGovernanceState {
            agent_id: agent_id.clone(),
            created_at: Utc::now(),
            resource_quota: self.config.read().default_resource_quota.clone(),
            active_policies: vec![],
            permissions: vec![],
            compliance_status: true,
            lifecycle_phase: LifecyclePhase::Initializing,
            violations: vec![],
        };

        // Register with subsystems
        self.permission_system.register_agent(&agent_id).await?;
        self.lifecycle_governor.register_agent(&agent_id).await?;
        self.compliance_integration
            .register_agent(&agent_id)
            .await?;

        // Store state
        self.active_agents.insert(agent_id.clone(), state);

        // Log the registration
        self.log_audit(
            Some(agent_id),
            "agent_registered",
            "success",
            serde_json::json!({
                "timestamp": Utc::now().to_rfc3339()
            }),
        );

        Ok(())
    }

    /// Evaluate a governance decision for an agent
    pub async fn evaluate_decision(
        &self,
        agent_id: &AgentId,
        decision_type: DecisionType,
    ) -> Result<GovernanceDecision> {
        debug!(
            "Evaluating decision for agent {:?}: {:?}",
            agent_id, decision_type
        );

        // Check if agent exists
        let agent_state = self
            .active_agents
            .get(agent_id)
            .ok_or_else(|| GovernanceError::InternalError("Agent not found".to_string()))?;

        // Check emergency override
        if self.kill_switch.is_global_kill_active() {
            return Ok(GovernanceDecision::Denied(
                "Emergency kill switch activated".to_string(),
            ));
        }

        // Capture decision type for logging before match consumes it
        let decision_type_str = format!("{:?}", decision_type);

        // Evaluate based on decision type
        let decision = match decision_type {
            DecisionType::ResourceAllocation(request) => {
                self.evaluate_resource_allocation(agent_id, request).await?
            }
            DecisionType::Evolution(request) => {
                self.evaluate_evolution_request(agent_id, request).await?
            }
            DecisionType::Coordination(request) => {
                self.evaluate_coordination_request(agent_id, request)
                    .await?
            }
            DecisionType::PermissionRequest(permission) => {
                self.evaluate_permission_request(agent_id, permission)
                    .await?
            }
        };

        // Log the decision
        self.log_audit(
            Some(agent_id.clone()),
            "governance_decision",
            &format!("{:?}", decision),
            serde_json::json!({
                "decision_type": decision_type_str,
                "timestamp": Utc::now().to_rfc3339()
            }),
        );

        Ok(decision)
    }

    /// Evaluate resource allocation request
    async fn evaluate_resource_allocation(
        &self,
        agent_id: &AgentId,
        request: ResourceRequest,
    ) -> Result<GovernanceDecision> {
        // Check against quota
        let agent_state = self.active_agents.get(agent_id)
            .ok_or_else(|| GovernanceError::InternalError(format!("Agent not found: {:?}", agent_id)))?;

        if request.memory_mb > agent_state.resource_quota.max_memory_mb {
            return Ok(GovernanceDecision::Denied(
                "Memory request exceeds quota".to_string(),
            ));
        }

        if request.cpu_cores > agent_state.resource_quota.max_cpu_cores {
            return Ok(GovernanceDecision::Denied(
                "CPU request exceeds quota".to_string(),
            ));
        }

        // Check policies
        let policies = self
            .policy_manager
            .get_applicable_policies(agent_id, PolicyType::Resource)
            .await?;
        for policy in policies {
            if !self
                .policy_manager
                .evaluate_policy(&policy, &request)
                .await?
            {
                return Ok(GovernanceDecision::Denied(format!(
                    "Policy {} violation",
                    policy.id
                )));
            }
        }

        Ok(GovernanceDecision::Approved)
    }

    /// Evaluate evolution request
    async fn evaluate_evolution_request(
        &self,
        agent_id: &AgentId,
        request: EvolutionRequest,
    ) -> Result<GovernanceDecision> {
        // Check lifecycle state
        let agent_state = self.active_agents.get(agent_id)
            .ok_or_else(|| GovernanceError::InternalError(format!("Agent not found: {:?}", agent_id)))?;
        if agent_state.lifecycle_phase != LifecyclePhase::Active {
            return Ok(GovernanceDecision::Denied(
                "Agent not in active phase".to_string(),
            ));
        }

        // Check with lifecycle governor
        let lifecycle_decision = self
            .lifecycle_governor
            .evaluate_evolution(agent_id, &request)
            .await?;
        if !matches!(lifecycle_decision, LifecycleDecision::Approved) {
            return Ok(GovernanceDecision::Denied(
                "Lifecycle governor rejected evolution".to_string(),
            ));
        }

        // Check compliance
        if self.config.read().strict_compliance {
            let compliance_ok = self
                .compliance_integration
                .check_evolution_compliance(&request)
                .await?;
            if !compliance_ok {
                return Ok(GovernanceDecision::Denied(
                    "Compliance check failed".to_string(),
                ));
            }
        }

        Ok(GovernanceDecision::Approved)
    }

    /// Evaluate coordination request
    async fn evaluate_coordination_request(
        &self,
        agent_id: &AgentId,
        request: CoordinationRequest,
    ) -> Result<GovernanceDecision> {
        // Check permissions
        let has_permission = self
            .permission_system
            .check_permission(agent_id, &Permission::Coordinate)
            .await?;

        if !has_permission {
            return Ok(GovernanceDecision::Denied(
                "Missing coordination permission".to_string(),
            ));
        }

        // Check with coordination manager
        let coordination_ok = self.coordination_manager.evaluate_request(&request).await?;
        if !coordination_ok {
            return Ok(GovernanceDecision::Denied(
                "Coordination conflicts detected".to_string(),
            ));
        }

        Ok(GovernanceDecision::Approved)
    }

    /// Evaluate permission request
    async fn evaluate_permission_request(
        &self,
        agent_id: &AgentId,
        permission: Permission,
    ) -> Result<GovernanceDecision> {
        // Check if permission can be granted
        let can_grant = self
            .permission_system
            .can_grant_permission(agent_id, &permission)
            .await?;

        if !can_grant {
            return Ok(GovernanceDecision::Denied(
                "Permission cannot be granted".to_string(),
            ));
        }

        // Grant the permission
        self.permission_system
            .grant_permission(agent_id, permission)
            .await?;

        Ok(GovernanceDecision::Approved)
    }

    /// Update agent lifecycle phase
    pub async fn update_lifecycle_phase(
        &self,
        agent_id: &AgentId,
        phase: LifecyclePhase,
    ) -> Result<()> {
        let mut agent_state = self
            .active_agents
            .get_mut(agent_id)
            .ok_or_else(|| GovernanceError::InternalError("Agent not found".to_string()))?;

        agent_state.lifecycle_phase = phase;

        self.log_audit(
            Some(agent_id.clone()),
            "lifecycle_phase_update",
            &format!("{:?}", phase),
            serde_json::json!({
                "new_phase": format!("{:?}", phase),
                "timestamp": Utc::now().to_rfc3339()
            }),
        );

        Ok(())
    }

    /// Record a policy violation
    pub async fn record_violation(
        &self,
        agent_id: &AgentId,
        violation: PolicyViolation,
    ) -> Result<()> {
        let mut agent_state = self
            .active_agents
            .get_mut(agent_id)
            .ok_or_else(|| GovernanceError::InternalError("Agent not found".to_string()))?;

        // Check if this is a critical violation
        if violation.severity == ViolationSeverity::Critical {
            warn!("Critical policy violation for agent {:?}", agent_id);

            // Suspend the agent
            agent_state.lifecycle_phase = LifecyclePhase::Suspended;

            // Trigger emergency response if configured
            if self.config.read().emergency_override_enabled {
                self.kill_switch
                    .activate_global_kill("Critical policy violation")
                    .await
                    .map_err(|e| GovernanceError::InternalError(format!("Kill switch activation failed: {}", e)))?;
            }
        }

        agent_state.violations.push(violation.clone());

        self.log_audit(
            Some(agent_id.clone()),
            "policy_violation",
            &format!("{:?}", violation.severity),
            serde_json::json!({
                "policy_id": violation.policy_id.to_string(),
                "violation_type": violation.violation_type,
                "details": violation.details,
                "timestamp": violation.timestamp.to_rfc3339()
            }),
        );

        Ok(())
    }

    /// Get governance metrics
    pub async fn get_metrics(&self) -> crate::monitoring_governance::GovernanceMetrics {
        self.monitor
            .collect_metrics(&self.active_agents, &self.audit_log.read())
            .await
    }

    /// Log an audit entry
    fn log_audit(
        &self,
        agent_id: Option<AgentId>,
        action: &str,
        result: &str,
        details: serde_json::Value,
    ) {
        let entry = AuditEntry {
            timestamp: Utc::now(),
            agent_id,
            action: action.to_string(),
            result: result.to_string(),
            details,
        };

        self.audit_log.write().push(entry);
    }

    /// Clean up old audit logs
    pub async fn cleanup_audit_logs(&self) {
        let retention_days = self.config.read().audit_retention_days;
        let cutoff = Utc::now() - chrono::Duration::days(retention_days as i64);

        self.audit_log
            .write()
            .retain(|entry| entry.timestamp > cutoff);
    }
}

/// Types of governance decisions
#[derive(Debug, Clone)]
pub enum DecisionType {
    ResourceAllocation(ResourceRequest),
    Evolution(EvolutionRequest),
    Coordination(CoordinationRequest),
    PermissionRequest(Permission),
}

/// Resource allocation request
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceRequest {
    pub memory_mb: u64,
    pub cpu_cores: f32,
    pub gpu_memory_mb: u64,
    pub duration_seconds: Option<u64>,
}

/// Evolution request
#[derive(Debug, Clone)]
pub struct EvolutionRequest {
    pub evolution_type: String,
    pub target_capabilities: Vec<String>,
    pub resource_requirements: ResourceRequest,
}

/// Governance decision result
#[derive(Debug, Clone)]
pub enum GovernanceDecision {
    Approved,
    Denied(String),
    Deferred(String),
}

/// Governance metrics for monitoring
#[derive(Debug, Clone)]
pub struct GovernanceMetrics {
    pub total_agents: usize,
    pub active_agents: usize,
    pub suspended_agents: usize,
    pub total_violations: usize,
    pub critical_violations: usize,
    pub decisions_made: usize,
    pub approvals: usize,
    pub denials: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_governance_engine_creation() {
        let config = GovernanceConfig::default();
        let engine = GovernanceEngine::new(config).await?;
        assert_eq!(engine.active_agents.len(), 0);
    }

    #[test]
    async fn test_agent_registration() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();

        engine.register_agent(agent_id.clone()).await?;
        assert_eq!(engine.active_agents.len(), 1);

        let state = engine.active_agents.get(&agent_id)?;
        assert_eq!(state.lifecycle_phase, LifecyclePhase::Initializing);
    }

    #[test]
    async fn test_max_agents_limit() {
        let mut config = GovernanceConfig::default();
        config.max_agents = 2;
        let engine = GovernanceEngine::new(config).await?;

        // Register two agents (should succeed)
        engine.register_agent(AgentId::new()).await?;
        engine.register_agent(AgentId::new()).await?;

        // Try to register third agent (should fail)
        let result = engine.register_agent(AgentId::new()).await;
        assert!(matches!(
            result,
            Err(GovernanceError::ResourceLimitExceeded(_))
        ));
    }

    #[test]
    async fn test_resource_allocation_within_quota() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        let request = ResourceRequest {
            memory_mb: 512,
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let decision = engine
            .evaluate_decision(&agent_id, DecisionType::ResourceAllocation(request))
            .await
            .unwrap();

        assert!(matches!(decision, GovernanceDecision::Approved));
    }

    #[test]
    async fn test_resource_allocation_exceeds_quota() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        let request = ResourceRequest {
            memory_mb: 2048, // Exceeds default quota of 1024
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let decision = engine
            .evaluate_decision(&agent_id, DecisionType::ResourceAllocation(request))
            .await
            .unwrap();

        assert!(matches!(decision, GovernanceDecision::Denied(_)));
    }

    #[test]
    async fn test_lifecycle_phase_update() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await?;

        let state = engine.active_agents.get(&agent_id)?;
        assert_eq!(state.lifecycle_phase, LifecyclePhase::Active);
    }

    #[test]
    async fn test_policy_violation_recording() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        let violation = PolicyViolation {
            timestamp: Utc::now(),
            policy_id: Uuid::new_v4(),
            violation_type: "resource_abuse".to_string(),
            severity: ViolationSeverity::Medium,
            details: "Exceeded CPU usage limit".to_string(),
        };

        engine.record_violation(&agent_id, violation).await?;

        let state = engine.active_agents.get(&agent_id)?;
        assert_eq!(state.violations.len(), 1);
    }

    #[test]
    async fn test_critical_violation_suspends_agent() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;
        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await?;

        let violation = PolicyViolation {
            timestamp: Utc::now(),
            policy_id: Uuid::new_v4(),
            violation_type: "security_breach".to_string(),
            severity: ViolationSeverity::Critical,
            details: "Attempted unauthorized access".to_string(),
        };

        engine.record_violation(&agent_id, violation).await?;

        let state = engine.active_agents.get(&agent_id)?;
        assert_eq!(state.lifecycle_phase, LifecyclePhase::Suspended);
    }

    #[test]
    async fn test_evolution_request_requires_active_phase() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        let request = EvolutionRequest {
            evolution_type: "capability_expansion".to_string(),
            target_capabilities: vec!["advanced_reasoning".to_string()],
            resource_requirements: ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 1024,
                duration_seconds: None,
            },
        };

        // Should fail because agent is in Initializing phase
        let decision = engine
            .evaluate_decision(&agent_id, DecisionType::Evolution(request.clone()))
            .await
            .unwrap();

        assert!(matches!(decision, GovernanceDecision::Denied(_)));

        // Update to active phase and retry
        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        let decision = engine
            .evaluate_decision(&agent_id, DecisionType::Evolution(request))
            .await
            .unwrap();

        // Should now pass lifecycle check (may still fail other checks)
        // In real implementation, this would depend on other subsystems
    }

    #[test]
    async fn test_permission_request() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        let decision = engine
            .evaluate_decision(
                &agent_id,
                DecisionType::PermissionRequest(Permission::ReadData),
            )
            .await
            .unwrap();

        // Basic permissions should be grantable
        assert!(matches!(decision, GovernanceDecision::Approved));
    }

    #[test]
    async fn test_coordination_request_requires_permission() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        let request = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: agent_id.clone(),
            target_agents: vec![AgentId::new()],
            coordination_type: "data_sharing".to_string(),
            duration: Some(3600),
        };

        // Should fail without coordination permission
        let decision = engine
            .evaluate_decision(&agent_id, DecisionType::Coordination(request))
            .await
            .unwrap();

        assert!(matches!(decision, GovernanceDecision::Denied(_)));
    }

    #[test]
    async fn test_audit_log_creation() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();

        engine.register_agent(agent_id.clone()).await?;

        let audit_log = engine.audit_log.read();
        assert!(audit_log.len() > 0);
        assert_eq!(audit_log[0].action, "agent_registered");
    }

    #[test]
    async fn test_audit_log_cleanup() {
        let mut config = GovernanceConfig::default();
        config.audit_retention_days = 0; // Immediate cleanup
        let engine = GovernanceEngine::new(config).await?;

        // Create an old audit entry
        engine.log_audit(None, "test_action", "success", serde_json::json!({}));

        assert_eq!(engine.audit_log.read().len(), 1);

        // Run cleanup
        engine.cleanup_audit_logs().await;

        // Should be cleaned up
        assert_eq!(engine.audit_log.read().len(), 0);
    }

    #[test]
    async fn test_kill_switch_blocks_decisions() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        // Activate kill switch
        engine.kill_switch.activate("Test emergency").await?;

        // Any decision should be denied
        let request = ResourceRequest {
            memory_mb: 512,
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let decision = engine
            .evaluate_decision(&agent_id, DecisionType::ResourceAllocation(request))
            .await
            .unwrap();

        assert!(matches!(decision, GovernanceDecision::Denied(_)));
    }

    #[test]
    async fn test_concurrent_agent_registration() {
        let engine = Arc::new(
            GovernanceEngine::new(GovernanceConfig::default())
                .await
                .unwrap(),
        );
        let mut handles = vec![];

        // Spawn 10 concurrent registrations
        for _ in 0..10 {
            let engine_clone = engine.clone();
            let handle = tokio::spawn(async move {
                let agent_id = AgentId::new();
                engine_clone.register_agent(agent_id).await
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            let result = handle.await?;
            assert!(result.is_ok());
        }

        assert_eq!(engine.active_agents.len(), 10);
    }

    #[test]
    async fn test_governance_metrics() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();

        // Register some agents
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        engine.register_agent(agent1.clone()).await?;
        engine.register_agent(agent2.clone()).await?;

        // Update phases
        engine
            .update_lifecycle_phase(&agent1, LifecyclePhase::Active)
            .await
            .unwrap();
        engine
            .update_lifecycle_phase(&agent2, LifecyclePhase::Suspended)
            .await
            .unwrap();

        // Add a violation
        let violation = PolicyViolation {
            timestamp: Utc::now(),
            policy_id: Uuid::new_v4(),
            violation_type: "test".to_string(),
            severity: ViolationSeverity::Low,
            details: "Test violation".to_string(),
        };
        engine.record_violation(&agent1, violation).await?;

        let metrics = engine.get_metrics().await;
        assert_eq!(metrics.total_agents, 2);
        assert_eq!(metrics.active_agents, 1);
        assert_eq!(metrics.suspended_agents, 1);
        assert_eq!(metrics.total_violations, 1);
    }

    #[test]
    async fn test_config_update() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();

        // Update config
        {
            let mut config = engine.config.write();
            config.max_agents = 500;
            config.strict_compliance = false;
        }

        // Verify update
        assert_eq!(engine.config.read().max_agents, 500);
        assert_eq!(engine.config.read().strict_compliance, false);
    }

    #[test]
    async fn test_resource_quota_update() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        // Update agent's resource quota
        {
            let mut agent_state = engine.active_agents.get_mut(&agent_id)?;
            agent_state.resource_quota.max_memory_mb = 2048;
        }

        // Now larger request should succeed
        let request = ResourceRequest {
            memory_mb: 2048,
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let decision = engine
            .evaluate_decision(&agent_id, DecisionType::ResourceAllocation(request))
            .await
            .unwrap();

        assert!(matches!(decision, GovernanceDecision::Approved));
    }

    #[test]
    async fn test_multiple_violations_tracking() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        // Add multiple violations
        for i in 0..5 {
            let violation = PolicyViolation {
                timestamp: Utc::now(),
                policy_id: Uuid::new_v4(),
                violation_type: format!("violation_{}", i),
                severity: ViolationSeverity::Low,
                details: format!("Test violation {}", i),
            };
            engine.record_violation(&agent_id, violation).await?;
        }

        let state = engine.active_agents.get(&agent_id)?;
        assert_eq!(state.violations.len(), 5);
    }

    #[test]
    async fn test_emergency_override_disabled() {
        let mut config = GovernanceConfig::default();
        config.emergency_override_enabled = false;
        let engine = GovernanceEngine::new(config).await?;
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        // Critical violation should not activate kill switch
        let violation = PolicyViolation {
            timestamp: Utc::now(),
            policy_id: Uuid::new_v4(),
            violation_type: "critical_breach".to_string(),
            severity: ViolationSeverity::Critical,
            details: "Test critical violation".to_string(),
        };

        engine.record_violation(&agent_id, violation).await?;

        // Kill switch should not be activated
        assert!(!engine.kill_switch.is_global_kill_active());
    }

    #[test]
    async fn test_agent_not_found_error() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new(); // Not registered

        let request = ResourceRequest {
            memory_mb: 512,
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let result = engine
            .evaluate_decision(&agent_id, DecisionType::ResourceAllocation(request))
            .await;

        assert!(matches!(result, Err(GovernanceError::InternalError(_))));
    }
}
