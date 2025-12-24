//! Agent lifecycle management module
//!
//! The LifecycleGovernor manages agent creation, evolution, and termination,
//! enforcing resource quotas and evolution approval workflows.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::governance_engine::{EvolutionRequest, ResourceQuota};
use crate::{GovernanceError, Result};
use stratoswarm_agent_core::agent::AgentId;
use stratoswarm_evolution_engines::traits::EvolutionStrategy;

/// Lifecycle governor for managing agent lifecycle
pub struct LifecycleGovernor {
    agent_states: DashMap<AgentId, AgentLifecycleState>,
    evolution_requests: DashMap<Uuid, EvolutionWorkflow>,
    resource_allocations: DashMap<AgentId, ResourceAllocation>,
    default_quota: ResourceQuota,
    evolution_approvers: Arc<RwLock<Vec<ApproverId>>>,
    termination_queue: Arc<RwLock<Vec<TerminationRequest>>>,
}

/// Agent lifecycle state
#[derive(Debug, Clone)]
pub struct AgentLifecycleState {
    pub agent_id: AgentId,
    pub phase: LifecyclePhase,
    pub created_at: DateTime<Utc>,
    pub last_transition: DateTime<Utc>,
    pub evolution_count: u32,
    pub parent_agent: Option<AgentId>,
    pub child_agents: Vec<AgentId>,
    pub metadata: serde_json::Value,
}

/// Lifecycle phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecyclePhase {
    Pending,
    Initializing,
    Active,
    Evolving,
    Suspended,
    Terminating,
    Terminated,
}

/// Evolution workflow state
#[derive(Debug, Clone)]
pub struct EvolutionWorkflow {
    pub request_id: Uuid,
    pub agent_id: AgentId,
    pub request: EvolutionRequest,
    pub status: WorkflowStatus,
    pub approvals: Vec<Approval>,
    pub required_approvals: u32,
    pub created_at: DateTime<Utc>,
    pub deadline: DateTime<Utc>,
}

/// Workflow status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
    InProgress,
    Completed,
    Failed,
}

/// Approval record
#[derive(Debug, Clone)]
pub struct Approval {
    pub approver_id: ApproverId,
    pub decision: ApprovalDecision,
    pub timestamp: DateTime<Utc>,
    pub comments: Option<String>,
}

/// Approver identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ApproverId {
    Agent(AgentId),
    System,
    Human(String),
}

/// Approval decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    Approve,
    Reject,
    Defer,
}

/// Resource allocation for an agent
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub agent_id: AgentId,
    pub quota: ResourceQuota,
    pub current_usage: ResourceUsage,
    pub allocated_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

/// Current resource usage
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    pub memory_mb: u64,
    pub cpu_cores: f32,
    pub gpu_memory_mb: u64,
    pub network_bandwidth_mbps: u64,
    pub storage_gb: u64,
}

/// Termination request
#[derive(Debug, Clone)]
pub struct TerminationRequest {
    pub agent_id: AgentId,
    pub reason: TerminationReason,
    pub requested_at: DateTime<Utc>,
    pub requested_by: Option<ApproverId>,
    pub force: bool,
}

/// Reasons for termination
#[derive(Debug, Clone)]
pub enum TerminationReason {
    UserRequested,
    PolicyViolation(String),
    ResourceExhaustion,
    EvolutionFailure,
    SystemShutdown,
    Other(String),
}

/// Lifecycle decision result
#[derive(Debug, Clone)]
pub enum LifecycleDecision {
    Approved,
    Rejected(String),
    Deferred(String),
    RequiresApproval(Uuid), // Workflow ID
}

impl LifecycleGovernor {
    /// Create a new lifecycle governor
    pub fn new(default_quota: ResourceQuota) -> Self {
        Self {
            agent_states: DashMap::new(),
            evolution_requests: DashMap::new(),
            resource_allocations: DashMap::new(),
            default_quota,
            evolution_approvers: Arc::new(RwLock::new(vec![ApproverId::System])),
            termination_queue: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a new agent
    pub async fn register_agent(&self, agent_id: &AgentId) -> Result<()> {
        info!("Registering agent in lifecycle governor: {:?}", agent_id);

        let state = AgentLifecycleState {
            agent_id: agent_id.clone(),
            phase: LifecyclePhase::Pending,
            created_at: Utc::now(),
            last_transition: Utc::now(),
            evolution_count: 0,
            parent_agent: None,
            child_agents: Vec::new(),
            metadata: serde_json::json!({}),
        };

        let allocation = ResourceAllocation {
            agent_id: agent_id.clone(),
            quota: self.default_quota.clone(),
            current_usage: ResourceUsage::default(),
            allocated_at: Utc::now(),
            last_updated: Utc::now(),
        };

        self.agent_states.insert(agent_id.clone(), state);
        self.resource_allocations
            .insert(agent_id.clone(), allocation);

        Ok(())
    }

    /// Transition agent to a new lifecycle phase
    pub async fn transition_phase(
        &self,
        agent_id: &AgentId,
        new_phase: LifecyclePhase,
    ) -> Result<()> {
        let mut state = self
            .agent_states
            .get_mut(agent_id)
            .ok_or_else(|| GovernanceError::InternalError("Agent not found".to_string()))?;

        // Validate transition
        if !self.is_valid_transition(state.phase, new_phase) {
            return Err(GovernanceError::LifecycleError(format!(
                "Invalid transition from {:?} to {:?}",
                state.phase, new_phase
            )));
        }

        info!(
            "Transitioning agent {:?} from {:?} to {:?}",
            agent_id, state.phase, new_phase
        );

        state.phase = new_phase;
        state.last_transition = Utc::now();

        // Handle special transitions
        match new_phase {
            LifecyclePhase::Terminating => {
                self.initiate_termination(agent_id, TerminationReason::UserRequested)
                    .await?;
            }
            LifecyclePhase::Terminated => {
                self.cleanup_agent_resources(agent_id).await?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Check if a phase transition is valid
    fn is_valid_transition(&self, from: LifecyclePhase, to: LifecyclePhase) -> bool {
        match (from, to) {
            // Pending can go to Initializing
            (LifecyclePhase::Pending, LifecyclePhase::Initializing) => true,

            // Initializing can go to Active or Terminated
            (LifecyclePhase::Initializing, LifecyclePhase::Active) => true,
            (LifecyclePhase::Initializing, LifecyclePhase::Terminated) => true,

            // Active can go to Evolving, Suspended, or Terminating
            (LifecyclePhase::Active, LifecyclePhase::Evolving) => true,
            (LifecyclePhase::Active, LifecyclePhase::Suspended) => true,
            (LifecyclePhase::Active, LifecyclePhase::Terminating) => true,

            // Evolving can go back to Active or to Terminated
            (LifecyclePhase::Evolving, LifecyclePhase::Active) => true,
            (LifecyclePhase::Evolving, LifecyclePhase::Terminated) => true,

            // Suspended can go to Active or Terminating
            (LifecyclePhase::Suspended, LifecyclePhase::Active) => true,
            (LifecyclePhase::Suspended, LifecyclePhase::Terminating) => true,

            // Terminating can only go to Terminated
            (LifecyclePhase::Terminating, LifecyclePhase::Terminated) => true,

            // All other transitions are invalid
            _ => false,
        }
    }

    /// Evaluate an evolution request
    pub async fn evaluate_evolution(
        &self,
        agent_id: &AgentId,
        request: &EvolutionRequest,
    ) -> Result<LifecycleDecision> {
        debug!("Evaluating evolution request for agent {:?}", agent_id);

        // Check agent state
        let state = self
            .agent_states
            .get(agent_id)
            .ok_or_else(|| GovernanceError::InternalError("Agent not found".to_string()))?;

        if state.phase != LifecyclePhase::Active {
            return Ok(LifecycleDecision::Rejected(
                "Agent must be in Active phase".to_string(),
            ));
        }

        // Check resource requirements
        if !self
            .check_resource_availability(agent_id, &request.resource_requirements)
            .await?
        {
            return Ok(LifecycleDecision::Rejected(
                "Insufficient resources".to_string(),
            ));
        }

        // Check evolution count limit
        if state.evolution_count >= 10 {
            return Ok(LifecycleDecision::Rejected(
                "Evolution count limit reached".to_string(),
            ));
        }

        // Create evolution workflow
        let workflow = EvolutionWorkflow {
            request_id: Uuid::new_v4(),
            agent_id: agent_id.clone(),
            request: request.clone(),
            status: WorkflowStatus::Pending,
            approvals: Vec::new(),
            required_approvals: self.get_required_approvals(&request),
            created_at: Utc::now(),
            deadline: Utc::now() + chrono::Duration::hours(24),
        };

        let workflow_id = workflow.request_id;
        self.evolution_requests.insert(workflow_id, workflow);

        // Check if auto-approval is possible
        if self.can_auto_approve(&request) {
            self.auto_approve_evolution(workflow_id).await?;
            return Ok(LifecycleDecision::Approved);
        }

        Ok(LifecycleDecision::RequiresApproval(workflow_id))
    }

    /// Check if resources are available for allocation
    async fn check_resource_availability(
        &self,
        agent_id: &AgentId,
        request: &crate::governance_engine::ResourceRequest,
    ) -> Result<bool> {
        let allocation = self.resource_allocations.get(agent_id).ok_or_else(|| {
            GovernanceError::InternalError("Resource allocation not found".to_string())
        })?;

        // Check against quota
        let projected_memory = allocation.current_usage.memory_mb + request.memory_mb;
        let projected_cpu = allocation.current_usage.cpu_cores + request.cpu_cores;
        let projected_gpu = allocation.current_usage.gpu_memory_mb + request.gpu_memory_mb;

        Ok(projected_memory <= allocation.quota.max_memory_mb
            && projected_cpu <= allocation.quota.max_cpu_cores
            && projected_gpu <= allocation.quota.max_gpu_memory_mb)
    }

    /// Get required approvals for an evolution request
    fn get_required_approvals(&self, request: &EvolutionRequest) -> u32 {
        // More complex evolutions require more approvals
        match request.evolution_type.as_str() {
            "minor_capability" => 0,
            "major_capability" => 1,
            "architecture_change" => 2,
            _ => 1,
        }
    }

    /// Check if a request can be auto-approved
    fn can_auto_approve(&self, request: &EvolutionRequest) -> bool {
        request.evolution_type == "minor_capability"
            && request.resource_requirements.memory_mb < 512
            && request.resource_requirements.cpu_cores < 1.0
    }

    /// Auto-approve an evolution workflow
    async fn auto_approve_evolution(&self, workflow_id: Uuid) -> Result<()> {
        let mut workflow = self
            .evolution_requests
            .get_mut(&workflow_id)
            .ok_or_else(|| GovernanceError::InternalError("Workflow not found".to_string()))?;

        workflow.approvals.push(Approval {
            approver_id: ApproverId::System,
            decision: ApprovalDecision::Approve,
            timestamp: Utc::now(),
            comments: Some("Auto-approved: minor evolution".to_string()),
        });

        workflow.status = WorkflowStatus::Approved;

        Ok(())
    }

    /// Submit approval for an evolution workflow
    pub async fn submit_approval(
        &self,
        workflow_id: Uuid,
        approver: ApproverId,
        decision: ApprovalDecision,
        comments: Option<String>,
    ) -> Result<()> {
        let mut workflow = self
            .evolution_requests
            .get_mut(&workflow_id)
            .ok_or_else(|| GovernanceError::InternalError("Workflow not found".to_string()))?;

        // Check if already decided
        if workflow.status != WorkflowStatus::Pending {
            return Err(GovernanceError::LifecycleError(
                "Workflow already decided".to_string(),
            ));
        }

        // Check if approver already voted
        if workflow.approvals.iter().any(|a| a.approver_id == approver) {
            return Err(GovernanceError::LifecycleError(
                "Approver already voted".to_string(),
            ));
        }

        // Add approval
        workflow.approvals.push(Approval {
            approver_id: approver,
            decision,
            timestamp: Utc::now(),
            comments,
        });

        // Check if we have enough approvals
        let approve_count = workflow
            .approvals
            .iter()
            .filter(|a| a.decision == ApprovalDecision::Approve)
            .count() as u32;

        let reject_count = workflow
            .approvals
            .iter()
            .filter(|a| a.decision == ApprovalDecision::Reject)
            .count() as u32;

        if approve_count >= workflow.required_approvals {
            workflow.status = WorkflowStatus::Approved;
            info!("Evolution workflow {} approved", workflow_id);
        } else if reject_count > 0 {
            workflow.status = WorkflowStatus::Rejected;
            info!("Evolution workflow {} rejected", workflow_id);
        }

        Ok(())
    }

    /// Execute an approved evolution
    pub async fn execute_evolution(&self, workflow_id: Uuid) -> Result<()> {
        let mut workflow = self
            .evolution_requests
            .get_mut(&workflow_id)
            .ok_or_else(|| GovernanceError::InternalError("Workflow not found".to_string()))?;

        if workflow.status != WorkflowStatus::Approved {
            return Err(GovernanceError::LifecycleError(
                "Workflow not approved".to_string(),
            ));
        }

        workflow.status = WorkflowStatus::InProgress;
        let agent_id = workflow.agent_id.clone();

        // Transition agent to evolving state
        self.transition_phase(&agent_id, LifecyclePhase::Evolving)
            .await?;

        // Update evolution count
        if let Some(mut state) = self.agent_states.get_mut(&agent_id) {
            state.evolution_count += 1;
        }

        // In a real implementation, this would trigger the actual evolution process
        workflow.status = WorkflowStatus::Completed;

        // Transition back to active
        self.transition_phase(&agent_id, LifecyclePhase::Active)
            .await?;

        Ok(())
    }

    /// Update resource usage for an agent
    pub async fn update_resource_usage(
        &self,
        agent_id: &AgentId,
        usage: ResourceUsage,
    ) -> Result<()> {
        let mut allocation = self.resource_allocations.get_mut(agent_id).ok_or_else(|| {
            GovernanceError::InternalError("Resource allocation not found".to_string())
        })?;

        // Check if usage exceeds quota
        if usage.memory_mb > allocation.quota.max_memory_mb
            || usage.cpu_cores > allocation.quota.max_cpu_cores
            || usage.gpu_memory_mb > allocation.quota.max_gpu_memory_mb
        {
            return Err(GovernanceError::ResourceLimitExceeded(
                "Usage exceeds quota".to_string(),
            ));
        }

        allocation.current_usage = usage;
        allocation.last_updated = Utc::now();

        Ok(())
    }

    /// Update resource quota for an agent
    pub async fn update_resource_quota(
        &self,
        agent_id: &AgentId,
        new_quota: ResourceQuota,
    ) -> Result<()> {
        let mut allocation = self.resource_allocations.get_mut(agent_id).ok_or_else(|| {
            GovernanceError::InternalError("Resource allocation not found".to_string())
        })?;

        allocation.quota = new_quota;
        allocation.last_updated = Utc::now();

        Ok(())
    }

    /// Initiate agent termination
    async fn initiate_termination(
        &self,
        agent_id: &AgentId,
        reason: TerminationReason,
    ) -> Result<()> {
        let request = TerminationRequest {
            agent_id: agent_id.clone(),
            reason,
            requested_at: Utc::now(),
            requested_by: None,
            force: false,
        };

        self.termination_queue.write().push(request);

        Ok(())
    }

    /// Process termination queue
    pub async fn process_terminations(&self) -> Result<Vec<AgentId>> {
        let mut terminated = Vec::new();
        let requests: Vec<_> = self.termination_queue.write().drain(..).collect();

        for request in requests {
            match self.terminate_agent(&request).await {
                Ok(_) => {
                    terminated.push(request.agent_id);
                }
                Err(e) => {
                    error!("Failed to terminate agent {:?}: {}", request.agent_id, e);
                    if !request.force {
                        // Re-queue non-forced terminations
                        self.termination_queue.write().push(request);
                    }
                }
            }
        }

        Ok(terminated)
    }

    /// Terminate an agent
    async fn terminate_agent(&self, request: &TerminationRequest) -> Result<()> {
        info!(
            "Terminating agent {:?}, reason: {:?}",
            request.agent_id, request.reason
        );

        // Check if agent has children
        if let Some(state) = self.agent_states.get(&request.agent_id) {
            if !state.child_agents.is_empty() && !request.force {
                return Err(GovernanceError::LifecycleError(
                    "Cannot terminate agent with active children".to_string(),
                ));
            }
        }

        // Transition to terminated
        self.transition_phase(&request.agent_id, LifecyclePhase::Terminated)
            .await?;

        Ok(())
    }

    /// Clean up resources for a terminated agent
    async fn cleanup_agent_resources(&self, agent_id: &AgentId) -> Result<()> {
        // Remove from resource allocations
        self.resource_allocations.remove(agent_id);

        // Cancel any pending evolution workflows
        let workflow_ids: Vec<_> = self
            .evolution_requests
            .iter()
            .filter(|entry| entry.value().agent_id == *agent_id)
            .map(|entry| *entry.key())
            .collect();

        for workflow_id in workflow_ids {
            self.evolution_requests.remove(&workflow_id);
        }

        Ok(())
    }

    /// Get lifecycle state for an agent
    pub async fn get_agent_state(&self, agent_id: &AgentId) -> Result<AgentLifecycleState> {
        self.agent_states
            .get(agent_id)
            .map(|entry| entry.clone())
            .ok_or_else(|| GovernanceError::InternalError("Agent not found".to_string()))
    }

    /// Get resource allocation for an agent
    pub async fn get_resource_allocation(&self, agent_id: &AgentId) -> Result<ResourceAllocation> {
        self.resource_allocations
            .get(agent_id)
            .map(|entry| entry.clone())
            .ok_or_else(|| {
                GovernanceError::InternalError("Resource allocation not found".to_string())
            })
    }

    /// Get all evolution workflows
    pub async fn get_evolution_workflows(&self) -> Vec<EvolutionWorkflow> {
        self.evolution_requests
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Add an evolution approver
    pub async fn add_approver(&self, approver: ApproverId) {
        self.evolution_approvers.write().push(approver);
    }

    /// Remove an evolution approver
    pub async fn remove_approver(&self, approver: &ApproverId) {
        self.evolution_approvers.write().retain(|a| a != approver);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_lifecycle_governor_creation() {
        let quota = ResourceQuota::default();
        let governor = LifecycleGovernor::new(quota);
        assert_eq!(governor.agent_states.len(), 0);
    }

    #[test]
    async fn test_agent_registration() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();

        governor.register_agent(&agent_id).await?;

        let state = governor.get_agent_state(&agent_id).await?;
        assert_eq!(state.phase, LifecyclePhase::Pending);
        assert_eq!(state.evolution_count, 0);
    }

    #[test]
    async fn test_valid_phase_transition() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;

        // Valid transition: Pending -> Initializing
        governor
            .transition_phase(&agent_id, LifecyclePhase::Initializing)
            .await
            ?;

        let state = governor.get_agent_state(&agent_id).await?;
        assert_eq!(state.phase, LifecyclePhase::Initializing);
    }

    #[test]
    async fn test_invalid_phase_transition() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;

        // Invalid transition: Pending -> Active (must go through Initializing)
        let result = governor
            .transition_phase(&agent_id, LifecyclePhase::Active)
            .await;
        assert!(matches!(result, Err(GovernanceError::LifecycleError(_))));
    }

    #[test]
    async fn test_evolution_request_evaluation() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;

        // Transition to active state
        governor
            .transition_phase(&agent_id, LifecyclePhase::Initializing)
            .await
            ?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        let request = EvolutionRequest {
            evolution_type: "minor_capability".to_string(),
            target_capabilities: vec!["enhanced_processing".to_string()],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 256,
                cpu_cores: 0.5,
                gpu_memory_mb: 0,
                duration_seconds: Some(3600),
            },
        };

        let decision = governor
            .evaluate_evolution(&agent_id, &request)
            .await
            .unwrap();
        assert!(matches!(decision, LifecycleDecision::Approved));
    }

    #[test]
    async fn test_evolution_requires_active_phase() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;

        let request = EvolutionRequest {
            evolution_type: "major_capability".to_string(),
            target_capabilities: vec![],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 0,
                duration_seconds: None,
            },
        };

        let decision = governor
            .evaluate_evolution(&agent_id, &request)
            .await
            .unwrap();
        assert!(matches!(decision, LifecycleDecision::Rejected(_)));
    }

    #[test]
    async fn test_evolution_workflow_creation() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Initializing)
            .await
            ?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        let request = EvolutionRequest {
            evolution_type: "major_capability".to_string(),
            target_capabilities: vec!["advanced_reasoning".to_string()],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 1024,
                cpu_cores: 2.0,
                gpu_memory_mb: 512,
                duration_seconds: None,
            },
        };

        let decision = governor
            .evaluate_evolution(&agent_id, &request)
            .await
            .unwrap();

        if let LifecycleDecision::RequiresApproval(workflow_id) = decision {
            let workflows = governor.get_evolution_workflows().await;
            assert!(workflows.iter().any(|w| w.request_id == workflow_id));
        } else {
            panic!("Expected RequiresApproval decision");
        }
    }

    #[test]
    async fn test_evolution_approval_workflow() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Initializing)
            .await
            ?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        let request = EvolutionRequest {
            evolution_type: "major_capability".to_string(),
            target_capabilities: vec![],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 0,
                duration_seconds: None,
            },
        };

        let decision = governor
            .evaluate_evolution(&agent_id, &request)
            .await
            .unwrap();

        if let LifecycleDecision::RequiresApproval(workflow_id) = decision {
            // Submit approval
            governor
                .submit_approval(
                    workflow_id,
                    ApproverId::System,
                    ApprovalDecision::Approve,
                    Some("Looks good".to_string()),
                )
                .await
                .unwrap();

            // Check workflow status
            let workflow = governor.evolution_requests.get(&workflow_id)?;
            assert_eq!(workflow.status, WorkflowStatus::Approved);
        }
    }

    #[test]
    async fn test_evolution_rejection() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Initializing)
            .await
            ?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        let request = EvolutionRequest {
            evolution_type: "architecture_change".to_string(),
            target_capabilities: vec![],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 2048,
                cpu_cores: 4.0,
                gpu_memory_mb: 2048,
                duration_seconds: None,
            },
        };

        let decision = governor
            .evaluate_evolution(&agent_id, &request)
            .await
            .unwrap();

        if let LifecycleDecision::RequiresApproval(workflow_id) = decision {
            // Submit rejection
            governor
                .submit_approval(
                    workflow_id,
                    ApproverId::System,
                    ApprovalDecision::Reject,
                    Some("Too risky".to_string()),
                )
                .await
                .unwrap();

            let workflow = governor.evolution_requests.get(&workflow_id)?;
            assert_eq!(workflow.status, WorkflowStatus::Rejected);
        }
    }

    #[test]
    async fn test_duplicate_approval_prevention() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Initializing)
            .await
            ?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        let request = EvolutionRequest {
            evolution_type: "major_capability".to_string(),
            target_capabilities: vec![],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 0,
                duration_seconds: None,
            },
        };

        let decision = governor
            .evaluate_evolution(&agent_id, &request)
            .await
            .unwrap();

        if let LifecycleDecision::RequiresApproval(workflow_id) = decision {
            // First approval
            governor
                .submit_approval(
                    workflow_id,
                    ApproverId::System,
                    ApprovalDecision::Approve,
                    None,
                )
                .await
                .unwrap();

            // Duplicate approval should fail
            let result = governor
                .submit_approval(
                    workflow_id,
                    ApproverId::System,
                    ApprovalDecision::Approve,
                    None,
                )
                .await;

            assert!(matches!(result, Err(GovernanceError::LifecycleError(_))));
        }
    }

    #[test]
    async fn test_resource_usage_update() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;

        let usage = ResourceUsage {
            memory_mb: 512,
            cpu_cores: 1.5,
            gpu_memory_mb: 256,
            network_bandwidth_mbps: 50,
            storage_gb: 2,
        };

        governor
            .update_resource_usage(&agent_id, usage.clone())
            .await
            .unwrap();

        let allocation = governor.get_resource_allocation(&agent_id).await?;
        assert_eq!(allocation.current_usage.memory_mb, 512);
        assert_eq!(allocation.current_usage.cpu_cores, 1.5);
    }

    #[test]
    async fn test_resource_usage_exceeds_quota() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;

        let usage = ResourceUsage {
            memory_mb: 2048, // Exceeds default quota of 1024
            cpu_cores: 1.0,
            gpu_memory_mb: 0,
            network_bandwidth_mbps: 0,
            storage_gb: 0,
        };

        let result = governor.update_resource_usage(&agent_id, usage).await;
        assert!(matches!(
            result,
            Err(GovernanceError::ResourceLimitExceeded(_))
        ));
    }

    #[test]
    async fn test_resource_quota_update() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;

        let new_quota = ResourceQuota {
            max_memory_mb: 2048,
            max_cpu_cores: 4.0,
            max_gpu_memory_mb: 4096,
            max_network_bandwidth_mbps: 200,
            max_storage_gb: 20,
        };

        governor
            .update_resource_quota(&agent_id, new_quota.clone())
            .await
            .unwrap();

        let allocation = governor.get_resource_allocation(&agent_id).await?;
        assert_eq!(allocation.quota.max_memory_mb, 2048);
        assert_eq!(allocation.quota.max_cpu_cores, 4.0);
    }

    #[test]
    async fn test_evolution_count_tracking() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Initializing)
            .await
            ?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        let request = EvolutionRequest {
            evolution_type: "minor_capability".to_string(),
            target_capabilities: vec![],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 256,
                cpu_cores: 0.5,
                gpu_memory_mb: 0,
                duration_seconds: None,
            },
        };

        // Execute evolution
        let decision = governor
            .evaluate_evolution(&agent_id, &request)
            .await
            .unwrap();
        assert!(matches!(decision, LifecycleDecision::Approved));

        // Find and execute the workflow
        let workflows = governor.get_evolution_workflows().await;
        if let Some(workflow) = workflows.iter().find(|w| w.agent_id == agent_id) {
            governor
                .execute_evolution(workflow.request_id)
                .await
                .unwrap();
        }

        let state = governor.get_agent_state(&agent_id).await?;
        assert_eq!(state.evolution_count, 1);
    }

    #[test]
    async fn test_evolution_count_limit() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Initializing)
            .await
            ?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        // Set evolution count to limit
        {
            let mut state = governor.agent_states.get_mut(&agent_id)?;
            state.evolution_count = 10;
        }

        let request = EvolutionRequest {
            evolution_type: "minor_capability".to_string(),
            target_capabilities: vec![],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 256,
                cpu_cores: 0.5,
                gpu_memory_mb: 0,
                duration_seconds: None,
            },
        };

        let decision = governor
            .evaluate_evolution(&agent_id, &request)
            .await
            .unwrap();
        assert!(matches!(decision, LifecycleDecision::Rejected(_)));
    }

    #[test]
    async fn test_termination_queue() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Initializing)
            .await
            ?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        // Initiate termination
        governor
            .transition_phase(&agent_id, LifecyclePhase::Terminating)
            .await
            .unwrap();

        // Process terminations
        let terminated = governor.process_terminations().await?;
        assert_eq!(terminated.len(), 1);
        assert_eq!(terminated[0], agent_id);

        let state = governor.get_agent_state(&agent_id).await?;
        assert_eq!(state.phase, LifecyclePhase::Terminated);
    }

    #[test]
    async fn test_resource_cleanup_on_termination() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Initializing)
            .await
            ?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();
        governor
            .transition_phase(&agent_id, LifecyclePhase::Terminating)
            .await
            .unwrap();
        governor
            .transition_phase(&agent_id, LifecyclePhase::Terminated)
            .await
            .unwrap();

        // Resource allocation should be cleaned up
        let result = governor.get_resource_allocation(&agent_id).await;
        assert!(result.is_err());
    }

    #[test]
    async fn test_approver_management() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());

        let approver = ApproverId::Human("alice".to_string());
        governor.add_approver(approver.clone()).await;

        assert_eq!(governor.evolution_approvers.read().len(), 2); // System + alice

        governor.remove_approver(&approver).await;
        assert_eq!(governor.evolution_approvers.read().len(), 1); // Just System
    }

    #[test]
    async fn test_workflow_execution() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let agent_id = AgentId::new();
        governor.register_agent(&agent_id).await?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Initializing)
            .await
            ?;
        governor
            .transition_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        let request = EvolutionRequest {
            evolution_type: "minor_capability".to_string(),
            target_capabilities: vec![],
            resource_requirements: crate::governance_engine::ResourceRequest {
                memory_mb: 256,
                cpu_cores: 0.5,
                gpu_memory_mb: 0,
                duration_seconds: None,
            },
        };

        let decision = governor
            .evaluate_evolution(&agent_id, &request)
            .await
            .unwrap();
        assert!(matches!(decision, LifecycleDecision::Approved));

        // Find the approved workflow
        let workflows = governor.get_evolution_workflows().await;
        let workflow = workflows
            .iter()
            .find(|w| w.agent_id == agent_id && w.status == WorkflowStatus::Approved)
            .unwrap();

        // Execute it
        governor
            .execute_evolution(workflow.request_id)
            .await
            .unwrap();

        // Verify completion
        let updated_workflow = governor
            .evolution_requests
            .get(&workflow.request_id)
            .unwrap();
        assert_eq!(updated_workflow.status, WorkflowStatus::Completed);
    }

    #[test]
    async fn test_child_agent_termination_prevention() {
        let governor = LifecycleGovernor::new(ResourceQuota::default());
        let parent_id = AgentId::new();
        let child_id = AgentId::new();

        governor.register_agent(&parent_id).await?;
        governor.register_agent(&child_id).await?;

        // Set up parent-child relationship
        {
            let mut parent_state = governor.agent_states.get_mut(&parent_id)?;
            parent_state.child_agents.push(child_id.clone());
        }
        {
            let mut child_state = governor.agent_states.get_mut(&child_id)?;
            child_state.parent_agent = Some(parent_id.clone());
        }

        governor
            .transition_phase(&parent_id, LifecyclePhase::Initializing)
            .await
            .unwrap();
        governor
            .transition_phase(&parent_id, LifecyclePhase::Active)
            .await
            .unwrap();
        governor
            .transition_phase(&parent_id, LifecyclePhase::Terminating)
            .await
            .unwrap();

        // Process termination should fail
        let terminated = governor.process_terminations().await?;
        assert_eq!(terminated.len(), 0);

        // Parent should still be terminating, not terminated
        let state = governor.get_agent_state(&parent_id).await?;
        assert_eq!(state.phase, LifecyclePhase::Terminating);
    }
}
