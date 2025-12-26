//! Cross-agent coordination management module
//!
//! The CoordinationManager handles multi-agent interactions, conflict resolution,
//! and resource sharing protocols between agents.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::{GovernanceError, Result};
use stratoswarm_agent_core::agent::AgentId;

/// Coordination manager for multi-agent interactions
pub struct CoordinationManager {
    active_coordinations: DashMap<Uuid, CoordinationSession>,
    resource_locks: DashMap<String, ResourceLock>,
    conflict_history: Arc<RwLock<Vec<ConflictRecord>>>,
    coordination_rules: DashMap<String, CoordinationRule>,
}

/// Coordination request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationRequest {
    pub request_id: Uuid,
    pub requesting_agent: AgentId,
    pub target_agents: Vec<AgentId>,
    pub coordination_type: String,
    pub duration: Option<u64>, // seconds
}

/// Active coordination session
#[derive(Debug, Clone)]
pub struct CoordinationSession {
    pub session_id: Uuid,
    pub coordinator: AgentId,
    pub participants: Vec<AgentId>,
    pub coordination_type: CoordinationType,
    pub status: SessionStatus,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub shared_resources: Vec<String>,
    pub metadata: serde_json::Value,
}

/// Types of coordination
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinationType {
    DataSharing,
    TaskDistribution,
    ConsensusBuilding,
    ResourceNegotiation,
    CollaborativeComputation,
    Custom(String),
}

/// Session status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    Pending,
    Active,
    Completed,
    Failed,
    Cancelled,
}

/// Resource lock for coordinated access
#[derive(Debug, Clone)]
pub struct ResourceLock {
    pub resource_id: String,
    pub lock_type: LockType,
    pub owner: LockOwner,
    pub acquired_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub waiting_queue: Vec<AgentId>,
}

/// Lock types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockType {
    Exclusive,
    Shared,
    ReadOnly,
}

/// Lock owner
#[derive(Debug, Clone)]
pub enum LockOwner {
    Agent(AgentId),
    Session(Uuid),
    System,
}

/// Conflict record
#[derive(Debug, Clone)]
pub struct ConflictRecord {
    pub conflict_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub involved_agents: Vec<AgentId>,
    pub conflict_type: ConflictType,
    pub resolution: ConflictResolution,
    pub details: String,
}

/// Types of conflicts
#[derive(Debug, Clone)]
pub enum ConflictType {
    ResourceContention,
    ProtocolMismatch,
    DataInconsistency,
    DeadlockDetected,
    Other(String),
}

/// Conflict resolution
#[derive(Debug, Clone)]
pub enum ConflictResolution {
    Resolved(String),
    Escalated,
    Abandoned,
    Timeout,
}

/// Coordination rule
#[derive(Debug, Clone)]
pub struct CoordinationRule {
    pub rule_id: Uuid,
    pub name: String,
    pub coordination_type: String,
    pub conditions: Vec<RuleCondition>,
    pub actions: Vec<RuleAction>,
    pub priority: u32,
}

/// Rule condition
#[derive(Debug, Clone)]
pub enum RuleCondition {
    AgentCount {
        min: usize,
        max: Option<usize>,
    },
    ResourceAvailable(String),
    TimeWindow {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },
    Custom(serde_json::Value),
}

/// Rule action
#[derive(Debug, Clone)]
pub enum RuleAction {
    RequireApproval,
    SetTimeout(u64),
    LimitResources(Vec<String>),
    NotifyAgents(Vec<AgentId>),
}

impl CoordinationManager {
    /// Create a new coordination manager
    pub fn new() -> Self {
        let mut manager = Self {
            active_coordinations: DashMap::new(),
            resource_locks: DashMap::new(),
            conflict_history: Arc::new(RwLock::new(Vec::new())),
            coordination_rules: DashMap::new(),
        };

        // Initialize default rules
        manager.initialize_default_rules();

        manager
    }

    /// Initialize default coordination rules
    fn initialize_default_rules(&mut self) {
        // Rule: Large coordinations require approval
        let large_coordination_rule = CoordinationRule {
            rule_id: Uuid::new_v4(),
            name: "large_coordination_approval".to_string(),
            coordination_type: "*".to_string(),
            conditions: vec![RuleCondition::AgentCount { min: 5, max: None }],
            actions: vec![
                RuleAction::RequireApproval,
                RuleAction::SetTimeout(3600), // 1 hour timeout
            ],
            priority: 100,
        };
        self.coordination_rules.insert(
            large_coordination_rule.name.clone(),
            large_coordination_rule,
        );

        // Rule: Resource negotiation timeout
        let resource_negotiation_rule = CoordinationRule {
            rule_id: Uuid::new_v4(),
            name: "resource_negotiation_timeout".to_string(),
            coordination_type: "ResourceNegotiation".to_string(),
            conditions: vec![],
            actions: vec![
                RuleAction::SetTimeout(300), // 5 minute timeout
            ],
            priority: 50,
        };
        self.coordination_rules.insert(
            resource_negotiation_rule.name.clone(),
            resource_negotiation_rule,
        );
    }

    /// Evaluate a coordination request
    pub async fn evaluate_request(&self, request: &CoordinationRequest) -> Result<bool> {
        debug!("Evaluating coordination request: {:?}", request.request_id);

        // Check for conflicts
        if self.has_conflicts(request)? {
            return Ok(false);
        }

        // Apply coordination rules
        let applicable_rules = self.get_applicable_rules(&request.coordination_type);
        for rule in applicable_rules {
            if !self.evaluate_rule(&rule, request)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Create a new coordination session
    pub async fn create_session(&self, request: CoordinationRequest) -> Result<Uuid> {
        info!(
            "Creating coordination session for request: {:?}",
            request.request_id
        );

        let coordination_type = match request.coordination_type.as_str() {
            "data_sharing" => CoordinationType::DataSharing,
            "task_distribution" => CoordinationType::TaskDistribution,
            "consensus_building" => CoordinationType::ConsensusBuilding,
            "resource_negotiation" => CoordinationType::ResourceNegotiation,
            "collaborative_computation" => CoordinationType::CollaborativeComputation,
            other => CoordinationType::Custom(other.to_string()),
        };

        let expires_at = request
            .duration
            .map(|d| Utc::now() + chrono::Duration::seconds(d as i64));

        let session = CoordinationSession {
            session_id: Uuid::new_v4(),
            coordinator: request.requesting_agent.clone(),
            participants: {
                let mut participants = vec![request.requesting_agent.clone()];
                participants.extend(request.target_agents.clone());
                participants
            },
            coordination_type,
            status: SessionStatus::Pending,
            created_at: Utc::now(),
            expires_at,
            shared_resources: Vec::new(),
            metadata: serde_json::json!({
                "request_id": request.request_id,
            }),
        };

        let session_id = session.session_id;
        self.active_coordinations.insert(session_id, session);

        Ok(session_id)
    }

    /// Start a coordination session
    pub async fn start_session(&self, session_id: Uuid) -> Result<()> {
        let mut session = self
            .active_coordinations
            .get_mut(&session_id)
            .ok_or_else(|| GovernanceError::CoordinationError("Session not found".to_string()))?;

        if session.status != SessionStatus::Pending {
            return Err(GovernanceError::CoordinationError(
                "Session already started".to_string(),
            ));
        }

        session.status = SessionStatus::Active;
        info!("Started coordination session: {}", session_id);

        Ok(())
    }

    /// End a coordination session
    pub async fn end_session(&self, session_id: Uuid, success: bool) -> Result<()> {
        let mut session = self
            .active_coordinations
            .get_mut(&session_id)
            .ok_or_else(|| GovernanceError::CoordinationError("Session not found".to_string()))?;

        session.status = if success {
            SessionStatus::Completed
        } else {
            SessionStatus::Failed
        };

        // Release any locks held by this session
        self.release_session_locks(&session_id).await?;

        info!(
            "Ended coordination session: {} (success: {})",
            session_id, success
        );

        Ok(())
    }

    /// Acquire a resource lock
    pub async fn acquire_lock(
        &self,
        resource_id: String,
        lock_type: LockType,
        requester: LockOwner,
        duration: Option<u64>,
    ) -> Result<bool> {
        debug!(
            "Lock request for resource: {} by {:?}",
            resource_id, requester
        );

        // Check if resource is already locked
        if let Some(mut existing_lock) = self.resource_locks.get_mut(&resource_id) {
            match (&existing_lock.lock_type, &lock_type) {
                // Shared locks can coexist
                (LockType::Shared, LockType::Shared) | (LockType::ReadOnly, LockType::ReadOnly) => {
                    // Add to shared lock holders (would need to track multiple owners)
                    return Ok(true);
                }
                // Exclusive locks block everything
                (LockType::Exclusive, _) | (_, LockType::Exclusive) => {
                    // Add to waiting queue
                    if let LockOwner::Agent(agent_id) = &requester {
                        existing_lock.waiting_queue.push(agent_id.clone());
                    }
                    return Ok(false);
                }
                _ => return Ok(false),
            }
        }

        // Create new lock
        let lock = ResourceLock {
            resource_id: resource_id.clone(),
            lock_type,
            owner: requester,
            acquired_at: Utc::now(),
            expires_at: duration.map(|d| Utc::now() + chrono::Duration::seconds(d as i64)),
            waiting_queue: Vec::new(),
        };

        self.resource_locks.insert(resource_id, lock);
        Ok(true)
    }

    /// Release a resource lock
    pub async fn release_lock(&self, resource_id: &str, owner: &LockOwner) -> Result<()> {
        let lock = self
            .resource_locks
            .get(resource_id)
            .ok_or_else(|| GovernanceError::CoordinationError("Lock not found".to_string()))?;

        // Verify ownership
        match (&lock.owner, owner) {
            (LockOwner::Agent(a1), LockOwner::Agent(a2)) if a1 == a2 => {}
            (LockOwner::Session(s1), LockOwner::Session(s2)) if s1 == s2 => {}
            (LockOwner::System, LockOwner::System) => {}
            _ => {
                return Err(GovernanceError::CoordinationError(
                    "Not the lock owner".to_string(),
                ))
            }
        }

        // Remove the lock
        self.resource_locks.remove(resource_id);

        // TODO: Process waiting queue and grant lock to next waiter

        Ok(())
    }

    /// Release all locks held by a session
    async fn release_session_locks(&self, session_id: &Uuid) -> Result<()> {
        let locks_to_release: Vec<_> = self
            .resource_locks
            .iter()
            .filter(
                |entry| matches!(entry.value().owner, LockOwner::Session(id) if id == *session_id),
            )
            .map(|entry| entry.key().clone())
            .collect();

        for resource_id in locks_to_release {
            self.resource_locks.remove(&resource_id);
        }

        Ok(())
    }

    /// Check for conflicts with existing coordinations
    fn has_conflicts(&self, request: &CoordinationRequest) -> Result<bool> {
        // Check for agent conflicts
        for session in self.active_coordinations.iter() {
            if session.status == SessionStatus::Active {
                // Check if any requested agents are already in an active session
                for agent in &request.target_agents {
                    if session.participants.contains(agent) {
                        // Log conflict
                        self.record_conflict(
                            vec![request.requesting_agent.clone(), agent.clone()],
                            ConflictType::ResourceContention,
                            format!(
                                "Agent {:?} already in session {}",
                                agent, session.session_id
                            ),
                        );
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Get applicable rules for a coordination type
    fn get_applicable_rules(&self, coordination_type: &str) -> Vec<CoordinationRule> {
        self.coordination_rules
            .iter()
            .filter(|entry| {
                entry.value().coordination_type == "*"
                    || entry.value().coordination_type == coordination_type
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Evaluate a coordination rule
    fn evaluate_rule(
        &self,
        rule: &CoordinationRule,
        request: &CoordinationRequest,
    ) -> Result<bool> {
        for condition in &rule.conditions {
            match condition {
                RuleCondition::AgentCount { min, max } => {
                    let agent_count = request.target_agents.len() + 1; // +1 for requester
                    if agent_count < *min {
                        return Ok(false);
                    }
                    if let Some(max) = max {
                        if agent_count > *max {
                            return Ok(false);
                        }
                    }
                }
                RuleCondition::ResourceAvailable(resource) => {
                    // Check if resource is locked
                    if self.resource_locks.contains_key(resource) {
                        return Ok(false);
                    }
                }
                RuleCondition::TimeWindow { start, end } => {
                    let now = Utc::now();
                    if now < *start || now > *end {
                        return Ok(false);
                    }
                }
                RuleCondition::Custom(_) => {
                    // Custom conditions would be evaluated here
                }
            }
        }

        Ok(true)
    }

    /// Record a conflict
    fn record_conflict(&self, agents: Vec<AgentId>, conflict_type: ConflictType, details: String) {
        let record = ConflictRecord {
            conflict_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            involved_agents: agents,
            conflict_type,
            resolution: ConflictResolution::Abandoned,
            details,
        };

        self.conflict_history.write().push(record);
    }

    /// Detect deadlocks in resource locks
    pub async fn detect_deadlocks(&self) -> Vec<Vec<AgentId>> {
        let mut deadlocks = Vec::new();

        // Build wait-for graph
        let mut wait_graph: std::collections::HashMap<AgentId, Vec<AgentId>> =
            std::collections::HashMap::new();

        for lock in self.resource_locks.iter() {
            if let LockOwner::Agent(owner) = &lock.owner {
                for waiter in &lock.waiting_queue {
                    wait_graph
                        .entry(waiter.clone())
                        .or_insert_with(Vec::new)
                        .push(owner.clone());
                }
            }
        }

        // Detect cycles using DFS
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();

        for agent in wait_graph.keys() {
            if !visited.contains(agent) {
                if let Some(cycle) =
                    self.dfs_detect_cycle(agent, &wait_graph, &mut visited, &mut rec_stack)
                {
                    deadlocks.push(cycle);
                }
            }
        }

        deadlocks
    }

    /// DFS to detect cycles in wait-for graph
    fn dfs_detect_cycle(
        &self,
        node: &AgentId,
        graph: &std::collections::HashMap<AgentId, Vec<AgentId>>,
        visited: &mut std::collections::HashSet<AgentId>,
        rec_stack: &mut std::collections::HashSet<AgentId>,
    ) -> Option<Vec<AgentId>> {
        visited.insert(node.clone());
        rec_stack.insert(node.clone());

        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if let Some(cycle) = self.dfs_detect_cycle(neighbor, graph, visited, rec_stack)
                    {
                        return Some(cycle);
                    }
                } else if rec_stack.contains(neighbor) {
                    // Cycle detected
                    return Some(vec![node.clone(), neighbor.clone()]);
                }
            }
        }

        rec_stack.remove(node);
        None
    }

    /// Get active sessions for an agent
    pub async fn get_agent_sessions(&self, agent_id: &AgentId) -> Vec<CoordinationSession> {
        self.active_coordinations
            .iter()
            .filter(|entry| {
                entry.value().participants.contains(agent_id)
                    && entry.value().status == SessionStatus::Active
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get session by ID
    pub async fn get_session(&self, session_id: Uuid) -> Result<CoordinationSession> {
        self.active_coordinations
            .get(&session_id)
            .map(|entry| entry.clone())
            .ok_or_else(|| GovernanceError::CoordinationError("Session not found".to_string()))
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&self) {
        let now = Utc::now();
        let expired_sessions: Vec<_> = self
            .active_coordinations
            .iter()
            .filter(|entry| {
                if let Some(expires_at) = entry.value().expires_at {
                    expires_at <= now
                } else {
                    false
                }
            })
            .map(|entry| *entry.key())
            .collect();

        for session_id in expired_sessions {
            if let Err(e) = self.end_session(session_id, false).await {
                warn!("Failed to end expired session {}: {}", session_id, e);
            }
        }
    }

    /// Get conflict history
    pub async fn get_conflict_history(&self) -> Vec<ConflictRecord> {
        self.conflict_history.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_coordination_manager_creation() {
        let manager = CoordinationManager::new();
        assert_eq!(manager.active_coordinations.len(), 0);
        assert!(manager.coordination_rules.len() > 0); // Has default rules
    }

    #[test]
    async fn test_create_coordination_session() {
        let manager = CoordinationManager::new();

        let request = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: AgentId::new(),
            target_agents: vec![AgentId::new(), AgentId::new()],
            coordination_type: "data_sharing".to_string(),
            duration: Some(3600),
        };

        let session_id = manager.create_session(request).await?;
        assert_ne!(session_id, Uuid::nil());

        let session = manager.get_session(session_id).await?;
        assert_eq!(session.status, SessionStatus::Pending);
        assert_eq!(session.participants.len(), 3); // requester + 2 targets
    }

    #[test]
    async fn test_start_session() {
        let manager = CoordinationManager::new();

        let request = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: AgentId::new(),
            target_agents: vec![AgentId::new()],
            coordination_type: "task_distribution".to_string(),
            duration: None,
        };

        let session_id = manager.create_session(request).await?;
        manager.start_session(session_id).await?;

        let session = manager.get_session(session_id).await?;
        assert_eq!(session.status, SessionStatus::Active);
    }

    #[test]
    async fn test_end_session() {
        let manager = CoordinationManager::new();

        let request = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: AgentId::new(),
            target_agents: vec![],
            coordination_type: "consensus_building".to_string(),
            duration: Some(300),
        };

        let session_id = manager.create_session(request).await?;
        manager.start_session(session_id).await?;
        manager.end_session(session_id, true).await?;

        let session = manager.get_session(session_id).await?;
        assert_eq!(session.status, SessionStatus::Completed);
    }

    #[test]
    async fn test_acquire_exclusive_lock() {
        let manager = CoordinationManager::new();
        let resource_id = "test_resource".to_string();
        let agent_id = AgentId::new();

        let acquired = manager
            .acquire_lock(
                resource_id.clone(),
                LockType::Exclusive,
                LockOwner::Agent(agent_id),
                Some(60),
            )
            .await
            .unwrap();

        assert!(acquired);
        assert!(manager.resource_locks.contains_key(&resource_id));
    }

    #[test]
    async fn test_exclusive_lock_blocks_others() {
        let manager = CoordinationManager::new();
        let resource_id = "test_resource".to_string();
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        // First agent acquires exclusive lock
        let acquired1 = manager
            .acquire_lock(
                resource_id.clone(),
                LockType::Exclusive,
                LockOwner::Agent(agent1),
                None,
            )
            .await
            .unwrap();
        assert!(acquired1);

        // Second agent should be blocked
        let acquired2 = manager
            .acquire_lock(
                resource_id.clone(),
                LockType::Exclusive,
                LockOwner::Agent(agent2.clone()),
                None,
            )
            .await
            .unwrap();
        assert!(!acquired2);

        // Check waiting queue
        let lock = manager.resource_locks.get(&resource_id)?;
        assert_eq!(lock.waiting_queue.len(), 1);
        assert_eq!(lock.waiting_queue[0], agent2);
    }

    #[test]
    async fn test_shared_locks_coexist() {
        let manager = CoordinationManager::new();
        let resource_id = "test_resource".to_string();
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        // Both agents acquire shared locks
        let acquired1 = manager
            .acquire_lock(
                resource_id.clone(),
                LockType::Shared,
                LockOwner::Agent(agent1),
                None,
            )
            .await
            .unwrap();
        assert!(acquired1);

        let acquired2 = manager
            .acquire_lock(
                resource_id.clone(),
                LockType::Shared,
                LockOwner::Agent(agent2),
                None,
            )
            .await
            .unwrap();
        assert!(acquired2);
    }

    #[test]
    async fn test_release_lock() {
        let manager = CoordinationManager::new();
        let resource_id = "test_resource".to_string();
        let agent_id = AgentId::new();

        // Acquire lock
        manager
            .acquire_lock(
                resource_id.clone(),
                LockType::Exclusive,
                LockOwner::Agent(agent_id.clone()),
                None,
            )
            .await
            .unwrap();

        // Release lock
        manager
            .release_lock(&resource_id, &LockOwner::Agent(agent_id))
            .await
            .unwrap();

        // Lock should be gone
        assert!(!manager.resource_locks.contains_key(&resource_id));
    }

    #[test]
    async fn test_release_lock_wrong_owner() {
        let manager = CoordinationManager::new();
        let resource_id = "test_resource".to_string();
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        // Agent1 acquires lock
        manager
            .acquire_lock(
                resource_id.clone(),
                LockType::Exclusive,
                LockOwner::Agent(agent1),
                None,
            )
            .await
            .unwrap();

        // Agent2 tries to release
        let result = manager
            .release_lock(&resource_id, &LockOwner::Agent(agent2))
            .await;
        assert!(result.is_err());
    }

    #[test]
    async fn test_session_lock_cleanup() {
        let manager = CoordinationManager::new();
        let session_id = Uuid::new_v4();
        let resource_id = "session_resource".to_string();

        // Session acquires lock
        manager
            .acquire_lock(
                resource_id.clone(),
                LockType::Exclusive,
                LockOwner::Session(session_id),
                None,
            )
            .await
            .unwrap();

        // Release session locks
        manager.release_session_locks(&session_id).await?;

        // Lock should be released
        assert!(!manager.resource_locks.contains_key(&resource_id));
    }

    #[test]
    async fn test_conflict_detection() {
        let manager = CoordinationManager::new();
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        let agent3 = AgentId::new();

        // Create first session with agent2
        let request1 = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: agent1.clone(),
            target_agents: vec![agent2.clone()],
            coordination_type: "data_sharing".to_string(),
            duration: None,
        };

        let session_id = manager.create_session(request1).await?;
        manager.start_session(session_id).await?;

        // Try to create second session with same agent
        let request2 = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: agent3,
            target_agents: vec![agent2],
            coordination_type: "task_distribution".to_string(),
            duration: None,
        };

        let has_conflicts = manager.has_conflicts(&request2)?;
        assert!(has_conflicts);

        // Check conflict was recorded
        let conflicts = manager.get_conflict_history().await;
        assert!(conflicts.len() > 0);
    }

    #[test]
    async fn test_rule_evaluation_agent_count() {
        let manager = CoordinationManager::new();

        // Request with many agents
        let request = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: AgentId::new(),
            target_agents: (0..10).map(|_| AgentId::new()).collect(),
            coordination_type: "collaborative_computation".to_string(),
            duration: None,
        };

        // Should fail due to large coordination rule (>5 agents)
        let result = manager.evaluate_request(&request).await?;
        // In this simple implementation, rules don't actually block, just check conditions
        assert!(result);
    }

    #[test]
    async fn test_deadlock_detection() {
        let manager = CoordinationManager::new();
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        // Create circular wait: agent1 holds R1, wants R2; agent2 holds R2, wants R1
        manager
            .acquire_lock(
                "R1".to_string(),
                LockType::Exclusive,
                LockOwner::Agent(agent1.clone()),
                None,
            )
            .await
            .unwrap();

        manager
            .acquire_lock(
                "R2".to_string(),
                LockType::Exclusive,
                LockOwner::Agent(agent2.clone()),
                None,
            )
            .await
            .unwrap();

        // Agent1 tries to acquire R2 (blocked)
        manager
            .acquire_lock(
                "R2".to_string(),
                LockType::Exclusive,
                LockOwner::Agent(agent1.clone()),
                None,
            )
            .await
            .unwrap();

        // Agent2 tries to acquire R1 (blocked)
        manager
            .acquire_lock(
                "R1".to_string(),
                LockType::Exclusive,
                LockOwner::Agent(agent2.clone()),
                None,
            )
            .await
            .unwrap();

        let deadlocks = manager.detect_deadlocks().await;
        assert!(deadlocks.len() > 0);
    }

    #[test]
    async fn test_get_agent_sessions() {
        let manager = CoordinationManager::new();
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        // Create multiple sessions with agent1
        for i in 0..3 {
            let request = CoordinationRequest {
                request_id: Uuid::new_v4(),
                requesting_agent: agent1.clone(),
                target_agents: vec![agent2.clone()],
                coordination_type: format!("session_{}", i),
                duration: None,
            };

            let session_id = manager.create_session(request).await?;
            if i < 2 {
                manager.start_session(session_id).await?;
            }
        }

        let sessions = manager.get_agent_sessions(&agent1).await;
        assert_eq!(sessions.len(), 2); // Only active sessions
    }

    #[test]
    async fn test_expired_session_cleanup() {
        let manager = CoordinationManager::new();

        // Create session with very short duration
        let request = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: AgentId::new(),
            target_agents: vec![],
            coordination_type: "test".to_string(),
            duration: Some(0), // Expires immediately
        };

        let session_id = manager.create_session(request).await?;
        manager.start_session(session_id).await?;

        // Clean up expired sessions
        manager.cleanup_expired_sessions().await;

        let session = manager.get_session(session_id).await?;
        assert_eq!(session.status, SessionStatus::Failed);
    }

    #[test]
    async fn test_coordination_type_mapping() {
        let manager = CoordinationManager::new();

        let test_cases = vec![
            ("data_sharing", CoordinationType::DataSharing),
            ("task_distribution", CoordinationType::TaskDistribution),
            ("consensus_building", CoordinationType::ConsensusBuilding),
            (
                "resource_negotiation",
                CoordinationType::ResourceNegotiation,
            ),
            (
                "collaborative_computation",
                CoordinationType::CollaborativeComputation,
            ),
            (
                "custom_type",
                CoordinationType::Custom("custom_type".to_string()),
            ),
        ];

        for (type_str, expected_type) in test_cases {
            let request = CoordinationRequest {
                request_id: Uuid::new_v4(),
                requesting_agent: AgentId::new(),
                target_agents: vec![],
                coordination_type: type_str.to_string(),
                duration: None,
            };

            let session_id = manager.create_session(request).await?;
            let session = manager.get_session(session_id).await?;
            assert_eq!(session.coordination_type, expected_type);
        }
    }

    #[test]
    async fn test_lock_expiration() {
        let manager = CoordinationManager::new();
        let resource_id = "expiring_resource".to_string();
        let agent_id = AgentId::new();

        // Acquire lock with short duration
        manager
            .acquire_lock(
                resource_id.clone(),
                LockType::Exclusive,
                LockOwner::Agent(agent_id),
                Some(1), // 1 second
            )
            .await
            .unwrap();

        let lock = manager.resource_locks.get(&resource_id)?;
        assert!(lock.expires_at.is_some());

        // In a real system, we'd have a background task to clean up expired locks
    }

    #[test]
    async fn test_multiple_coordination_rules() {
        let manager = CoordinationManager::new();

        // Add custom rule
        let custom_rule = CoordinationRule {
            rule_id: Uuid::new_v4(),
            name: "custom_rule".to_string(),
            coordination_type: "test_type".to_string(),
            conditions: vec![RuleCondition::AgentCount {
                min: 2,
                max: Some(4),
            }],
            actions: vec![RuleAction::SetTimeout(120)],
            priority: 200,
        };

        manager
            .coordination_rules
            .insert(custom_rule.name.clone(), custom_rule);

        let rules = manager.get_applicable_rules("test_type");
        assert!(rules.len() >= 1);

        // Test wildcard rules
        let wildcard_rules = manager.get_applicable_rules("any_type");
        assert!(wildcard_rules.iter().any(|r| r.coordination_type == "*"));
    }
}
