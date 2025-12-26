//! Subnet Mesh Orchestrator
//!
//! Orchestrates automatic subnet assignment during node registration
//! in cluster-mesh, integrating policy evaluation and event publishing.

use super::node_mapper::{ClusterNodeInfo, NodeClassMapper};
use crate::events::SubnetEventPublisher;
use crate::models::{Subnet, SubnetAssignment, SubnetStatus};
use crate::policy_engine::{NodeAttributes, PolicyEngine, PolicyEvaluator};
use crate::service::SubnetManager;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

/// Error types for mesh orchestration
#[derive(Debug, Error)]
pub enum OrchestrationError {
    #[error("No matching policy found for node")]
    NoPolicyMatch,

    #[error("Target subnet is not available: {0}")]
    SubnetNotAvailable(String),

    #[error("IP allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Node already assigned to subnet {0}")]
    AlreadyAssigned(Uuid),

    #[error("Node not found: {0}")]
    NodeNotFound(Uuid),

    #[error("Assignment not found for node {0}")]
    AssignmentNotFound(Uuid),

    #[error("Event publishing failed: {0}")]
    EventPublishFailed(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Information about a node's subnet assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSubnetInfo {
    pub subnet_id: Uuid,
    pub subnet_name: String,
    pub assigned_ip: Ipv4Addr,
    pub wg_interface: String,
    pub assigned_at: DateTime<Utc>,
    pub policy_id: Option<Uuid>,
    pub assignment_method: String,
}

impl From<(&SubnetAssignment, &Subnet)> for NodeSubnetInfo {
    fn from((assignment, subnet): (&SubnetAssignment, &Subnet)) -> Self {
        Self {
            subnet_id: subnet.id,
            subnet_name: subnet.name.clone(),
            assigned_ip: assignment.assigned_ip,
            wg_interface: subnet.wg_interface.clone(),
            assigned_at: assignment.assigned_at,
            policy_id: assignment.policy_id,
            assignment_method: assignment.assignment_method.clone(),
        }
    }
}

impl NodeSubnetInfo {
    /// Convert to a tuple of core values for cross-crate compatibility.
    ///
    /// This can be used to create cluster-mesh's NodeSubnetAssignment:
    /// ```rust,ignore
    /// let info = orchestrator.on_node_joined(&node).await?;
    /// let (subnet_id, subnet_name, assigned_ip, wg_interface, assigned_at) = info.to_assignment_tuple();
    /// node.subnet_info = Some(NodeSubnetAssignment {
    ///     subnet_id, subnet_name, assigned_ip, wg_interface, assigned_at
    /// });
    /// ```
    pub fn to_assignment_tuple(&self) -> (Uuid, String, Ipv4Addr, String, DateTime<Utc>) {
        (
            self.subnet_id,
            self.subnet_name.clone(),
            self.assigned_ip,
            self.wg_interface.clone(),
            self.assigned_at,
        )
    }
}

/// Subnet affinity for job scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubnetAffinity {
    /// Node must be in this specific subnet
    Required(Uuid),
    /// Prefer nodes in these subnets (weighted)
    Preferred(Vec<(Uuid, f64)>),
    /// Must be in the same subnet as another node
    SameAs(Uuid),
    /// Must not be in these subnets
    Excluded(Vec<Uuid>),
    /// No subnet preference
    None,
}

impl Default for SubnetAffinity {
    fn default() -> Self {
        SubnetAffinity::None
    }
}

impl SubnetAffinity {
    /// Check if a node in a given subnet satisfies this affinity
    pub fn is_satisfied(&self, node_subnet_id: Uuid, all_assignments: &HashMap<Uuid, Uuid>) -> bool {
        match self {
            SubnetAffinity::Required(subnet_id) => node_subnet_id == *subnet_id,
            SubnetAffinity::Preferred(_) => true, // Always satisfied, just affects scoring
            SubnetAffinity::SameAs(other_node_id) => {
                all_assignments
                    .get(other_node_id)
                    .map(|s| *s == node_subnet_id)
                    .unwrap_or(true) // If other node not assigned yet, consider satisfied
            }
            SubnetAffinity::Excluded(subnets) => !subnets.contains(&node_subnet_id),
            SubnetAffinity::None => true,
        }
    }

    /// Calculate an affinity score for a node in a given subnet
    pub fn score(&self, node_subnet_id: Uuid) -> f64 {
        match self {
            SubnetAffinity::Required(subnet_id) => {
                if node_subnet_id == *subnet_id {
                    1.0
                } else {
                    0.0
                }
            }
            SubnetAffinity::Preferred(prefs) => {
                prefs
                    .iter()
                    .find(|(id, _)| *id == node_subnet_id)
                    .map(|(_, weight)| *weight)
                    .unwrap_or(0.5)
            }
            SubnetAffinity::SameAs(_) => 1.0, // Score doesn't apply here
            SubnetAffinity::Excluded(subnets) => {
                if subnets.contains(&node_subnet_id) {
                    0.0
                } else {
                    1.0
                }
            }
            SubnetAffinity::None => 1.0,
        }
    }
}

/// Configuration for the mesh orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Whether to automatically assign nodes on join
    pub auto_assign: bool,
    /// Whether to publish events
    pub publish_events: bool,
    /// Default assignment method label
    pub default_assignment_method: String,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            auto_assign: true,
            publish_events: true,
            default_assignment_method: "policy-auto".to_string(),
        }
    }
}

/// Main orchestrator for subnet-mesh integration
pub struct SubnetMeshOrchestrator {
    manager: Arc<SubnetManager>,
    policy_engine: Arc<PolicyEngine>,
    event_publisher: Arc<SubnetEventPublisher>,
    node_mapper: NodeClassMapper,
    config: OrchestratorConfig,
    /// Cache of node -> subnet assignments
    assignment_cache: RwLock<HashMap<Uuid, Uuid>>,
}

impl SubnetMeshOrchestrator {
    /// Create a new orchestrator
    pub fn new(
        manager: Arc<SubnetManager>,
        policy_engine: Arc<PolicyEngine>,
        event_publisher: Arc<SubnetEventPublisher>,
    ) -> Self {
        Self {
            manager,
            policy_engine,
            event_publisher,
            node_mapper: NodeClassMapper::new(),
            config: OrchestratorConfig::default(),
            assignment_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Create an orchestrator with hpc-channels integration.
    ///
    /// This enables platform-wide event broadcast for subnet assignments,
    /// migrations, and topology changes via hpc-channels.
    ///
    /// Requires the `hpc-channels` feature to be enabled.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let orchestrator = SubnetMeshOrchestrator::with_hpc_channels(
    ///     manager,
    ///     policy_engine,
    /// );
    /// // Node assignments now broadcast via hpc-channels
    /// orchestrator.on_node_joined(&node).await?;
    /// ```
    #[cfg(feature = "hpc-channels")]
    pub fn with_hpc_channels(
        manager: Arc<SubnetManager>,
        policy_engine: Arc<PolicyEngine>,
    ) -> Self {
        let event_publisher = Arc::new(SubnetEventPublisher::with_hpc_channels());
        Self::new(manager, policy_engine, event_publisher)
    }

    /// Create with custom configuration
    pub fn with_config(mut self, config: OrchestratorConfig) -> Self {
        self.config = config;
        self
    }

    /// Set a custom node mapper
    pub fn with_mapper(mut self, mapper: NodeClassMapper) -> Self {
        self.node_mapper = mapper;
        self
    }

    /// Handle a node joining the mesh
    #[instrument(skip(self, node), fields(node_id = %node.id, hostname = %node.hostname))]
    pub async fn on_node_joined(
        &self,
        node: &ClusterNodeInfo,
    ) -> Result<NodeSubnetInfo, OrchestrationError> {
        info!("Node joined mesh, processing subnet assignment");

        // Check if already assigned
        if let Some(subnet_id) = self.assignment_cache.read().get(&node.id) {
            warn!(subnet_id = %subnet_id, "Node already assigned to subnet");
            return Err(OrchestrationError::AlreadyAssigned(*subnet_id));
        }

        // Map node to attributes
        let attrs = self.node_mapper.map_to_attributes(node);

        // Evaluate policy
        let subnet_id = match self.policy_engine.evaluate(&attrs) {
            Ok(subnet_id) => subnet_id,
            Err(_) => {
                warn!("No matching policy for node");
                return Err(OrchestrationError::NoPolicyMatch);
            }
        };

        // Get the WireGuard public key
        let wg_public_key = node
            .wg_public_key
            .clone()
            .ok_or_else(|| OrchestrationError::Internal("No WireGuard public key".to_string()))?;

        // Assign node to subnet
        let info = self
            .assign_node(node.id, subnet_id, &wg_public_key, &attrs)
            .await?;

        info!(
            subnet = %info.subnet_name,
            ip = %info.assigned_ip,
            "Node assigned to subnet"
        );

        Ok(info)
    }

    /// Handle a node leaving the mesh
    #[instrument(skip(self), fields(node_id = %node_id))]
    pub async fn on_node_left(&self, node_id: Uuid) -> Result<(), OrchestrationError> {
        info!("Node left mesh, releasing subnet assignment");

        // Get current assignment
        let subnet_id = self
            .assignment_cache
            .read()
            .get(&node_id)
            .copied()
            .ok_or(OrchestrationError::AssignmentNotFound(node_id))?;

        // Release the assignment
        self.unassign_node(node_id, subnet_id, "node_left").await?;

        Ok(())
    }

    /// Manually assign a node to a specific subnet
    #[instrument(skip(self), fields(node_id = %node_id, subnet_id = %subnet_id))]
    pub async fn assign_node(
        &self,
        node_id: Uuid,
        subnet_id: Uuid,
        wg_public_key: &str,
        _attrs: &NodeAttributes,
    ) -> Result<NodeSubnetInfo, OrchestrationError> {
        use crate::service::AssignNodeRequest;

        // Get subnet first for status check
        let subnet = self
            .manager
            .get_subnet(subnet_id)
            .ok_or_else(|| OrchestrationError::SubnetNotAvailable(subnet_id.to_string()))?;

        // Check subnet status
        if subnet.status != SubnetStatus::Active {
            return Err(OrchestrationError::SubnetNotAvailable(format!(
                "Subnet {} is {:?}",
                subnet.name, subnet.status
            )));
        }

        // Use SubnetManager to handle assignment (it handles IP allocation internally)
        let request = AssignNodeRequest {
            node_id,
            wg_public_key: wg_public_key.to_string(),
            subnet_id: Some(subnet_id),
            attributes: None,
            method: self.config.default_assignment_method.clone(),
        };

        let assignment = self
            .manager
            .assign_node(request)
            .map_err(|e| OrchestrationError::AllocationFailed(e.to_string()))?;

        // Update cache
        self.assignment_cache.write().insert(node_id, subnet_id);

        // Publish event
        if self.config.publish_events {
            if let Err(e) = self.event_publisher.node_assigned(&assignment).await {
                error!(error = %e, "Failed to publish node assigned event");
            }
        }

        Ok(NodeSubnetInfo::from((&assignment, &subnet)))
    }

    /// Unassign a node from a subnet
    #[instrument(skip(self), fields(node_id = %node_id, subnet_id = %subnet_id))]
    pub async fn unassign_node(
        &self,
        node_id: Uuid,
        subnet_id: Uuid,
        reason: &str,
    ) -> Result<(), OrchestrationError> {
        // Get assigned IP from manager (would need to track this)
        // For now, we'll need to look it up or require it as a parameter
        // This is a simplification - in production, we'd track assignments

        // Release from cache
        self.assignment_cache.write().remove(&node_id);

        // Publish event
        if self.config.publish_events {
            // Note: We'd need to track the IP to publish properly
            // For now, use a placeholder
            let placeholder_ip = Ipv4Addr::new(0, 0, 0, 0);
            if let Err(e) = self
                .event_publisher
                .node_unassigned(node_id, subnet_id, placeholder_ip, reason)
                .await
            {
                error!(error = %e, "Failed to publish node unassigned event");
            }
        }

        info!("Node unassigned from subnet");
        Ok(())
    }

    /// Get subnet info for a node
    pub fn get_node_subnet(&self, node_id: Uuid) -> Option<Uuid> {
        self.assignment_cache.read().get(&node_id).copied()
    }

    /// Get all node-to-subnet assignments
    pub fn all_assignments(&self) -> HashMap<Uuid, Uuid> {
        self.assignment_cache.read().clone()
    }

    /// Calculate subnet affinity score for a node
    pub fn calculate_affinity_score(
        &self,
        node_id: Uuid,
        affinity: &SubnetAffinity,
    ) -> f64 {
        match self.assignment_cache.read().get(&node_id) {
            Some(subnet_id) => affinity.score(*subnet_id),
            None => 0.0, // Node not assigned
        }
    }

    /// Check if a node satisfies a subnet affinity requirement
    pub fn satisfies_affinity(&self, node_id: Uuid, affinity: &SubnetAffinity) -> bool {
        let assignments = self.assignment_cache.read();
        match assignments.get(&node_id) {
            Some(subnet_id) => affinity.is_satisfied(*subnet_id, &assignments),
            None => false, // Node not assigned
        }
    }

    /// Re-evaluate all nodes and migrate if policies changed
    #[instrument(skip(self))]
    pub async fn reevaluate_all(&self) -> Vec<(Uuid, Uuid, Uuid)> {
        let mut migrations = Vec::new();

        // This would iterate over all known nodes and check if their
        // current subnet assignment still matches the best policy
        // For now, return empty - full implementation would require
        // tracking node attributes

        debug!("Re-evaluation complete, {} migrations needed", migrations.len());
        migrations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::InMemoryTransport;
    use crate::models::{AssignmentPolicy, PolicyRule, SubnetPurpose};
    use crate::policy_engine::PolicyEngine;
    use ipnet::Ipv4Net;
    use std::str::FromStr;

    fn create_test_orchestrator() -> SubnetMeshOrchestrator {
        let manager = Arc::new(SubnetManager::new());
        let policy_engine = Arc::new(PolicyEngine::new());
        let transport = Arc::new(InMemoryTransport::new());
        let event_publisher = Arc::new(SubnetEventPublisher::with_transport(transport));

        SubnetMeshOrchestrator::new(manager, policy_engine, event_publisher)
    }

    #[test]
    fn test_subnet_affinity_required() {
        let subnet_id = Uuid::new_v4();
        let other_subnet = Uuid::new_v4();

        let affinity = SubnetAffinity::Required(subnet_id);

        let assignments = HashMap::new();
        assert!(affinity.is_satisfied(subnet_id, &assignments));
        assert!(!affinity.is_satisfied(other_subnet, &assignments));
    }

    #[test]
    fn test_subnet_affinity_excluded() {
        let excluded1 = Uuid::new_v4();
        let excluded2 = Uuid::new_v4();
        let allowed = Uuid::new_v4();

        let affinity = SubnetAffinity::Excluded(vec![excluded1, excluded2]);

        let assignments = HashMap::new();
        assert!(affinity.is_satisfied(allowed, &assignments));
        assert!(!affinity.is_satisfied(excluded1, &assignments));
        assert!(!affinity.is_satisfied(excluded2, &assignments));
    }

    #[test]
    fn test_subnet_affinity_same_as() {
        let node1 = Uuid::new_v4();
        let subnet = Uuid::new_v4();
        let other_subnet = Uuid::new_v4();

        let affinity = SubnetAffinity::SameAs(node1);

        let mut assignments = HashMap::new();
        assignments.insert(node1, subnet);

        // Same subnet as node1 - satisfied
        assert!(affinity.is_satisfied(subnet, &assignments));
        // Different subnet - not satisfied
        assert!(!affinity.is_satisfied(other_subnet, &assignments));
    }

    #[test]
    fn test_affinity_scoring() {
        let subnet1 = Uuid::new_v4();
        let subnet2 = Uuid::new_v4();
        let subnet3 = Uuid::new_v4();

        let affinity = SubnetAffinity::Preferred(vec![
            (subnet1, 1.0),
            (subnet2, 0.8),
        ]);

        assert_eq!(affinity.score(subnet1), 1.0);
        assert_eq!(affinity.score(subnet2), 0.8);
        assert_eq!(affinity.score(subnet3), 0.5); // Default for unspecified
    }

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let orchestrator = create_test_orchestrator();
        assert!(orchestrator.all_assignments().is_empty());
    }
}
