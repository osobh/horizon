//! Distributed coordination for fault tolerance across multiple nodes

use crate::checkpoint::CheckpointId;
use crate::error::{FaultToleranceError, FtResult, HealthStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Node identifier in the distributed system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(Uuid);

impl NodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

/// Node information in the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: NodeId,
    pub address: SocketAddr,
    pub role: NodeRole,
    pub status: NodeStatus,
    pub last_heartbeat: u64,
    pub capabilities: Vec<String>,
}

/// Node role in the cluster
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeRole {
    Leader,
    Follower,
    Observer,
}

/// Node operational status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeStatus {
    Active,
    Degraded,
    Failed,
    Recovering,
}

/// Coordination message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationMessage {
    Heartbeat {
        node_id: NodeId,
        timestamp: u64,
        status: NodeStatus,
    },
    CheckpointRequest {
        initiator: NodeId,
        checkpoint_id: CheckpointId,
        priority: Priority,
    },
    CheckpointComplete {
        node_id: NodeId,
        checkpoint_id: CheckpointId,
        success: bool,
    },
    RecoveryRequest {
        initiator: NodeId,
        checkpoint_id: CheckpointId,
        target_nodes: Vec<NodeId>,
    },
    RecoveryProgress {
        node_id: NodeId,
        checkpoint_id: CheckpointId,
        progress_percent: u8,
    },
    LeaderElection {
        candidate: NodeId,
        term: u64,
    },
    HealthCheck {
        requester: NodeId,
    },
    HealthResponse {
        node_id: NodeId,
        health: HealthStatus,
        metrics: HashMap<String, f64>,
    },
}

/// Message priority levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Normal,
    Low,
}

/// Distributed coordinator manages fault tolerance across nodes
pub struct DistributedCoordinator {
    node_id: NodeId,
    node_info: NodeInfo,
    known_nodes: HashMap<NodeId, NodeInfo>,
    is_leader: bool,
    election_term: u64,
    heartbeat_interval: Duration,
}

impl DistributedCoordinator {
    /// Create new distributed coordinator
    pub fn new() -> anyhow::Result<Self> {
        let node_id = NodeId::new();
        let node_info = NodeInfo {
            id: node_id.clone(),
            address: "127.0.0.1:8080".parse()?, // Mock address
            role: NodeRole::Follower,
            status: NodeStatus::Active,
            last_heartbeat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                ?
                .as_secs(),
            capabilities: vec!["checkpoint".to_string(), "recovery".to_string()],
        };

        Ok(Self {
            node_id,
            node_info,
            known_nodes: HashMap::new(),
            is_leader: false,
            election_term: 0,
            heartbeat_interval: Duration::from_secs(30),
        })
    }

    /// Create coordinator with specific address
    pub fn with_address(address: SocketAddr) -> anyhow::Result<Self> {
        let mut coordinator = Self::new()?;
        coordinator.node_info.address = address;
        Ok(coordinator)
    }

    /// Join cluster by connecting to existing nodes
    pub async fn join_cluster(&mut self, bootstrap_nodes: Vec<SocketAddr>) -> FtResult<()> {
        for addr in bootstrap_nodes {
            // Mock joining cluster - in real implementation would:
            // 1. Connect to bootstrap node
            // 2. Exchange node information
            // 3. Sync cluster membership
            // 4. Start heartbeating

            let bootstrap_node = NodeInfo {
                id: NodeId::new(),
                address: addr,
                role: NodeRole::Leader, // Assume bootstrap node is leader
                status: NodeStatus::Active,
                last_heartbeat: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                capabilities: vec!["checkpoint".to_string(), "recovery".to_string()],
            };

            self.known_nodes
                .insert(bootstrap_node.id.clone(), bootstrap_node);
        }

        Ok(())
    }

    /// Coordinate distributed checkpoint creation
    pub async fn coordinate_checkpoint(&self, checkpoint_id: CheckpointId) -> FtResult<()> {
        if !self.is_leader {
            return Err(FaultToleranceError::CoordinationError(
                "Only leader can coordinate checkpoints".to_string(),
            ));
        }

        // Send checkpoint request to all nodes
        for node in self.known_nodes.values() {
            if node.status == NodeStatus::Active {
                let message = CoordinationMessage::CheckpointRequest {
                    initiator: self.node_id.clone(),
                    checkpoint_id: checkpoint_id.clone(),
                    priority: Priority::Normal,
                };

                self.send_message(node, message).await?;
            }
        }

        // Wait for all nodes to complete checkpointing
        self.wait_for_checkpoint_completion(checkpoint_id).await?;

        Ok(())
    }

    /// Coordinate distributed recovery
    pub async fn coordinate_recovery(
        &self,
        checkpoint_id: CheckpointId,
        target_nodes: Vec<NodeId>,
    ) -> FtResult<()> {
        if !self.is_leader {
            return Err(FaultToleranceError::CoordinationError(
                "Only leader can coordinate recovery".to_string(),
            ));
        }

        let message = CoordinationMessage::RecoveryRequest {
            initiator: self.node_id.clone(),
            checkpoint_id,
            target_nodes: target_nodes.clone(),
        };

        for target_node_id in &target_nodes {
            if let Some(node_info) = self.known_nodes.get(target_node_id) {
                if node_info.status != NodeStatus::Failed {
                    self.send_message(node_info, message.clone()).await?;
                }
            }
        }

        Ok(())
    }

    /// Send heartbeat to maintain cluster membership
    pub async fn send_heartbeat(&mut self) -> FtResult<()> {
        let message = CoordinationMessage::Heartbeat {
            node_id: self.node_id.clone(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                ?
                .as_secs(),
            status: self.node_info.status.clone(),
        };

        // Broadcast heartbeat to all known nodes
        for node in self.known_nodes.values() {
            if node.id != self.node_id {
                self.send_message(node, message.clone()).await?;
            }
        }

        self.node_info.last_heartbeat = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(())
    }

    /// Initiate leader election
    pub async fn initiate_leader_election(&mut self) -> FtResult<bool> {
        self.election_term += 1;

        let message = CoordinationMessage::LeaderElection {
            candidate: self.node_id.clone(),
            term: self.election_term,
        };

        let mut votes = 1; // Vote for self
        let total_nodes = self.known_nodes.len() + 1; // +1 for self

        // Send election message to all nodes
        for node in self.known_nodes.values() {
            if node.status == NodeStatus::Active {
                self.send_message(node, message.clone()).await?;
                // Mock receiving vote - in real implementation would wait for responses
                votes += 1;
            }
        }

        // Check if won election (majority)
        if votes > total_nodes / 2 {
            self.is_leader = true;
            self.node_info.role = NodeRole::Leader;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get overall system health
    pub async fn system_health(&self) -> HealthStatus {
        let active_nodes = self
            .known_nodes
            .values()
            .filter(|node| node.status == NodeStatus::Active)
            .count();

        let total_nodes = self.known_nodes.len();

        if total_nodes == 0 {
            return HealthStatus::Healthy; // Single node system
        }

        let healthy_ratio = active_nodes as f64 / total_nodes as f64;

        match healthy_ratio {
            r if r >= 0.8 => HealthStatus::Healthy,
            r if r >= 0.5 => HealthStatus::Degraded,
            _ => HealthStatus::Failed,
        }
    }

    /// Get cluster status summary
    pub fn cluster_status(&self) -> HashMap<String, serde_json::Value> {
        let mut status = HashMap::new();

        status.insert(
            "total_nodes".to_string(),
            serde_json::Value::Number((self.known_nodes.len() + 1).into()),
        );

        let active_count = self
            .known_nodes
            .values()
            .filter(|node| node.status == NodeStatus::Active)
            .count()
            + if self.node_info.status == NodeStatus::Active {
                1
            } else {
                0
            };

        status.insert(
            "active_nodes".to_string(),
            serde_json::Value::Number(active_count.into()),
        );
        status.insert(
            "is_leader".to_string(),
            serde_json::Value::Bool(self.is_leader),
        );
        status.insert(
            "election_term".to_string(),
            serde_json::Value::Number(self.election_term.into()),
        );

        status
    }

    /// Send message to specific node (mock implementation)
    async fn send_message(
        &self,
        _target: &NodeInfo,
        _message: CoordinationMessage,
    ) -> FtResult<()> {
        // Mock message sending - in real implementation would:
        // 1. Serialize message
        // 2. Send over network (TCP/UDP)
        // 3. Handle network errors
        // 4. Implement retries and timeouts

        tokio::time::sleep(Duration::from_millis(10)).await; // Simulate network delay
        Ok(())
    }

    /// Wait for checkpoint completion across all nodes
    async fn wait_for_checkpoint_completion(&self, _checkpoint_id: CheckpointId) -> FtResult<()> {
        // Mock waiting - in real implementation would:
        // 1. Track checkpoint completion messages
        // 2. Handle timeouts
        // 3. Retry failed nodes
        // 4. Ensure consistency

        tokio::time::sleep(Duration::from_millis(100)).await; // Simulate waiting
        Ok(())
    }

    /// Handle incoming coordination message
    pub async fn handle_message(&mut self, message: CoordinationMessage) -> FtResult<()> {
        match message {
            CoordinationMessage::Heartbeat {
                node_id,
                timestamp,
                status,
            } => {
                // Update node information
                if let Some(node_info) = self.known_nodes.get_mut(&node_id) {
                    node_info.last_heartbeat = timestamp;
                    node_info.status = status;
                }
            }
            CoordinationMessage::CheckpointRequest { checkpoint_id, .. } => {
                // In real implementation, would create local checkpoint
                let _response = CoordinationMessage::CheckpointComplete {
                    node_id: self.node_id.clone(),
                    checkpoint_id,
                    success: true,
                };
                // Would send response back to initiator
            }
            CoordinationMessage::LeaderElection { candidate, term } => {
                // Handle leader election
                if term > self.election_term {
                    self.election_term = term;
                    self.is_leader = false;
                    self.node_info.role = NodeRole::Follower;
                    // Would vote for candidate
                }
            }
            CoordinationMessage::HealthCheck { requester: _ } => {
                // Respond with current health status
                let _response = CoordinationMessage::HealthResponse {
                    node_id: self.node_id.clone(),
                    health: HealthStatus::Healthy, // Mock health
                    metrics: HashMap::new(),
                };
                // Would send response
            }
            _ => {
                // Handle other message types
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let coordinator = DistributedCoordinator::new();
        assert!(coordinator.is_ok());
    }

    #[tokio::test]
    async fn test_coordinator_with_address() {
        let addr: SocketAddr = "192.168.1.100:9000".parse()?;
        let coordinator = DistributedCoordinator::with_address(addr);
        assert!(coordinator.is_ok());

        let coord = coordinator?;
        assert_eq!(coord.node_info.address, addr);
    }

    #[tokio::test]
    async fn test_join_cluster() {
        let mut coordinator = DistributedCoordinator::new()?;
        let bootstrap_nodes = vec!["127.0.0.1:8081".parse()?];

        let result = coordinator.join_cluster(bootstrap_nodes).await;
        assert!(result.is_ok());
        assert_eq!(coordinator.known_nodes.len(), 1);
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let mut coordinator = DistributedCoordinator::new()?;

        // Add a mock node
        let mock_node = NodeInfo {
            id: NodeId::new(),
            address: "127.0.0.1:8081".parse()?,
            role: NodeRole::Follower,
            status: NodeStatus::Active,
            last_heartbeat: 0,
            capabilities: vec![],
        };
        coordinator
            .known_nodes
            .insert(mock_node.id.clone(), mock_node);

        let result = coordinator.send_heartbeat().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_leader_election() {
        let mut coordinator = DistributedCoordinator::new()?;

        let won_election = coordinator.initiate_leader_election().await?;
        assert!(won_election); // Should win with no other nodes
        assert!(coordinator.is_leader);
        assert_eq!(coordinator.node_info.role, NodeRole::Leader);
    }

    #[tokio::test]
    async fn test_coordinate_checkpoint_as_follower() {
        let coordinator = DistributedCoordinator::new()?;
        let checkpoint_id = CheckpointId::new();

        // Should fail if not leader
        let result = coordinator.coordinate_checkpoint(checkpoint_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_coordinate_checkpoint_as_leader() {
        let mut coordinator = DistributedCoordinator::new()?;
        coordinator.is_leader = true; // Make it leader

        let checkpoint_id = CheckpointId::new();
        let result = coordinator.coordinate_checkpoint(checkpoint_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_system_health_single_node() {
        let coordinator = DistributedCoordinator::new()?;
        let health = coordinator.system_health().await;
        assert_eq!(health, HealthStatus::Healthy); // Single node should be healthy
    }

    #[tokio::test]
    async fn test_system_health_multiple_nodes() {
        let mut coordinator = DistributedCoordinator::new()?;

        // Add healthy node
        let healthy_node = NodeInfo {
            id: NodeId::new(),
            address: "127.0.0.1:8081".parse()?,
            role: NodeRole::Follower,
            status: NodeStatus::Active,
            last_heartbeat: 0,
            capabilities: vec![],
        };
        coordinator
            .known_nodes
            .insert(healthy_node.id.clone(), healthy_node);

        // Add failed node
        let failed_node = NodeInfo {
            id: NodeId::new(),
            address: "127.0.0.1:8082".parse()?,
            role: NodeRole::Follower,
            status: NodeStatus::Failed,
            last_heartbeat: 0,
            capabilities: vec![],
        };
        coordinator
            .known_nodes
            .insert(failed_node.id.clone(), failed_node);

        let health = coordinator.system_health().await;
        assert!(matches!(
            health,
            HealthStatus::Degraded | HealthStatus::Failed
        ));
    }

    #[tokio::test]
    async fn test_cluster_status() {
        let mut coordinator = DistributedCoordinator::new()?;
        coordinator.is_leader = true;

        let status = coordinator.cluster_status();
        assert!(status.contains_key("total_nodes"));
        assert!(status.contains_key("active_nodes"));
        assert!(status.contains_key("is_leader"));

        if let Some(serde_json::Value::Bool(is_leader)) = status.get("is_leader") {
            assert!(is_leader);
        } else {
            panic!("Expected is_leader to be boolean true");
        }
    }

    #[tokio::test]
    async fn test_handle_heartbeat_message() {
        let mut coordinator = DistributedCoordinator::new()?;
        let node_id = NodeId::new();

        // Add node first
        let node_info = NodeInfo {
            id: node_id.clone(),
            address: "127.0.0.1:8081".parse()?,
            role: NodeRole::Follower,
            status: NodeStatus::Active,
            last_heartbeat: 0,
            capabilities: vec![],
        };
        coordinator.known_nodes.insert(node_id.clone(), node_info);

        let heartbeat = CoordinationMessage::Heartbeat {
            node_id: node_id.clone(),
            timestamp: 123456789,
            status: NodeStatus::Degraded,
        };

        let result = coordinator.handle_message(heartbeat).await;
        assert!(result.is_ok());

        // Check that node status was updated
        let updated_node = coordinator.known_nodes.get(&node_id)?;
        assert_eq!(updated_node.status, NodeStatus::Degraded);
        assert_eq!(updated_node.last_heartbeat, 123456789);
    }

    #[test]
    fn test_node_id_creation() {
        let id1 = NodeId::new();
        let id2 = NodeId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_message_serialization() {
        let message = CoordinationMessage::Heartbeat {
            node_id: NodeId::new(),
            timestamp: 123456789,
            status: NodeStatus::Active,
        };

        let serialized = serde_json::to_string(&message);
        assert!(serialized.is_ok());

        let deserialized: CoordinationMessage = serde_json::from_str(&serialized?)?;
        assert!(matches!(
            deserialized,
            CoordinationMessage::Heartbeat { .. }
        ));
    }

    #[test]
    fn test_priority_levels() {
        let priorities = vec![
            Priority::Critical,
            Priority::High,
            Priority::Normal,
            Priority::Low,
        ];

        for priority in priorities {
            let serialized = serde_json::to_string(&priority)?;
            let _deserialized: Priority = serde_json::from_str(&serialized)?;
        }
    }

    #[tokio::test]
    async fn test_coordinate_recovery_as_follower() {
        let coordinator = DistributedCoordinator::new()?;
        let checkpoint_id = CheckpointId::new();
        let target_nodes = vec![NodeId::new()];

        // Should fail if not leader
        let result = coordinator
            .coordinate_recovery(checkpoint_id, target_nodes)
            .await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FaultToleranceError::CoordinationError(_)
        ));
    }

    #[tokio::test]
    async fn test_coordinate_recovery_as_leader() {
        let mut coordinator = DistributedCoordinator::new()?;
        coordinator.is_leader = true; // Make it leader

        // Add target nodes to known_nodes
        let target_node1 = NodeId::new();
        let target_node2 = NodeId::new();

        let node_info1 = NodeInfo {
            id: target_node1.clone(),
            address: "127.0.0.1:8081".parse()?,
            role: NodeRole::Follower,
            status: NodeStatus::Active,
            last_heartbeat: 0,
            capabilities: vec![],
        };
        let node_info2 = NodeInfo {
            id: target_node2.clone(),
            address: "127.0.0.1:8082".parse()?,
            role: NodeRole::Follower,
            status: NodeStatus::Failed, // This node should be skipped
            last_heartbeat: 0,
            capabilities: vec![],
        };

        coordinator
            .known_nodes
            .insert(target_node1.clone(), node_info1);
        coordinator
            .known_nodes
            .insert(target_node2.clone(), node_info2);

        let checkpoint_id = CheckpointId::new();
        let target_nodes = vec![target_node1, target_node2];

        let result = coordinator
            .coordinate_recovery(checkpoint_id, target_nodes)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_coordinate_recovery_unknown_nodes() {
        let mut coordinator = DistributedCoordinator::new()?;
        coordinator.is_leader = true;

        let checkpoint_id = CheckpointId::new();
        let unknown_node = NodeId::new(); // Not in known_nodes
        let target_nodes = vec![unknown_node];

        let result = coordinator
            .coordinate_recovery(checkpoint_id, target_nodes)
            .await;
        assert!(result.is_ok()); // Should succeed even with unknown nodes
    }

    #[tokio::test]
    async fn test_handle_checkpoint_request_message() {
        let mut coordinator = DistributedCoordinator::new()?;
        let checkpoint_id = CheckpointId::new();

        let checkpoint_request = CoordinationMessage::CheckpointRequest {
            initiator: NodeId::new(),
            checkpoint_id: checkpoint_id.clone(),
            priority: Priority::High,
        };

        let result = coordinator.handle_message(checkpoint_request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_checkpoint_complete_message() {
        let mut coordinator = DistributedCoordinator::new()?;

        let checkpoint_complete = CoordinationMessage::CheckpointComplete {
            node_id: NodeId::new(),
            checkpoint_id: CheckpointId::new(),
            success: true,
        };

        let result = coordinator.handle_message(checkpoint_complete).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_recovery_request_message() {
        let mut coordinator = DistributedCoordinator::new()?;

        let recovery_request = CoordinationMessage::RecoveryRequest {
            initiator: NodeId::new(),
            checkpoint_id: CheckpointId::new(),
            target_nodes: vec![NodeId::new()],
        };

        let result = coordinator.handle_message(recovery_request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_recovery_progress_message() {
        let mut coordinator = DistributedCoordinator::new()?;

        let recovery_progress = CoordinationMessage::RecoveryProgress {
            node_id: NodeId::new(),
            checkpoint_id: CheckpointId::new(),
            progress_percent: 75,
        };

        let result = coordinator.handle_message(recovery_progress).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_health_response_message() {
        let mut coordinator = DistributedCoordinator::new()?;

        let health_response = CoordinationMessage::HealthResponse {
            node_id: NodeId::new(),
            health: HealthStatus::Degraded,
            metrics: HashMap::new(),
        };

        let result = coordinator.handle_message(health_response).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_leader_election_with_newer_term() {
        let mut coordinator = DistributedCoordinator::new()?;
        coordinator.is_leader = true;
        coordinator.election_term = 5;

        let leader_election = CoordinationMessage::LeaderElection {
            candidate: NodeId::new(),
            term: 10, // Newer term
        };

        let result = coordinator.handle_message(leader_election).await;
        assert!(result.is_ok());

        // Should step down as leader and update term
        assert!(!coordinator.is_leader);
        assert_eq!(coordinator.election_term, 10);
        assert_eq!(coordinator.node_info.role, NodeRole::Follower);
    }

    #[tokio::test]
    async fn test_handle_leader_election_with_older_term() {
        let mut coordinator = DistributedCoordinator::new()?;
        coordinator.is_leader = true;
        coordinator.election_term = 10;

        let leader_election = CoordinationMessage::LeaderElection {
            candidate: NodeId::new(),
            term: 5, // Older term
        };

        let result = coordinator.handle_message(leader_election).await;
        assert!(result.is_ok());

        // Should remain leader with current term
        assert!(coordinator.is_leader);
        assert_eq!(coordinator.election_term, 10);
    }

    #[tokio::test]
    async fn test_heartbeat_excludes_self() {
        let mut coordinator = DistributedCoordinator::new()?;

        // Add a node with same ID as coordinator (should be excluded)
        let self_node = NodeInfo {
            id: coordinator.node_id.clone(),
            address: "127.0.0.1:8081".parse()?,
            role: NodeRole::Follower,
            status: NodeStatus::Active,
            last_heartbeat: 0,
            capabilities: vec![],
        };
        coordinator
            .known_nodes
            .insert(coordinator.node_id.clone(), self_node);

        // Add another node
        let other_node = NodeInfo {
            id: NodeId::new(),
            address: "127.0.0.1:8082".parse()?,
            role: NodeRole::Follower,
            status: NodeStatus::Active,
            last_heartbeat: 0,
            capabilities: vec![],
        };
        coordinator
            .known_nodes
            .insert(other_node.id.clone(), other_node);

        let result = coordinator.send_heartbeat().await;
        assert!(result.is_ok());

        // Verify heartbeat timestamp was updated
        assert!(coordinator.node_info.last_heartbeat > 0);
    }

    #[tokio::test]
    async fn test_leader_election_with_multiple_nodes() {
        let mut coordinator = DistributedCoordinator::new()?;

        // Add multiple active nodes to test majority calculation
        for i in 0..4 {
            let node = NodeInfo {
                id: NodeId::new(),
                address: format!("127.0.0.1:808{}", i).parse()?,
                role: NodeRole::Follower,
                status: NodeStatus::Active,
                last_heartbeat: 0,
                capabilities: vec![],
            };
            coordinator.known_nodes.insert(node.id.clone(), node);
        }

        let won_election = coordinator.initiate_leader_election().await?;
        assert!(won_election); // Should win majority (mock votes for all)
        assert!(coordinator.is_leader);
        assert_eq!(coordinator.node_info.role, NodeRole::Leader);
    }

    #[tokio::test]
    async fn test_system_health_degraded_scenario() {
        let mut coordinator = DistributedCoordinator::new()?;

        // Add nodes: 2 active, 2 failed (60% healthy - should be degraded)
        for i in 0..4 {
            let status = if i < 2 {
                NodeStatus::Active
            } else {
                NodeStatus::Failed
            };
            let node = NodeInfo {
                id: NodeId::new(),
                address: format!("127.0.0.1:808{}", i).parse()?,
                role: NodeRole::Follower,
                status,
                last_heartbeat: 0,
                capabilities: vec![],
            };
            coordinator.known_nodes.insert(node.id.clone(), node);
        }

        let health = coordinator.system_health().await;
        assert_eq!(health, HealthStatus::Degraded);
    }

    #[tokio::test]
    async fn test_system_health_failed_scenario() {
        let mut coordinator = DistributedCoordinator::new()?;

        // Add nodes: 1 active, 3 failed (25% healthy - should be failed)
        for i in 0..4 {
            let status = if i == 0 {
                NodeStatus::Active
            } else {
                NodeStatus::Failed
            };
            let node = NodeInfo {
                id: NodeId::new(),
                address: format!("127.0.0.1:808{}", i).parse()?,
                role: NodeRole::Follower,
                status,
                last_heartbeat: 0,
                capabilities: vec![],
            };
            coordinator.known_nodes.insert(node.id.clone(), node);
        }

        let health = coordinator.system_health().await;
        assert_eq!(health, HealthStatus::Failed);
    }

    #[tokio::test]
    async fn test_cluster_status_with_inactive_self() {
        let mut coordinator = DistributedCoordinator::new()?;
        coordinator.node_info.status = NodeStatus::Failed; // Self is failed

        // Add one active node
        let node = NodeInfo {
            id: NodeId::new(),
            address: "127.0.0.1:8081".parse()?,
            role: NodeRole::Follower,
            status: NodeStatus::Active,
            last_heartbeat: 0,
            capabilities: vec![],
        };
        coordinator.known_nodes.insert(node.id.clone(), node);

        let status = coordinator.cluster_status();

        if let Some(serde_json::Value::Number(active_count)) = status.get("active_nodes") {
            assert_eq!(active_count.as_u64()?, 1); // Only the other node is active
        } else {
            panic!("Expected active_nodes to be a number");
        }
    }

    #[test]
    fn test_node_id_from_uuid() {
        let uuid = Uuid::new_v4();
        let node_id1 = NodeId::from_uuid(uuid);
        let node_id2 = NodeId::from_uuid(uuid);
        assert_eq!(node_id1, node_id2);
    }

    #[test]
    fn test_coordination_message_variants() {
        let node_id = NodeId::new();
        let checkpoint_id = CheckpointId::new();

        let messages = vec![
            CoordinationMessage::Heartbeat {
                node_id: node_id.clone(),
                timestamp: 123456789,
                status: NodeStatus::Active,
            },
            CoordinationMessage::CheckpointRequest {
                initiator: node_id.clone(),
                checkpoint_id: checkpoint_id.clone(),
                priority: Priority::Critical,
            },
            CoordinationMessage::CheckpointComplete {
                node_id: node_id.clone(),
                checkpoint_id: checkpoint_id.clone(),
                success: true,
            },
            CoordinationMessage::RecoveryRequest {
                initiator: node_id.clone(),
                checkpoint_id: checkpoint_id.clone(),
                target_nodes: vec![node_id.clone()],
            },
            CoordinationMessage::RecoveryProgress {
                node_id: node_id.clone(),
                checkpoint_id: checkpoint_id.clone(),
                progress_percent: 50,
            },
            CoordinationMessage::LeaderElection {
                candidate: node_id.clone(),
                term: 5,
            },
            CoordinationMessage::HealthCheck {
                requester: node_id.clone(),
            },
            CoordinationMessage::HealthResponse {
                node_id: node_id.clone(),
                health: HealthStatus::Healthy,
                metrics: HashMap::new(),
            },
        ];

        for message in messages {
            let serialized = serde_json::to_string(&message)?;
            let _deserialized: CoordinationMessage = serde_json::from_str(&serialized)?;
        }
    }
}
