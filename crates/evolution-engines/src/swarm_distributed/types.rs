//! Core types for distributed swarm implementation

use super::config::{DistributedSwarmConfig, RecoveryStrategy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Represents a node in the distributed swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmNode {
    /// Unique node identifier
    pub node_id: String,
    /// Node network address
    pub address: String,
    /// Node status
    pub status: NodeStatus,
    /// Number of particles on this node
    pub particle_count: usize,
    /// Node computational capacity
    pub capacity: f64,
    /// Current load (0.0 - 1.0)
    pub current_load: f64,
    /// Last heartbeat timestamp
    pub last_heartbeat: u64,
}

/// Node status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    /// Node is active and processing
    Active,
    /// Node is starting up
    Starting,
    /// Node is shutting down gracefully
    Stopping,
    /// Node has failed
    Failed,
    /// Node is temporarily unavailable
    Unavailable,
}

/// Cluster status information
#[derive(Debug, Clone, Default)]
pub struct ClusterStatus {
    /// Total number of nodes in the cluster
    pub total_nodes: usize,
    /// Number of active nodes
    pub active_nodes: usize,
    /// Total particles across all nodes
    pub total_particles: usize,
    /// Current generation
    pub generation: u32,
    /// Global best fitness
    pub global_best_fitness: Option<f64>,
    /// Average load across nodes
    pub average_load: f64,
    /// Health status of the cluster
    pub health: ClusterHealth,
}

/// Cluster health status
#[derive(Debug, Clone, Default)]
pub enum ClusterHealth {
    #[default]
    Healthy,
    Degraded,
    Critical,
}

/// Node discovery service
pub struct DiscoveryService {
    /// Bootstrap peers
    pub bootstrap_peers: Vec<String>,
    /// Discovery protocol handler
    pub protocol: Arc<RwLock<DiscoveryProtocol>>,
}

/// Discovery protocol implementation
pub struct DiscoveryProtocol {
    /// Known nodes
    pub known_nodes: HashMap<String, SwarmNode>,
    /// Pending discovery requests
    pub pending_requests: Vec<DiscoveryRequest>,
}

/// Discovery request
pub struct DiscoveryRequest {
    /// Request ID
    pub id: String,
    /// Target node
    pub target: String,
    /// Request timestamp
    pub timestamp: u64,
}

impl SwarmNode {
    /// Create new swarm node
    pub fn new(node_id: String, address: String) -> Self {
        Self {
            node_id,
            address,
            status: NodeStatus::Starting,
            particle_count: 0,
            capacity: 1.0,
            current_load: 0.0,
            last_heartbeat: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Update node load
    pub fn update_load(&mut self, load: f64) {
        self.current_load = load.clamp(0.0, 1.0);
        self.last_heartbeat = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
    }

    /// Check if node is healthy based on heartbeat
    #[inline]
    pub fn is_healthy(&self, timeout_ms: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        now - self.last_heartbeat < timeout_ms && self.status == NodeStatus::Active
    }
}

impl DiscoveryService {
    /// Create new discovery service
    pub async fn new(config: DistributedSwarmConfig) -> crate::error::EvolutionEngineResult<Self> {
        Ok(Self {
            bootstrap_peers: config.network_config.bootstrap_peers,
            protocol: Arc::new(RwLock::new(DiscoveryProtocol {
                known_nodes: HashMap::new(),
                pending_requests: Vec::new(),
            })),
        })
    }

    /// Discover nodes in the network
    pub async fn discover_nodes(&self) -> crate::error::EvolutionEngineResult<Vec<SwarmNode>> {
        // Stub implementation for now
        Ok(Vec::new())
    }

    /// Register a discovered node
    pub async fn register_node(&self, node: SwarmNode) -> crate::error::EvolutionEngineResult<()> {
        let mut protocol = self.protocol.write().await;
        protocol.known_nodes.insert(node.node_id.clone(), node);
        Ok(())
    }
}

impl DiscoveryProtocol {
    /// Add discovery request
    pub fn add_request(&mut self, request: DiscoveryRequest) {
        self.pending_requests.push(request);
    }

    /// Get known nodes
    #[inline]
    pub fn get_known_nodes(&self) -> Vec<SwarmNode> {
        self.known_nodes.values().cloned().collect()
    }
}

/// Individual migration operation  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Migration {
    /// Particle to migrate
    pub particle_id: String,
    /// Source node
    pub from_node: String,
    /// Destination node
    pub to_node: String,
    /// Migration priority (higher = more urgent)
    pub priority: f64,
    /// Expected benefit of this migration
    pub expected_benefit: f64,
}

/// Migration plan containing multiple migrations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// List of migrations to execute
    pub migrations: Vec<Migration>,
    /// Expected overall improvement
    pub expected_improvement: f64,
    /// Total migration cost
    pub migration_cost: f64,
}
