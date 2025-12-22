//! Node management for distributed swarm

use super::{
    config::DistributedSwarmConfig,
    types::{DiscoveryService, SwarmNode},
};
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Manages node discovery, registration, and health monitoring
pub struct NodeManager {
    /// This node's information
    pub local_node: SwarmNode,
    /// Known peer nodes
    pub peer_nodes: HashMap<String, SwarmNode>,
    /// Node discovery service
    pub discovery_service: Arc<RwLock<DiscoveryService>>,
}

impl NodeManager {
    /// Create new node manager
    pub async fn new(config: DistributedSwarmConfig) -> EvolutionEngineResult<Self> {
        let local_node = SwarmNode::new(
            config.node_id.clone(),
            format!(
                "{}:{}",
                config.network_config.listen_addr, config.network_config.port
            ),
        );

        let discovery_service = Arc::new(RwLock::new(DiscoveryService::new(config).await?));

        Ok(Self {
            local_node,
            peer_nodes: HashMap::new(),
            discovery_service,
        })
    }

    /// Join the cluster
    pub async fn join_cluster(&mut self) -> EvolutionEngineResult<()> {
        // Update local node status
        self.local_node.status = super::types::NodeStatus::Active;

        // Discover peers from bootstrap nodes
        let discovery = self.discovery_service.read().await;
        let discovered_nodes = discovery.discover_nodes().await?;

        for node in discovered_nodes {
            self.peer_nodes.insert(node.node_id.clone(), node);
        }

        Ok(())
    }

    /// Leave the cluster gracefully
    pub async fn leave_cluster(&mut self) -> EvolutionEngineResult<()> {
        self.local_node.status = super::types::NodeStatus::Stopping;
        // In a real implementation, would notify peers
        Ok(())
    }

    /// Add a peer node
    pub fn add_peer(&mut self, node: SwarmNode) {
        self.peer_nodes.insert(node.node_id.clone(), node);
    }

    /// Remove a peer node
    pub fn remove_peer(&mut self, node_id: &str) {
        self.peer_nodes.remove(node_id);
    }

    /// Get peer nodes
    pub fn get_peer_nodes(&self) -> &HashMap<String, SwarmNode> {
        &self.peer_nodes
    }

    /// Get peer count
    pub fn get_peer_count(&self) -> usize {
        self.peer_nodes.len()
    }

    /// Update local node statistics
    pub fn update_local_stats(&mut self, particle_count: usize, load: f64) {
        self.local_node.particle_count = particle_count;
        self.local_node.update_load(load);
    }

    /// Check node health
    pub fn check_node_health(&self, node_id: &str, timeout_ms: u64) -> bool {
        if let Some(node) = self.peer_nodes.get(node_id) {
            node.is_healthy(timeout_ms)
        } else {
            false
        }
    }

    /// Get healthy nodes
    pub fn get_healthy_nodes(&self, timeout_ms: u64) -> Vec<&SwarmNode> {
        self.peer_nodes
            .values()
            .filter(|node| node.is_healthy(timeout_ms))
            .collect()
    }

    /// Handle node failure
    pub fn handle_node_failure(&mut self, node_id: &str) {
        if let Some(node) = self.peer_nodes.get_mut(node_id) {
            node.status = super::types::NodeStatus::Failed;
        }
    }

    /// Get load distribution statistics
    pub fn get_load_distribution(&self) -> HashMap<String, f64> {
        let mut distribution = HashMap::new();
        distribution.insert(
            self.local_node.node_id.clone(),
            self.local_node.current_load,
        );

        for (id, node) in &self.peer_nodes {
            distribution.insert(id.clone(), node.current_load);
        }

        distribution
    }

    /// Find least loaded node
    pub fn find_least_loaded_node(&self) -> Option<&str> {
        let mut min_load = self.local_node.current_load;
        let mut min_node = &self.local_node.node_id;

        for (id, node) in &self.peer_nodes {
            if node.status == super::types::NodeStatus::Active && node.current_load < min_load {
                min_load = node.current_load;
                min_node = id;
            }
        }

        Some(min_node)
    }

    /// Find nodes for rebalancing
    pub fn find_rebalance_candidates(&self, threshold: f64) -> (Vec<&str>, Vec<&str>) {
        let avg_load = self.calculate_average_load();
        let mut overloaded = Vec::new();
        let mut underloaded = Vec::new();

        // Check local node
        if self.local_node.current_load > avg_load + threshold {
            overloaded.push(self.local_node.node_id.as_str());
        } else if self.local_node.current_load < avg_load - threshold {
            underloaded.push(self.local_node.node_id.as_str());
        }

        // Check peer nodes
        for (id, node) in &self.peer_nodes {
            if node.status != super::types::NodeStatus::Active {
                continue;
            }

            if node.current_load > avg_load + threshold {
                overloaded.push(id.as_str());
            } else if node.current_load < avg_load - threshold {
                underloaded.push(id.as_str());
            }
        }

        (overloaded, underloaded)
    }

    /// Calculate average load across all nodes
    fn calculate_average_load(&self) -> f64 {
        let mut total_load = self.local_node.current_load;
        let mut node_count = 1;

        for node in self.peer_nodes.values() {
            if node.status == super::types::NodeStatus::Active {
                total_load += node.current_load;
                node_count += 1;
            }
        }

        total_load / node_count as f64
    }
}
