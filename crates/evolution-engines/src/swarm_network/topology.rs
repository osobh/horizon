//! Network topology manager for distributed SwarmAgentic systems

use super::load_balancer::LoadBalancer;
use super::network_graph::NetworkGraph;
use super::partition_manager::PartitionManager;
use super::types::{EdgeWeight, LoadBalanceMetrics, MigrationPlan, NodeCapacity};
use crate::error::EvolutionEngineResult;
use crate::swarm_distributed::{DistributedSwarmConfig, SwarmNode};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Network topology manager for distributed particle distribution
pub struct NetworkTopology {
    /// Configuration
    pub(crate) config: DistributedSwarmConfig,
    /// Partition manager for distributing particles
    pub(crate) partition_manager: Arc<RwLock<PartitionManager>>,
    /// Load balancer for dynamic load distribution
    pub(crate) load_balancer: Arc<RwLock<LoadBalancer>>,
    /// Network graph representation
    pub(crate) network_graph: Arc<RwLock<NetworkGraph>>,
}

impl NetworkTopology {
    /// Create new network topology manager
    pub async fn new(config: DistributedSwarmConfig) -> EvolutionEngineResult<Self> {
        Ok(Self {
            config: config.clone(),
            partition_manager: Arc::new(RwLock::new(PartitionManager::new(config.clone()).await?)),
            load_balancer: Arc::new(RwLock::new(LoadBalancer::new(config.clone()).await?)),
            network_graph: Arc::new(RwLock::new(NetworkGraph::new().await?)),
        })
    }

    /// Add node to the topology
    pub async fn add_node(&self, node: SwarmNode) -> EvolutionEngineResult<()> {
        // Add to network graph
        self.network_graph
            .write()
            .await
            .add_node(node.clone())
            .await?;

        // Update load balancer
        self.load_balancer
            .write()
            .await
            .add_node_capacity(
                node.node_id.clone(),
                NodeCapacity {
                    max_particles: self.config.load_balance_config.target_particles_per_node,
                    compute_capacity: node.capacity,
                    memory_capacity: 1024 * 1024 * 1024, // 1GB default
                    bandwidth_capacity: 1000.0,          // 1Gbps default
                    current_utilization: node.current_load,
                },
            )
            .await?;

        Ok(())
    }

    /// Remove node from the topology
    pub async fn remove_node(&self, node_id: &str) -> EvolutionEngineResult<MigrationPlan> {
        // Get particles that need to be migrated
        let particles_to_migrate = self
            .partition_manager
            .read()
            .await
            .get_particles_on_node(node_id);

        // Create migration plan
        let migration_plan = self
            .load_balancer
            .read()
            .await
            .create_evacuation_plan(node_id, &particles_to_migrate)
            .await?;

        // Remove from network graph
        self.network_graph
            .write()
            .await
            .remove_node(node_id)
            .await?;

        // Remove from load balancer
        self.load_balancer
            .write()
            .await
            .remove_node(node_id)
            .await?;

        Ok(migration_plan)
    }

    /// Get current load balance metrics
    pub async fn get_load_balance_metrics(&self) -> EvolutionEngineResult<LoadBalanceMetrics> {
        self.load_balancer.read().await.get_metrics().await
    }

    /// Create migration plan for rebalancing
    pub async fn create_migration_plan(&self) -> EvolutionEngineResult<MigrationPlan> {
        self.load_balancer
            .read()
            .await
            .create_migration_plan()
            .await
    }

    /// Apply migration plan
    pub async fn apply_migration_plan(&self, plan: &MigrationPlan) -> EvolutionEngineResult<()> {
        let mut partition_manager = self.partition_manager.write().await;

        for migration in &plan.migrations {
            partition_manager
                .migrate_particle(
                    &migration.particle_id,
                    &migration.from_node,
                    &migration.to_node,
                )
                .await?;
        }

        Ok(())
    }

    /// Update network graph with connectivity information
    pub async fn update_network_connectivity(
        &self,
        from_node: &str,
        to_node: &str,
        edge_weight: EdgeWeight,
    ) -> EvolutionEngineResult<()> {
        self.network_graph
            .write()
            .await
            .update_edge(from_node, to_node, edge_weight)
            .await
    }

    /// Get network partitions for fault tolerance
    pub async fn get_network_partitions(
        &self,
    ) -> EvolutionEngineResult<Vec<super::types::NetworkPartition>> {
        self.network_graph.read().await.get_partitions().await
    }
}
