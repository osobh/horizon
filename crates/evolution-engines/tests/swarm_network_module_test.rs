//! Test to verify swarm_network module structure works correctly

use exorust_evolution_engines::swarm_distributed::DistributedSwarmConfig;
use exorust_evolution_engines::swarm_network::{
    EdgeWeight, LoadBalanceMetrics, LoadBalancer, Migration, MigrationPlan, NetworkGraph,
    NetworkPartition, NetworkTopology, NodeCapacity, ParticleMetadata, PartitionManager,
    PartitionStrategy,
};

#[test]
fn test_swarm_network_module_imports() {
    // Test that we can create config types
    let _strategy = PartitionStrategy::RoundRobin;
    let _capacity = NodeCapacity::default();
    let _edge_weight = EdgeWeight::default();

    // Test enums work
    let _ = PartitionStrategy::NetworkAware;
    let _ = PartitionStrategy::LoadAware;
}

#[test]
fn test_data_structures() {
    let metadata = ParticleMetadata {
        id: "test".to_string(),
        current_node: "node1".to_string(),
        compute_cost: 1.0,
        communication_frequency: std::collections::HashMap::new(),
        migration_count: 0,
        last_migration: 0,
    };
    assert_eq!(metadata.id, "test");

    let capacity = NodeCapacity::default();
    assert!(capacity.compute_capacity > 0.0);
}

#[tokio::test]
async fn test_network_topology_creation() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let topology = NetworkTopology::new(config).await;
    assert!(topology.is_ok());
}

#[tokio::test]
async fn test_partition_manager_creation() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let manager = PartitionManager::new(config).await;
    assert!(manager.is_ok());
}

#[tokio::test]
async fn test_load_balancer_creation() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let balancer = LoadBalancer::new(config).await;
    assert!(balancer.is_ok());
}
