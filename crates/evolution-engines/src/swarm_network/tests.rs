//! Tests for swarm network module

use super::*;
use crate::swarm_distributed::{DistributedSwarmConfig, NodeStatus};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[tokio::test]
async fn test_network_topology_creation() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let topology = NetworkTopology::new(config).await;
    assert!(topology.is_ok());
}

#[tokio::test]
async fn test_add_remove_node() -> TestResult {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let topology = NetworkTopology::new(config).await?;

    let node =
        crate::swarm_distributed::SwarmNode::new("node1".to_string(), "127.0.0.1:8001".to_string());
    assert!(topology.add_node(node).await.is_ok());

    let migration_plan = topology.remove_node("node1").await?;
    assert_eq!(migration_plan.migrations.len(), 0); // No particles to migrate
    Ok(())
}

#[tokio::test]
async fn test_partition_manager_creation() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let manager = PartitionManager::new(config).await;
    assert!(manager.is_ok());
}

#[tokio::test]
async fn test_partition_manager_assignment() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let mut manager = PartitionManager::new(config).await.unwrap();

    assert!(manager
        .assign_particle("particle1".to_string(), "node1".to_string())
        .await
        .is_ok());

    let particles = manager.get_particles_on_node("node1");
    assert_eq!(particles, vec!["particle1"]);
}

#[tokio::test]
async fn test_particle_migration() -> TestResult {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let mut manager = PartitionManager::new(config).await?;

    // Assign particle to node1
    manager
        .assign_particle("particle1".to_string(), "node1".to_string())
        .await?;

    // Migrate to node2
    assert!(manager
        .migrate_particle("particle1", "node1", "node2")
        .await
        .is_ok());

    // Verify migration
    assert_eq!(manager.get_particles_on_node("node1").len(), 0);
    assert_eq!(manager.get_particles_on_node("node2"), vec!["particle1"]);
    Ok(())
}

#[tokio::test]
async fn test_load_balancer_creation() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let balancer = LoadBalancer::new(config).await;
    assert!(balancer.is_ok());
}

#[tokio::test]
async fn test_load_balancer_node_management() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let mut balancer = LoadBalancer::new(config).await.unwrap();

    let capacity = NodeCapacity::default();
    assert!(balancer
        .add_node_capacity("node1".to_string(), capacity)
        .await
        .is_ok());
    assert!(balancer.remove_node("node1").await.is_ok());
}

#[tokio::test]
async fn test_load_balance_metrics() -> TestResult {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let mut balancer = LoadBalancer::new(config).await?;

    // Add some nodes
    let capacity = NodeCapacity::default();
    balancer
        .add_node_capacity("node1".to_string(), capacity.clone())
        .await?;
    balancer
        .add_node_capacity("node2".to_string(), capacity)
        .await?;

    let metrics = balancer.get_metrics().await?;
    assert_eq!(metrics.active_nodes, 2);
    Ok(())
}

#[tokio::test]
async fn test_network_graph_operations() {
    let mut graph = NetworkGraph::new().await.unwrap();

    let node =
        crate::swarm_distributed::SwarmNode::new("node1".to_string(), "127.0.0.1:8001".to_string());
    assert!(graph.add_node(node).await.is_ok());

    let edge_weight = EdgeWeight::default();
    assert!(graph
        .update_edge("node1", "node2", edge_weight)
        .await
        .is_ok());

    assert!(graph.remove_node("node1").await.is_ok());
}

#[test]
fn test_partition_strategies() {
    let strategy = PartitionStrategy::default();
    assert!(matches!(strategy, PartitionStrategy::LoadAware));

    let strategies = vec![
        PartitionStrategy::RoundRobin,
        PartitionStrategy::HashBased,
        PartitionStrategy::RangeBased,
        PartitionStrategy::NetworkAware,
        PartitionStrategy::LoadAware,
    ];

    for strategy in strategies {
        // Test serialization
        let serialized = serde_json::to_string(&strategy).unwrap();
        let _deserialized: PartitionStrategy = serde_json::from_str(&serialized).unwrap();
    }
}

#[test]
fn test_node_capacity() -> TestResult {
    let capacity = NodeCapacity::default();
    assert_eq!(capacity.max_particles, 100);
    assert_eq!(capacity.compute_capacity, 1.0);

    // Test serialization
    let serialized = serde_json::to_string(&capacity)?;
    let deserialized: NodeCapacity = serde_json::from_str(&serialized)?;
    assert_eq!(deserialized.max_particles, capacity.max_particles);
    Ok(())
}

#[test]
fn test_edge_weight() -> TestResult {
    let weight = EdgeWeight::default();
    assert_eq!(weight.latency, 10.0);
    assert_eq!(weight.bandwidth, 100.0);
    assert_eq!(weight.reliability, 0.99);

    // Test serialization
    let serialized = serde_json::to_string(&weight)?;
    let _deserialized: EdgeWeight = serde_json::from_str(&serialized)?;
    Ok(())
}

#[test]
fn test_migration_plan() {
    let migration = Migration {
        particle_id: "particle1".to_string(),
        from_node: "node1".to_string(),
        to_node: "node2".to_string(),
        priority: 1.0,
        expected_benefit: 0.5,
    };

    let plan = MigrationPlan {
        migrations: vec![migration],
        expected_improvement: 0.3,
        migration_cost: 0.1,
    };

    assert_eq!(plan.migrations.len(), 1);
    assert_eq!(plan.expected_improvement, 0.3);
}

#[test]
fn test_load_balance_metrics_struct() {
    let metrics = LoadBalanceMetrics {
        imbalance_factor: 0.1,
        total_particles: 100,
        active_nodes: 5,
        avg_particles_per_node: 20.0,
        distribution_std_dev: 2.5,
        pending_migrations: 3,
    };

    assert_eq!(metrics.imbalance_factor, 0.1);
    assert_eq!(metrics.total_particles, 100);
    assert_eq!(metrics.active_nodes, 5);
}
