//! Tests for distributed swarm module

use super::*;
use crate::traits::{AgentGenome, ArchitectureGenes, BehaviorGenes};
use std::collections::HashMap;

#[test]
fn test_swarm_node_creation() {
    let node = SwarmNode::new("test_node".to_string(), "127.0.0.1:8000".to_string());
    assert_eq!(node.node_id, "test_node");
    assert_eq!(node.address, "127.0.0.1:8000");
    assert_eq!(node.status, NodeStatus::Starting);
    assert_eq!(node.particle_count, 0);
}

#[test]
fn test_swarm_node_health_check() {
    let mut node = SwarmNode::new("test_node".to_string(), "127.0.0.1:8000".to_string());
    node.status = NodeStatus::Active;

    // Should be healthy with recent heartbeat
    assert!(node.is_healthy(5000));

    // Simulate old heartbeat
    node.last_heartbeat = 0;
    assert!(!node.is_healthy(5000));
}

#[test]
fn test_distributed_swarm_config_creation() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    assert_eq!(config.node_id, "test_node");
    assert_eq!(config.network_config.port, 7001);
    assert_eq!(config.load_balance_config.target_particles_per_node, 50);
}

#[test]
fn test_distributed_swarm_config_validation() {
    let mut config = DistributedSwarmConfig::new("test_node".to_string());
    assert!(config.validate().is_ok());

    // Invalid port
    config.network_config.port = 0;
    assert!(config.validate().is_err());

    // Reset port, invalid max connections
    config.network_config.port = 7001;
    config.network_config.max_connections = 0;
    assert!(config.validate().is_err());

    // Reset connections, invalid rebalance threshold
    config.network_config.max_connections = 100;
    config.load_balance_config.rebalance_threshold = 1.5;
    assert!(config.validate().is_err());
}

#[tokio::test]
async fn test_distributed_swarm_engine_lifecycle() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let engine = DistributedSwarmEngine::new(config).await.unwrap();

    // Test start
    assert!(engine.start().await.is_ok());

    // Test stop
    assert!(engine.stop().await.is_ok());
}

#[tokio::test]
async fn test_cluster_status() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let engine = DistributedSwarmEngine::new(config).await.unwrap();

    let status = engine.get_cluster_status().await.unwrap();
    assert_eq!(status.total_nodes, 1); // Just the local node
    assert_eq!(status.generation, 0);
}

#[tokio::test]
async fn test_particle_migration() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let engine = DistributedSwarmEngine::new(config).await.unwrap();

    let particle_ids = vec!["particle1".to_string(), "particle2".to_string()];
    let result = engine
        .migrate_particles(particle_ids, "target_node".to_string())
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_node_manager_creation() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let manager = NodeManager::new(config).await;
    assert!(manager.is_ok());
}

#[tokio::test]
async fn test_node_manager_cluster_operations() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let mut manager = NodeManager::new(config).await.unwrap();

    // Test join cluster
    assert!(manager.join_cluster().await.is_ok());

    // Test leave cluster
    assert!(manager.leave_cluster().await.is_ok());
}

#[test]
fn test_node_manager_peer_management() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut manager = rt.block_on(NodeManager::new(config)).unwrap();

    let peer_node = SwarmNode::new("peer1".to_string(), "127.0.0.1:8001".to_string());

    // Test add peer
    manager.add_peer(peer_node.clone());
    assert!(manager.get_peer_nodes().contains_key("peer1"));

    // Test remove peer
    manager.remove_peer("peer1");
    assert!(!manager.get_peer_nodes().contains_key("peer1"));
}

#[tokio::test]
async fn test_message_bus_creation() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let bus = MessageBus::new(config).await;
    assert!(bus.is_ok());
}

#[tokio::test]
async fn test_message_bus_messaging() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let bus = MessageBus::new(config).await.unwrap();

    let message = DistributedMessage::Heartbeat {
        node_id: "test_node".to_string(),
        timestamp: 123456789,
        load: 0.5,
    };

    // Test send message
    assert!(bus
        .send_message("target_node", message.clone())
        .await
        .is_ok());

    // Test broadcast message
    assert!(bus.broadcast_message(message).await.is_ok());
}

#[test]
fn test_message_serialization() {
    let message = DistributedMessage::Heartbeat {
        node_id: "test_node".to_string(),
        timestamp: 123456789,
        load: 0.5,
    };

    let serialized = serde_json::to_string(&message)?;
    let deserialized: DistributedMessage = serde_json::from_str(&serialized)?;

    match deserialized {
        DistributedMessage::Heartbeat {
            node_id,
            timestamp,
            load,
        } => {
            assert_eq!(node_id, "test_node");
            assert_eq!(timestamp, 123456789);
            assert_eq!(load, 0.5);
        }
        _ => panic!("Wrong message type"),
    }
}

#[test]
fn test_migration_particle_serialization() {
    let genome = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new(
            "test".to_string(),
            stratoswarm_agent_core::GoalPriority::Normal,
        ),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    let particle = MigrationParticle {
        id: "particle1".to_string(),
        genome,
        velocity: vec![0.1, 0.2, 0.3],
        personal_best_fitness: 0.8,
        current_fitness: Some(0.7),
    };

    let serialized = serde_json::to_string(&particle).unwrap();
    let deserialized: MigrationParticle = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized.id, "particle1");
    assert_eq!(deserialized.velocity, vec![0.1, 0.2, 0.3]);
    assert_eq!(deserialized.personal_best_fitness, 0.8);
}

#[test]
fn test_checkpoint_data_serialization() {
    let checkpoint = CheckpointData {
        generation: 42,
        global_best: None,
        global_best_fitness: Some(0.95),
        node_assignments: HashMap::new(),
        timestamp: 123456789,
    };

    let serialized = serde_json::to_string(&checkpoint)?;
    let deserialized: CheckpointData = serde_json::from_str(&serialized)?;

    assert_eq!(deserialized.generation, 42);
    assert_eq!(deserialized.global_best_fitness, Some(0.95));
    assert_eq!(deserialized.timestamp, 123456789);
}

#[tokio::test]
async fn test_discovery_service_creation() {
    let mut config = DistributedSwarmConfig::new("test_node".to_string());
    config.network_config.bootstrap_peers = vec!["127.0.0.1:8001".to_string()];

    let service = DiscoveryService::new(config).await;
    assert!(service.is_ok());
}

#[tokio::test]
async fn test_node_discovery() {
    let config = DistributedSwarmConfig::new("test_node".to_string());
    let service = DiscoveryService::new(config).await.unwrap();

    let nodes = service.discover_nodes().await.unwrap();
    assert_eq!(nodes.len(), 0); // Stub implementation returns empty
}

#[test]
fn test_connection_pool_creation() {
    let pool = ConnectionPool::new(10);
    assert_eq!(pool.max_size, 10);
    assert_eq!(pool.available.len(), 0);
    assert_eq!(pool.active.len(), 0);
}

#[test]
fn test_connection_pool_operations() {
    let mut pool = ConnectionPool::new(10);

    // Test get connection (should return None for stub)
    let conn = pool.get_connection("peer1");
    assert!(conn.is_none());

    // Test return connection (stub implementation)
    let connection = Connection {
        peer_id: "peer1".to_string(),
        status: ConnectionStatus::Connected,
        last_activity: 123456789,
    };
    pool.return_connection(connection);
}

#[test]
fn test_load_balance_strategy_serialization() {
    let strategies = vec![
        LoadBalanceStrategy::EvenDistribution,
        LoadBalanceStrategy::LoadBased,
        LoadBalanceStrategy::TopologyAware,
        LoadBalanceStrategy::Adaptive,
    ];

    for strategy in strategies {
        let serialized = serde_json::to_string(&strategy)?;
        let _deserialized: LoadBalanceStrategy = serde_json::from_str(&serialized)?;
    }
}

#[test]
fn test_recovery_strategy_serialization() {
    let strategies = vec![
        RecoveryStrategy::Redistribute,
        RecoveryStrategy::Checkpoint,
        RecoveryStrategy::Hybrid,
    ];

    for strategy in strategies {
        let serialized = serde_json::to_string(&strategy)?;
        let _deserialized: RecoveryStrategy = serde_json::from_str(&serialized)?;
    }
}
