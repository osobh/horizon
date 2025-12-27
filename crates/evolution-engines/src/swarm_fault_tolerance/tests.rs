//! Tests for swarm fault tolerance module

use super::*;
use crate::swarm_distributed::SwarmNode;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[tokio::test]
async fn test_fault_detector_creation() {
    let config = FaultToleranceConfig::default();
    let detector = FaultDetector::new(config).await;
    assert!(detector.is_ok());
}

#[tokio::test]
async fn test_node_monitoring() -> TestResult {
    let config = FaultToleranceConfig::default();
    let detector = FaultDetector::new(config).await?;

    let node = SwarmNode::new("test_node".to_string(), "127.0.0.1:8001".to_string());
    assert!(detector.start_monitoring(node).await.is_ok());

    let health = detector.get_cluster_health().await?;
    assert!(health.contains_key("test_node"));

    assert!(detector.stop_monitoring("test_node").await.is_ok());
    Ok(())
}

#[tokio::test]
async fn test_health_update() -> TestResult {
    let config = FaultToleranceConfig::default();
    let detector = FaultDetector::new(config).await?;

    let node = SwarmNode::new("test_node".to_string(), "127.0.0.1:8001".to_string());
    detector.start_monitoring(node).await?;

    assert!(detector
        .update_node_health("test_node", 50.0, 0.3, 0.4)
        .await
        .is_ok());

    let health = detector.get_cluster_health().await?;
    let node_health = health.get("test_node").unwrap();
    assert_eq!(node_health.cpu_utilization, 0.3);
    assert_eq!(node_health.memory_utilization, 0.4);
    Ok(())
}

#[tokio::test]
async fn test_failure_detection() -> TestResult {
    let mut config = FaultToleranceConfig::default();
    config.failure_timeout_ms = 100; // Very short timeout for testing

    let detector = FaultDetector::new(config).await?;

    let node = SwarmNode::new("test_node".to_string(), "127.0.0.1:8001".to_string());
    detector.start_monitoring(node).await?;

    // Wait for timeout
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

    let failed_nodes = detector.check_for_failures().await?;
    assert!(failed_nodes.contains(&"test_node".to_string()));
    Ok(())
}

#[tokio::test]
async fn test_recovery_manager_creation() {
    let config = FaultToleranceConfig::default();
    let checkpoint_manager = Arc::new(RwLock::new(
        CheckpointManager::new(config.clone()).await.unwrap(),
    ));
    let recovery_manager = RecoveryManager::new(config, checkpoint_manager).await;
    assert!(recovery_manager.is_ok());
}

#[tokio::test]
async fn test_recovery_execution() -> TestResult {
    let config = FaultToleranceConfig::default();
    let checkpoint_manager = Arc::new(RwLock::new(
        CheckpointManager::new(config.clone()).await?,
    ));
    let mut recovery_manager = RecoveryManager::new(config, checkpoint_manager).await?;

    let affected_particles = vec!["particle1".to_string(), "particle2".to_string()];
    let migration_plan = recovery_manager
        .execute_recovery("failed_node", &affected_particles)
        .await?;

    assert!(migration_plan.migrations.len() > 0);
    assert_eq!(recovery_manager.get_recovery_history().len(), 1);
    Ok(())
}

#[tokio::test]
async fn test_checkpoint_manager_creation() {
    let config = FaultToleranceConfig::default();
    let manager = CheckpointManager::new(config).await;
    assert!(manager.is_ok());
}

#[tokio::test]
async fn test_checkpoint_creation() {
    let config = FaultToleranceConfig::default();
    let mut manager = CheckpointManager::new(config).await.unwrap();

    let mut node_states = HashMap::new();
    node_states.insert(
        "node1".to_string(),
        NodeState {
            node_id: "node1".to_string(),
            particles: vec![],
            config: serde_json::Value::Null,
            local_best: None,
            local_best_fitness: None,
        },
    );

    let checkpoint_id = manager
        .create_checkpoint(1, node_states, None, Some(0.8))
        .await
        .unwrap();
    assert!(!checkpoint_id.is_empty());

    let checkpoints = manager.list_checkpoints();
    assert_eq!(checkpoints.len(), 1);
}

#[tokio::test]
async fn test_checkpoint_retrieval() {
    let config = FaultToleranceConfig::default();
    let mut manager = CheckpointManager::new(config).await.unwrap();

    let mut node_states = HashMap::new();
    node_states.insert(
        "node1".to_string(),
        NodeState {
            node_id: "node1".to_string(),
            particles: vec![],
            config: serde_json::Value::Null,
            local_best: None,
            local_best_fitness: None,
        },
    );

    let checkpoint_id = manager
        .create_checkpoint(1, node_states, None, Some(0.8))
        .await
        .unwrap();

    let retrieved = manager
        .restore_from_checkpoint(&checkpoint_id)
        .await
        .unwrap();
    assert_eq!(retrieved.id, checkpoint_id);
    assert_eq!(retrieved.generation, 1);

    let latest = manager.get_latest_checkpoint().await.unwrap();
    assert!(latest.is_some());
    assert_eq!(latest.unwrap().id, checkpoint_id);
}

#[tokio::test]
async fn test_checkpoint_cleanup() {
    let config = FaultToleranceConfig::default();
    let mut manager = CheckpointManager::new(config).await.unwrap();

    // Create multiple checkpoints
    for i in 1..=5 {
        let mut node_states = HashMap::new();
        node_states.insert(
            format!("node{}", i),
            NodeState {
                node_id: format!("node{}", i),
                particles: vec![],
                config: serde_json::Value::Null,
                local_best: None,
                local_best_fitness: None,
            },
        );

        manager
            .create_checkpoint(i, node_states, None, Some(0.8))
            .await
            .unwrap();
    }

    assert_eq!(manager.list_checkpoints().len(), 5);

    let deleted_count = manager.cleanup_old_checkpoints(3).await.unwrap();
    assert_eq!(deleted_count, 2);
    assert_eq!(manager.list_checkpoints().len(), 3);
}

#[tokio::test]
async fn test_checkpoint_storage() -> TestResult {
    let storage = CheckpointStorage::new(StorageType::Local);

    let test_data = r#"{"id":"test","data":"checkpoint"}"#;
    assert!(storage.store_checkpoint("test_id", test_data).await.is_ok());

    let loaded = storage.load_checkpoint("test_id").await?;
    assert!(loaded.contains("test_id"));

    assert!(storage.delete_checkpoint("test_id").await.is_ok());
    Ok(())
}

#[test]
fn test_redistribute_recovery() -> TestResult {
    let recovery = RedistributeRecovery::new(vec!["node1".to_string(), "node2".to_string()]);
    let affected_particles = vec!["particle1".to_string(), "particle2".to_string()];

    let plan = recovery.execute_recovery("failed_node", &affected_particles, None)?;
    assert_eq!(plan.migrations.len(), 2);
    assert!(plan.migrations.iter().all(|m| m.from_node == "failed_node"));
    Ok(())
}

#[test]
fn test_checkpoint_recovery() -> TestResult {
    use std::sync::Arc;
    use tokio::sync::RwLock;

    let rt = tokio::runtime::Runtime::new()?;
    let checkpoint_manager = Arc::new(RwLock::new(rt.block_on(async {
        CheckpointManager::new(FaultToleranceConfig::default()).await
    })?));

    let recovery = CheckpointRecovery::new(checkpoint_manager);
    let affected_particles = vec!["particle1".to_string()];

    // Create a checkpoint with node state
    let mut node_states = HashMap::new();
    node_states.insert(
        "failed_node".to_string(),
        NodeState {
            node_id: "failed_node".to_string(),
            particles: vec![],
            config: serde_json::Value::Null,
            local_best: None,
            local_best_fitness: None,
        },
    );

    let checkpoint = CheckpointSnapshot {
        id: "test_checkpoint".to_string(),
        generation: 1,
        timestamp: 123456789,
        node_states,
        global_best: None,
        global_best_fitness: None,
        size_bytes: 0,
    };

    let plan = recovery.execute_recovery("failed_node", &affected_particles, Some(&checkpoint))?;
    assert_eq!(plan.migrations.len(), 1);
    Ok(())
}

#[test]
fn test_hybrid_recovery() -> TestResult {
    use std::sync::Arc;
    use tokio::sync::RwLock;

    let rt = tokio::runtime::Runtime::new()?;
    let checkpoint_manager = Arc::new(RwLock::new(rt.block_on(async {
        CheckpointManager::new(FaultToleranceConfig::default()).await
    })?));

    let recovery = HybridRecovery::new(checkpoint_manager);
    let affected_particles = vec!["particle1".to_string(), "particle2".to_string()];

    let plan = recovery.execute_recovery("failed_node", &affected_particles, None)?;
    assert!(plan.migrations.len() >= 2);
    Ok(())
}

#[test]
fn test_health_status_serialization() -> TestResult {
    let statuses = vec![
        HealthStatus::Healthy,
        HealthStatus::Degraded,
        HealthStatus::Suspect,
        HealthStatus::Failed,
        HealthStatus::Recovering,
    ];

    for status in statuses {
        let serialized = serde_json::to_string(&status)?;
        let _deserialized: HealthStatus = serde_json::from_str(&serialized)?;
    }
    Ok(())
}

#[test]
fn test_node_health_serialization() {
    let health = NodeHealth {
        node_id: "test_node".to_string(),
        status: HealthStatus::Healthy,
        last_heartbeat: 123456789,
        response_times: vec![10.0, 20.0, 15.0],
        error_count: 2,
        cpu_utilization: 0.3,
        memory_utilization: 0.4,
        network_score: 0.95,
    };

    let serialized = serde_json::to_string(&health).unwrap();
    let deserialized: NodeHealth = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized.node_id, "test_node");
    assert_eq!(deserialized.error_count, 2);
    assert_eq!(deserialized.response_times.len(), 3);
}

#[test]
fn test_recovery_event_serialization() {
    let event = RecoveryEvent {
        id: "event1".to_string(),
        failed_node: "node1".to_string(),
        strategy: crate::swarm_distributed::RecoveryStrategy::Hybrid,
        failure_time: 123456789,
        recovery_start_time: 123456790,
        recovery_completion_time: Some(123456800),
        status: RecoveryStatus::Completed,
        particles_affected: 5,
        success_rate: 0.9,
    };

    let serialized = serde_json::to_string(&event).unwrap();
    let deserialized: RecoveryEvent = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized.id, "event1");
    assert_eq!(deserialized.particles_affected, 5);
    assert_eq!(deserialized.success_rate, 0.9);
}

#[test]
fn test_checkpoint_metadata_serialization() {
    let metadata = CheckpointMetadata {
        id: "checkpoint1".to_string(),
        created_at: 123456789,
        size_bytes: 2048,
        node_count: 3,
        particle_count: 150,
        compression: CompressionAlgorithm::Gzip,
        checksum: "abc123".to_string(),
    };

    let serialized = serde_json::to_string(&metadata).unwrap();
    let deserialized: CheckpointMetadata = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized.id, "checkpoint1");
    assert_eq!(deserialized.node_count, 3);
    assert_eq!(deserialized.particle_count, 150);
}

#[test]
fn test_default_implementations() {
    let _thresholds = AlertThresholds::default();
    let _compression = CompressionAlgorithm::default();
}
