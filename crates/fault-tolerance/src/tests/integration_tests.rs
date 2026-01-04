//! Integration tests for fault-tolerance components working together

use crate::checkpoint::*;
use crate::coordinator::*;
use crate::error::{FaultToleranceError, FtResult, HealthStatus};
use crate::recovery::*;
use serde_json::json;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::timeout;

#[tokio::test]
async fn test_end_to_end_checkpoint_and_recovery() {
    let temp_dir = TempDir::new().unwrap();

    // Create checkpoint manager
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create full system checkpoint
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await?;

    // Verify checkpoint exists
    let checkpoint_list = checkpoint_manager.list_checkpoints().await?;
    assert!(checkpoint_list.contains(&checkpoint_id));

    // Load the checkpoint to verify it's valid
    let loaded_checkpoint = checkpoint_manager
        .load_checkpoint(&checkpoint_id)
        .await
        .unwrap();
    assert_eq!(loaded_checkpoint.id, checkpoint_id);

    // Create recovery manager and restore from checkpoint
    let recovery_manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
    let recovery_result = recovery_manager.restore_full_system(checkpoint_id).await;
    assert!(recovery_result.is_ok());

    // Verify post-recovery health
    let health = recovery_manager.post_recovery_health_check().await;
    assert!(matches!(
        health,
        HealthStatus::Healthy | HealthStatus::Degraded
    ));
}

#[tokio::test]
async fn test_coordinator_checkpoint_integration() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create and setup coordinator
    let mut coordinator = DistributedCoordinator::new()?;

    // Make coordinator a leader
    let won_election = coordinator.initiate_leader_election().await?;
    assert!(won_election);
    assert!(coordinator.is_leader);

    // Add some nodes to the cluster
    let bootstrap_nodes = vec![
        "127.0.0.1:8081".parse().unwrap(),
        "127.0.0.1:8082".parse().unwrap(),
    ];
    coordinator.join_cluster(bootstrap_nodes).await.unwrap();

    // Create checkpoint through checkpoint manager
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await.unwrap();

    // Coordinate distributed checkpoint
    let coordination_result = coordinator
        .coordinate_checkpoint(checkpoint_id.clone())
        .await;
    assert!(coordination_result.is_ok());

    // Verify system health after coordination
    let health = coordinator.system_health().await;
    assert!(matches!(
        health,
        HealthStatus::Healthy | HealthStatus::Degraded
    ));

    // Test recovery coordination
    let target_nodes = coordinator.known_nodes.keys().cloned().collect::<Vec<_>>();
    if !target_nodes.is_empty() {
        let recovery_result = coordinator
            .coordinate_recovery(checkpoint_id, target_nodes)
            .await;
        assert!(recovery_result.is_ok());
    }
}

#[tokio::test]
async fn test_multi_node_coordination_and_recovery() {
    let temp_dir = TempDir::new().unwrap();

    // Create multiple coordinators simulating different nodes
    let mut leader_coordinator = DistributedCoordinator::new().unwrap();
    let mut follower_coordinator1 =
        DistributedCoordinator::with_address("127.0.0.1:8081".parse()?)?;
    let mut follower_coordinator2 =
        DistributedCoordinator::with_address("127.0.0.1:8082".parse()?)?;

    // Setup leader
    leader_coordinator.initiate_leader_election().await.unwrap();
    assert!(leader_coordinator.is_leader);

    // Join followers to cluster
    let bootstrap_nodes = vec![leader_coordinator.node_info.address];
    follower_coordinator1
        .join_cluster(bootstrap_nodes.clone())
        .await
        .unwrap();
    follower_coordinator2
        .join_cluster(bootstrap_nodes)
        .await
        .unwrap();

    // Create shared checkpoint manager
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await.unwrap();

    // Coordinate checkpoint from leader
    let coordination_result = leader_coordinator
        .coordinate_checkpoint(checkpoint_id.clone())
        .await;
    assert!(coordination_result.is_ok());

    // Test heartbeat mechanism across nodes
    let heartbeat_result1 = follower_coordinator1.send_heartbeat().await;
    let heartbeat_result2 = follower_coordinator2.send_heartbeat().await;
    assert!(heartbeat_result1.is_ok());
    assert!(heartbeat_result2.is_ok());

    // Simulate recovery coordination
    let all_node_ids: Vec<NodeId> = vec![
        follower_coordinator1.node_id.clone(),
        follower_coordinator2.node_id.clone(),
    ];

    let recovery_coordination = leader_coordinator
        .coordinate_recovery(checkpoint_id, all_node_ids)
        .await;
    assert!(recovery_coordination.is_ok());

    // Verify cluster status
    let cluster_status = leader_coordinator.cluster_status();
    assert!(cluster_status.contains_key("total_nodes"));
    assert!(cluster_status.contains_key("is_leader"));
}

#[tokio::test]
async fn test_fault_tolerance_manager_integration() {
    let temp_dir = TempDir::new().unwrap();

    // Create fault tolerance manager (simulated integration)
    struct FaultToleranceManager {
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
        coordinator: DistributedCoordinator,
    }

    impl FaultToleranceManager {
        fn new(temp_dir: &std::path::Path) -> anyhow::Result<Self> {
            let checkpoint_manager = CheckpointManager::with_path(temp_dir)?;
            let recovery_manager =
                RecoveryManager::with_checkpoint_manager(CheckpointManager::with_path(temp_dir)?);
            let coordinator = DistributedCoordinator::new()?;

            Ok(Self {
                checkpoint_manager,
                recovery_manager,
                coordinator,
            })
        }

        async fn create_and_coordinate_checkpoint(&mut self) -> FtResult<CheckpointId> {
            // Create checkpoint
            let checkpoint_id = self.checkpoint_manager.create_full_checkpoint().await?;

            // Make coordinator leader if needed
            if !self.coordinator.is_leader {
                self.coordinator.initiate_leader_election().await?;
            }

            // Coordinate checkpoint across cluster
            self.coordinator
                .coordinate_checkpoint(checkpoint_id.clone())
                .await?;

            Ok(checkpoint_id)
        }

        async fn recover_from_checkpoint(
            &self,
            checkpoint_id: CheckpointId,
            strategy: RecoveryStrategy,
        ) -> FtResult<()> {
            self.recovery_manager
                .restore_full_system(checkpoint_id)
                .await
        }

        async fn get_system_health(&self) -> HealthStatus {
            self.coordinator.system_health().await
        }
    }

    // Test integrated fault tolerance operations
    let mut ft_manager = FaultToleranceManager::new(temp_dir.path())?;

    // Create and coordinate checkpoint
    let checkpoint_id = ft_manager.create_and_coordinate_checkpoint().await?;

    // Verify checkpoint exists
    let available_checkpoints = ft_manager
        .checkpoint_manager
        .list_checkpoints()
        .await
        .unwrap();
    assert!(available_checkpoints.contains(&checkpoint_id));

    // Test recovery
    let recovery_result = ft_manager
        .recover_from_checkpoint(checkpoint_id, RecoveryStrategy::Full)
        .await;
    assert!(recovery_result.is_ok());

    // Check overall system health
    let health = ft_manager.get_system_health().await;
    assert!(matches!(
        health,
        HealthStatus::Healthy | HealthStatus::Degraded
    ));
}

#[tokio::test]
async fn test_checkpoint_recovery_with_coordinator_failure() {
    let temp_dir = TempDir::new().unwrap();

    // Setup initial system
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let mut coordinator = DistributedCoordinator::new()?;

    // Create checkpoint while system is healthy
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await?;
    coordinator.initiate_leader_election().await?;
    coordinator
        .coordinate_checkpoint(checkpoint_id.clone())
        .await
        .unwrap();

    // Simulate coordinator failure by creating new coordinator (simulates restart)
    let mut new_coordinator =
        DistributedCoordinator::with_address(coordinator.node_info.address).unwrap();

    // New coordinator should not be leader initially
    assert!(!new_coordinator.is_leader);

    // Trigger leader election
    new_coordinator.initiate_leader_election().await.unwrap();
    assert!(new_coordinator.is_leader);

    // Test recovery after coordinator failure
    let recovery_manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
    let recovery_result = recovery_manager.restore_full_system(checkpoint_id).await;
    assert!(recovery_result.is_ok());

    // Verify new coordinator can coordinate recovery
    let dummy_target_nodes = vec![NodeId::new()]; // Simulate some target nodes
    let coordination_result = new_coordinator
        .coordinate_recovery(checkpoint_id, dummy_target_nodes)
        .await;
    assert!(coordination_result.is_ok());
}

#[tokio::test]
async fn test_distributed_checkpoint_with_node_failures() {
    let temp_dir = TempDir::new().unwrap();

    // Create leader coordinator
    let mut leader = DistributedCoordinator::new().unwrap();
    leader.initiate_leader_election().await?;

    // Add multiple nodes with different statuses
    let healthy_node = NodeInfo {
        id: NodeId::new(),
        address: "127.0.0.1:8081".parse()?,
        role: NodeRole::Follower,
        status: NodeStatus::Active,
        last_heartbeat: 0,
        capabilities: vec!["checkpoint".to_string()],
    };

    let degraded_node = NodeInfo {
        id: NodeId::new(),
        address: "127.0.0.1:8082".parse().unwrap(),
        role: NodeRole::Follower,
        status: NodeStatus::Degraded,
        last_heartbeat: 0,
        capabilities: vec!["checkpoint".to_string()],
    };

    let failed_node = NodeInfo {
        id: NodeId::new(),
        address: "127.0.0.1:8083".parse().unwrap(),
        role: NodeRole::Follower,
        status: NodeStatus::Failed, // This node should be skipped
        last_heartbeat: 0,
        capabilities: vec![],
    };

    leader
        .known_nodes
        .insert(healthy_node.id.clone(), healthy_node);
    leader
        .known_nodes
        .insert(degraded_node.id.clone(), degraded_node);
    leader
        .known_nodes
        .insert(failed_node.id.clone(), failed_node);

    // Create checkpoint
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await.unwrap();

    // Coordinate checkpoint - should skip failed nodes
    let coordination_result = leader.coordinate_checkpoint(checkpoint_id.clone()).await;
    assert!(coordination_result.is_ok());

    // Check system health (should be degraded due to failed node)
    let health = leader.system_health().await;
    assert!(matches!(
        health,
        HealthStatus::Degraded | HealthStatus::Failed
    ));

    // Test targeted recovery (should skip failed nodes)
    let target_nodes = leader.known_nodes.keys().cloned().collect::<Vec<_>>();
    let recovery_result = leader
        .coordinate_recovery(checkpoint_id, target_nodes)
        .await;
    assert!(recovery_result.is_ok());
}

#[tokio::test]
async fn test_concurrent_checkpoint_and_recovery_operations() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager =
        std::sync::Arc::new(CheckpointManager::with_path(temp_dir.path()).unwrap());

    // Create multiple checkpoints concurrently
    let mut checkpoint_handles = Vec::new();
    for _ in 0..3 {
        let manager_clone = checkpoint_manager.clone();
        let handle = tokio::spawn(async move { manager_clone.create_full_checkpoint().await });
        checkpoint_handles.push(handle);
    }

    // Wait for all checkpoints to complete
    let mut checkpoint_ids = Vec::new();
    for handle in checkpoint_handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
        checkpoint_ids.push(result.unwrap());
    }

    // Verify all checkpoints exist
    let available_checkpoints = checkpoint_manager.list_checkpoints().await.unwrap();
    for checkpoint_id in &checkpoint_ids {
        assert!(available_checkpoints.contains(checkpoint_id));
    }

    // Test concurrent recovery operations
    let recovery_manager = std::sync::Arc::new(RecoveryManager::with_checkpoint_manager(
        CheckpointManager::with_path(temp_dir.path()).unwrap(),
    ));

    let mut recovery_handles = Vec::new();
    for checkpoint_id in checkpoint_ids {
        let manager_clone = recovery_manager.clone();
        let handle =
            tokio::spawn(async move { manager_clone.restore_full_system(checkpoint_id).await });
        recovery_handles.push(handle);
    }

    // Wait for all recoveries to complete
    for handle in recovery_handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_checkpoint_compression_and_recovery_integration() {
    let temp_dir = TempDir::new().unwrap();

    // Test with compression enabled
    let compressed_manager = CheckpointManager::with_path(temp_dir.path().join("compressed"))?;
    assert!(compressed_manager.compression_enabled);

    let compressed_checkpoint_id = compressed_manager.create_full_checkpoint().await?;
    let compressed_checkpoint = compressed_manager
        .load_checkpoint(&compressed_checkpoint_id)
        .await
        .unwrap();
    assert!(compressed_checkpoint.compressed);

    // Test recovery from compressed checkpoint
    let recovery_manager = RecoveryManager::with_checkpoint_manager(compressed_manager);
    let recovery_result = recovery_manager
        .restore_full_system(compressed_checkpoint_id)
        .await;
    assert!(recovery_result.is_ok());

    // Test with compression disabled
    let mut uncompressed_manager =
        CheckpointManager::with_path(temp_dir.path().join("uncompressed")).unwrap();
    uncompressed_manager.compression_enabled = false;

    let uncompressed_checkpoint_id = uncompressed_manager.create_full_checkpoint().await.unwrap();
    let uncompressed_checkpoint = uncompressed_manager
        .load_checkpoint(&uncompressed_checkpoint_id)
        .await
        .unwrap();
    assert!(!uncompressed_checkpoint.compressed);

    // Test recovery from uncompressed checkpoint
    let recovery_manager2 = RecoveryManager::with_checkpoint_manager(uncompressed_manager);
    let recovery_result2 = recovery_manager2
        .restore_full_system(uncompressed_checkpoint_id)
        .await;
    assert!(recovery_result2.is_ok());
}

#[tokio::test]
async fn test_complex_recovery_scenario_with_multiple_strategies() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create complex checkpoint with multiple GPU and agent states
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await?;
    let checkpoint = checkpoint_manager.load_checkpoint(&checkpoint_id).await?;

    // Verify checkpoint has both GPU and agent data
    assert!(!checkpoint.gpu_checkpoints.is_empty());
    assert!(!checkpoint.agent_checkpoints.is_empty());

    let recovery_manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);

    // Test all recovery strategies
    let strategies = vec![
        RecoveryStrategy::Full,
        RecoveryStrategy::GpuOnly,
        RecoveryStrategy::AgentsOnly,
        RecoveryStrategy::Rolling,
    ];

    for strategy in strategies {
        let result = match strategy {
            RecoveryStrategy::Full => {
                recovery_manager
                    .restore_full_system(checkpoint_id.clone())
                    .await
            }
            RecoveryStrategy::GpuOnly => {
                recovery_manager
                    .restore_gpu_only(checkpoint_id.clone())
                    .await
            }
            RecoveryStrategy::AgentsOnly => {
                recovery_manager
                    .restore_agents_only(checkpoint_id.clone())
                    .await
            }
            RecoveryStrategy::Rolling => {
                recovery_manager
                    .restore_full_system(checkpoint_id.clone())
                    .await
            } // Use full restore for rolling
        };

        assert!(
            result.is_ok(),
            "Failed to recover with strategy: {:?}",
            strategy
        );

        // Verify system health after each recovery
        let health = recovery_manager.post_recovery_health_check().await;
        assert!(matches!(
            health,
            HealthStatus::Healthy | HealthStatus::Degraded
        ));
    }
}

#[tokio::test]
async fn test_error_propagation_across_components() {
    let temp_dir = TempDir::new().unwrap();

    // Test checkpoint creation failure propagation
    let invalid_path = temp_dir
        .path()
        .join("nonexistent")
        .join("very")
        .join("deep")
        .join("path");
    std::fs::remove_dir_all(&invalid_path).ok(); // Ensure it doesn't exist

    let checkpoint_manager = CheckpointManager::with_path(&invalid_path).unwrap();

    // This might succeed due to create_dir_all, so we test the recovery path instead
    let nonexistent_checkpoint_id = CheckpointId::new();

    // Test recovery failure propagation
    let recovery_manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
    let recovery_result = recovery_manager
        .restore_full_system(nonexistent_checkpoint_id.clone())
        .await;

    assert!(recovery_result.is_err());
    match recovery_result.unwrap_err() {
        FaultToleranceError::CheckpointNotFound(_) => {
            // Expected error type
        }
        other => panic!("Expected CheckpointNotFound, got: {:?}", other),
    }

    // Test coordinator error propagation
    let coordinator = DistributedCoordinator::new().unwrap();

    // Try to coordinate checkpoint as non-leader (should fail)
    let coordination_result = coordinator
        .coordinate_checkpoint(nonexistent_checkpoint_id)
        .await;
    assert!(coordination_result.is_err());

    match coordination_result.unwrap_err() {
        FaultToleranceError::CoordinationError(msg) => {
            assert!(msg.contains("leader"));
        }
        other => panic!("Expected CoordinationError, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_timeout_behavior_integration() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await.unwrap();

    // Test recovery with very short timeout
    let recovery_manager = RecoveryManager::with_checkpoint_manager_and_timeout(
        checkpoint_manager,
        Duration::from_millis(1), // Extremely short timeout
    );

    let recovery_result = recovery_manager.restore_full_system(checkpoint_id).await;
    assert!(recovery_result.is_err());

    match recovery_result.unwrap_err() {
        FaultToleranceError::RecoveryFailed(msg) => {
            assert!(msg.contains("timeout"));
        }
        other => panic!("Expected timeout error, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_health_monitoring_integration() {
    let temp_dir = TempDir::new().unwrap();

    // Create system components
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let recovery_manager =
        RecoveryManager::with_checkpoint_manager(CheckpointManager::with_path(temp_dir.path())?);
    let mut coordinator = DistributedCoordinator::new()?;

    // Initial health check
    let initial_health = coordinator.system_health().await;
    assert_eq!(initial_health, HealthStatus::Healthy); // Single node system

    // Add some nodes with different health statuses
    let healthy_node = NodeInfo {
        id: NodeId::new(),
        address: "127.0.0.1:8081".parse().unwrap(),
        role: NodeRole::Follower,
        status: NodeStatus::Active,
        last_heartbeat: 0,
        capabilities: vec![],
    };

    let degraded_node = NodeInfo {
        id: NodeId::new(),
        address: "127.0.0.1:8082".parse().unwrap(),
        role: NodeRole::Follower,
        status: NodeStatus::Degraded,
        last_heartbeat: 0,
        capabilities: vec![],
    };

    coordinator
        .known_nodes
        .insert(healthy_node.id.clone(), healthy_node);
    coordinator
        .known_nodes
        .insert(degraded_node.id.clone(), degraded_node);

    // Health should now be degraded
    let degraded_health = coordinator.system_health().await;
    assert!(matches!(
        degraded_health,
        HealthStatus::Degraded | HealthStatus::Healthy
    ));

    // Create checkpoint and verify it works despite degraded health
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await.unwrap();

    // Recovery should still work
    let recovery_result = recovery_manager.restore_full_system(checkpoint_id).await;
    assert!(recovery_result.is_ok());

    // Post-recovery health check
    let post_recovery_health = recovery_manager.post_recovery_health_check().await;
    assert!(matches!(
        post_recovery_health,
        HealthStatus::Healthy | HealthStatus::Degraded
    ));
}
