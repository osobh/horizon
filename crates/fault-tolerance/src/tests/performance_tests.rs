//! Performance and benchmark tests for fault-tolerance components

use crate::checkpoint::*;
use crate::coordinator::*;
use crate::error::{FaultToleranceError, HealthStatus};
use crate::recovery::*;
use serde_json::json;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tempfile::TempDir;

#[tokio::test]
async fn test_checkpoint_creation_performance() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Measure time for single checkpoint creation
    let start = Instant::now();
    let checkpoint_id = manager.create_full_checkpoint().await?;
    let single_duration = start.elapsed();

    println!("Single checkpoint creation: {:?}", single_duration);
    assert!(single_duration < Duration::from_secs(5)); // Should be reasonably fast

    // Verify checkpoint was created successfully
    let checkpoint = manager.load_checkpoint(&checkpoint_id).await.unwrap();
    assert_eq!(checkpoint.id, checkpoint_id);

    // Measure time for multiple checkpoint creations
    let batch_size = 10;
    let start = Instant::now();

    for _ in 0..batch_size {
        manager.create_full_checkpoint().await.unwrap();
    }

    let batch_duration = start.elapsed();
    let avg_duration = batch_duration / batch_size;

    println!(
        "Batch checkpoint creation: {:?} total, {:?} average",
        batch_duration, avg_duration
    );
    assert!(batch_duration < Duration::from_secs(30)); // Batch should complete in reasonable time

    // Verify all checkpoints exist
    let checkpoint_list = manager.list_checkpoints().await.unwrap();
    assert!(checkpoint_list.len() >= batch_size as usize);
}

#[tokio::test]
async fn test_checkpoint_load_performance() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create a checkpoint to load
    let checkpoint_id = manager.create_full_checkpoint().await?;

    // Measure time for single load
    let start = Instant::now();
    let loaded_checkpoint = manager.load_checkpoint(&checkpoint_id).await?;
    let single_load_duration = start.elapsed();

    println!("Single checkpoint load: {:?}", single_load_duration);
    assert!(single_load_duration < Duration::from_secs(2));
    assert_eq!(loaded_checkpoint.id, checkpoint_id);

    // Measure time for repeated loads (testing caching behavior)
    let iterations = 50;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = manager.load_checkpoint(&checkpoint_id).await.unwrap();
    }

    let repeated_load_duration = start.elapsed();
    let avg_load_duration = repeated_load_duration / iterations;

    println!(
        "Repeated loads: {:?} total, {:?} average",
        repeated_load_duration, avg_load_duration
    );
    assert!(avg_load_duration < Duration::from_millis(100)); // Should be fast after first load
}

#[tokio::test]
async fn test_large_checkpoint_performance() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create a checkpoint with larger data to test scaling
    let start = Instant::now();
    let checkpoint_id = manager.create_full_checkpoint().await?;
    let creation_duration = start.elapsed();

    println!("Large checkpoint creation: {:?}", creation_duration);

    // Load the large checkpoint
    let start = Instant::now();
    let loaded_checkpoint = manager.load_checkpoint(&checkpoint_id).await.unwrap();
    let load_duration = start.elapsed();

    println!("Large checkpoint load: {:?}", load_duration);

    // Verify checkpoint integrity
    assert_eq!(loaded_checkpoint.id, checkpoint_id);
    assert!(!loaded_checkpoint.gpu_checkpoints.is_empty());
    assert!(!loaded_checkpoint.agent_checkpoints.is_empty());

    // Both operations should complete in reasonable time even for larger data
    assert!(creation_duration < Duration::from_secs(10));
    assert!(load_duration < Duration::from_secs(5));
}

#[tokio::test]
async fn test_checkpoint_compression_performance() {
    let temp_dir = TempDir::new().unwrap();

    // Test with compression enabled
    let compressed_manager =
        CheckpointManager::with_path(temp_dir.path().join("compressed"))?;
    assert!(compressed_manager.compression_enabled);

    let start = Instant::now();
    let compressed_id = compressed_manager.create_full_checkpoint().await?;
    let compressed_creation_time = start.elapsed();

    let start = Instant::now();
    let _ = compressed_manager
        .load_checkpoint(&compressed_id)
        .await
        .unwrap();
    let compressed_load_time = start.elapsed();

    // Test with compression disabled
    let mut uncompressed_manager =
        CheckpointManager::with_path(temp_dir.path().join("uncompressed")).unwrap();
    uncompressed_manager.compression_enabled = false;

    let start = Instant::now();
    let uncompressed_id = uncompressed_manager.create_full_checkpoint().await.unwrap();
    let uncompressed_creation_time = start.elapsed();

    let start = Instant::now();
    let _ = uncompressed_manager
        .load_checkpoint(&uncompressed_id)
        .await
        .unwrap();
    let uncompressed_load_time = start.elapsed();

    println!(
        "Compressed: create {:?}, load {:?}",
        compressed_creation_time, compressed_load_time
    );
    println!(
        "Uncompressed: create {:?}, load {:?}",
        uncompressed_creation_time, uncompressed_load_time
    );

    // Both should complete in reasonable time
    assert!(compressed_creation_time < Duration::from_secs(5));
    assert!(compressed_load_time < Duration::from_secs(5));
    assert!(uncompressed_creation_time < Duration::from_secs(5));
    assert!(uncompressed_load_time < Duration::from_secs(5));

    // Compressed operations might be slightly slower due to compression overhead,
    // but file sizes should be smaller (we can't directly test this without access to file system)
}

#[tokio::test]
async fn test_recovery_performance() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create checkpoint for recovery testing
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await?;

    let recovery_manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);

    // Test full system recovery performance
    let start = Instant::now();
    let result = recovery_manager
        .restore_full_system(checkpoint_id.clone())
        .await;
    let full_recovery_duration = start.elapsed();

    assert!(result.is_ok());
    println!("Full system recovery: {:?}", full_recovery_duration);
    assert!(full_recovery_duration < Duration::from_secs(10));

    // Test GPU-only recovery performance
    let start = Instant::now();
    let result = recovery_manager
        .restore_gpu_only(checkpoint_id.clone())
        .await;
    let gpu_recovery_duration = start.elapsed();

    assert!(result.is_ok());
    println!("GPU-only recovery: {:?}", gpu_recovery_duration);
    assert!(gpu_recovery_duration < Duration::from_secs(5));

    // Test agents-only recovery performance
    let start = Instant::now();
    let result = recovery_manager
        .restore_agents_only(checkpoint_id.clone())
        .await;
    let agents_recovery_duration = start.elapsed();

    assert!(result.is_ok());
    println!("Agents-only recovery: {:?}", agents_recovery_duration);
    assert!(agents_recovery_duration < Duration::from_secs(5));

    // Partial recoveries should generally be faster than full recovery
    assert!(gpu_recovery_duration <= full_recovery_duration + Duration::from_millis(100));
    assert!(agents_recovery_duration <= full_recovery_duration + Duration::from_millis(100));
}

#[tokio::test]
async fn test_coordinator_message_handling_performance() {
    let mut coordinator = DistributedCoordinator::new().unwrap();

    // Add multiple nodes to test scaling
    for i in 0..100 {
        let node = NodeInfo {
            id: NodeId::new(),
            address: format!("127.0.0.1:80{:02}", i % 100).parse()?,
            role: NodeRole::Follower,
            status: NodeStatus::Active,
            last_heartbeat: 0,
            capabilities: vec!["checkpoint".to_string()],
        };
        coordinator.known_nodes.insert(node.id.clone(), node);
    }

    // Test heartbeat performance with many nodes
    let start = Instant::now();
    let result = coordinator.send_heartbeat().await;
    let heartbeat_duration = start.elapsed();

    assert!(result.is_ok());
    println!("Heartbeat to 100 nodes: {:?}", heartbeat_duration);
    assert!(heartbeat_duration < Duration::from_secs(5));

    // Test message handling performance
    let node_ids: Vec<NodeId> = coordinator.known_nodes.keys().cloned().collect();
    let test_messages = vec![
        CoordinationMessage::Heartbeat {
            node_id: node_ids[0].clone(),
            timestamp: 1234567890,
            status: NodeStatus::Active,
        },
        CoordinationMessage::HealthCheck {
            requester: node_ids[1].clone(),
        },
        CoordinationMessage::LeaderElection {
            candidate: node_ids[2].clone(),
            term: 5,
        },
    ];

    let start = Instant::now();
    for message in test_messages {
        coordinator.handle_message(message).await.unwrap();
    }
    let message_handling_duration = start.elapsed();

    println!(
        "Message handling (3 messages): {:?}",
        message_handling_duration
    );
    assert!(message_handling_duration < Duration::from_millis(100));
}

#[tokio::test]
async fn test_concurrent_operations_performance() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager =
        std::sync::Arc::new(CheckpointManager::with_path(temp_dir.path()).unwrap());

    // Test concurrent checkpoint creation
    let concurrent_operations = 10;
    let start = Instant::now();

    let mut handles = Vec::new();
    for _ in 0..concurrent_operations {
        let manager_clone = checkpoint_manager.clone();
        let handle = tokio::spawn(async move { manager_clone.create_full_checkpoint().await });
        handles.push(handle);
    }

    // Wait for all operations to complete
    let mut checkpoint_ids = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
        checkpoint_ids.push(result.unwrap());
    }

    let concurrent_creation_duration = start.elapsed();
    println!(
        "Concurrent checkpoint creation ({}): {:?}",
        concurrent_operations, concurrent_creation_duration
    );
    assert!(concurrent_creation_duration < Duration::from_secs(30));

    // Test concurrent checkpoint loading
    let recovery_manager = std::sync::Arc::new(RecoveryManager::with_checkpoint_manager(
        CheckpointManager::with_path(temp_dir.path()).unwrap(),
    ));

    let start = Instant::now();
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

    let concurrent_recovery_duration = start.elapsed();
    println!(
        "Concurrent recovery ({}): {:?}",
        concurrent_operations, concurrent_recovery_duration
    );
    assert!(concurrent_recovery_duration < Duration::from_secs(60));
}

#[tokio::test]
async fn test_system_health_calculation_performance() {
    let mut coordinator = DistributedCoordinator::new().unwrap();

    // Add a large number of nodes to test health calculation scaling
    let node_count = 1000;
    for i in 0..node_count {
        let status = match i % 4 {
            0 => NodeStatus::Active,
            1 => NodeStatus::Active,
            2 => NodeStatus::Degraded,
            _ => NodeStatus::Failed,
        }; // 50% active, 25% degraded, 25% failed

        let node = NodeInfo {
            id: NodeId::new(),
            address: format!("127.0.0.1:{}", 8000 + (i % 65535)).parse().unwrap(),
            role: NodeRole::Follower,
            status,
            last_heartbeat: i as u64,
            capabilities: vec![],
        };
        coordinator.known_nodes.insert(node.id.clone(), node);
    }

    // Measure health calculation performance
    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        let _health = coordinator.system_health().await;
    }

    let health_calculation_duration = start.elapsed();
    let avg_health_calculation = health_calculation_duration / iterations;

    println!(
        "Health calculation with {} nodes: {:?} total, {:?} average",
        node_count, health_calculation_duration, avg_health_calculation
    );

    // Health calculation should be fast even with many nodes
    assert!(avg_health_calculation < Duration::from_millis(10));

    // Verify the health calculation is correct
    let health = coordinator.system_health().await;
    assert!(matches!(
        health,
        HealthStatus::Degraded | HealthStatus::Failed
    )); // With 50% active nodes
}

#[tokio::test]
async fn test_checkpoint_cleanup_performance() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    manager.max_checkpoints = 5; // Set low limit to trigger frequent cleanup

    // Create many checkpoints to trigger cleanup
    let checkpoint_count = 20;
    let start = Instant::now();

    for _ in 0..checkpoint_count {
        manager.create_full_checkpoint().await?;
    }

    let creation_with_cleanup_duration = start.elapsed();
    println!(
        "Creation with cleanup ({} checkpoints): {:?}",
        checkpoint_count, creation_with_cleanup_duration
    );

    // Verify cleanup worked correctly
    let remaining_checkpoints = manager.list_checkpoints().await.unwrap();
    assert_eq!(remaining_checkpoints.len(), manager.max_checkpoints);

    // Should complete in reasonable time despite cleanup operations
    assert!(creation_with_cleanup_duration < Duration::from_secs(30));
}

#[tokio::test]
async fn test_memory_usage_patterns() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create multiple checkpoints and verify they can all be loaded
    let checkpoint_count = 10;
    let mut checkpoint_ids = Vec::new();

    for _ in 0..checkpoint_count {
        let id = manager.create_full_checkpoint().await?;
        checkpoint_ids.push(id);
    }

    // Load all checkpoints to test memory usage
    let start = Instant::now();
    let mut loaded_checkpoints = Vec::new();

    for checkpoint_id in &checkpoint_ids {
        let checkpoint = manager.load_checkpoint(checkpoint_id).await.unwrap();
        loaded_checkpoints.push(checkpoint);
    }

    let load_all_duration = start.elapsed();
    println!(
        "Loading {} checkpoints: {:?}",
        checkpoint_count, load_all_duration
    );

    // Verify all checkpoints are loaded correctly
    assert_eq!(loaded_checkpoints.len(), checkpoint_count);
    for (i, checkpoint) in loaded_checkpoints.iter().enumerate() {
        assert_eq!(checkpoint.id, checkpoint_ids[i]);
    }

    // Should handle multiple checkpoints in memory efficiently
    assert!(load_all_duration < Duration::from_secs(10));
}

#[tokio::test]
async fn test_recovery_time_estimation_performance() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await.unwrap();

    let recovery_manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);

    // Test estimation performance for different strategies
    let strategies = vec![
        RecoveryStrategy::Full,
        RecoveryStrategy::GpuOnly,
        RecoveryStrategy::AgentsOnly,
        RecoveryStrategy::Rolling,
    ];

    let iterations = 100;

    for strategy in strategies {
        let start = Instant::now();

        for _ in 0..iterations {
            let _estimate = recovery_manager
                .estimate_recovery_time(checkpoint_id.clone(), strategy.clone())
                .await
                .unwrap();
        }

        let estimation_duration = start.elapsed();
        let avg_estimation = estimation_duration / iterations;

        println!(
            "Recovery time estimation for {:?}: {:?} total, {:?} average",
            strategy, estimation_duration, avg_estimation
        );

        // Estimation should be very fast
        assert!(avg_estimation < Duration::from_millis(10));
    }
}

#[tokio::test]
async fn test_coordinator_scaling_performance() {
    // Test coordinator performance with increasing numbers of nodes
    let node_counts = vec![10, 50, 100, 500];

    for node_count in node_counts {
        let mut coordinator = DistributedCoordinator::new()?;

        // Add nodes
        for i in 0..node_count {
            let node = NodeInfo {
                id: NodeId::new(),
                address: format!("127.0.0.1:{}", 8000 + i).parse().unwrap(),
                role: NodeRole::Follower,
                status: NodeStatus::Active,
                last_heartbeat: 0,
                capabilities: vec!["checkpoint".to_string()],
            };
            coordinator.known_nodes.insert(node.id.clone(), node);
        }

        // Make coordinator leader
        coordinator.initiate_leader_election().await.unwrap();

        // Measure checkpoint coordination performance
        let checkpoint_id = CheckpointId::new();
        let start = Instant::now();
        let result = coordinator.coordinate_checkpoint(checkpoint_id).await;
        let coordination_duration = start.elapsed();

        assert!(result.is_ok());
        println!(
            "Checkpoint coordination with {} nodes: {:?}",
            node_count, coordination_duration
        );

        // Should scale reasonably with node count
        let expected_max_duration = Duration::from_millis(50 * node_count as u64);
        assert!(coordination_duration < expected_max_duration);
    }
}

#[tokio::test]
async fn test_serialization_performance() {
    // Test serialization performance for different checkpoint sizes
    let test_cases = vec![
        ("small", 100),     // 100 bytes
        ("medium", 10000),  // 10KB
        ("large", 1000000), // 1MB
    ];

    for (size_name, data_size) in test_cases {
        // Create test checkpoint with specified size
        let gpu_checkpoint = GpuCheckpoint {
            memory_snapshot: vec![42u8; data_size],
            kernel_states: HashMap::new(),
            timestamp: 1234567890,
            size_bytes: data_size,
        };

        let agent_checkpoint = AgentCheckpoint {
            agent_id: "test_agent".to_string(),
            state_data: vec![1u8; data_size / 10], // Smaller agent data
            memory_contents: {
                let mut contents = HashMap::new();
                contents.insert("key".to_string(), json!("value"));
                contents
            },
            goals: vec!["test_goal".to_string()],
            metadata: HashMap::new(),
        };

        let checkpoint = SystemCheckpoint {
            id: CheckpointId::new(),
            timestamp: 1234567890,
            gpu_checkpoints: vec![gpu_checkpoint],
            agent_checkpoints: vec![agent_checkpoint],
            system_metadata: HashMap::new(),
            compressed: true,
        };

        // Test serialization performance
        let iterations = 10;
        let start = Instant::now();

        for _ in 0..iterations {
            let _serialized = bincode::serialize(&checkpoint).unwrap();
        }

        let serialization_duration = start.elapsed();
        let avg_serialization = serialization_duration / iterations;

        println!(
            "Serialization {} checkpoint: {:?} total, {:?} average",
            size_name, serialization_duration, avg_serialization
        );

        // Serialization should complete in reasonable time
        let max_duration = match size_name {
            "small" => Duration::from_millis(10),
            "medium" => Duration::from_millis(100),
            "large" => Duration::from_secs(1),
            _ => Duration::from_secs(5),
        };
        assert!(avg_serialization < max_duration);
    }
}

#[tokio::test]
async fn test_stress_test_mixed_operations() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager =
        std::sync::Arc::new(CheckpointManager::with_path(temp_dir.path()).unwrap());
    let recovery_manager = std::sync::Arc::new(RecoveryManager::with_checkpoint_manager(
        CheckpointManager::with_path(temp_dir.path())?,
    ));

    // Run mixed operations concurrently for stress testing
    let duration = Duration::from_secs(10);
    let start_time = Instant::now();
    let mut handles = Vec::new();

    // Checkpoint creation workers
    for _ in 0..3 {
        let manager_clone = checkpoint_manager.clone();
        let handle = tokio::spawn(async move {
            let mut count = 0;
            let start = Instant::now();
            while start.elapsed() < duration {
                if manager_clone.create_full_checkpoint().await.is_ok() {
                    count += 1;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            count
        });
        handles.push(handle);
    }

    // Wait for initial checkpoints to be created
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Recovery workers
    for _ in 0..2 {
        let manager_clone = recovery_manager.clone();
        let checkpoint_mgr_clone = checkpoint_manager.clone();
        let handle = tokio::spawn(async move {
            let mut count = 0;
            let start = Instant::now();
            while start.elapsed() < duration {
                if let Ok(checkpoints) = checkpoint_mgr_clone.list_checkpoints().await {
                    if let Some(checkpoint_id) = checkpoints.first() {
                        if manager_clone
                            .restore_full_system(checkpoint_id.clone())
                            .await
                            .is_ok()
                        {
                            count += 1;
                        }
                    }
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            count
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    let mut total_operations = 0;
    for handle in handles {
        let operations = handle.await.unwrap();
        total_operations += operations;
    }

    let actual_duration = start_time.elapsed();
    println!(
        "Stress test: {} operations in {:?}",
        total_operations, actual_duration
    );

    // Should handle mixed concurrent operations without errors
    assert!(total_operations > 0);
    assert!(actual_duration <= duration + Duration::from_millis(100)); // Allow small tolerance

    // Verify system is still functional after stress test
    let final_checkpoints = checkpoint_manager.list_checkpoints().await.unwrap();
    assert!(!final_checkpoints.is_empty());
}
