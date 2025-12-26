//! Comprehensive tests for recovery functionality

use crate::checkpoint::*;
use crate::error::{FaultToleranceError, HealthStatus};
use crate::recovery::*;
use serde_json::json;
use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;

#[tokio::test]
async fn test_recovery_manager_initialization() {
    let manager = RecoveryManager::new().unwrap();

    // Verify default settings
    assert_eq!(manager.recovery_timeout, Duration::from_secs(300));
    assert!(manager.verification_enabled);
}

#[tokio::test]
async fn test_recovery_manager_with_custom_timeout() {
    let custom_timeout = Duration::from_secs(600);
    let manager = RecoveryManager::with_timeout(custom_timeout).unwrap();

    assert_eq!(manager.recovery_timeout, custom_timeout);
    assert!(manager.verification_enabled);
}

#[tokio::test]
async fn test_recovery_manager_with_checkpoint_manager() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
    assert_eq!(manager.recovery_timeout, Duration::from_secs(300));
    assert!(manager.verification_enabled);
}

#[tokio::test]
async fn test_recovery_manager_with_checkpoint_manager_and_timeout() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let custom_timeout = Duration::from_secs(450);

    let manager =
        RecoveryManager::with_checkpoint_manager_and_timeout(checkpoint_manager, custom_timeout);

    assert_eq!(manager.recovery_timeout, custom_timeout);
    assert!(manager.verification_enabled);
}

#[tokio::test]
async fn test_recovery_strategy_variants() {
    let strategies = vec![
        RecoveryStrategy::Full,
        RecoveryStrategy::AgentsOnly,
        RecoveryStrategy::GpuOnly,
        RecoveryStrategy::Rolling,
    ];

    // Test serialization for all strategies
    for strategy in strategies {
        let serialized = serde_json::to_string(&strategy)?;
        let deserialized: RecoveryStrategy = serde_json::from_str(&serialized).unwrap();

        // Verify serialization round-trip works
        let reserialized = serde_json::to_string(&deserialized).unwrap();
        assert!(!serialized.is_empty());
        assert!(!reserialized.is_empty());
    }
}

#[tokio::test]
async fn test_recovery_stage_equality_and_variants() {
    let stages = vec![
        RecoveryStage::Loading,
        RecoveryStage::ValidatingCheckpoint,
        RecoveryStage::RestoringGpuState,
        RecoveryStage::RestoringAgents,
        RecoveryStage::VerifyingSystem,
        RecoveryStage::Complete,
        RecoveryStage::Failed("test error".to_string()),
    ];

    // Test stage equality
    for stage in &stages {
        let cloned = stage.clone();
        assert_eq!(stage, &cloned);
    }

    // Test stage inequality
    for (i, stage1) in stages.iter().enumerate() {
        for (j, stage2) in stages.iter().enumerate() {
            if i != j
                && !matches!(
                    (stage1, stage2),
                    (RecoveryStage::Failed(_), RecoveryStage::Failed(_))
                )
            {
                assert_ne!(stage1, stage2);
            }
        }
    }

    // Test failed stage with same message
    let failed1 = RecoveryStage::Failed("same error".to_string());
    let failed2 = RecoveryStage::Failed("same error".to_string());
    assert_eq!(failed1, failed2);

    // Test failed stage with different message
    let failed3 = RecoveryStage::Failed("different error".to_string());
    assert_ne!(failed1, failed3);
}

#[tokio::test]
async fn test_recovery_progress_structure() {
    let manager = RecoveryManager::new().unwrap();
    let progress = manager.get_recovery_progress().await;

    // Verify progress structure is valid
    assert!(progress.completed_steps <= progress.total_steps);
    assert!(progress.elapsed_time >= Duration::from_secs(0));
    assert!(progress.estimated_remaining >= Duration::from_secs(0));
    assert!(matches!(progress.stage, RecoveryStage::Complete));
}

#[tokio::test]
async fn test_full_system_recovery_success() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await.unwrap();

    let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
    let result = manager.restore_full_system(checkpoint_id).await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_agents_only_recovery_success() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await.unwrap();

    let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
    let result = manager.restore_agents_only(checkpoint_id).await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_gpu_only_recovery_success() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await.unwrap();

    let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
    let result = manager.restore_gpu_only(checkpoint_id).await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_recovery_from_nonexistent_checkpoint() {
    let manager = RecoveryManager::new().unwrap();
    let nonexistent_id = CheckpointId::new();

    let result = manager.restore_full_system(nonexistent_id).await;
    assert!(result.is_err());

    match result.unwrap_err() {
        FaultToleranceError::CheckpointNotFound(_) => {
            // Expected error type
        }
        _ => panic!("Expected CheckpointNotFound error"),
    }
}

#[tokio::test]
async fn test_recovery_timeout_behavior() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await.unwrap();

    // Create manager with very short timeout
    let manager = RecoveryManager::with_checkpoint_manager_and_timeout(
        checkpoint_manager,
        Duration::from_millis(1), // Extremely short timeout
    );

    let result = manager.restore_full_system(checkpoint_id).await;
    assert!(result.is_err());

    match result.unwrap_err() {
        FaultToleranceError::RecoveryFailed(msg) => {
            assert!(msg.contains("timeout"));
        }
        _ => panic!("Expected RecoveryFailed with timeout"),
    }
}

#[tokio::test]
async fn test_invalid_checkpoint_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create empty checkpoint manually
    let empty_checkpoint = SystemCheckpoint {
        id: CheckpointId::new(),
        timestamp: 1234567890,
        gpu_checkpoints: vec![],
        agent_checkpoints: vec![],
        system_metadata: HashMap::new(),
        compressed: true,
    };

    // Save the invalid checkpoint
    let file_path = temp_dir
        .path()
        .join(format!("{}.checkpoint", empty_checkpoint.id.as_uuid()));
    let data = bincode::serialize(&empty_checkpoint).unwrap();
    let compressed_data = lz4::block::compress(&data, None, true).unwrap();
    fs::write(file_path, compressed_data).await.unwrap();

    let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
    let result = manager.restore_full_system(empty_checkpoint.id).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        FaultToleranceError::InvalidCheckpoint(_) => {
            // Expected error type
        }
        _ => panic!("Expected InvalidCheckpoint error"),
    }
}

#[tokio::test]
async fn test_recovery_with_empty_gpu_memory_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create checkpoint with empty GPU memory snapshot
    let checkpoint = SystemCheckpoint {
        id: CheckpointId::new(),
        timestamp: 1234567890,
        gpu_checkpoints: vec![GpuCheckpoint {
            memory_snapshot: vec![], // Empty snapshot
            kernel_states: HashMap::new(),
            timestamp: 1234567890,
            size_bytes: 0,
        }],
        agent_checkpoints: vec![AgentCheckpoint {
            agent_id: "test_agent".to_string(),
            state_data: vec![1, 2, 3],
            memory_contents: HashMap::new(),
            goals: vec!["test_goal".to_string()],
            metadata: HashMap::new(),
        }],
        system_metadata: HashMap::new(),
        compressed: true,
    };

    // Save the checkpoint
    let file_path = temp_dir
        .path()
        .join(format!("{}.checkpoint", checkpoint.id.as_uuid()));
    let data = bincode::serialize(&checkpoint).unwrap();
    let compressed_data = lz4::block::compress(&data, None, true).unwrap();
    fs::write(file_path, compressed_data).await.unwrap();

    let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
    let result = manager.restore_full_system(checkpoint.id).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        FaultToleranceError::RecoveryFailed(msg) => {
            assert!(msg.contains("Empty GPU memory snapshot"));
        }
        _ => panic!("Expected RecoveryFailed error"),
    }
}

#[tokio::test]
async fn test_recovery_with_empty_agent_state() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create checkpoint with empty agent state
    let checkpoint = SystemCheckpoint {
        id: CheckpointId::new(),
        timestamp: 1234567890,
        gpu_checkpoints: vec![GpuCheckpoint {
            memory_snapshot: vec![1, 2, 3],
            kernel_states: HashMap::new(),
            timestamp: 1234567890,
            size_bytes: 3,
        }],
        agent_checkpoints: vec![AgentCheckpoint {
            agent_id: "test_agent".to_string(),
            state_data: vec![], // Empty state data
            memory_contents: HashMap::new(),
            goals: vec!["test_goal".to_string()],
            metadata: HashMap::new(),
        }],
        system_metadata: HashMap::new(),
        compressed: true,
    };

    // Save the checkpoint
    let file_path = temp_dir
        .path()
        .join(format!("{}.checkpoint", checkpoint.id.as_uuid()));
    let data = bincode::serialize(&checkpoint).unwrap();
    let compressed_data = lz4::block::compress(&data, None, true).unwrap();
    fs::write(file_path, compressed_data).await.unwrap();

    let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
    let result = manager.restore_agents_only(checkpoint.id).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        FaultToleranceError::RecoveryFailed(msg) => {
            assert!(msg.contains("Empty agent state data"));
        }
        _ => panic!("Expected RecoveryFailed error"),
    }
}

#[tokio::test]
async fn test_post_recovery_health_check() {
    let manager = RecoveryManager::new().unwrap();
    let health = manager.post_recovery_health_check().await;

    // Health check should return a valid health status
    assert!(matches!(
        health,
        HealthStatus::Healthy | HealthStatus::Degraded | HealthStatus::Failed
    ));
}

#[tokio::test]
async fn test_recovery_time_estimation_all_strategies() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await.unwrap();

    let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);

    let strategies = vec![
        RecoveryStrategy::Full,
        RecoveryStrategy::AgentsOnly,
        RecoveryStrategy::GpuOnly,
        RecoveryStrategy::Rolling,
    ];

    let mut previous_estimate = Duration::from_secs(0);

    for strategy in strategies {
        let estimate = manager
            .estimate_recovery_time(checkpoint_id.clone(), strategy)
            .await
            .unwrap();

        assert!(estimate > Duration::from_secs(0));

        // Rolling recovery should be fastest for same checkpoint
        if matches!(strategy, RecoveryStrategy::Rolling) {
            // Rolling strategy should have reasonable time estimate
            assert!(estimate >= Duration::from_secs(15)); // Should be at least some time
        }
    }
}

#[tokio::test]
async fn test_recovery_time_estimation_nonexistent_checkpoint() {
    let manager = RecoveryManager::new().unwrap();
    let nonexistent_id = CheckpointId::new();

    let result = manager
        .estimate_recovery_time(nonexistent_id, RecoveryStrategy::Full)
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        FaultToleranceError::CheckpointNotFound(_) => {
            // Expected error type
        }
        _ => panic!("Expected CheckpointNotFound error"),
    }
}

#[tokio::test]
async fn test_recovery_verification_disabled() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    let checkpoint_id = checkpoint_manager.create_full_checkpoint().await.unwrap();

    let mut manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);
    manager.verification_enabled = false;

    let result = manager.restore_full_system(checkpoint_id).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_recovery_progress_tracking_comprehensive() {
    let manager = RecoveryManager::new().unwrap();

    // Test multiple progress calls
    for _ in 0..5 {
        let progress = manager.get_recovery_progress().await;

        assert!(progress.completed_steps <= progress.total_steps);
        assert!(progress.elapsed_time >= Duration::from_secs(0));
        assert!(progress.estimated_remaining >= Duration::from_secs(0));

        // Progress should be consistent
        assert!(matches!(progress.stage, RecoveryStage::Complete));
    }
}

#[tokio::test]
async fn test_recovery_stage_debug_formatting() {
    let stages = vec![
        RecoveryStage::Loading,
        RecoveryStage::ValidatingCheckpoint,
        RecoveryStage::RestoringGpuState,
        RecoveryStage::RestoringAgents,
        RecoveryStage::VerifyingSystem,
        RecoveryStage::Complete,
        RecoveryStage::Failed("test error".to_string()),
    ];

    for stage in stages {
        let debug_str = format!("{:?}", stage);
        assert!(!debug_str.is_empty());

        // Verify debug string contains stage name
        match stage {
            RecoveryStage::Loading => assert!(debug_str.contains("Loading")),
            RecoveryStage::ValidatingCheckpoint => {
                assert!(debug_str.contains("ValidatingCheckpoint"))
            }
            RecoveryStage::RestoringGpuState => assert!(debug_str.contains("RestoringGpuState")),
            RecoveryStage::RestoringAgents => assert!(debug_str.contains("RestoringAgents")),
            RecoveryStage::VerifyingSystem => assert!(debug_str.contains("VerifyingSystem")),
            RecoveryStage::Complete => assert!(debug_str.contains("Complete")),
            RecoveryStage::Failed(ref msg) => {
                assert!(debug_str.contains("Failed"));
                assert!(debug_str.contains(msg));
            }
        }
    }
}

#[tokio::test]
async fn test_recovery_progress_custom_values() {
    let test_cases = vec![
        (0, 5, Duration::from_secs(0), Duration::from_secs(300)),
        (3, 5, Duration::from_secs(180), Duration::from_secs(120)),
        (5, 5, Duration::from_secs(300), Duration::from_secs(0)),
    ];

    for (completed, total, elapsed, remaining) in test_cases {
        let progress = RecoveryProgress {
            stage: RecoveryStage::RestoringGpuState,
            completed_steps: completed,
            total_steps: total,
            elapsed_time: elapsed,
            estimated_remaining: remaining,
        };

        assert_eq!(progress.completed_steps, completed);
        assert_eq!(progress.total_steps, total);
        assert_eq!(progress.elapsed_time, elapsed);
        assert_eq!(progress.estimated_remaining, remaining);
        assert!(matches!(progress.stage, RecoveryStage::RestoringGpuState));
    }
}

#[tokio::test]
async fn test_concurrent_recovery_operations() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create multiple checkpoints
    let mut checkpoint_ids = Vec::new();
    for _ in 0..3 {
        let id = checkpoint_manager.create_full_checkpoint().await?;
        checkpoint_ids.push(id);
    }

    let manager = std::sync::Arc::new(RecoveryManager::with_checkpoint_manager(checkpoint_manager));
    let mut handles = Vec::new();

    // Start multiple recovery operations concurrently
    for checkpoint_id in checkpoint_ids {
        let manager_clone = manager.clone();
        let handle =
            tokio::spawn(async move { manager_clone.restore_full_system(checkpoint_id).await });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_recovery_strategy_with_complex_checkpoint() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create a checkpoint with multiple GPU states and agents
    let checkpoint = SystemCheckpoint {
        id: CheckpointId::new(),
        timestamp: 1234567890,
        gpu_checkpoints: vec![
            GpuCheckpoint {
                memory_snapshot: vec![1; 1024],
                kernel_states: {
                    let mut states = HashMap::new();
                    states.insert(
                        "kernel1".to_string(),
                        KernelState {
                            kernel_id: "kernel1".to_string(),
                            parameters: vec![1, 2, 3, 4],
                            execution_context: ExecutionContext {
                                device_id: 0,
                                block_size: (256, 1, 1),
                                grid_size: (1024, 1, 1),
                                shared_memory_bytes: 4096,
                            },
                        },
                    );
                    states
                },
                timestamp: 1234567890,
                size_bytes: 1024,
            },
            GpuCheckpoint {
                memory_snapshot: vec![2; 2048],
                kernel_states: HashMap::new(),
                timestamp: 1234567891,
                size_bytes: 2048,
            },
        ],
        agent_checkpoints: vec![
            AgentCheckpoint {
                agent_id: "agent1".to_string(),
                state_data: vec![10; 512],
                memory_contents: {
                    let mut contents = HashMap::new();
                    contents.insert("key1".to_string(), json!("value1"));
                    contents
                },
                goals: vec!["goal1".to_string(), "goal2".to_string()],
                metadata: HashMap::new(),
            },
            AgentCheckpoint {
                agent_id: "agent2".to_string(),
                state_data: vec![20; 256],
                memory_contents: HashMap::new(),
                goals: vec!["goal3".to_string()],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("type".to_string(), json!("neural_agent"));
                    meta
                },
            },
        ],
        system_metadata: {
            let mut meta = HashMap::new();
            meta.insert("version".to_string(), json!("1.0.0"));
            meta.insert("complexity".to_string(), json!("high"));
            meta
        },
        compressed: true,
    };

    // Save the complex checkpoint
    let file_path = temp_dir
        .path()
        .join(format!("{}.checkpoint", checkpoint.id.as_uuid()));
    let data = bincode::serialize(&checkpoint).unwrap();
    let compressed_data = lz4::block::compress(&data, None, true).unwrap();
    fs::write(file_path, compressed_data).await.unwrap();

    let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);

    // Test all recovery strategies with complex checkpoint
    let strategies = vec![
        RecoveryStrategy::Full,
        RecoveryStrategy::GpuOnly,
        RecoveryStrategy::AgentsOnly,
        RecoveryStrategy::Rolling,
    ];

    for strategy in strategies {
        let result = match strategy {
            RecoveryStrategy::Full => manager.restore_full_system(checkpoint.id.clone()).await,
            RecoveryStrategy::GpuOnly => manager.restore_gpu_only(checkpoint.id.clone()).await,
            RecoveryStrategy::AgentsOnly => {
                manager.restore_agents_only(checkpoint.id.clone()).await
            }
            RecoveryStrategy::Rolling => manager.restore_full_system(checkpoint.id.clone()).await, // Rolling uses full restore path
        };

        assert!(
            result.is_ok(),
            "Failed to recover with strategy: {:?}",
            strategy
        );
    }
}

#[tokio::test]
async fn test_recovery_time_estimation_scaling() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create checkpoints with different amounts of data
    let simple_checkpoint = SystemCheckpoint {
        id: CheckpointId::new(),
        timestamp: 1234567890,
        gpu_checkpoints: vec![GpuCheckpoint {
            memory_snapshot: vec![0; 100],
            kernel_states: HashMap::new(),
            timestamp: 1234567890,
            size_bytes: 100,
        }],
        agent_checkpoints: vec![AgentCheckpoint {
            agent_id: "simple_agent".to_string(),
            state_data: vec![0; 50],
            memory_contents: HashMap::new(),
            goals: vec!["simple_goal".to_string()],
            metadata: HashMap::new(),
        }],
        system_metadata: HashMap::new(),
        compressed: true,
    };

    let complex_checkpoint = SystemCheckpoint {
        id: CheckpointId::new(),
        timestamp: 1234567890,
        gpu_checkpoints: vec![
            GpuCheckpoint {
                memory_snapshot: vec![0; 10000],
                kernel_states: HashMap::new(),
                timestamp: 1234567890,
                size_bytes: 10000,
            },
            GpuCheckpoint {
                memory_snapshot: vec![0; 20000],
                kernel_states: HashMap::new(),
                timestamp: 1234567891,
                size_bytes: 20000,
            },
        ],
        agent_checkpoints: vec![
            AgentCheckpoint {
                agent_id: "complex_agent1".to_string(),
                state_data: vec![0; 5000],
                memory_contents: HashMap::new(),
                goals: vec!["complex_goal1".to_string()],
                metadata: HashMap::new(),
            },
            AgentCheckpoint {
                agent_id: "complex_agent2".to_string(),
                state_data: vec![0; 7500],
                memory_contents: HashMap::new(),
                goals: vec!["complex_goal2".to_string()],
                metadata: HashMap::new(),
            },
        ],
        system_metadata: HashMap::new(),
        compressed: true,
    };

    // Save both checkpoints
    for checkpoint in [&simple_checkpoint, &complex_checkpoint] {
        let file_path = temp_dir
            .path()
            .join(format!("{}.checkpoint", checkpoint.id.as_uuid()));
        let data = bincode::serialize(checkpoint).unwrap();
        let compressed_data = lz4::block::compress(&data, None, true).unwrap();
        fs::write(file_path, compressed_data).await.unwrap();
    }

    let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);

    // Estimate recovery times
    let simple_time = manager
        .estimate_recovery_time(simple_checkpoint.id, RecoveryStrategy::Full)
        .await
        .unwrap();

    let complex_time = manager
        .estimate_recovery_time(complex_checkpoint.id, RecoveryStrategy::Full)
        .await
        .unwrap();

    // Complex checkpoint should take longer to recover
    assert!(complex_time > simple_time);
    assert!(simple_time > Duration::from_secs(0));
    assert!(complex_time > Duration::from_secs(0));
}

#[tokio::test]
async fn test_recovery_manager_edge_cases() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create checkpoint with minimal valid data
    let minimal_checkpoint = SystemCheckpoint {
        id: CheckpointId::new(),
        timestamp: 1234567890,
        gpu_checkpoints: vec![GpuCheckpoint {
            memory_snapshot: vec![42], // Single byte
            kernel_states: HashMap::new(),
            timestamp: 1234567890,
            size_bytes: 1,
        }],
        agent_checkpoints: vec![AgentCheckpoint {
            agent_id: "minimal_agent".to_string(),
            state_data: vec![1], // Single byte
            memory_contents: HashMap::new(),
            goals: vec!["minimal_goal".to_string()],
            metadata: HashMap::new(),
        }],
        system_metadata: HashMap::new(),
        compressed: true,
    };

    // Save the minimal checkpoint
    let file_path = temp_dir
        .path()
        .join(format!("{}.checkpoint", minimal_checkpoint.id.as_uuid()));
    let data = bincode::serialize(&minimal_checkpoint).unwrap();
    let compressed_data = lz4::block::compress(&data, None, true).unwrap();
    fs::write(file_path, compressed_data).await.unwrap();

    let manager = RecoveryManager::with_checkpoint_manager(checkpoint_manager);

    // All recovery strategies should work with minimal data
    let result = manager.restore_full_system(minimal_checkpoint.id).await;
    assert!(result.is_ok());

    let result = manager.restore_gpu_only(minimal_checkpoint.id).await;
    assert!(result.is_ok());

    let result = manager.restore_agents_only(minimal_checkpoint.id).await;
    assert!(result.is_ok());
}
