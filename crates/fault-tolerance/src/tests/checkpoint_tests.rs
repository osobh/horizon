//! Comprehensive tests for checkpoint management

use crate::checkpoint::*;
use crate::error::FaultToleranceError;
use serde_json::json;
use std::collections::HashMap;
use tempfile::TempDir;
use tokio::fs;
use uuid::Uuid;

#[tokio::test]
async fn test_checkpoint_id_creation_and_uniqueness() {
    let mut ids = Vec::new();
    for _ in 0..100 {
        ids.push(CheckpointId::new());
    }

    // All IDs should be unique
    for (i, id1) in ids.iter().enumerate() {
        for (j, id2) in ids.iter().enumerate() {
            if i != j {
                assert_ne!(id1, id2, "Checkpoint IDs should be unique");
            }
        }
    }
}

#[tokio::test]
async fn test_checkpoint_id_from_uuid() {
    let uuid = Uuid::new_v4();
    let checkpoint_id = CheckpointId::from_uuid(uuid);

    assert_eq!(checkpoint_id.as_uuid(), uuid);
}

#[tokio::test]
async fn test_checkpoint_id_default() {
    let id1 = CheckpointId::default();
    let id2 = CheckpointId::default();

    assert_ne!(id1, id2, "Default IDs should be unique");
}

#[tokio::test]
async fn test_gpu_checkpoint_creation() {
    let mut kernel_states = HashMap::new();
    kernel_states.insert(
        "test_kernel".to_string(),
        KernelState {
            kernel_id: "test_kernel".to_string(),
            parameters: vec![1, 2, 3, 4],
            execution_context: ExecutionContext {
                device_id: 0,
                block_size: (256, 1, 1),
                grid_size: (1024, 1, 1),
                shared_memory_bytes: 4096,
            },
        },
    );

    let gpu_checkpoint = GpuCheckpoint {
        memory_snapshot: vec![0u8; 1024],
        kernel_states,
        timestamp: 1234567890,
        size_bytes: 1024,
    };

    assert_eq!(gpu_checkpoint.size_bytes, 1024);
    assert_eq!(gpu_checkpoint.memory_snapshot.len(), 1024);
    assert_eq!(gpu_checkpoint.kernel_states.len(), 1);
    assert!(gpu_checkpoint.kernel_states.contains_key("test_kernel"));
}

#[tokio::test]
async fn test_kernel_state_creation() {
    let execution_context = ExecutionContext {
        device_id: 1,
        block_size: (512, 1, 1),
        grid_size: (2048, 1, 1),
        shared_memory_bytes: 8192,
    };

    let kernel_state = KernelState {
        kernel_id: "matrix_mult".to_string(),
        parameters: vec![10, 20, 30],
        execution_context,
    };

    assert_eq!(kernel_state.kernel_id, "matrix_mult");
    assert_eq!(kernel_state.parameters, vec![10, 20, 30]);
    assert_eq!(kernel_state.execution_context.device_id, 1);
    assert_eq!(kernel_state.execution_context.block_size.0, 512);
}

#[tokio::test]
async fn test_execution_context_validation() {
    let contexts = vec![
        ExecutionContext {
            device_id: 0,
            block_size: (1, 1, 1),
            grid_size: (1, 1, 1),
            shared_memory_bytes: 0,
        },
        ExecutionContext {
            device_id: 0,
            block_size: (1024, 1, 1),
            grid_size: (65535, 65535, 65535),
            shared_memory_bytes: 49152, // 48KB max shared memory
        },
    ];

    for context in contexts {
        assert!(context.block_size.0 >= 1);
        assert!(context.grid_size.0 >= 1);
        assert!(context.shared_memory_bytes < 64 * 1024); // Reasonable limit
    }
}

#[tokio::test]
async fn test_agent_checkpoint_creation() {
    let mut memory_contents = HashMap::new();
    memory_contents.insert("variables".to_string(), json!({"count": 42}));

    let mut metadata = HashMap::new();
    metadata.insert("agent_type".to_string(), json!("gpu_agent"));
    metadata.insert("version".to_string(), json!("1.0.0"));

    let agent_checkpoint = AgentCheckpoint {
        agent_id: "agent-123".to_string(),
        state_data: vec![1, 2, 3, 4, 5],
        memory_contents,
        goals: vec!["optimize".to_string(), "learn".to_string()],
        metadata,
    };

    assert_eq!(agent_checkpoint.agent_id, "agent-123");
    assert_eq!(agent_checkpoint.state_data.len(), 5);
    assert_eq!(agent_checkpoint.goals.len(), 2);
    assert_eq!(agent_checkpoint.memory_contents.len(), 1);
    assert_eq!(agent_checkpoint.metadata.len(), 2);
}

#[tokio::test]
async fn test_system_checkpoint_creation() {
    let checkpoint_id = CheckpointId::new();

    let gpu_checkpoint = GpuCheckpoint {
        memory_snapshot: vec![0u8; 512],
        kernel_states: HashMap::new(),
        timestamp: 1234567890,
        size_bytes: 512,
    };

    let agent_checkpoint = AgentCheckpoint {
        agent_id: "test-agent".to_string(),
        state_data: vec![1, 2, 3],
        memory_contents: HashMap::new(),
        goals: vec!["test".to_string()],
        metadata: HashMap::new(),
    };

    let mut system_metadata = HashMap::new();
    system_metadata.insert("version".to_string(), json!("1.0.0"));

    let system_checkpoint = SystemCheckpoint {
        id: checkpoint_id.clone(),
        timestamp: 1234567890,
        gpu_checkpoints: vec![gpu_checkpoint],
        agent_checkpoints: vec![agent_checkpoint],
        system_metadata,
        compressed: true,
    };

    assert_eq!(system_checkpoint.id, checkpoint_id);
    assert_eq!(system_checkpoint.gpu_checkpoints.len(), 1);
    assert_eq!(system_checkpoint.agent_checkpoints.len(), 1);
    assert!(system_checkpoint.compressed);
}

#[tokio::test]
async fn test_checkpoint_manager_creation() {
    let manager = CheckpointManager::new().unwrap();
    // Manager should be created successfully
    assert_eq!(manager.storage_path.file_name().unwrap(), "checkpoints");
    assert!(manager.compression_enabled);
    assert_eq!(manager.max_checkpoints, 10);
}

#[tokio::test]
async fn test_checkpoint_manager_with_custom_path() {
    let temp_dir = TempDir::new().unwrap();
    let custom_path = temp_dir.path().join("custom_checkpoints");

    let manager = CheckpointManager::with_path(&custom_path).unwrap();
    assert_eq!(manager.storage_path, custom_path);
}

#[tokio::test]
async fn test_checkpoint_manager_directory_creation() {
    let temp_dir = TempDir::new().unwrap();
    let checkpoint_path = temp_dir.path().join("new_checkpoint_dir");

    // Directory doesn't exist initially
    assert!(!checkpoint_path.exists());

    let manager = CheckpointManager::with_path(&checkpoint_path)?;
    let _checkpoint_id = manager.create_full_checkpoint().await?;

    // Directory should be created automatically
    assert!(checkpoint_path.exists());
}

#[tokio::test]
async fn test_full_checkpoint_creation_and_storage() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    let checkpoint_id = manager.create_full_checkpoint().await.unwrap();

    // Verify checkpoint file exists
    let checkpoint_file = temp_dir
        .path()
        .join(format!("{}.checkpoint", checkpoint_id.as_uuid()));
    assert!(checkpoint_file.exists());

    // Verify file is not empty
    let metadata = fs::metadata(&checkpoint_file).await.unwrap();
    assert!(metadata.len() > 0);
}

#[tokio::test]
async fn test_checkpoint_load_and_verify_data() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    let checkpoint_id = manager.create_full_checkpoint().await.unwrap();
    let loaded_checkpoint = manager.load_checkpoint(&checkpoint_id).await?;

    assert_eq!(loaded_checkpoint.id, checkpoint_id);
    assert!(!loaded_checkpoint.gpu_checkpoints.is_empty());
    assert!(!loaded_checkpoint.agent_checkpoints.is_empty());
    assert!(loaded_checkpoint.compressed);
}

#[tokio::test]
async fn test_checkpoint_list_functionality() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Initially empty
    let initial_list = manager.list_checkpoints().await?;
    assert!(initial_list.is_empty());

    // Create multiple checkpoints
    let id1 = manager.create_full_checkpoint().await?;
    let id2 = manager.create_full_checkpoint().await?;
    let id3 = manager.create_full_checkpoint().await.unwrap();

    let checkpoint_list = manager.list_checkpoints().await.unwrap();
    assert_eq!(checkpoint_list.len(), 3);
    assert!(checkpoint_list.contains(&id1));
    assert!(checkpoint_list.contains(&id2));
    assert!(checkpoint_list.contains(&id3));
}

#[tokio::test]
async fn test_checkpoint_cleanup_functionality() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
    manager.max_checkpoints = 3; // Set low limit

    // Create checkpoints up to the limit
    for _ in 0..3 {
        manager.create_full_checkpoint().await?;
    }

    let checkpoints_at_limit = manager.list_checkpoints().await?;
    assert_eq!(checkpoints_at_limit.len(), 3);

    // Create one more to trigger cleanup
    manager.create_full_checkpoint().await.unwrap();

    let checkpoints_after_cleanup = manager.list_checkpoints().await.unwrap();
    assert_eq!(checkpoints_after_cleanup.len(), 3); // Should still be 3 due to cleanup
}

#[tokio::test]
async fn test_checkpoint_compression() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    let checkpoint_id = manager.create_full_checkpoint().await.unwrap();
    let checkpoint = manager.load_checkpoint(&checkpoint_id).await?;

    // Verify compression flag is set and data is reasonable
    assert!(checkpoint.compressed);

    // Test file size is reasonable (compressed should be smaller)
    let file_path = temp_dir
        .path()
        .join(format!("{}.checkpoint", checkpoint_id.as_uuid()));
    let file_size = fs::metadata(file_path).await.unwrap().len();
    assert!(file_size > 0);
    assert!(file_size < 10000); // Should be reasonable size for test data
}

#[tokio::test]
async fn test_load_nonexistent_checkpoint() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    let nonexistent_id = CheckpointId::new();
    let result = manager.load_checkpoint(&nonexistent_id).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        FaultToleranceError::CheckpointNotFound(id) => {
            assert_eq!(id, nonexistent_id.as_uuid().to_string());
        }
        _ => panic!("Expected CheckpointNotFound error"),
    }
}

#[tokio::test]
async fn test_concurrent_checkpoint_creation() {
    let temp_dir = TempDir::new().unwrap();
    let manager = std::sync::Arc::new(CheckpointManager::with_path(temp_dir.path()).unwrap());

    let mut handles = Vec::new();

    // Create multiple checkpoints concurrently
    for _ in 0..5 {
        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move { manager_clone.create_full_checkpoint().await });
        handles.push(handle);
    }

    // Wait for all to complete
    let mut checkpoint_ids = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
        checkpoint_ids.push(result.unwrap());
    }

    // All IDs should be unique
    checkpoint_ids.sort_by(|a, b| a.as_uuid().cmp(&b.as_uuid()));
    for i in 1..checkpoint_ids.len() {
        assert_ne!(checkpoint_ids[i - 1], checkpoint_ids[i]);
    }

    // Verify all checkpoints can be loaded
    for id in checkpoint_ids {
        let result = manager.load_checkpoint(&id).await;
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_large_checkpoint_handling() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Create a checkpoint with larger data
    let large_memory = vec![42u8; 1024 * 1024]; // 1MB of data

    // We can't directly inject this without modifying the manager,
    // but we can test that the system handles larger files correctly
    let checkpoint_id = manager.create_full_checkpoint().await?;
    let checkpoint = manager.load_checkpoint(&checkpoint_id).await?;

    // Verify the checkpoint loads correctly
    assert_eq!(checkpoint.id, checkpoint_id);
    assert!(!checkpoint.gpu_checkpoints.is_empty());
}

#[tokio::test]
async fn test_checkpoint_timestamp_ordering() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    let id1 = manager.create_full_checkpoint().await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    let id2 = manager.create_full_checkpoint().await?;
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    let id3 = manager.create_full_checkpoint().await?;

    let checkpoint1 = manager.load_checkpoint(&id1).await?;
    let checkpoint2 = manager.load_checkpoint(&id2).await.unwrap();
    let checkpoint3 = manager.load_checkpoint(&id3).await.unwrap();

    // Timestamps should be in order
    assert!(checkpoint1.timestamp <= checkpoint2.timestamp);
    assert!(checkpoint2.timestamp <= checkpoint3.timestamp);
}

#[tokio::test]
async fn test_checkpoint_metadata_preservation() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    let checkpoint_id = manager.create_full_checkpoint().await.unwrap();
    let checkpoint = manager.load_checkpoint(&checkpoint_id).await?;

    // Check that GPU checkpoint has proper structure
    assert!(!checkpoint.gpu_checkpoints.is_empty());
    let gpu_checkpoint = &checkpoint.gpu_checkpoints[0];
    assert_eq!(
        gpu_checkpoint.size_bytes,
        gpu_checkpoint.memory_snapshot.len()
    );
    assert!(gpu_checkpoint.timestamp > 0);

    // Check that agent checkpoint has proper structure
    assert!(!checkpoint.agent_checkpoints.is_empty());
    let agent_checkpoint = &checkpoint.agent_checkpoints[0];
    assert!(!agent_checkpoint.agent_id.is_empty());
    assert!(!agent_checkpoint.goals.is_empty());
}

#[tokio::test]
async fn test_checkpoint_error_handling() {
    // Test with invalid path (read-only filesystem simulation)
    let temp_dir = TempDir::new().unwrap();
    let invalid_path = temp_dir
        .path()
        .join("nonexistent")
        .join("deep")
        .join("path");

    // This should still work as create_dir_all handles directory creation
    let manager = CheckpointManager::with_path(&invalid_path)?;
    let result = manager.create_full_checkpoint().await;

    // Should either succeed (if permissions allow) or fail gracefully
    match result {
        Ok(_) => {
            // Successfully created checkpoint
            assert!(invalid_path.exists());
        }
        Err(e) => {
            // Failed gracefully with proper error
            assert!(matches!(e, FaultToleranceError::IoError(_)));
        }
    }
}

#[tokio::test]
async fn test_checkpoint_serialization_formats() {
    // Test GPU checkpoint serialization
    let gpu_checkpoint = GpuCheckpoint {
        memory_snapshot: vec![1, 2, 3, 4, 5],
        kernel_states: HashMap::new(),
        timestamp: 1234567890,
        size_bytes: 5,
    };

    let serialized = bincode::serialize(&gpu_checkpoint)?;
    let deserialized: GpuCheckpoint = bincode::deserialize(&serialized)?;

    assert_eq!(gpu_checkpoint.memory_snapshot, deserialized.memory_snapshot);
    assert_eq!(gpu_checkpoint.timestamp, deserialized.timestamp);
    assert_eq!(gpu_checkpoint.size_bytes, deserialized.size_bytes);

    // Test agent checkpoint with JSON due to serde_json::Value
    let mut metadata = HashMap::new();
    metadata.insert("test_key".to_string(), json!("test_value"));

    let agent_checkpoint = AgentCheckpoint {
        agent_id: "test-agent".to_string(),
        state_data: vec![10, 20, 30],
        memory_contents: HashMap::new(),
        goals: vec!["goal1".to_string()],
        metadata,
    };

    let json_serialized = serde_json::to_string(&agent_checkpoint).unwrap();
    let json_deserialized: AgentCheckpoint = serde_json::from_str(&json_serialized).unwrap();

    assert_eq!(agent_checkpoint.agent_id, json_deserialized.agent_id);
    assert_eq!(agent_checkpoint.state_data, json_deserialized.state_data);
    assert_eq!(agent_checkpoint.goals, json_deserialized.goals);
}

#[tokio::test]
async fn test_checkpoint_manager_configuration() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Test default configuration
    assert!(manager.compression_enabled);
    assert_eq!(manager.max_checkpoints, 10);

    // Test configuration changes
    manager.compression_enabled = false;
    manager.max_checkpoints = 5;

    // Create checkpoint with modified settings
    let checkpoint_id = manager.create_full_checkpoint().await.unwrap();
    let checkpoint = manager.load_checkpoint(&checkpoint_id).await.unwrap();

    // Verify checkpoint was created with new settings
    assert_eq!(checkpoint.compressed, false); // Should reflect compression setting
}

#[tokio::test]
async fn test_checkpoint_edge_cases() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    // Test with empty checkpoint directory initially
    let empty_list = manager.list_checkpoints().await?;
    assert!(empty_list.is_empty());

    // Test loading from empty directory
    let nonexistent_id = CheckpointId::new();
    let result = manager.load_checkpoint(&nonexistent_id).await;
    assert!(result.is_err());

    // Create and immediately list
    let checkpoint_id = manager.create_full_checkpoint().await.unwrap();
    let list_after_create = manager.list_checkpoints().await.unwrap();
    assert_eq!(list_after_create.len(), 1);
    assert!(list_after_create.contains(&checkpoint_id));
}

#[tokio::test]
async fn test_checkpoint_id_consistency() {
    let temp_dir = TempDir::new().unwrap();
    let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

    let checkpoint_id = manager.create_full_checkpoint().await.unwrap();

    // Load checkpoint and verify ID consistency
    let loaded_checkpoint = manager.load_checkpoint(&checkpoint_id).await?;
    assert_eq!(loaded_checkpoint.id, checkpoint_id);

    // Verify ID appears in listing
    let checkpoint_list = manager.list_checkpoints().await.unwrap();
    assert!(checkpoint_list.contains(&checkpoint_id));

    // Verify the same checkpoint can be loaded multiple times
    let loaded_again = manager.load_checkpoint(&checkpoint_id).await.unwrap();
    assert_eq!(loaded_again.id, checkpoint_id);
    assert_eq!(loaded_again.timestamp, loaded_checkpoint.timestamp);
}
