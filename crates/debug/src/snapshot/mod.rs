//! Memory snapshot system for time-travel debugging
//!
//! This module provides a comprehensive snapshot system for capturing and managing
//! memory states during agent execution. It supports various storage backends,
//! automatic cleanup, and snapshot comparison functionality.

pub mod manager;
pub mod storage;
pub mod types;

// Re-export main types for convenience
pub use types::{
    ExecutionContext, KernelParameters, MemorySnapshot, SnapshotConfig, SnapshotMetadata,
    SnapshotSession, SnapshotStorage, StorageStats,
};

pub use storage::{FileStorage, RingBufferStorage};

pub use manager::{MemoryDiff, SnapshotDiff, SnapshotManager};

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_integration_full_workflow() {
        // Test complete snapshot workflow
        let config = SnapshotConfig {
            auto_snapshot: true,
            default_ttl_seconds: 3600,
            max_snapshot_size_mb: 100,
            compression_enabled: true,
            encryption_enabled: false,
            cleanup_interval_seconds: 300,
        };

        let manager = SnapshotManager::new(config);
        let container_id = Uuid::new_v4();

        // Start debugging session
        manager.start_session(container_id).await.unwrap();

        // Create execution context
        let mut context = ExecutionContext::new("Integration test goal".to_string());
        context.set_agent_info("test_agent_123".to_string(), 5, 2);
        context.add_env_var("GPU_DEVICE".to_string(), "0".to_string());

        // Create kernel parameters
        let kernel_params = KernelParameters::new((8, 8, 1), (256, 1, 1))
            .with_shared_memory(1024)
            .with_stream(42);

        // Create a snapshot
        let snapshot_id = manager
            .create_snapshot(
                container_id,
                vec![1; 1000], // 1KB host memory
                vec![2; 1000], // 1KB device memory
                kernel_params,
                context,
                "Integration test snapshot".to_string(),
            )
            .await
            .unwrap();

        // Retrieve and verify snapshot
        let snapshot = manager.get_snapshot(snapshot_id).await.unwrap();
        assert_eq!(snapshot.container_id, container_id);
        assert_eq!(snapshot.host_memory.len(), 1000);
        assert_eq!(snapshot.device_memory.len(), 1000);
        assert_eq!(
            snapshot.execution_context.agent_id,
            Some("test_agent_123".to_string())
        );
        assert_eq!(snapshot.execution_context.generation, 5);
        assert_eq!(snapshot.execution_context.mutation_count, 2);
        assert_eq!(snapshot.kernel_parameters.grid_size, (8, 8, 1));
        assert_eq!(snapshot.kernel_parameters.shared_memory_size, 1024);
        assert_eq!(snapshot.kernel_parameters.stream_id, Some(42));

        // Test auto snapshot
        let auto_snapshot_id = manager
            .auto_snapshot(
                container_id,
                vec![3; 500],
                vec![4; 500],
                KernelParameters::default(),
                ExecutionContext::new("Auto snapshot test".to_string()),
            )
            .await
            .unwrap();

        assert!(auto_snapshot_id.is_some());

        // List snapshots
        let snapshot_list = manager.list_snapshots(container_id).await.unwrap();
        assert_eq!(snapshot_list.len(), 2);
        assert!(snapshot_list.contains(&snapshot_id));
        assert!(snapshot_list.contains(&auto_snapshot_id.unwrap()));

        // Compare snapshots
        let diff = manager
            .compare_snapshots(snapshot_id, auto_snapshot_id.unwrap())
            .await
            .unwrap();

        match diff.host_memory_diff {
            MemoryDiff::Different { bytes_changed } => {
                assert!(bytes_changed > 0); // Should be different
            }
            MemoryDiff::Identical => panic!("Expected different snapshots"),
        }

        // Get statistics
        let stats = manager.get_stats().await.unwrap();
        assert_eq!(stats.expired_snapshots, 0); // None should be expired yet

        // Clean up session
        manager.end_session(container_id, true).await.unwrap();

        // Verify cleanup
        let session_info = manager.get_session_info(container_id).await;
        assert!(session_info.is_none());
    }

    #[tokio::test]
    async fn test_storage_backend_compatibility() {
        // Test that different storage backends work with the manager
        let config = SnapshotConfig::default();

        // Test with ring buffer storage (default)
        let ring_buffer_manager = SnapshotManager::new(config.clone());
        let container_id1 = Uuid::new_v4();

        ring_buffer_manager
            .start_session(container_id1)
            .await
            .unwrap();
        let snapshot_id1 = ring_buffer_manager
            .create_snapshot(
                container_id1,
                vec![1, 2, 3],
                vec![4, 5, 6],
                KernelParameters::default(),
                ExecutionContext::new("ring buffer test".to_string()),
                "ring buffer snapshot".to_string(),
            )
            .await
            .unwrap();

        let retrieved1 = ring_buffer_manager
            .get_snapshot(snapshot_id1)
            .await
            .unwrap();
        assert_eq!(retrieved1.host_memory, vec![1, 2, 3]);

        // Test with file storage
        let temp_dir = std::env::temp_dir().join("snapshot_tests");
        let file_storage = std::sync::Arc::new(storage::FileStorage::new(temp_dir));
        let file_manager = SnapshotManager::with_storage(config, file_storage);
        let container_id2 = Uuid::new_v4();

        file_manager.start_session(container_id2).await.unwrap();
        let snapshot_id2 = file_manager
            .create_snapshot(
                container_id2,
                vec![7, 8, 9],
                vec![10, 11, 12],
                KernelParameters::default(),
                ExecutionContext::new("file storage test".to_string()),
                "file storage snapshot".to_string(),
            )
            .await
            .unwrap();

        let retrieved2 = file_manager.get_snapshot(snapshot_id2).await.unwrap();
        assert_eq!(retrieved2.host_memory, vec![7, 8, 9]);
    }

    #[tokio::test]
    async fn test_compression_and_decompression() {
        let mut config = SnapshotConfig::default();
        config.compression_enabled = true;

        let manager = SnapshotManager::new(config);
        let container_id = Uuid::new_v4();

        manager.start_session(container_id).await.unwrap();

        // Create snapshot with compression
        let snapshot_id = manager
            .create_snapshot(
                container_id,
                vec![1; 1000], // Repetitive data compresses well
                vec![2; 1000],
                KernelParameters::default(),
                ExecutionContext::new("compression test".to_string()),
                "compression test snapshot".to_string(),
            )
            .await
            .unwrap();

        // Retrieve snapshot (should decompress automatically)
        let snapshot = manager.get_snapshot(snapshot_id).await.unwrap();
        assert_eq!(snapshot.host_memory.len(), 1000);
        assert_eq!(snapshot.device_memory.len(), 1000);
        assert!(snapshot.host_memory.iter().all(|&x| x == 1));
        assert!(snapshot.device_memory.iter().all(|&x| x == 2));
    }

    #[tokio::test]
    async fn test_error_handling() {
        let config = SnapshotConfig::default();
        let manager = SnapshotManager::new(config);
        let container_id = Uuid::new_v4();

        // Test creating snapshot without session
        let result = manager
            .create_snapshot(
                container_id,
                vec![1, 2, 3],
                vec![4, 5, 6],
                KernelParameters::default(),
                ExecutionContext::new("no session test".to_string()),
                "should fail".to_string(),
            )
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            crate::DebugError::SessionNotFound { .. } => {} // Expected
            other => panic!("Unexpected error: {:?}", other),
        }

        // Test retrieving non-existent snapshot
        let fake_id = Uuid::new_v4();
        let result = manager.get_snapshot(fake_id).await;
        assert!(result.is_err());

        // Test deleting non-existent snapshot
        let result = manager.delete_snapshot(fake_id).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_snapshot_metadata_operations() {
        let mut snapshot = MemorySnapshot::new(
            Uuid::new_v4(),
            vec![1, 2, 3],
            vec![4, 5, 6],
            KernelParameters::default(),
            ExecutionContext::new("metadata test".to_string()),
            "testing metadata".to_string(),
        );

        // Test tag operations
        snapshot.add_tag("environment".to_string(), "test".to_string());
        snapshot.add_tag("version".to_string(), "1.0".to_string());

        assert!(snapshot.has_tag("environment", "test"));
        assert!(snapshot.has_tag("version", "1.0"));
        assert!(!snapshot.has_tag("environment", "production"));

        let removed = snapshot.remove_tag("version");
        assert_eq!(removed, Some("1.0".to_string()));
        assert!(!snapshot.has_tag("version", "1.0"));

        // Test size calculation
        assert_eq!(snapshot.total_size(), 6); // 3 + 3 bytes

        // Test expiration
        assert!(!snapshot.is_expired()); // Should not be expired immediately
    }

    #[test]
    fn test_execution_context_operations() {
        let mut context = ExecutionContext::new("Test context operations".to_string());

        // Test environment variables
        context.add_env_var("GPU_DEVICE".to_string(), "0".to_string());
        context.add_env_var("CUDA_VISIBLE_DEVICES".to_string(), "0,1".to_string());

        assert_eq!(context.get_env_var("GPU_DEVICE"), Some(&"0".to_string()));
        assert_eq!(
            context.get_env_var("CUDA_VISIBLE_DEVICES"),
            Some(&"0,1".to_string())
        );
        assert_eq!(context.get_env_var("NON_EXISTENT"), None);

        // Test agent info
        context.set_agent_info("agent_456".to_string(), 10, 5);
        assert_eq!(context.agent_id, Some("agent_456".to_string()));
        assert_eq!(context.generation, 10);
        assert_eq!(context.mutation_count, 5);
    }

    #[test]
    fn test_kernel_parameters_builder() {
        let params = KernelParameters::new((16, 16, 1), (512, 1, 1))
            .with_shared_memory(2048)
            .with_args(vec![0xDE, 0xAD, 0xBE, 0xEF])
            .with_stream(123);

        assert_eq!(params.grid_size, (16, 16, 1));
        assert_eq!(params.block_size, (512, 1, 1));
        assert_eq!(params.shared_memory_size, 2048);
        assert_eq!(params.kernel_args, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(params.stream_id, Some(123));
    }
}
