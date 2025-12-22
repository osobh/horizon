//! Async storage operation tests following TDD RED phase
//! These tests define the expected behavior for async storage operations

#[cfg(test)]
mod async_tests {
    use crate::error::StorageError;
    use crate::graph_format::NodeRecord;
    use crate::graph_storage::GraphStorage;
    use crate::graph_wal::GraphWAL;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::time::Duration;

    /// TDD RED: Test concurrent async write operations should not deadlock
    #[tokio::test]
    async fn test_concurrent_async_writes() {
        let dir = tempdir().unwrap();
        let storage = GraphStorage::create(dir.path().to_path_buf(), 1000)
            .await
            .expect("Failed to create storage");

        let storage = Arc::new(storage);

        // Spawn multiple concurrent write operations
        let mut handles = Vec::new();
        for i in 0..10 {
            let storage_clone = Arc::clone(&storage);
            let handle = tokio::spawn(async move {
                let node = NodeRecord::new(i, i as u32);

                storage_clone.write_node(&node).await
            });
            handles.push(handle);
        }

        // All operations should complete without deadlock
        let results: Vec<Result<(), StorageError>> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // All writes should succeed
        for result in results {
            assert!(result.is_ok(), "Concurrent write should succeed");
        }
    }

    /// TDD RED: Test async WAL operations with proper await usage
    #[tokio::test]
    async fn test_async_wal_operations() {
        let dir = tempdir().unwrap();
        let wal = GraphWAL::new(dir.path().to_path_buf())
            .await
            .expect("Failed to create WAL");

        // Test writing to WAL without deadlock
        let entry = crate::graph_wal::WALEntry::NodeWrite {
            id: 1,
            data: NodeRecord::new(1, 1),
        };

        // This should not deadlock and should complete quickly
        let result = tokio::time::timeout(Duration::from_secs(5), wal.append(entry))
            .await
            .expect("WAL append should not timeout");

        assert!(result.is_ok(), "WAL append should succeed");
    }

    /// TDD RED: Test async read operations with proper mutex handling
    #[tokio::test]
    async fn test_async_read_operations() {
        let dir = tempdir().unwrap();
        let storage = GraphStorage::create(dir.path().to_path_buf(), 1000)
            .await
            .expect("Failed to create storage");

        // Write a node first
        let node = NodeRecord::new(42, 1);

        storage
            .write_node(&node)
            .await
            .expect("Write should succeed");

        // Read should work without deadlock
        let read_result = tokio::time::timeout(Duration::from_secs(5), storage.read_node(42))
            .await
            .expect("Read should not timeout");

        let read_node = read_result.expect("Read should succeed");
        assert_eq!(read_node.id, 42);
        assert_eq!(read_node.type_id, 1);
    }

    /// TDD RED: Test that tokio::sync::Mutex doesn't have poisoning behavior
    #[tokio::test]
    async fn test_no_mutex_poisoning() {
        let dir = tempdir().unwrap();
        let storage = Arc::new(
            GraphStorage::create(dir.path().to_path_buf(), 1000)
                .await
                .expect("Failed to create storage"),
        );

        let storage_clone = Arc::clone(&storage);

        // Spawn a task that panics while holding a mutex
        let panic_handle = tokio::spawn(async move {
            let node = NodeRecord::new(1, 1);

            // This should panic during write
            let _result = storage_clone.write_node(&node).await;
            panic!("Intentional panic for testing");
        });

        // Wait for the panic
        let _ = panic_handle.await;

        // Storage should still be usable (no poisoning with tokio::sync::Mutex)
        let node = NodeRecord::new(2, 2);

        let result = storage.write_node(&node).await;
        assert!(
            result.is_ok(),
            "Storage should recover from panic in other task"
        );
    }

    /// TDD RED: Test async checkpoint operations
    #[tokio::test]
    async fn test_async_checkpoint_operations() {
        let dir = tempdir().unwrap();
        let wal = GraphWAL::new(dir.path().to_path_buf())
            .await
            .expect("Failed to create WAL");

        // Add some entries first
        for i in 0..5 {
            let entry = crate::graph_wal::WALEntry::NodeWrite {
                id: i,
                data: NodeRecord::new(i, i as u32),
            };
            wal.append(entry).await.expect("Append should succeed");
        }

        // Checkpoint should complete without deadlock
        let result = tokio::time::timeout(Duration::from_secs(10), wal.checkpoint())
            .await
            .expect("Checkpoint should not timeout");

        assert!(result.is_ok(), "Checkpoint should succeed");
    }
}
