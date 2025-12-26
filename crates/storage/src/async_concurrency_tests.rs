//! Async concurrency tests replacing mutex poisoning tests
//! Since tokio::sync::Mutex doesn't have poisoning, we test concurrent access instead

#[cfg(test)]
mod async_concurrency_tests {
    use crate::graph_format::NodeRecord;
    use crate::graph_storage::GraphStorage;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::time::{timeout, Duration};

    #[tokio::test]
    async fn test_concurrent_write_operations() {
        let dir = tempdir().unwrap();
        let storage = Arc::new(
            GraphStorage::create(dir.path().to_path_buf(), 1000)
                .await
                .unwrap(),
        );

        // Test concurrent writes don't deadlock
        let storage1 = Arc::clone(&storage);
        let storage2 = Arc::clone(&storage);
        let storage3 = Arc::clone(&storage);

        let handle1 = tokio::spawn(async move {
            for i in 0..10 {
                let node = NodeRecord::new(i, i as u32);
                storage1.write_node(&node).await.unwrap();
            }
        });

        let handle2 = tokio::spawn(async move {
            for i in 10..20 {
                let node = NodeRecord::new(i, i as u32);
                storage2.write_node(&node).await.unwrap();
            }
        });

        let handle3 = tokio::spawn(async move {
            for i in 20..30 {
                let node = NodeRecord::new(i, i as u32);
                storage3.write_node(&node).await.unwrap();
            }
        });

        // All operations should complete within reasonable time
        let result = timeout(Duration::from_secs(10), async {
            handle1.await.unwrap();
            handle2.await.unwrap();
            handle3.await.unwrap();
        })
        .await;

        assert!(
            result.is_ok(),
            "Concurrent writes should complete without deadlock"
        );
    }

    #[tokio::test]
    async fn test_concurrent_read_write_operations() {
        let dir = tempdir().unwrap();
        let storage = Arc::new(
            GraphStorage::create(dir.path().to_path_buf(), 1000)
                .await
                .unwrap(),
        );

        // Write initial data
        for i in 0..5 {
            let node = NodeRecord::new(i, i as u32);
            storage.write_node(&node).await.unwrap();
        }

        let storage_reader = Arc::clone(&storage);
        let storage_writer = Arc::clone(&storage);

        let read_handle = tokio::spawn(async move {
            for _ in 0..20 {
                for i in 0..5 {
                    let _node = storage_reader.read_node(i).await.unwrap();
                }
                tokio::task::yield_now().await;
            }
        });

        let write_handle = tokio::spawn(async move {
            for i in 5..10 {
                let node = NodeRecord::new(i, i as u32);
                storage_writer.write_node(&node).await.unwrap();
                tokio::task::yield_now().await;
            }
        });

        let result = timeout(Duration::from_secs(15), async {
            read_handle.await.unwrap();
            write_handle.await.unwrap();
        })
        .await;

        assert!(
            result.is_ok(),
            "Concurrent read/write should complete without deadlock"
        );
    }

    #[tokio::test]
    async fn test_concurrent_edge_operations() {
        let dir = tempdir().unwrap();
        let storage = Arc::new(
            GraphStorage::create(dir.path().to_path_buf(), 1000)
                .await
                .unwrap(),
        );

        // Write nodes first
        for i in 0..10 {
            let node = NodeRecord::new(i, i as u32);
            storage.write_node(&node).await.unwrap();
        }

        let storage1 = Arc::clone(&storage);
        let storage2 = Arc::clone(&storage);

        let add_edges_handle = tokio::spawn(async move {
            for i in 0..5 {
                storage1.add_edge(i, i + 1, 1, 1.0).await.unwrap();
                tokio::task::yield_now().await;
            }
        });

        let get_edges_handle = tokio::spawn(async move {
            for i in 0..5 {
                let _edges = storage2.get_edges(i).await.unwrap();
                tokio::task::yield_now().await;
            }
        });

        let result = timeout(Duration::from_secs(10), async {
            add_edges_handle.await.unwrap();
            get_edges_handle.await.unwrap();
        })
        .await;

        assert!(result.is_ok(), "Concurrent edge operations should complete");
    }

    #[tokio::test]
    async fn test_high_concurrency_stress() {
        let dir = tempdir().unwrap();
        let storage = Arc::new(
            GraphStorage::create(dir.path().to_path_buf(), 1000)
                .await
                .unwrap(),
        );

        // Spawn many concurrent tasks
        let mut handles = Vec::new();

        for worker_id in 0..10 {
            let storage_clone = Arc::clone(&storage);
            let handle = tokio::spawn(async move {
                for i in 0..5 {
                    let node_id = worker_id * 10 + i;
                    let node = NodeRecord::new(node_id, node_id as u32);
                    storage_clone.write_node(&node).await.unwrap();

                    // Read back the node
                    let _read_node = storage_clone.read_node(node_id).await.unwrap();

                    tokio::task::yield_now().await;
                }
            });
            handles.push(handle);
        }

        // All tasks should complete within reasonable time
        let result = timeout(Duration::from_secs(30), async {
            for handle in handles {
                handle.await.unwrap();
            }
        })
        .await;

        assert!(
            result.is_ok(),
            "High concurrency operations should complete"
        );
    }

    #[tokio::test]
    async fn test_async_mutex_recovery_after_panic() {
        let dir = tempdir().unwrap();
        let storage = Arc::new(
            GraphStorage::create(dir.path().to_path_buf(), 1000)
                .await
                .unwrap(),
        );

        let storage_panic = Arc::clone(&storage);
        let storage_recover = Arc::clone(&storage);

        // Spawn a task that panics
        let panic_handle = tokio::spawn(async move {
            let node = NodeRecord::new(1, 1);
            let _result = storage_panic.write_node(&node).await;
            panic!("Intentional panic for testing");
        });

        // Wait for the panic
        let _ = panic_handle.await;

        // Storage should still be usable (no poisoning with tokio::sync::Mutex)
        let node = NodeRecord::new(2, 2);
        let result = storage_recover.write_node(&node).await;
        assert!(
            result.is_ok(),
            "Storage should recover from panic in other task"
        );

        // Should be able to read the node
        let read_result = storage_recover.read_node(2).await;
        assert!(
            read_result.is_ok(),
            "Should be able to read after panic recovery"
        );
    }

    #[tokio::test]
    async fn test_long_running_concurrent_operations() {
        let dir = tempdir().unwrap();
        let storage = Arc::new(
            GraphStorage::create(dir.path().to_path_buf(), 1000)
                .await
                .unwrap(),
        );

        let storage1 = Arc::clone(&storage);
        let storage2 = Arc::clone(&storage);

        let long_write_handle = tokio::spawn(async move {
            for i in 0..100 {
                let node = NodeRecord::new(i, i as u32);
                storage1.write_node(&node).await.unwrap();
                if i % 10 == 0 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }
        });

        let long_read_handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await; // Let some writes happen first
            for i in 0..50 {
                if let Ok(_node) = storage2.read_node(i).await {
                    // Successfully read node
                }
                tokio::task::yield_now().await;
            }
        });

        let result = timeout(Duration::from_secs(60), async {
            long_write_handle.await.unwrap();
            long_read_handle.await.unwrap();
        })
        .await;

        assert!(
            result.is_ok(),
            "Long running concurrent operations should complete"
        );
    }
}
