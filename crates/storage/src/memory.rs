//! In-memory storage implementation for development and testing

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{Storage, StorageError, StorageStats};

/// In-memory storage implementation
pub struct MemoryStorage {
    data: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    total_capacity: u64,
}

impl MemoryStorage {
    /// Create new in-memory storage
    pub fn new(capacity: u64) -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
            total_capacity: capacity,
        }
    }

    fn calculate_used_bytes(&self, data: &HashMap<String, Vec<u8>>) -> u64 {
        data.iter()
            .map(|(key, value)| key.len() + value.len())
            .sum::<usize>() as u64
    }
}

#[async_trait::async_trait]
impl Storage for MemoryStorage {
    async fn store(&self, key: &str, data: &[u8]) -> Result<(), StorageError> {
        let mut storage = self.data.lock().map_err(|e| {
            StorageError::Io(std::io::Error::other(format!(
                "Failed to acquire lock: {e}"
            )))
        })?;

        let current_used = self.calculate_used_bytes(&storage);
        let required_space = (key.len() + data.len()) as u64;

        // Check if we have existing entry
        if let Some(existing) = storage.get(key) {
            let freed_space = (key.len() + existing.len()) as u64;
            if current_used - freed_space + required_space > self.total_capacity {
                return Err(StorageError::StorageFull {
                    available: self.total_capacity - (current_used - freed_space),
                });
            }
        } else if current_used + required_space > self.total_capacity {
            return Err(StorageError::StorageFull {
                available: self.total_capacity - current_used,
            });
        }

        storage.insert(key.to_string(), data.to_vec());
        Ok(())
    }

    async fn retrieve(&self, key: &str) -> Result<Vec<u8>, StorageError> {
        let storage = self.data.lock().map_err(|e| {
            StorageError::Io(std::io::Error::other(format!(
                "Failed to acquire lock: {e}"
            )))
        })?;

        storage
            .get(key)
            .cloned()
            .ok_or_else(|| StorageError::KeyNotFound {
                key: key.to_string(),
            })
    }

    async fn delete(&self, key: &str) -> Result<(), StorageError> {
        let mut storage = self.data.lock().map_err(|e| {
            StorageError::Io(std::io::Error::other(format!(
                "Failed to acquire lock: {e}"
            )))
        })?;

        storage
            .remove(key)
            .map(|_| ())
            .ok_or_else(|| StorageError::KeyNotFound {
                key: key.to_string(),
            })
    }

    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        let storage = self.data.lock().map_err(|e| {
            StorageError::Io(std::io::Error::other(format!(
                "Failed to acquire lock: {e}"
            )))
        })?;

        Ok(storage
            .keys()
            .filter(|key| key.starts_with(prefix))
            .cloned()
            .collect())
    }

    async fn stats(&self) -> Result<StorageStats, StorageError> {
        let storage = self.data.lock().map_err(|e| {
            StorageError::Io(std::io::Error::other(format!(
                "Failed to acquire lock: {e}"
            )))
        })?;

        let used_bytes = self.calculate_used_bytes(&storage);
        let total_files = storage.len() as u64;

        Ok(StorageStats {
            total_bytes: self.total_capacity,
            used_bytes,
            available_bytes: self.total_capacity - used_bytes,
            total_files,
            read_throughput_mbps: 1000.0,  // Mock value for in-memory
            write_throughput_mbps: 1000.0, // Mock value for in-memory
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_storage_creation() {
        let storage = MemoryStorage::new(1024);

        let stats = storage.stats().await.expect("Failed to get stats");
        assert_eq!(stats.total_bytes, 1024);
        assert_eq!(stats.used_bytes, 0);
        assert_eq!(stats.available_bytes, 1024);
        assert_eq!(stats.total_files, 0);
    }

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let storage = MemoryStorage::new(1024);
        let data = b"Hello, World!";

        storage
            .store("test_key", data)
            .await
            .expect("Failed to store data");

        let retrieved = storage
            .retrieve("test_key")
            .await
            .expect("Failed to retrieve data");
        assert_eq!(retrieved, data);

        let stats = storage.stats().await.expect("Failed to get stats");
        assert!(stats.used_bytes > 0);
        assert_eq!(stats.total_files, 1);
    }

    #[tokio::test]
    async fn test_key_not_found() {
        let storage = MemoryStorage::new(1024);

        let result = storage.retrieve("nonexistent").await;
        assert!(matches!(result, Err(StorageError::KeyNotFound { key }) if key == "nonexistent"));
    }

    #[tokio::test]
    async fn test_storage_full() {
        let storage = MemoryStorage::new(20); // Small capacity
        let large_data = vec![0u8; 50]; // Larger than capacity

        let result = storage.store("large_key", &large_data).await;
        assert!(matches!(result, Err(StorageError::StorageFull { .. })));
    }

    #[tokio::test]
    async fn test_overwrite_data() {
        let storage = MemoryStorage::new(1024);

        storage
            .store("key", b"original")
            .await
            .expect("Failed to store original");
        let retrieved1 = storage
            .retrieve("key")
            .await
            .expect("Failed to retrieve original");
        assert_eq!(retrieved1, b"original");

        storage
            .store("key", b"updated")
            .await
            .expect("Failed to overwrite");
        let retrieved2 = storage
            .retrieve("key")
            .await
            .expect("Failed to retrieve updated");
        assert_eq!(retrieved2, b"updated");

        let stats = storage.stats().await.expect("Failed to get stats");
        assert_eq!(stats.total_files, 1); // Still only one file
    }

    #[tokio::test]
    async fn test_delete() {
        let storage = MemoryStorage::new(1024);

        storage
            .store("to_delete", b"data")
            .await
            .expect("Failed to store");
        assert!(storage.retrieve("to_delete").await.is_ok());

        storage.delete("to_delete").await.expect("Failed to delete");

        let result = storage.retrieve("to_delete").await;
        assert!(matches!(result, Err(StorageError::KeyNotFound { .. })));
    }

    #[tokio::test]
    async fn test_delete_nonexistent() {
        let storage = MemoryStorage::new(1024);

        let result = storage.delete("nonexistent").await;
        assert!(matches!(result, Err(StorageError::KeyNotFound { key }) if key == "nonexistent"));
    }

    #[tokio::test]
    async fn test_list_keys() {
        let storage = MemoryStorage::new(1024);

        storage
            .store("prefix_1", b"data1")
            .await
            .expect("Failed to store 1");
        storage
            .store("prefix_2", b"data2")
            .await
            .expect("Failed to store 2");
        storage
            .store("other_key", b"data3")
            .await
            .expect("Failed to store 3");

        let prefix_keys = storage
            .list_keys("prefix_")
            .await
            .expect("Failed to list keys");
        assert_eq!(prefix_keys.len(), 2);
        assert!(prefix_keys.contains(&"prefix_1".to_string()));
        assert!(prefix_keys.contains(&"prefix_2".to_string()));

        let all_keys = storage
            .list_keys("")
            .await
            .expect("Failed to list all keys");
        assert_eq!(all_keys.len(), 3);
    }

    #[tokio::test]
    async fn test_capacity_management() {
        let storage = MemoryStorage::new(100);

        // Fill storage almost to capacity
        storage
            .store("key1", &vec![0u8; 40])
            .await
            .expect("Failed to store key1");
        storage
            .store("key2", &vec![0u8; 40])
            .await
            .expect("Failed to store key2");

        // This should fail - not enough space
        let result = storage.store("key3", &vec![0u8; 30]).await;
        assert!(matches!(result, Err(StorageError::StorageFull { .. })));

        // Delete one key, now we should have space
        storage.delete("key1").await.expect("Failed to delete key1");
        storage
            .store("key3", &vec![0u8; 30])
            .await
            .expect("Failed to store key3 after delete");
    }

    #[tokio::test]
    async fn test_overwrite_with_insufficient_space() {
        let storage = MemoryStorage::new(60);

        // Store initial data
        // "key1" (4 bytes) + data (20 bytes) = 24 bytes
        // "key2" (4 bytes) + data (20 bytes) = 24 bytes
        // Total: 48 bytes
        storage
            .store("key1", &vec![0u8; 20])
            .await
            .expect("Failed to store key1");
        storage
            .store("key2", &vec![0u8; 20])
            .await
            .expect("Failed to store key2");

        // Try to overwrite key1 with larger data that won't fit
        // Current: 48 bytes used
        // After removing key1: 48 - 24 = 24 bytes used
        // New key1 would be 4 + 40 = 44 bytes
        // Total would be 24 + 44 = 68 > 60
        let result = storage.store("key1", &vec![0u8; 40]).await;
        assert!(matches!(result, Err(StorageError::StorageFull { available }) if available == 36));
    }

    #[tokio::test]
    async fn test_list_keys_empty() {
        let storage = MemoryStorage::new(1024);

        let keys = storage.list_keys("").await.expect("Failed to list keys");
        assert_eq!(keys.len(), 0);

        let prefix_keys = storage
            .list_keys("prefix_")
            .await
            .expect("Failed to list keys");
        assert_eq!(prefix_keys.len(), 0);
    }

    #[tokio::test]
    async fn test_multiple_overwrites() {
        let storage = MemoryStorage::new(200);

        // Initial store
        storage
            .store("key", &vec![1u8; 50])
            .await
            .expect("Failed to store initial");

        // Overwrite with smaller
        storage
            .store("key", &vec![2u8; 30])
            .await
            .expect("Failed to overwrite smaller");
        let data = storage.retrieve("key").await.expect("Failed to retrieve");
        assert_eq!(data.len(), 30);
        assert_eq!(data[0], 2);

        // Overwrite with larger
        storage
            .store("key", &vec![3u8; 60])
            .await
            .expect("Failed to overwrite larger");
        let data = storage.retrieve("key").await.expect("Failed to retrieve");
        assert_eq!(data.len(), 60);
        assert_eq!(data[0], 3);

        let stats = storage.stats().await.expect("Failed to get stats");
        assert_eq!(stats.total_files, 1);
        assert_eq!(stats.used_bytes, "key".len() as u64 + 60);
    }

    #[tokio::test]
    async fn test_storage_with_empty_data() {
        let storage = MemoryStorage::new(100);

        // Store empty data
        storage
            .store("empty", &[])
            .await
            .expect("Failed to store empty");

        let data = storage.retrieve("empty").await.expect("Failed to retrieve");
        assert_eq!(data.len(), 0);

        let stats = storage.stats().await.expect("Failed to get stats");
        assert_eq!(stats.used_bytes, "empty".len() as u64);
    }

    #[tokio::test]
    async fn test_exact_capacity() {
        let storage = MemoryStorage::new(50);

        // Store exactly the capacity
        // "key" = 3 bytes, data = 47 bytes, total = 50
        storage
            .store("key", &vec![0u8; 47])
            .await
            .expect("Failed to store exact capacity");

        // Try to store even 1 more byte
        let result = storage.store("a", &[0]).await;
        assert!(matches!(
            result,
            Err(StorageError::StorageFull { available: 0 })
        ));

        // Overwrite with same size should work
        storage
            .store("key", &vec![1u8; 47])
            .await
            .expect("Failed to overwrite same size");
    }

    // Test to increase coverage by simulating lock poisoning
    #[tokio::test]
    async fn test_concurrent_access() {
        use std::sync::Arc;

        let storage = Arc::new(MemoryStorage::new(1024));
        let storage1 = storage.clone();
        let storage2 = storage.clone();
        let storage3 = storage.clone();

        // Spawn multiple tasks accessing storage concurrently
        let handle1 = tokio::spawn(async move {
            for i in 0..10 {
                let key = format!("task1_{i}");
                storage1.store(&key, &[i as u8]).await?;
            }
        });

        let handle2 = tokio::spawn(async move {
            for i in 0..10 {
                let key = format!("task2_{i}");
                storage2.store(&key, &[i as u8]).await?;
            }
        });

        let handle3 = tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
            let keys = storage3.list_keys("task").await?;
            assert!(keys.len() > 0);
        });

        handle1.await?;
        handle2.await?;
        handle3.await?;

        // Verify all data was stored
        let all_keys = storage.list_keys("").await?;
        assert_eq!(all_keys.len(), 20);
    }

    #[tokio::test]
    async fn test_mutex_poisoning_store() {
        use crate::test_helpers::tests::PoisonedMemoryStorage;

        let storage = PoisonedMemoryStorage::new(1024);

        // Create a storage-like object with poisoned mutex
        let storage_impl = MemoryStorage {
            data: storage.data.clone(),
            total_capacity: storage.total_capacity,
        };

        // Try to store - should fail with lock error
        let result = storage_impl.store("key", b"data").await;
        assert!(result.is_err());

        match result {
            Err(StorageError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_retrieve() {
        use crate::test_helpers::tests::PoisonedMemoryStorage;

        let storage = PoisonedMemoryStorage::new(1024);
        let storage_impl = MemoryStorage {
            data: storage.data.clone(),
            total_capacity: storage.total_capacity,
        };

        // Try to retrieve - should fail with lock error
        let result = storage_impl.retrieve("key").await;
        assert!(result.is_err());

        match result {
            Err(StorageError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_delete() {
        use crate::test_helpers::tests::PoisonedMemoryStorage;

        let storage = PoisonedMemoryStorage::new(1024);
        let storage_impl = MemoryStorage {
            data: storage.data.clone(),
            total_capacity: storage.total_capacity,
        };

        // Try to delete - should fail with lock error
        let result = storage_impl.delete("key").await;
        assert!(result.is_err());

        match result {
            Err(StorageError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_list_keys() {
        use crate::test_helpers::tests::PoisonedMemoryStorage;

        let storage = PoisonedMemoryStorage::new(1024);
        let storage_impl = MemoryStorage {
            data: storage.data.clone(),
            total_capacity: storage.total_capacity,
        };

        // Try to list keys - should fail with lock error
        let result = storage_impl.list_keys("").await;
        assert!(result.is_err());

        match result {
            Err(StorageError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_stats() {
        use crate::test_helpers::tests::PoisonedMemoryStorage;

        let storage = PoisonedMemoryStorage::new(1024);
        let storage_impl = MemoryStorage {
            data: storage.data.clone(),
            total_capacity: storage.total_capacity,
        };

        // Try to get stats - should fail with lock error
        let result = storage_impl.stats().await;
        assert!(result.is_err());

        match result {
            Err(StorageError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }
}
