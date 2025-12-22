//! Storage backend implementations for memory snapshots

use super::types::{MemorySnapshot, SnapshotStorage, StorageStats};
use crate::DebugError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Ring buffer storage for snapshots with fixed capacity
pub struct RingBufferStorage {
    capacity: usize,
    snapshots: Arc<RwLock<HashMap<Uuid, MemorySnapshot>>>,
    container_snapshots: Arc<RwLock<HashMap<Uuid, Vec<Uuid>>>>,
}

impl RingBufferStorage {
    /// Create a new ring buffer storage with specified capacity
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity: capacity_bytes,
            snapshots: Arc::new(RwLock::new(HashMap::new())),
            container_snapshots: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get current storage utilization
    pub async fn get_utilization(&self) -> f64 {
        let snapshots = self.snapshots.read().await;
        let current_size: usize = snapshots.values().map(|s| s.total_size()).sum();
        current_size as f64 / self.capacity as f64
    }

    /// Evict oldest snapshots if capacity is exceeded
    async fn evict_if_needed(&self, new_snapshot_size: usize) -> Result<(), DebugError> {
        let mut snapshots = self.snapshots.write().await;
        let mut container_snapshots = self.container_snapshots.write().await;

        let current_size: usize = snapshots.values().map(|s| s.total_size()).sum();

        if current_size + new_snapshot_size > self.capacity {
            // Find oldest snapshots and remove them
            let mut snapshot_ages: Vec<(Uuid, u64)> = snapshots
                .iter()
                .map(|(id, snapshot)| (*id, snapshot.timestamp))
                .collect();

            snapshot_ages.sort_by_key(|&(_, timestamp)| timestamp);

            let mut freed_size = 0;
            for (snapshot_id, _) in snapshot_ages {
                if current_size + new_snapshot_size - freed_size <= self.capacity {
                    break;
                }

                if let Some(snapshot) = snapshots.remove(&snapshot_id) {
                    freed_size += snapshot.total_size();

                    // Remove from container tracking
                    if let Some(container_list) =
                        container_snapshots.get_mut(&snapshot.container_id)
                    {
                        container_list.retain(|&id| id != snapshot_id);
                        if container_list.is_empty() {
                            container_snapshots.remove(&snapshot.container_id);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[async_trait]
impl SnapshotStorage for RingBufferStorage {
    async fn store_snapshot(&self, snapshot: MemorySnapshot) -> Result<(), DebugError> {
        let snapshot_size = snapshot.total_size();
        let snapshot_id = snapshot.snapshot_id;
        let container_id = snapshot.container_id;

        // Evict old snapshots if needed
        self.evict_if_needed(snapshot_size).await?;

        // Store the snapshot
        {
            let mut snapshots = self.snapshots.write().await;
            snapshots.insert(snapshot_id, snapshot);
        }

        // Update container tracking
        {
            let mut container_snapshots = self.container_snapshots.write().await;
            container_snapshots
                .entry(container_id)
                .or_insert_with(Vec::new)
                .push(snapshot_id);
        }

        Ok(())
    }

    async fn get_snapshot(&self, snapshot_id: Uuid) -> Result<MemorySnapshot, DebugError> {
        let snapshots = self.snapshots.read().await;

        snapshots
            .get(&snapshot_id)
            .cloned()
            .ok_or_else(|| DebugError::SnapshotNotFound { snapshot_id })
    }

    async fn list_snapshots(&self, container_id: Uuid) -> Result<Vec<Uuid>, DebugError> {
        let container_snapshots = self.container_snapshots.read().await;

        Ok(container_snapshots
            .get(&container_id)
            .cloned()
            .unwrap_or_default())
    }

    async fn delete_snapshot(&self, snapshot_id: Uuid) -> Result<(), DebugError> {
        let container_id = {
            let mut snapshots = self.snapshots.write().await;
            let snapshot = snapshots
                .remove(&snapshot_id)
                .ok_or_else(|| DebugError::SnapshotNotFound { snapshot_id })?;
            snapshot.container_id
        };

        // Remove from container tracking
        let mut container_snapshots = self.container_snapshots.write().await;
        if let Some(container_list) = container_snapshots.get_mut(&container_id) {
            container_list.retain(|&id| id != snapshot_id);
            if container_list.is_empty() {
                container_snapshots.remove(&container_id);
            }
        }

        Ok(())
    }

    async fn get_stats(&self) -> Result<StorageStats, DebugError> {
        let snapshots = self.snapshots.read().await;
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let expired_count = snapshots
            .values()
            .filter(|snapshot| snapshot.is_expired())
            .count() as u64;

        let oldest_age = snapshots
            .values()
            .map(|snapshot| current_time.saturating_sub(snapshot.timestamp))
            .max()
            .unwrap_or(0);

        Ok(StorageStats {
            expired_snapshots: expired_count,
            oldest_snapshot_age_seconds: oldest_age,
        })
    }

    async fn cleanup_expired(&self, max_age_seconds: u64) -> Result<u64, DebugError> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let cutoff_time = current_time.saturating_sub(max_age_seconds);

        let expired_snapshots: Vec<Uuid> = {
            let snapshots = self.snapshots.read().await;
            snapshots
                .iter()
                .filter(|(_, snapshot)| snapshot.timestamp < cutoff_time)
                .map(|(id, _)| *id)
                .collect()
        };

        let count = expired_snapshots.len() as u64;

        for snapshot_id in expired_snapshots {
            self.delete_snapshot(snapshot_id).await?;
        }

        Ok(count)
    }
}

/// File-based storage backend (placeholder implementation)
pub struct FileStorage {
    base_path: std::path::PathBuf,
    snapshots: Arc<RwLock<HashMap<Uuid, MemorySnapshot>>>,
    container_snapshots: Arc<RwLock<HashMap<Uuid, Vec<Uuid>>>>,
}

impl FileStorage {
    /// Create a new file storage backend
    pub fn new(base_path: std::path::PathBuf) -> Self {
        Self {
            base_path,
            snapshots: Arc::new(RwLock::new(HashMap::new())),
            container_snapshots: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get file path for a snapshot
    fn get_snapshot_path(&self, snapshot_id: Uuid) -> std::path::PathBuf {
        self.base_path.join(format!("{}.snapshot", snapshot_id))
    }
}

#[async_trait]
impl SnapshotStorage for FileStorage {
    async fn store_snapshot(&self, snapshot: MemorySnapshot) -> Result<(), DebugError> {
        let snapshot_id = snapshot.snapshot_id;
        let container_id = snapshot.container_id;

        // Serialize and write to file (placeholder)
        let serialized =
            serde_json::to_string(&snapshot).map_err(|e| DebugError::SerializationFailed {
                reason: format!("Failed to serialize snapshot: {}", e),
            })?;

        let file_path = self.get_snapshot_path(snapshot_id);
        tokio::fs::write(&file_path, serialized)
            .await
            .map_err(|e| DebugError::StorageError {
                reason: format!("Failed to write snapshot to file: {}", e),
            })?;

        // Update in-memory tracking
        {
            let mut snapshots = self.snapshots.write().await;
            snapshots.insert(snapshot_id, snapshot);
        }

        {
            let mut container_snapshots = self.container_snapshots.write().await;
            container_snapshots
                .entry(container_id)
                .or_insert_with(Vec::new)
                .push(snapshot_id);
        }

        Ok(())
    }

    async fn get_snapshot(&self, snapshot_id: Uuid) -> Result<MemorySnapshot, DebugError> {
        // Try in-memory first
        {
            let snapshots = self.snapshots.read().await;
            if let Some(snapshot) = snapshots.get(&snapshot_id) {
                return Ok(snapshot.clone());
            }
        }

        // Load from file
        let file_path = self.get_snapshot_path(snapshot_id);
        let content = tokio::fs::read_to_string(&file_path)
            .await
            .map_err(|_| DebugError::SnapshotNotFound { snapshot_id })?;

        let snapshot: MemorySnapshot =
            serde_json::from_str(&content).map_err(|e| DebugError::SerializationFailed {
                reason: format!("Failed to deserialize snapshot: {}", e),
            })?;

        // Cache in memory
        {
            let mut snapshots = self.snapshots.write().await;
            snapshots.insert(snapshot_id, snapshot.clone());
        }

        Ok(snapshot)
    }

    async fn list_snapshots(&self, container_id: Uuid) -> Result<Vec<Uuid>, DebugError> {
        let container_snapshots = self.container_snapshots.read().await;

        Ok(container_snapshots
            .get(&container_id)
            .cloned()
            .unwrap_or_default())
    }

    async fn delete_snapshot(&self, snapshot_id: Uuid) -> Result<(), DebugError> {
        let container_id = {
            let mut snapshots = self.snapshots.write().await;
            let snapshot = snapshots
                .remove(&snapshot_id)
                .ok_or_else(|| DebugError::SnapshotNotFound { snapshot_id })?;
            snapshot.container_id
        };

        // Remove file
        let file_path = self.get_snapshot_path(snapshot_id);
        if file_path.exists() {
            tokio::fs::remove_file(&file_path)
                .await
                .map_err(|e| DebugError::StorageError {
                    reason: format!("Failed to delete snapshot file: {}", e),
                })?;
        }

        // Remove from container tracking
        let mut container_snapshots = self.container_snapshots.write().await;
        if let Some(container_list) = container_snapshots.get_mut(&container_id) {
            container_list.retain(|&id| id != snapshot_id);
            if container_list.is_empty() {
                container_snapshots.remove(&container_id);
            }
        }

        Ok(())
    }

    async fn get_stats(&self) -> Result<StorageStats, DebugError> {
        let snapshots = self.snapshots.read().await;
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let expired_count = snapshots
            .values()
            .filter(|snapshot| snapshot.is_expired())
            .count() as u64;

        let oldest_age = snapshots
            .values()
            .map(|snapshot| current_time.saturating_sub(snapshot.timestamp))
            .max()
            .unwrap_or(0);

        Ok(StorageStats {
            expired_snapshots: expired_count,
            oldest_snapshot_age_seconds: oldest_age,
        })
    }

    async fn cleanup_expired(&self, max_age_seconds: u64) -> Result<u64, DebugError> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let cutoff_time = current_time.saturating_sub(max_age_seconds);

        let expired_snapshots: Vec<Uuid> = {
            let snapshots = self.snapshots.read().await;
            snapshots
                .iter()
                .filter(|(_, snapshot)| snapshot.timestamp < cutoff_time)
                .map(|(id, _)| *id)
                .collect()
        };

        let count = expired_snapshots.len() as u64;

        for snapshot_id in expired_snapshots {
            self.delete_snapshot(snapshot_id).await?;
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::types::{ExecutionContext, KernelParameters};

    #[tokio::test]
    async fn test_ring_buffer_storage() {
        let storage = RingBufferStorage::new(1024 * 1024); // 1MB capacity

        let snapshot = MemorySnapshot::new(
            Uuid::new_v4(),
            vec![1; 100],
            vec![2; 100],
            KernelParameters::default(),
            ExecutionContext::new("test".to_string()),
            "test storage".to_string(),
        );

        let snapshot_id = snapshot.snapshot_id;
        let container_id = snapshot.container_id;

        // Store snapshot
        storage.store_snapshot(snapshot).await.unwrap();

        // Retrieve snapshot
        let retrieved = storage.get_snapshot(snapshot_id).await.unwrap();
        assert_eq!(retrieved.snapshot_id, snapshot_id);
        assert_eq!(retrieved.host_memory.len(), 100);

        // List snapshots for container
        let snapshot_list = storage.list_snapshots(container_id).await.unwrap();
        assert_eq!(snapshot_list, vec![snapshot_id]);

        // Get stats
        let stats = storage.get_stats().await.unwrap();
        assert_eq!(stats.expired_snapshots, 0);

        // Delete snapshot
        storage.delete_snapshot(snapshot_id).await.unwrap();

        // Verify deletion
        assert!(storage.get_snapshot(snapshot_id).await.is_err());
        let empty_list = storage.list_snapshots(container_id).await.unwrap();
        assert!(empty_list.is_empty());
    }

    #[tokio::test]
    async fn test_capacity_eviction() {
        let storage = RingBufferStorage::new(300); // Small capacity

        let mut snapshots = Vec::new();

        // Create several snapshots that exceed capacity
        for i in 0..5 {
            let snapshot = MemorySnapshot::new(
                Uuid::new_v4(),
                vec![1; 100], // 100 bytes
                vec![2; 100], // 100 bytes = 200 bytes total per snapshot
                KernelParameters::default(),
                ExecutionContext::new(format!("test {}", i)),
                format!("test snapshot {}", i),
            );
            snapshots.push(snapshot.snapshot_id);
            storage.store_snapshot(snapshot).await.unwrap();

            // Small delay to ensure different timestamps
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }

        // Only the latest snapshots should remain due to capacity limits
        let utilization = storage.get_utilization().await;
        assert!(utilization <= 1.0); // Should not exceed capacity
    }

    #[tokio::test]
    async fn test_cleanup_expired() {
        let storage = RingBufferStorage::new(1024 * 1024);

        // Create snapshot with short TTL
        let mut snapshot = MemorySnapshot::new(
            Uuid::new_v4(),
            vec![1; 100],
            vec![2; 100],
            KernelParameters::default(),
            ExecutionContext::new("test".to_string()),
            "test cleanup".to_string(),
        );

        // Set short TTL
        snapshot.metadata.ttl_seconds = 1;
        snapshot.timestamp = 1000; // Old timestamp

        let snapshot_id = snapshot.snapshot_id;
        storage.store_snapshot(snapshot).await.unwrap();

        // Clean up snapshots older than 60 seconds
        let cleaned = storage.cleanup_expired(60).await.unwrap();
        assert_eq!(cleaned, 1);

        // Verify snapshot was deleted
        assert!(storage.get_snapshot(snapshot_id).await.is_err());
    }
}
