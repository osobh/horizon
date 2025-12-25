//! Snapshot manager for coordinating snapshot operations

use super::storage::RingBufferStorage;
use super::types::{
    ExecutionContext, KernelParameters, MemorySnapshot, SnapshotConfig, SnapshotSession,
    SnapshotStorage, StorageStats,
};
use crate::DebugError;
use dashmap::DashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Main snapshot manager coordinating all snapshot operations
pub struct SnapshotManager {
    config: SnapshotConfig,
    storage: Arc<dyn SnapshotStorage>,
    active_sessions: Arc<DashMap<Uuid, SnapshotSession>>,
}

impl SnapshotManager {
    /// Create a new snapshot manager with default ring buffer storage
    pub fn new(config: SnapshotConfig) -> Self {
        let storage_capacity = (config.max_snapshot_size_mb * 1024 * 1024) as usize;
        let storage = Arc::new(RingBufferStorage::new(storage_capacity));

        Self {
            config,
            storage,
            active_sessions: Arc::new(DashMap::new()),
        }
    }

    /// Create a new snapshot manager with custom storage backend
    pub fn with_storage(config: SnapshotConfig, storage: Arc<dyn SnapshotStorage>) -> Self {
        Self {
            config,
            storage,
            active_sessions: Arc::new(DashMap::new()),
        }
    }

    /// Start a new debugging session for a container
    pub async fn start_session(&self, container_id: Uuid) -> Result<(), DebugError> {
        let session = SnapshotSession {
            container_id,
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            active_snapshots: Vec::new(),
            session_config: self.config.clone(),
        };

        self.active_sessions.insert(container_id, session);

        Ok(())
    }

    /// End a debugging session and optionally clean up snapshots
    pub async fn end_session(&self, container_id: Uuid, cleanup: bool) -> Result<(), DebugError> {
        let session = self.active_sessions.remove(&container_id).map(|(_, s)| s);

        if let Some(session) = session {
            if cleanup {
                // Delete all snapshots for this session
                for snapshot_id in session.active_snapshots {
                    let _ = self.storage.delete_snapshot(snapshot_id).await;
                }
            }
        }

        Ok(())
    }

    /// Create a memory snapshot for a container
    pub async fn create_snapshot(
        &self,
        container_id: Uuid,
        host_memory: Vec<u8>,
        device_memory: Vec<u8>,
        kernel_parameters: KernelParameters,
        execution_context: ExecutionContext,
        creation_reason: String,
    ) -> Result<Uuid, DebugError> {
        // Check if session exists
        if !self.active_sessions.contains_key(&container_id) {
            return Err(DebugError::SessionNotFound { container_id });
        }

        // Create the snapshot
        let mut snapshot = MemorySnapshot::new(
            container_id,
            host_memory,
            device_memory,
            kernel_parameters,
            execution_context,
            creation_reason,
        );

        // Apply configuration settings
        snapshot.metadata.ttl_seconds = self.config.default_ttl_seconds;
        snapshot.metadata.encrypted = self.config.encryption_enabled;

        // Compress if enabled
        if self.config.compression_enabled {
            snapshot.compress()?;
        }

        let snapshot_id = snapshot.snapshot_id;

        // Store the snapshot
        self.storage.store_snapshot(snapshot).await?;

        // Update session tracking
        if let Some(mut session) = self.active_sessions.get_mut(&container_id) {
            session.active_snapshots.push(snapshot_id);
        }

        Ok(snapshot_id)
    }

    /// Retrieve a snapshot by ID
    pub async fn get_snapshot(&self, snapshot_id: Uuid) -> Result<MemorySnapshot, DebugError> {
        let mut snapshot = self.storage.get_snapshot(snapshot_id).await?;

        // Decompress if needed
        if snapshot.metadata.compression_ratio < 1.0 {
            snapshot.decompress()?;
        }

        Ok(snapshot)
    }

    /// List all snapshots for a container
    pub async fn list_snapshots(&self, container_id: Uuid) -> Result<Vec<Uuid>, DebugError> {
        self.storage.list_snapshots(container_id).await
    }

    /// Delete a snapshot
    pub async fn delete_snapshot(&self, snapshot_id: Uuid) -> Result<(), DebugError> {
        // Get container ID before deletion
        let container_id = {
            let snapshot = self.storage.get_snapshot(snapshot_id).await?;
            snapshot.container_id
        };

        // Delete from storage
        self.storage.delete_snapshot(snapshot_id).await?;

        // Update session tracking
        if let Some(mut session) = self.active_sessions.get_mut(&container_id) {
            session.active_snapshots.retain(|&id| id != snapshot_id);
        }

        Ok(())
    }

    /// Get storage statistics
    pub async fn get_stats(&self) -> Result<StorageStats, DebugError> {
        self.storage.get_stats().await
    }

    /// Perform cleanup of expired snapshots
    pub async fn cleanup_expired(&self) -> Result<u64, DebugError> {
        let cleaned = self
            .storage
            .cleanup_expired(self.config.default_ttl_seconds)
            .await?;

        // Update session tracking to remove cleaned snapshots
        for mut entry in self.active_sessions.iter_mut() {
            let session = entry.value_mut();
            session.active_snapshots.retain(|&snapshot_id| {
                // Check if snapshot still exists
                matches!(
                    futures::executor::block_on(self.storage.get_snapshot(snapshot_id)),
                    Ok(_)
                )
            });
        }

        Ok(cleaned)
    }

    /// Run periodic cleanup task
    pub async fn run_cleanup_task(&self) -> Result<(), DebugError> {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(
            self.config.cleanup_interval_seconds,
        ));

        loop {
            interval.tick().await;

            match self.cleanup_expired().await {
                Ok(cleaned) => {
                    if cleaned > 0 {
                        // Log cleanup success (placeholder for actual logging)
                        // log::info!("Cleaned up {} expired snapshots", cleaned);
                    }
                }
                Err(_e) => {
                    // Log cleanup error (placeholder for actual logging)
                    // log::error!("Failed to clean up expired snapshots: {}", e);
                }
            }
        }
    }

    /// Get active session information
    pub async fn get_session_info(&self, container_id: Uuid) -> Option<SnapshotSession> {
        self.active_sessions
            .get(&container_id)
            .map(|entry| entry.clone())
    }

    /// List all active sessions
    pub async fn list_active_sessions(&self) -> Vec<Uuid> {
        self.active_sessions
            .iter()
            .map(|entry| *entry.key())
            .collect()
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: SnapshotConfig) {
        self.config = new_config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &SnapshotConfig {
        &self.config
    }

    /// Take an automatic snapshot if enabled
    pub async fn auto_snapshot(
        &self,
        container_id: Uuid,
        host_memory: Vec<u8>,
        device_memory: Vec<u8>,
        kernel_parameters: KernelParameters,
        execution_context: ExecutionContext,
    ) -> Result<Option<Uuid>, DebugError> {
        if !self.config.auto_snapshot {
            return Ok(None);
        }

        let snapshot_id = self
            .create_snapshot(
                container_id,
                host_memory,
                device_memory,
                kernel_parameters,
                execution_context,
                "automatic snapshot".to_string(),
            )
            .await?;

        Ok(Some(snapshot_id))
    }

    /// Compare two snapshots and return differences
    pub async fn compare_snapshots(
        &self,
        snapshot_id1: Uuid,
        snapshot_id2: Uuid,
    ) -> Result<SnapshotDiff, DebugError> {
        let snapshot1 = self.get_snapshot(snapshot_id1).await?;
        let snapshot2 = self.get_snapshot(snapshot_id2).await?;

        let host_memory_diff = if snapshot1.host_memory == snapshot2.host_memory {
            MemoryDiff::Identical
        } else {
            MemoryDiff::Different {
                bytes_changed: count_different_bytes(
                    &snapshot1.host_memory,
                    &snapshot2.host_memory,
                ),
            }
        };

        let device_memory_diff = if snapshot1.device_memory == snapshot2.device_memory {
            MemoryDiff::Identical
        } else {
            MemoryDiff::Different {
                bytes_changed: count_different_bytes(
                    &snapshot1.device_memory,
                    &snapshot2.device_memory,
                ),
            }
        };

        Ok(SnapshotDiff {
            snapshot_id1,
            snapshot_id2,
            timestamp_diff: snapshot2.timestamp.saturating_sub(snapshot1.timestamp),
            host_memory_diff,
            device_memory_diff,
            generation_diff: snapshot2
                .execution_context
                .generation
                .saturating_sub(snapshot1.execution_context.generation),
        })
    }
}

/// Snapshot comparison result
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    pub snapshot_id1: Uuid,
    pub snapshot_id2: Uuid,
    pub timestamp_diff: u64,
    pub host_memory_diff: MemoryDiff,
    pub device_memory_diff: MemoryDiff,
    pub generation_diff: u64,
}

/// Memory difference information
#[derive(Debug, Clone)]
pub enum MemoryDiff {
    Identical,
    Different { bytes_changed: usize },
}

/// Count different bytes between two byte arrays
fn count_different_bytes(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    let mut different = 0;

    for i in 0..min_len {
        if a[i] != b[i] {
            different += 1;
        }
    }

    // Add difference in length
    different += a.len().abs_diff(b.len());

    different
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::types::ExecutionContext;

    #[tokio::test]
    async fn test_snapshot_manager_session_lifecycle() {
        let config = SnapshotConfig::default();
        let manager = SnapshotManager::new(config);
        let container_id = Uuid::new_v4();

        // Start session
        manager.start_session(container_id).await.unwrap();

        // Verify session exists
        let session_info = manager.get_session_info(container_id).await;
        assert!(session_info.is_some());
        assert_eq!(session_info.unwrap().container_id, container_id);

        // End session
        manager.end_session(container_id, false).await.unwrap();

        // Verify session is gone
        let session_info = manager.get_session_info(container_id).await;
        assert!(session_info.is_none());
    }

    #[tokio::test]
    async fn test_snapshot_creation_and_retrieval() {
        let config = SnapshotConfig::default();
        let manager = SnapshotManager::new(config);
        let container_id = Uuid::new_v4();

        // Start session
        manager.start_session(container_id).await.unwrap();

        // Create snapshot
        let snapshot_id = manager
            .create_snapshot(
                container_id,
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                KernelParameters::default(),
                ExecutionContext::new("test context".to_string()),
                "test snapshot".to_string(),
            )
            .await
            .unwrap();

        // Retrieve snapshot
        let snapshot = manager.get_snapshot(snapshot_id).await.unwrap();
        assert_eq!(snapshot.snapshot_id, snapshot_id);
        assert_eq!(snapshot.host_memory, vec![1, 2, 3, 4]);
        assert_eq!(snapshot.device_memory, vec![5, 6, 7, 8]);

        // List snapshots
        let snapshot_list = manager.list_snapshots(container_id).await.unwrap();
        assert_eq!(snapshot_list, vec![snapshot_id]);

        // Delete snapshot
        manager.delete_snapshot(snapshot_id).await.unwrap();

        // Verify deletion
        assert!(manager.get_snapshot(snapshot_id).await.is_err());
    }

    #[tokio::test]
    async fn test_auto_snapshot() {
        let mut config = SnapshotConfig::default();
        config.auto_snapshot = true;

        let manager = SnapshotManager::new(config);
        let container_id = Uuid::new_v4();

        // Start session
        manager.start_session(container_id).await.unwrap();

        // Auto snapshot should work
        let snapshot_id = manager
            .auto_snapshot(
                container_id,
                vec![1, 2, 3],
                vec![4, 5, 6],
                KernelParameters::default(),
                ExecutionContext::new("auto test".to_string()),
            )
            .await
            .unwrap();

        assert!(snapshot_id.is_some());

        // Disable auto snapshot
        let mut config = SnapshotConfig::default();
        config.auto_snapshot = false;
        let manager = SnapshotManager::new(config);

        manager.start_session(container_id).await.unwrap();

        let no_snapshot = manager
            .auto_snapshot(
                container_id,
                vec![1, 2, 3],
                vec![4, 5, 6],
                KernelParameters::default(),
                ExecutionContext::new("auto test".to_string()),
            )
            .await
            .unwrap();

        assert!(no_snapshot.is_none());
    }

    #[tokio::test]
    async fn test_snapshot_comparison() {
        let config = SnapshotConfig::default();
        let manager = SnapshotManager::new(config);
        let container_id = Uuid::new_v4();

        manager.start_session(container_id).await.unwrap();

        // Create two different snapshots
        let snapshot_id1 = manager
            .create_snapshot(
                container_id,
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                KernelParameters::default(),
                ExecutionContext::new("snapshot 1".to_string()),
                "first snapshot".to_string(),
            )
            .await
            .unwrap();

        let snapshot_id2 = manager
            .create_snapshot(
                container_id,
                vec![1, 2, 9, 4], // Different byte at index 2
                vec![5, 6, 7, 8],
                KernelParameters::default(),
                ExecutionContext::new("snapshot 2".to_string()),
                "second snapshot".to_string(),
            )
            .await
            .unwrap();

        // Compare snapshots
        let diff = manager
            .compare_snapshots(snapshot_id1, snapshot_id2)
            .await
            .unwrap();

        assert_eq!(diff.snapshot_id1, snapshot_id1);
        assert_eq!(diff.snapshot_id2, snapshot_id2);

        match diff.host_memory_diff {
            MemoryDiff::Different { bytes_changed } => {
                assert_eq!(bytes_changed, 1); // One byte different
            }
            MemoryDiff::Identical => panic!("Expected different memory"),
        }

        match diff.device_memory_diff {
            MemoryDiff::Identical => {} // Device memory should be identical
            MemoryDiff::Different { .. } => panic!("Expected identical device memory"),
        }
    }

    #[test]
    fn test_count_different_bytes() {
        assert_eq!(count_different_bytes(&[1, 2, 3], &[1, 2, 3]), 0);
        assert_eq!(count_different_bytes(&[1, 2, 3], &[1, 9, 3]), 1);
        assert_eq!(count_different_bytes(&[1, 2, 3], &[9, 8, 7]), 3);
        assert_eq!(count_different_bytes(&[1, 2, 3], &[1, 2]), 1); // Length difference
        assert_eq!(count_different_bytes(&[1, 2], &[1, 2, 3, 4]), 2); // Length difference
    }
}
