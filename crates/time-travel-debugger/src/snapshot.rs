use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::error::{Result, TimeDebuggerError};

/// Represents a point-in-time snapshot of agent state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateSnapshot {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub agent_id: Uuid,
    pub state_data: serde_json::Value,
    pub memory_usage: u64,
    pub metadata: HashMap<String, String>,
}

/// Represents a diff between two states for efficient storage
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateDiff {
    pub id: Uuid,
    pub from_snapshot: Uuid,
    pub to_snapshot: Uuid,
    pub timestamp: DateTime<Utc>,
    pub changes: Vec<StateChange>,
    pub metadata: HashMap<String, String>,
}

/// Individual state change within a diff
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateChange {
    pub path: String,
    pub change_type: ChangeType,
    pub old_value: Option<serde_json::Value>,
    pub new_value: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChangeType {
    Added,
    Modified,
    Removed,
}

/// Configuration for snapshot management
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    pub max_snapshots: usize,
    pub compression_enabled: bool,
    pub diff_threshold: f32, // Percentage of changes before creating full snapshot
    pub cleanup_interval: std::time::Duration,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            max_snapshots: 1000,
            compression_enabled: true,
            diff_threshold: 0.3,                                   // 30% changes
            cleanup_interval: std::time::Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Manages state snapshots and diffs with efficient storage
pub struct SnapshotManager {
    snapshots: Arc<DashMap<Uuid, StateSnapshot>>,
    diffs: Arc<DashMap<Uuid, StateDiff>>,
    agent_snapshots: Arc<DashMap<Uuid, Vec<Uuid>>>, // agent_id -> snapshot_ids
    config: SnapshotConfig,
    cleanup_task: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl SnapshotManager {
    pub fn new(config: SnapshotConfig) -> Self {
        Self {
            snapshots: Arc::new(DashMap::new()),
            diffs: Arc::new(DashMap::new()),
            agent_snapshots: Arc::new(DashMap::new()),
            config,
            cleanup_task: Arc::new(RwLock::new(None)),
        }
    }

    /// Create a new state snapshot
    pub async fn create_snapshot(
        &self,
        agent_id: Uuid,
        state_data: serde_json::Value,
        memory_usage: u64,
        metadata: HashMap<String, String>,
    ) -> Result<Uuid> {
        let snapshot_id = Uuid::new_v4();
        let timestamp = Utc::now();

        let snapshot = StateSnapshot {
            id: snapshot_id,
            timestamp,
            agent_id,
            state_data,
            memory_usage,
            metadata,
        };

        // Store the snapshot
        self.snapshots.insert(snapshot_id, snapshot);

        // Update agent snapshot list
        self.agent_snapshots
            .entry(agent_id)
            .or_insert_with(Vec::new)
            .push(snapshot_id);

        // Check if cleanup is needed
        self.maybe_trigger_cleanup().await;

        Ok(snapshot_id)
    }

    /// Get a snapshot by ID
    pub async fn get_snapshot(&self, snapshot_id: Uuid) -> Result<StateSnapshot> {
        self.snapshots
            .get(&snapshot_id)
            .map(|entry| entry.clone())
            .ok_or(TimeDebuggerError::SnapshotNotFound { id: snapshot_id })
    }

    /// Get all snapshots for an agent, sorted by timestamp
    pub async fn get_agent_snapshots(&self, agent_id: Uuid) -> Result<Vec<StateSnapshot>> {
        let snapshot_ids = self
            .agent_snapshots
            .get(&agent_id)
            .map(|ids| ids.clone())
            .unwrap_or_default();

        let mut snapshots = Vec::new();
        for id in snapshot_ids {
            if let Some(snapshot) = self.snapshots.get(&id) {
                snapshots.push(snapshot.clone());
            }
        }

        // Sort by timestamp
        snapshots.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(snapshots)
    }

    /// Create a diff between two snapshots
    pub async fn create_diff(&self, from_snapshot_id: Uuid, to_snapshot_id: Uuid) -> Result<Uuid> {
        let from_snapshot = self.get_snapshot(from_snapshot_id).await?;
        let to_snapshot = self.get_snapshot(to_snapshot_id).await?;

        let changes = self.compute_changes(&from_snapshot.state_data, &to_snapshot.state_data)?;

        let diff_id = Uuid::new_v4();
        let diff = StateDiff {
            id: diff_id,
            from_snapshot: from_snapshot_id,
            to_snapshot: to_snapshot_id,
            timestamp: Utc::now(),
            changes,
            metadata: HashMap::new(),
        };

        self.diffs.insert(diff_id, diff);
        Ok(diff_id)
    }

    /// Get a diff by ID
    pub async fn get_diff(&self, diff_id: Uuid) -> Result<StateDiff> {
        self.diffs
            .get(&diff_id)
            .map(|entry| entry.clone())
            .ok_or(TimeDebuggerError::InvalidDiff {
                reason: format!("Diff {} not found", diff_id),
            })
    }

    /// Get snapshot at specific timestamp (closest match)
    pub async fn get_snapshot_at_time(
        &self,
        agent_id: Uuid,
        target_time: DateTime<Utc>,
    ) -> Result<StateSnapshot> {
        let snapshots = self.get_agent_snapshots(agent_id).await?;

        if snapshots.is_empty() {
            return Err(TimeDebuggerError::SnapshotNotFound { id: agent_id });
        }

        // Find closest snapshot
        let closest = snapshots
            .into_iter()
            .min_by_key(|snapshot| (snapshot.timestamp - target_time).num_milliseconds().abs())
            .unwrap();

        Ok(closest)
    }

    /// Delete a snapshot and associated diffs
    pub async fn delete_snapshot(&self, snapshot_id: Uuid) -> Result<()> {
        let snapshot = self.get_snapshot(snapshot_id).await?;

        // Remove from agent snapshots
        if let Some(mut agent_snapshots) = self.agent_snapshots.get_mut(&snapshot.agent_id) {
            agent_snapshots.retain(|&id| id != snapshot_id);
        }

        // Remove associated diffs
        let diffs_to_remove: Vec<Uuid> = self
            .diffs
            .iter()
            .filter_map(|entry| {
                let diff = entry.value();
                if diff.from_snapshot == snapshot_id || diff.to_snapshot == snapshot_id {
                    Some(diff.id)
                } else {
                    None
                }
            })
            .collect();

        for diff_id in diffs_to_remove {
            self.diffs.remove(&diff_id);
        }

        // Remove the snapshot
        self.snapshots.remove(&snapshot_id);
        Ok(())
    }

    /// Get memory usage statistics
    pub async fn get_memory_stats(&self) -> (usize, usize, u64) {
        let snapshot_count = self.snapshots.len();
        let diff_count = self.diffs.len();
        let total_memory: u64 = self
            .snapshots
            .iter()
            .map(|entry| entry.value().memory_usage)
            .sum();

        (snapshot_count, diff_count, total_memory)
    }

    /// Start automatic cleanup task
    pub async fn start_cleanup_task(&self) {
        let mut cleanup_task = self.cleanup_task.write().await;
        if cleanup_task.is_some() {
            return; // Already running
        }

        let snapshots = Arc::clone(&self.snapshots);
        let agent_snapshots = Arc::clone(&self.agent_snapshots);
        let config = self.config.clone();

        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.cleanup_interval);
            loop {
                interval.tick().await;
                Self::cleanup_old_snapshots(&snapshots, &agent_snapshots, &config).await;
            }
        });

        *cleanup_task = Some(task);
    }

    /// Stop the cleanup task
    pub async fn stop_cleanup_task(&self) {
        let mut cleanup_task = self.cleanup_task.write().await;
        if let Some(task) = cleanup_task.take() {
            task.abort();
        }
    }

    // Private helper methods

    fn compute_changes(
        &self,
        from_value: &serde_json::Value,
        to_value: &serde_json::Value,
    ) -> Result<Vec<StateChange>> {
        let mut changes = Vec::new();
        self.compute_changes_recursive("", from_value, to_value, &mut changes)?;
        Ok(changes)
    }

    fn compute_changes_recursive(
        &self,
        path: &str,
        from_value: &serde_json::Value,
        to_value: &serde_json::Value,
        changes: &mut Vec<StateChange>,
    ) -> Result<()> {
        use serde_json::Value;

        match (from_value, to_value) {
            (Value::Object(from_obj), Value::Object(to_obj)) => {
                // Check for removed or modified keys
                for (key, from_val) in from_obj {
                    let new_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", path, key)
                    };

                    match to_obj.get(key) {
                        Some(to_val) => {
                            self.compute_changes_recursive(&new_path, from_val, to_val, changes)?;
                        }
                        None => {
                            changes.push(StateChange {
                                path: new_path,
                                change_type: ChangeType::Removed,
                                old_value: Some(from_val.clone()),
                                new_value: None,
                            });
                        }
                    }
                }

                // Check for added keys
                for (key, to_val) in to_obj {
                    if !from_obj.contains_key(key) {
                        let new_path = if path.is_empty() {
                            key.clone()
                        } else {
                            format!("{}.{}", path, key)
                        };

                        changes.push(StateChange {
                            path: new_path,
                            change_type: ChangeType::Added,
                            old_value: None,
                            new_value: Some(to_val.clone()),
                        });
                    }
                }
            }
            _ => {
                if from_value != to_value {
                    changes.push(StateChange {
                        path: path.to_string(),
                        change_type: ChangeType::Modified,
                        old_value: Some(from_value.clone()),
                        new_value: Some(to_value.clone()),
                    });
                }
            }
        }

        Ok(())
    }

    async fn maybe_trigger_cleanup(&self) {
        if self.snapshots.len() > self.config.max_snapshots {
            let snapshots = Arc::clone(&self.snapshots);
            let agent_snapshots = Arc::clone(&self.agent_snapshots);
            let config = self.config.clone();

            tokio::spawn(async move {
                Self::cleanup_old_snapshots(&snapshots, &agent_snapshots, &config).await;
            });
        }
    }

    async fn cleanup_old_snapshots(
        snapshots: &DashMap<Uuid, StateSnapshot>,
        agent_snapshots: &DashMap<Uuid, Vec<Uuid>>,
        config: &SnapshotConfig,
    ) {
        if snapshots.len() <= config.max_snapshots {
            return;
        }

        // Collect all snapshots with timestamps
        let mut all_snapshots: Vec<(Uuid, DateTime<Utc>)> = snapshots
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().timestamp))
            .collect();

        // Sort by timestamp (oldest first)
        all_snapshots.sort_by(|a, b| a.1.cmp(&b.1));

        // Remove oldest snapshots
        let to_remove = all_snapshots.len().saturating_sub(config.max_snapshots);
        for (snapshot_id, _) in all_snapshots.into_iter().take(to_remove) {
            if let Some(snapshot) = snapshots.remove(&snapshot_id) {
                let snapshot = snapshot.1;
                // Remove from agent snapshots
                if let Some(mut agent_list) = agent_snapshots.get_mut(&snapshot.agent_id) {
                    agent_list.retain(|&id| id != snapshot_id);
                }
            }
        }
    }
}

impl Drop for SnapshotManager {
    fn drop(&mut self) {
        // Cancel cleanup task when dropping
        if let Ok(mut task) = self.cleanup_task.try_write() {
            if let Some(handle) = task.take() {
                handle.abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_manager() -> SnapshotManager {
        let config = SnapshotConfig {
            max_snapshots: 5,
            compression_enabled: false,
            diff_threshold: 0.3,
            cleanup_interval: std::time::Duration::from_millis(100),
        };
        SnapshotManager::new(config)
    }

    fn create_test_manager_concurrent() -> SnapshotManager {
        let config = SnapshotConfig {
            max_snapshots: 20, // Increased to handle concurrent test
            compression_enabled: false,
            diff_threshold: 0.3,
            cleanup_interval: std::time::Duration::from_secs(60), // Longer cleanup interval
        };
        SnapshotManager::new(config)
    }

    #[tokio::test]
    async fn test_create_and_get_snapshot() {
        let manager = create_test_manager();
        let agent_id = Uuid::new_v4();
        let state_data = json!({"key": "value", "number": 42});
        let metadata = HashMap::new();

        let snapshot_id = manager
            .create_snapshot(agent_id, state_data.clone(), 1024, metadata)
            .await
            .unwrap();

        let retrieved = manager.get_snapshot(snapshot_id).await?;
        assert_eq!(retrieved.agent_id, agent_id);
        assert_eq!(retrieved.state_data, state_data);
        assert_eq!(retrieved.memory_usage, 1024);
    }

    #[tokio::test]
    async fn test_get_nonexistent_snapshot() {
        let manager = create_test_manager();
        let fake_id = Uuid::new_v4();

        let result = manager.get_snapshot(fake_id).await;
        assert!(matches!(
            result,
            Err(TimeDebuggerError::SnapshotNotFound { .. })
        ));
    }

    #[tokio::test]
    async fn test_get_agent_snapshots() {
        let manager = create_test_manager();
        let agent_id = Uuid::new_v4();

        // Create multiple snapshots
        let snapshot1 = manager
            .create_snapshot(agent_id, json!({"version": 1}), 1024, HashMap::new())
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let snapshot2 = manager
            .create_snapshot(agent_id, json!({"version": 2}), 2048, HashMap::new())
            .await
            .unwrap();

        let snapshots = manager.get_agent_snapshots(agent_id).await?;
        assert_eq!(snapshots.len(), 2);

        // Should be sorted by timestamp
        assert!(snapshots[0].timestamp <= snapshots[1].timestamp);
        assert_eq!(snapshots[0].id, snapshot1);
        assert_eq!(snapshots[1].id, snapshot2);
    }

    #[tokio::test]
    async fn test_create_diff() {
        let manager = create_test_manager();
        let agent_id = Uuid::new_v4();

        let snapshot1_id = manager
            .create_snapshot(
                agent_id,
                json!({"key": "value1", "number": 42}),
                1024,
                HashMap::new(),
            )
            .await
            .unwrap();

        let snapshot2_id = manager
            .create_snapshot(
                agent_id,
                json!({"key": "value2", "number": 42, "new_key": "new_value"}),
                2048,
                HashMap::new(),
            )
            .await
            .unwrap();

        let diff_id = manager
            .create_diff(snapshot1_id, snapshot2_id)
            .await
            .unwrap();
        let diff = manager.get_diff(diff_id).await?;

        assert_eq!(diff.from_snapshot, snapshot1_id);
        assert_eq!(diff.to_snapshot, snapshot2_id);
        assert!(!diff.changes.is_empty());

        // Verify changes
        let key_change = diff.changes.iter().find(|c| c.path == "key")?;
        assert!(matches!(key_change.change_type, ChangeType::Modified));
        assert_eq!(key_change.old_value, Some(json!("value1")));
        assert_eq!(key_change.new_value, Some(json!("value2")));

        let new_key_change = diff.changes.iter().find(|c| c.path == "new_key")?;
        assert!(matches!(new_key_change.change_type, ChangeType::Added));
        assert_eq!(new_key_change.old_value, None);
        assert_eq!(new_key_change.new_value, Some(json!("new_value")));
    }

    #[tokio::test]
    async fn test_get_snapshot_at_time() {
        let manager = create_test_manager();
        let agent_id = Uuid::new_v4();

        let _time1 = Utc::now();
        let snapshot1_id = manager
            .create_snapshot(agent_id, json!({"version": 1}), 1024, HashMap::new())
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let time2 = Utc::now();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let snapshot2_id = manager
            .create_snapshot(agent_id, json!({"version": 2}), 2048, HashMap::new())
            .await
            .unwrap();

        // Query at time2 should return snapshot1 (closest)
        let closest = manager.get_snapshot_at_time(agent_id, time2).await?;
        assert_eq!(closest.id, snapshot1_id);

        // Query at current time should return snapshot2
        let latest = manager
            .get_snapshot_at_time(agent_id, Utc::now())
            .await
            .unwrap();
        assert_eq!(latest.id, snapshot2_id);
    }

    #[tokio::test]
    async fn test_delete_snapshot() {
        let manager = create_test_manager();
        let agent_id = Uuid::new_v4();

        let snapshot_id = manager
            .create_snapshot(agent_id, json!({"key": "value"}), 1024, HashMap::new())
            .await
            .unwrap();

        // Verify snapshot exists
        assert!(manager.get_snapshot(snapshot_id).await.is_ok());

        // Delete snapshot
        manager.delete_snapshot(snapshot_id).await?;

        // Verify snapshot is gone
        assert!(matches!(
            manager.get_snapshot(snapshot_id).await,
            Err(TimeDebuggerError::SnapshotNotFound { .. })
        ));

        // Verify it's removed from agent snapshots
        let agent_snapshots = manager.get_agent_snapshots(agent_id).await?;
        assert!(agent_snapshots.is_empty());
    }

    #[tokio::test]
    async fn test_memory_stats() {
        let manager = create_test_manager();
        let agent_id = Uuid::new_v4();

        let (count, diff_count, memory) = manager.get_memory_stats().await;
        assert_eq!(count, 0);
        assert_eq!(diff_count, 0);
        assert_eq!(memory, 0);

        manager
            .create_snapshot(agent_id, json!({"key": "value"}), 1024, HashMap::new())
            .await
            .unwrap();

        let (count, diff_count, memory) = manager.get_memory_stats().await;
        assert_eq!(count, 1);
        assert_eq!(diff_count, 0);
        assert_eq!(memory, 1024);
    }

    #[tokio::test]
    async fn test_cleanup_task() {
        let manager = create_test_manager();
        let agent_id = Uuid::new_v4();

        // Create more snapshots than the limit
        for i in 0..7 {
            manager
                .create_snapshot(agent_id, json!({"version": i}), 1024, HashMap::new())
                .await
                .unwrap();
        }

        // Wait for cleanup to trigger
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let (count, _, _) = manager.get_memory_stats().await;
        assert!(count <= 5); // Should be cleaned up to max_snapshots
    }

    #[tokio::test]
    async fn test_change_computation() {
        let manager = create_test_manager();

        let from_value = json!({
            "removed_key": "removed_value",
            "modified_key": "old_value",
            "unchanged_key": "same_value",
            "nested": {
                "nested_removed": "removed",
                "nested_modified": "old"
            }
        });

        let to_value = json!({
            "modified_key": "new_value",
            "unchanged_key": "same_value",
            "added_key": "added_value",
            "nested": {
                "nested_modified": "new",
                "nested_added": "added"
            }
        });

        let changes = manager.compute_changes(&from_value, &to_value)?;

        // Find specific changes
        let removed = changes.iter().find(|c| c.path == "removed_key")?;
        assert!(matches!(removed.change_type, ChangeType::Removed));

        let modified = changes.iter().find(|c| c.path == "modified_key")?;
        assert!(matches!(modified.change_type, ChangeType::Modified));

        let added = changes.iter().find(|c| c.path == "added_key")?;
        assert!(matches!(added.change_type, ChangeType::Added));

        let nested_removed = changes
            .iter()
            .find(|c| c.path == "nested.nested_removed")
            .unwrap();
        assert!(matches!(nested_removed.change_type, ChangeType::Removed));
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let manager = Arc::new(create_test_manager_concurrent());
        let agent_id = Uuid::new_v4();

        // Spawn multiple tasks creating snapshots concurrently
        let mut handles = Vec::new();
        for i in 0..10 {
            let manager_clone = Arc::clone(&manager);
            let handle = tokio::spawn(async move {
                manager_clone
                    .create_snapshot(agent_id, json!({"version": i}), 1024, HashMap::new())
                    .await
                    .unwrap()
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let snapshot_ids: Vec<Uuid> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|result| result.unwrap())
            .collect();

        assert_eq!(snapshot_ids.len(), 10);

        // Verify all snapshots exist
        for snapshot_id in snapshot_ids {
            assert!(manager.get_snapshot(snapshot_id).await.is_ok());
        }

        let agent_snapshots = manager.get_agent_snapshots(agent_id).await?;
        assert_eq!(agent_snapshots.len(), 10);
    }
}
