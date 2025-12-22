//! Backup and restore functionality for pruned entities
//!
//! This module handles backing up entities that are removed during pruning
//! and provides functionality to restore them if needed.

use super::types::RemovedEntity;
use crate::error::KnowledgeGraphResult;
use crate::graph::{KnowledgeGraph, Node};
use chrono::{DateTime, Utc};

/// Backup manager for removed entities
pub struct BackupManager {
    /// Backup of removed entities
    backup: Vec<RemovedEntity>,
    /// Maximum backup size
    max_size: usize,
}

impl BackupManager {
    /// Create a new backup manager
    pub fn new(max_size: usize) -> Self {
        Self {
            backup: Vec::new(),
            max_size,
        }
    }

    /// Backup a removed entity
    pub async fn backup_entity(
        &mut self,
        entity: Node,
        entity_type: &str,
        reason: &str,
    ) -> KnowledgeGraphResult<()> {
        let backup_entry = RemovedEntity {
            entity: serde_json::to_value(&entity)?,
            entity_type: entity_type.to_string(),
            removed_at: Utc::now(),
            reason: reason.to_string(),
        };

        self.backup.push(backup_entry);
        self.cleanup_if_needed();

        Ok(())
    }

    /// Get backup entries with optional limit
    pub fn get_backup(&self, limit: Option<usize>) -> Vec<&RemovedEntity> {
        match limit {
            Some(n) => self.backup.iter().take(n).collect(),
            None => self.backup.iter().collect(),
        }
    }

    /// Restore entity from backup by index
    pub async fn restore_entity(
        &mut self,
        backup_index: usize,
        graph: &mut KnowledgeGraph,
    ) -> KnowledgeGraphResult<String> {
        if backup_index >= self.backup.len() {
            return Err(crate::error::KnowledgeGraphError::Other(
                "Backup index out of range".to_string(),
            ));
        }

        let backup_entry = self.backup.remove(backup_index);

        if backup_entry.entity_type == "node" {
            let node: Node = serde_json::from_value(backup_entry.entity)?;
            let node_id = node.id.clone();
            graph.add_node(node)?;
            Ok(node_id)
        } else {
            Err(crate::error::KnowledgeGraphError::Other(
                "Only node restoration is implemented".to_string(),
            ))
        }
    }

    /// Get number of backup entries
    pub fn len(&self) -> usize {
        self.backup.len()
    }

    /// Check if backup is empty
    pub fn is_empty(&self) -> bool {
        self.backup.is_empty()
    }

    /// Clear all backups
    pub fn clear(&mut self) {
        self.backup.clear();
    }

    /// Update maximum backup size
    pub fn set_max_size(&mut self, max_size: usize) {
        self.max_size = max_size;
        self.cleanup_if_needed();
    }

    /// Get maximum backup size
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Cleanup backup to maintain size limit
    fn cleanup_if_needed(&mut self) {
        if self.backup.len() > self.max_size {
            // Sort by removal time and keep most recent
            self.backup.sort_by_key(|backup| backup.removed_at);
            let keep_count = self.max_size;
            let temp_backup = std::mem::take(&mut self.backup);
            self.backup = temp_backup.into_iter().rev().take(keep_count).collect();
        }
    }

    /// Get backup entries newer than specified timestamp
    pub fn get_recent_backups(&self, since: DateTime<Utc>) -> Vec<&RemovedEntity> {
        self.backup
            .iter()
            .filter(|entry| entry.removed_at > since)
            .collect()
    }

    /// Get backup entries by reason
    pub fn get_backups_by_reason(&self, reason: &str) -> Vec<&RemovedEntity> {
        self.backup
            .iter()
            .filter(|entry| entry.reason == reason)
            .collect()
    }

    /// Get backup entries by entity type
    pub fn get_backups_by_type(&self, entity_type: &str) -> Vec<&RemovedEntity> {
        self.backup
            .iter()
            .filter(|entry| entry.entity_type == entity_type)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{KnowledgeGraphConfig, NodeType};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_backup_and_restore() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;
        let mut backup_manager = BackupManager::new(10);

        let node = Node::new(NodeType::Memory, HashMap::new());
        let node_id = node.id.clone();
        let node_properties = node.properties.clone();
        graph.add_node(node.clone()).unwrap();

        // Backup the node
        backup_manager
            .backup_entity(node, "node", "test")
            .await
            .unwrap();

        assert_eq!(backup_manager.len(), 1);

        // Remove from graph
        graph.remove_node(&node_id).unwrap();
        assert!(graph.get_node(&node_id).is_err());

        // Restore from backup
        let restored_id = backup_manager.restore_entity(0, &mut graph).await.unwrap();

        assert_eq!(restored_id, node_id);
        assert_eq!(backup_manager.len(), 0);

        // Verify restoration
        let restored_node = graph.get_node(&node_id).unwrap();
        assert_eq!(restored_node.properties, node_properties);
    }

    #[test]
    fn test_backup_size_limit() {
        let mut backup_manager = BackupManager::new(3);

        // Add more backups than the limit
        for i in 0..5 {
            let backup = RemovedEntity {
                entity: serde_json::json!({"id": i}),
                entity_type: "node".to_string(),
                removed_at: Utc::now(),
                reason: "test".to_string(),
            };
            backup_manager.backup.push(backup);
        }

        backup_manager.cleanup_if_needed();

        // Should keep only the most recent 3
        assert_eq!(backup_manager.len(), 3);
    }

    #[test]
    fn test_backup_filtering() {
        let mut backup_manager = BackupManager::new(10);
        let now = Utc::now();

        // Add different types of backups
        for i in 0..3 {
            let backup = RemovedEntity {
                entity: serde_json::json!({"id": i}),
                entity_type: if i == 0 { "node" } else { "edge" }.to_string(),
                removed_at: now,
                reason: if i < 2 { "pruning" } else { "cleanup" }.to_string(),
            };
            backup_manager.backup.push(backup);
        }

        // Test filtering by type
        let node_backups = backup_manager.get_backups_by_type("node");
        assert_eq!(node_backups.len(), 1);

        let edge_backups = backup_manager.get_backups_by_type("edge");
        assert_eq!(edge_backups.len(), 2);

        // Test filtering by reason
        let pruning_backups = backup_manager.get_backups_by_reason("pruning");
        assert_eq!(pruning_backups.len(), 2);

        let cleanup_backups = backup_manager.get_backups_by_reason("cleanup");
        assert_eq!(cleanup_backups.len(), 1);
    }

    #[test]
    fn test_backup_max_size_update() {
        let mut backup_manager = BackupManager::new(5);

        // Fill with backups
        for i in 0..7 {
            let backup = RemovedEntity {
                entity: serde_json::json!({"id": i}),
                entity_type: "node".to_string(),
                removed_at: Utc::now(),
                reason: "test".to_string(),
            };
            backup_manager.backup.push(backup);
        }

        assert_eq!(backup_manager.len(), 7);

        // Reduce max size
        backup_manager.set_max_size(3);
        assert_eq!(backup_manager.len(), 3);
        assert_eq!(backup_manager.max_size(), 3);
    }

    #[test]
    fn test_backup_clear() {
        let mut backup_manager = BackupManager::new(10);

        // Add some backups
        for i in 0..3 {
            let backup = RemovedEntity {
                entity: serde_json::json!({"id": i}),
                entity_type: "node".to_string(),
                removed_at: Utc::now(),
                reason: "test".to_string(),
            };
            backup_manager.backup.push(backup);
        }

        assert_eq!(backup_manager.len(), 3);
        assert!(!backup_manager.is_empty());

        backup_manager.clear();

        assert_eq!(backup_manager.len(), 0);
        assert!(backup_manager.is_empty());
    }
}
