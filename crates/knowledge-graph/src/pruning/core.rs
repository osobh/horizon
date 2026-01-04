//! Core pruning system that coordinates all pruning functionality
//!
//! This module contains the main PruningSystem struct that coordinates
//! access tracking, statistics, backup management, and strategy execution.

use super::access_tracking::AccessTracker;
use super::backup::BackupManager;
use super::stats::StatsManager;
use super::strategies::StrategyExecutor;
use super::types::{PruningConfig, PruningStats};
use crate::error::KnowledgeGraphResult;
use crate::graph::{KnowledgeGraph, NodeType};
use chrono::{DateTime, Utc};
use std::collections::{HashMap, HashSet};

/// Main pruning system that coordinates all pruning functionality
pub struct PruningSystem {
    /// Configuration
    config: PruningConfig,
    /// Entity access tracking
    access_tracker: AccessTracker,
    /// Statistics manager
    stats_manager: StatsManager,
    /// Backup manager
    backup_manager: BackupManager,
    /// Strategy executor
    strategy_executor: StrategyExecutor,
    /// Last pruning time
    last_pruning: Option<DateTime<Utc>>,
}

impl PruningSystem {
    /// Create a new pruning system
    pub fn new(config: PruningConfig) -> Self {
        let backup_manager = BackupManager::new(config.max_backup_size);

        Self {
            config,
            access_tracker: AccessTracker::new(),
            stats_manager: StatsManager::new(),
            backup_manager,
            strategy_executor: StrategyExecutor::new(),
            last_pruning: None,
        }
    }

    /// Run pruning operation
    pub async fn prune(
        &mut self,
        graph: &mut KnowledgeGraph,
    ) -> KnowledgeGraphResult<PruningStats> {
        let start_time = std::time::Instant::now();
        let initial_stats = graph.stats();

        // Check if pruning is needed
        if !self.should_prune() {
            return Ok(self.stats_manager.stats().clone());
        }

        let mut all_nodes_to_remove: HashSet<String> = HashSet::new();

        // Apply each pruning strategy
        for strategy in &self.config.strategies.clone() {
            let nodes_to_remove = self
                .strategy_executor
                .execute_strategy(strategy, graph, &self.access_tracker)
                .await?;

            all_nodes_to_remove.extend(nodes_to_remove);
        }

        // Backup and remove entities
        let mut nodes_removed = 0;
        let mut nodes_by_type: HashMap<NodeType, usize> = HashMap::new();

        for node_id in all_nodes_to_remove {
            if let Ok(node) = graph.get_node(&node_id) {
                // Track node type
                *nodes_by_type.entry(node.node_type.clone()).or_insert(0) += 1;

                // Backup if enabled
                if self.config.backup_removed {
                    self.backup_manager
                        .backup_entity(node.clone(), "node", "pruning")
                        .await?;
                }

                // Remove node (this will also remove connected edges)
                graph.remove_node(&node_id)?;
                nodes_removed += 1;
            }
        }

        // Calculate statistics
        let final_stats = graph.stats();
        let duration = start_time.elapsed().as_millis() as f64;
        let edges_removed = initial_stats.edge_count - final_stats.edge_count;

        // Update statistics
        self.stats_manager.update_after_pruning(
            nodes_removed,
            edges_removed,
            duration,
            nodes_by_type,
        );

        self.last_pruning = Some(Utc::now());

        Ok(self.stats_manager.stats().clone())
    }

    /// Check if pruning should run
    fn should_prune(&self) -> bool {
        if !self.config.auto_pruning {
            return false;
        }

        if let Some(last) = self.last_pruning {
            let hours_since = (Utc::now() - last).num_hours();
            hours_since >= self.config.pruning_interval_hours
        } else {
            true // First time
        }
    }

    /// Record entity access
    pub fn record_access(&mut self, entity_id: String) {
        self.access_tracker.record_access(entity_id);
    }

    /// Get pruning statistics
    pub fn stats(&self) -> &PruningStats {
        self.stats_manager.stats()
    }

    /// Get removed entities backup
    pub fn get_backup(&self, limit: Option<usize>) -> Vec<&super::types::RemovedEntity> {
        self.backup_manager.get_backup(limit)
    }

    /// Restore entity from backup
    pub async fn restore_entity(
        &mut self,
        backup_index: usize,
        graph: &mut KnowledgeGraph,
    ) -> KnowledgeGraphResult<String> {
        self.backup_manager
            .restore_entity(backup_index, graph)
            .await
    }

    /// Force pruning regardless of interval
    pub async fn force_prune(
        &mut self,
        graph: &mut KnowledgeGraph,
    ) -> KnowledgeGraphResult<PruningStats> {
        self.last_pruning = None; // Reset to force pruning
        self.prune(graph).await
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PruningConfig) {
        // Update backup manager if max size changed
        if config.max_backup_size != self.config.max_backup_size {
            self.backup_manager.set_max_size(config.max_backup_size);
        }

        self.config = config;
    }

    /// Clear access tracking
    pub fn clear_access_tracking(&mut self) {
        self.access_tracker.clear();
    }

    /// Get configuration
    pub fn config(&self) -> &PruningConfig {
        &self.config
    }

    /// Get access tracking statistics
    pub fn access_tracking_stats(&self) -> (usize, bool) {
        (self.access_tracker.len(), self.access_tracker.is_empty())
    }

    /// Get backup statistics
    pub fn backup_stats(&self) -> (usize, usize, bool) {
        (
            self.backup_manager.len(),
            self.backup_manager.max_size(),
            self.backup_manager.is_empty(),
        )
    }

    /// Get importance score for a node (if cached)
    pub fn get_node_importance(&self, node_id: &str) -> Option<f64> {
        self.strategy_executor.get_importance(node_id)
    }

    /// Clear all caches and reset state (except configuration)
    pub fn reset_state(&mut self) {
        self.access_tracker.clear();
        self.backup_manager.clear();
        self.stats_manager.reset();
        self.strategy_executor.clear_importance_cache();
        self.last_pruning = None;
    }

    /// Check if the pruning system is healthy
    pub fn is_healthy(&self) -> bool {
        self.stats_manager.is_healthy()
    }

    /// Get a comprehensive system report
    pub fn system_report(&self) -> String {
        let (access_count, access_empty) = self.access_tracking_stats();
        let (backup_count, backup_max, backup_empty) = self.backup_stats();

        format!(
            "Pruning System Report:\n\
             ===================\n\
             Configuration:\n\
             - Auto Pruning: {}\n\
             - Pruning Interval: {} hours\n\
             - Strategies: {} configured\n\
             - Backup Enabled: {}\n\
             \n\
             Access Tracking:\n\
             - Tracked Entities: {}\n\
             - Status: {}\n\
             \n\
             Backup System:\n\
             - Current Backups: {}/{}\n\
             - Status: {}\n\
             \n\
             {}\n\
             \n\
             System Health: {}\n\
             Last Pruning: {}",
            self.config.auto_pruning,
            self.config.pruning_interval_hours,
            self.config.strategies.len(),
            self.config.backup_removed,
            access_count,
            if access_empty { "Empty" } else { "Active" },
            backup_count,
            backup_max,
            if backup_empty { "Empty" } else { "Active" },
            self.stats_manager.summary_report(),
            if self.is_healthy() {
                "✅ Healthy"
            } else {
                "⚠️ Unhealthy"
            },
            self.last_pruning
                .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| "Never".to_string())
        )
    }

    /// Cleanup old access records to free memory
    pub fn cleanup_old_access_records(&mut self, max_age_hours: i64) {
        self.access_tracker.cleanup_old_records(max_age_hours);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{KnowledgeGraphConfig, Node};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_pruning_system_creation() {
        let config = PruningConfig::default();
        let system = PruningSystem::new(config);

        assert_eq!(system.stats().total_operations, 0);
        assert!(system.is_healthy());
        let (backup_count, _, backup_empty) = system.backup_stats();
        assert_eq!(backup_count, 0);
        assert!(backup_empty);
    }

    #[tokio::test]
    async fn test_should_prune_logic() {
        let mut config = PruningConfig::default();
        config.auto_pruning = true;
        config.pruning_interval_hours = 1;

        let mut system = PruningSystem::new(config);

        // Should prune on first run
        assert!(system.should_prune());

        // Set recent pruning time
        system.last_pruning = Some(Utc::now());
        assert!(!system.should_prune());

        // Set old pruning time
        system.last_pruning = Some(Utc::now() - Duration::hours(2));
        assert!(system.should_prune());

        // Disable auto pruning
        let mut new_config = system.config().clone();
        new_config.auto_pruning = false;
        system.update_config(new_config);
        assert!(!system.should_prune());
    }

    #[tokio::test]
    async fn test_access_tracking_integration() {
        let config = PruningConfig::default();
        let mut system = PruningSystem::new(config);

        let entity_id = "test_entity".to_string();

        let (count, empty) = system.access_tracking_stats();
        assert_eq!(count, 0);
        assert!(empty);

        system.record_access(entity_id.clone());
        system.record_access(entity_id.clone());

        let (count, empty) = system.access_tracking_stats();
        assert_eq!(count, 1);
        assert!(!empty);

        system.clear_access_tracking();

        let (count, empty) = system.access_tracking_stats();
        assert_eq!(count, 0);
        assert!(empty);
    }

    #[tokio::test]
    async fn test_force_prune() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        let mut config = PruningConfig::default();
        config.pruning_interval_hours = 24; // Long interval
        let mut system = PruningSystem::new(config);

        // Set recent pruning time
        system.last_pruning = Some(Utc::now());

        // Normal prune should not run
        let stats = system.prune(&mut graph).await.unwrap();
        assert_eq!(stats.total_operations, 0);

        // Force prune should run
        let stats = system.force_prune(&mut graph).await.unwrap();
        assert_eq!(stats.total_operations, 1);
    }

    #[tokio::test]
    async fn test_config_update() {
        let config = PruningConfig::default();
        let mut system = PruningSystem::new(config);

        let mut new_config = PruningConfig::default();
        new_config.max_backup_size = 500;
        new_config.auto_pruning = false;

        system.update_config(new_config);

        assert!(!system.config().auto_pruning);
        assert_eq!(system.backup_stats().1, 500); // max_size
    }

    #[tokio::test]
    async fn test_system_reset() {
        let config = PruningConfig::default();
        let mut system = PruningSystem::new(config);

        // Add some state
        system.record_access("test".to_string());
        system.last_pruning = Some(Utc::now());

        let (access_count, _) = system.access_tracking_stats();
        assert_eq!(access_count, 1);
        assert!(system.last_pruning.is_some());

        // Reset state
        system.reset_state();

        let (access_count, access_empty) = system.access_tracking_stats();
        assert_eq!(access_count, 0);
        assert!(access_empty);
        assert!(system.last_pruning.is_none());
    }

    #[test]
    fn test_system_report() {
        let config = PruningConfig::default();
        let mut system = PruningSystem::new(config);

        // Add some activity
        system.record_access("test_entity".to_string());

        let report = system.system_report();

        assert!(report.contains("Pruning System Report"));
        assert!(report.contains("Configuration:"));
        assert!(report.contains("Access Tracking:"));
        assert!(report.contains("Backup System:"));
        assert!(report.contains("System Health:"));
    }

    #[tokio::test]
    async fn test_backup_integration() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        let mut config = PruningConfig::default();
        config.backup_removed = true;
        let mut system = PruningSystem::new(config);

        // Add a node to be pruned
        let node = Node::new(NodeType::Memory, HashMap::new());
        let node_id = node.id.clone();
        graph.add_node(node).unwrap();

        // Configure strategy to remove all nodes
        let mut new_config = system.config().clone();
        new_config.strategies = vec![super::types::PruningStrategy::SizeBased {
            max_nodes: 0,
            max_edges: 0,
            priority: super::types::RemovalPriority::OldestFirst,
        }];
        system.update_config(new_config);

        // Run pruning
        system.force_prune(&mut graph).await.unwrap();

        // Check backup was created
        let backups = system.get_backup(None);
        assert_eq!(backups.len(), 1);

        // Restore and verify
        let restored_id = system.restore_entity(0, &mut graph).await.unwrap();
        assert_eq!(restored_id, node_id);
        assert!(graph.get_node(&node_id).is_ok());
    }
}
