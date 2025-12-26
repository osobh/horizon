//! Statistics tracking for pruning operations
//!
//! This module handles collecting and calculating statistics for pruning
//! operations to provide insights into system performance and behavior.

use super::types::PruningStats;
use crate::graph::NodeType;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Statistics manager for pruning operations
pub struct StatsManager {
    /// Current statistics
    stats: PruningStats,
}

impl StatsManager {
    /// Create a new statistics manager
    pub fn new() -> Self {
        Self {
            stats: PruningStats {
                total_operations: 0,
                nodes_pruned: 0,
                edges_pruned: 0,
                pruned_by_type: HashMap::new(),
                last_pruning: None,
                avg_duration_ms: 0.0,
                space_saved_bytes: 0,
            },
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &PruningStats {
        &self.stats
    }

    /// Update statistics after a pruning operation
    pub fn update_after_pruning(
        &mut self,
        nodes_removed: usize,
        edges_removed: usize,
        duration_ms: f64,
        nodes_by_type: HashMap<NodeType, usize>,
    ) {
        self.stats.total_operations += 1;
        self.stats.nodes_pruned += nodes_removed;
        self.stats.edges_pruned += edges_removed;
        self.stats.last_pruning = Some(Utc::now());

        // Update average duration
        self.stats.avg_duration_ms =
            (self.stats.avg_duration_ms * (self.stats.total_operations - 1) as f64 + duration_ms)
                / self.stats.total_operations as f64;

        // Update nodes pruned by type
        for (node_type, count) in nodes_by_type {
            *self.stats.pruned_by_type.entry(node_type).or_insert(0) += count;
        }

        // Estimate space saved (rough calculation)
        let space_saved = (nodes_removed * 1000 + edges_removed * 500) as u64;
        self.stats.space_saved_bytes += space_saved;
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.stats = PruningStats {
            total_operations: 0,
            nodes_pruned: 0,
            edges_pruned: 0,
            pruned_by_type: HashMap::new(),
            last_pruning: None,
            avg_duration_ms: 0.0,
            space_saved_bytes: 0,
        };
    }

    /// Get total nodes pruned
    pub fn total_nodes_pruned(&self) -> usize {
        self.stats.nodes_pruned
    }

    /// Get total edges pruned
    pub fn total_edges_pruned(&self) -> usize {
        self.stats.edges_pruned
    }

    /// Get total operations
    pub fn total_operations(&self) -> u64 {
        self.stats.total_operations
    }

    /// Get average duration per operation
    pub fn avg_duration_ms(&self) -> f64 {
        self.stats.avg_duration_ms
    }

    /// Get last pruning timestamp
    pub fn last_pruning(&self) -> Option<DateTime<Utc>> {
        self.stats.last_pruning
    }

    /// Get estimated space saved
    pub fn space_saved_bytes(&self) -> u64 {
        self.stats.space_saved_bytes
    }

    /// Get nodes pruned by type
    pub fn pruned_by_type(&self) -> &HashMap<NodeType, usize> {
        &self.stats.pruned_by_type
    }

    /// Get pruning rate (operations per hour) based on last pruning time
    pub fn pruning_rate_per_hour(&self) -> Option<f64> {
        if let Some(last_pruning) = self.stats.last_pruning {
            if self.stats.total_operations > 0 {
                let duration = Utc::now().signed_duration_since(last_pruning);
                let hours = duration.num_minutes() as f64 / 60.0;
                if hours > 0.0 {
                    return Some(self.stats.total_operations as f64 / hours);
                }
            }
        }
        None
    }

    /// Get efficiency metrics (nodes pruned per ms)
    pub fn pruning_efficiency(&self) -> f64 {
        if self.stats.avg_duration_ms > 0.0 {
            self.stats.nodes_pruned as f64 / self.stats.avg_duration_ms
        } else {
            0.0
        }
    }

    /// Check if statistics show healthy pruning behavior
    pub fn is_healthy(&self) -> bool {
        // Basic health checks
        if self.stats.total_operations == 0 {
            return true; // No operations yet, considered healthy
        }

        // Check if average duration is reasonable (< 10 seconds)
        if self.stats.avg_duration_ms > 10_000.0 {
            return false;
        }

        // Check if we're actually pruning things
        if self.stats.nodes_pruned == 0 && self.stats.edges_pruned == 0 {
            return false;
        }

        true
    }

    /// Get a summary report of statistics
    pub fn summary_report(&self) -> String {
        format!(
            "Pruning Statistics Summary:\n\
             - Total Operations: {}\n\
             - Nodes Pruned: {}\n\
             - Edges Pruned: {}\n\
             - Average Duration: {:.2}ms\n\
             - Space Saved: {}KB\n\
             - Last Pruning: {}\n\
             - Efficiency: {:.2} nodes/ms\n\
             - Health Status: {}",
            self.stats.total_operations,
            self.stats.nodes_pruned,
            self.stats.edges_pruned,
            self.stats.avg_duration_ms,
            self.stats.space_saved_bytes / 1024,
            self.stats
                .last_pruning
                .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| "Never".to_string()),
            self.pruning_efficiency(),
            if self.is_healthy() {
                "Healthy"
            } else {
                "Unhealthy"
            }
        )
    }
}

impl Default for StatsManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_creation() {
        let manager = StatsManager::new();
        let stats = manager.stats();

        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.nodes_pruned, 0);
        assert_eq!(stats.edges_pruned, 0);
        assert!(stats.last_pruning.is_none());
    }

    #[test]
    fn test_stats_update() {
        let mut manager = StatsManager::new();

        let mut nodes_by_type = HashMap::new();
        nodes_by_type.insert(NodeType::Memory, 3);
        nodes_by_type.insert(NodeType::Agent, 1);

        manager.update_after_pruning(4, 5, 100.0, nodes_by_type);

        let stats = manager.stats();
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.nodes_pruned, 4);
        assert_eq!(stats.edges_pruned, 5);
        assert_eq!(stats.avg_duration_ms, 100.0);
        assert!(stats.last_pruning.is_some());
        assert_eq!(stats.space_saved_bytes, 6500); // 4*1000 + 5*500

        // Check nodes by type
        assert_eq!(stats.pruned_by_type[&NodeType::Memory], 3);
        assert_eq!(stats.pruned_by_type[&NodeType::Agent], 1);
    }

    #[test]
    fn test_average_duration_calculation() {
        let mut manager = StatsManager::new();

        // First operation: 100ms
        manager.update_after_pruning(1, 0, 100.0, HashMap::new());
        assert_eq!(manager.avg_duration_ms(), 100.0);

        // Second operation: 200ms
        manager.update_after_pruning(1, 0, 200.0, HashMap::new());
        assert_eq!(manager.avg_duration_ms(), 150.0); // (100 + 200) / 2

        // Third operation: 300ms
        manager.update_after_pruning(1, 0, 300.0, HashMap::new());
        assert_eq!(manager.avg_duration_ms(), 200.0); // (100 + 200 + 300) / 3
    }

    #[test]
    fn test_efficiency_calculation() {
        let mut manager = StatsManager::new();

        // Prune 10 nodes in 100ms
        manager.update_after_pruning(10, 0, 100.0, HashMap::new());
        assert_eq!(manager.pruning_efficiency(), 0.1); // 10 nodes / 100ms

        // No duration should give 0 efficiency
        manager.stats.avg_duration_ms = 0.0;
        assert_eq!(manager.pruning_efficiency(), 0.0);
    }

    #[test]
    fn test_health_check() {
        let mut manager = StatsManager::new();

        // No operations is healthy
        assert!(manager.is_healthy());

        // Normal operation is healthy
        manager.update_after_pruning(5, 3, 1000.0, HashMap::new());
        assert!(manager.is_healthy());

        // Too slow operation is unhealthy
        manager.stats.avg_duration_ms = 15_000.0; // 15 seconds
        assert!(!manager.is_healthy());

        // Reset and test no pruning
        manager.reset();
        manager.update_after_pruning(0, 0, 100.0, HashMap::new());
        assert!(!manager.is_healthy()); // No actual pruning happened
    }

    #[test]
    fn test_reset() {
        let mut manager = StatsManager::new();

        // Add some data
        manager.update_after_pruning(5, 3, 100.0, HashMap::new());
        assert_eq!(manager.total_operations(), 1);

        // Reset
        manager.reset();
        assert_eq!(manager.total_operations(), 0);
        assert_eq!(manager.total_nodes_pruned(), 0);
        assert_eq!(manager.total_edges_pruned(), 0);
        assert!(manager.last_pruning().is_none());
    }

    #[test]
    fn test_summary_report() {
        let mut manager = StatsManager::new();

        let mut nodes_by_type = HashMap::new();
        nodes_by_type.insert(NodeType::Memory, 2);

        manager.update_after_pruning(2, 1, 50.0, nodes_by_type);

        let report = manager.summary_report();
        assert!(report.contains("Total Operations: 1"));
        assert!(report.contains("Nodes Pruned: 2"));
        assert!(report.contains("Edges Pruned: 1"));
        assert!(report.contains("Average Duration: 50.00ms"));
        assert!(report.contains("Health Status: Healthy"));
    }
}
