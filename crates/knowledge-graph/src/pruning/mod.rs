//! Knowledge Graph Pruning System
//!
//! This module provides comprehensive pruning functionality for knowledge graphs,
//! including multiple pruning strategies, backup/restore capabilities, access tracking,
//! and detailed statistics.
//!
//! ## Module Structure
//!
//! The pruning system has been refactored using TDD methodology into logical modules:
//!
//! - `types`: Core type definitions and configuration
//! - `access_tracking`: Entity access tracking for usage-based pruning
//! - `backup`: Backup and restore functionality for pruned entities
//! - `stats`: Statistics collection and reporting
//! - `strategies`: Implementation of different pruning strategies
//! - `core`: Main system that coordinates all components
//!
//! ## Usage
//!
//! ```rust
//! use knowledge_graph::pruning::{PruningSystem, PruningConfig, PruningStrategy};
//!
//! let config = PruningConfig::default();
//! let mut system = PruningSystem::new(config);
//!
//! // Record entity access for usage-based pruning
//! system.record_access("entity_id".to_string());
//!
//! // Run pruning
//! let stats = system.prune(&mut graph).await?;
//! println!("Pruned {} nodes", stats.nodes_pruned);
//! ```

// Internal modules
mod access_tracking;
mod backup;
mod core;
mod stats;
mod strategies;
mod types;

// Re-export public API to maintain backward compatibility
pub use core::PruningSystem;
pub use types::{ImportanceMethod, PruningConfig, PruningStats, PruningStrategy, RemovalPriority};

// Internal types that may be useful for advanced usage
pub use access_tracking::AccessTracker;
pub use backup::BackupManager;
pub use stats::StatsManager;
pub use strategies::StrategyExecutor;

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::graph::{Edge, EdgeType, KnowledgeGraph, KnowledgeGraphConfig, Node, NodeType};
    use chrono::{Duration, Utc};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_full_pruning_integration() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        // Create a complex graph
        let mut nodes = Vec::new();

        // Add old nodes
        for i in 0..5 {
            let mut node = Node::new(NodeType::Memory, HashMap::new());
            node.created_at = Utc::now() - Duration::hours(48);
            nodes.push(node.id.clone());
            graph.add_node(node).unwrap();
        }

        // Add recent nodes with duplicate properties
        let properties = HashMap::from([(
            "type".to_string(),
            serde_json::Value::String("duplicate".to_string()),
        )]);

        for _ in 0..3 {
            let node = Node::new(NodeType::Pattern, properties.clone());
            nodes.push(node.id.clone());
            graph.add_node(node).unwrap();
        }

        // Add important hub node
        let hub = Node::new(NodeType::Agent, HashMap::new());
        let hub_id = hub.id.clone();
        nodes.push(hub_id.clone());
        graph.add_node(hub).unwrap();

        // Connect hub to make it important
        for i in 0..3 {
            let target = Node::new(NodeType::Goal, HashMap::new());
            let target_id = target.id.clone();
            graph.add_node(target).unwrap();

            let edge = Edge::new(hub_id.clone(), target_id, EdgeType::Has, 1.0);
            graph.add_edge(edge).unwrap();
        }

        // Configure comprehensive pruning
        let config = PruningConfig {
            strategies: vec![
                PruningStrategy::TimeBased {
                    max_age_hours: 24,
                    node_types: vec![NodeType::Memory],
                },
                PruningStrategy::RedundancyBased {
                    similarity_threshold: 0.9,
                    max_similar: 1,
                },
                PruningStrategy::SizeBased {
                    max_nodes: 8,
                    max_edges: 20,
                    priority: RemovalPriority::OldestFirst,
                },
                PruningStrategy::ImportanceBased {
                    min_importance: 0.01,
                    method: ImportanceMethod::DegreeCentrality,
                },
            ],
            auto_pruning: true,
            backup_removed: true,
            max_backup_size: 100,
            pruning_interval_hours: 1,
        };

        let mut system = PruningSystem::new(config);

        // Record some access patterns
        for _ in 0..5 {
            system.record_access(hub_id.clone());
        }

        let initial_count = graph.stats().node_count;
        println!("Initial node count: {}", initial_count);

        // Run pruning
        let stats = system.force_prune(&mut graph).await.unwrap();

        println!("Pruning results:");
        println!("- Nodes pruned: {}", stats.nodes_pruned);
        println!("- Edges pruned: {}", stats.edges_pruned);
        println!("- Duration: {:.2}ms", stats.avg_duration_ms);

        let final_count = graph.stats().node_count;
        println!("Final node count: {}", final_count);

        // Verify pruning worked
        assert!(stats.nodes_pruned > 0);
        assert!(final_count < initial_count);
        assert!(stats.total_operations == 1);

        // Verify hub remains (important node)
        assert!(graph.get_node(&hub_id).is_ok());

        // Verify backup system worked
        let backups = system.get_backup(None);
        assert!(backups.len() > 0);

        // Test system health
        assert!(system.is_healthy());

        // Test system report
        let report = system.system_report();
        assert!(report.contains("Pruning System Report"));
        assert!(report.contains("✅ Healthy"));
    }

    #[tokio::test]
    async fn test_modular_components_independence() {
        // Test that each module can work independently

        // Test AccessTracker
        let mut tracker = AccessTracker::new();
        tracker.record_access("test".to_string());
        assert_eq!(tracker.len(), 1);

        // Test BackupManager
        let mut backup_manager = BackupManager::new(10);
        assert!(backup_manager.is_empty());

        // Test StatsManager
        let mut stats_manager = StatsManager::new();
        assert_eq!(stats_manager.total_operations(), 0);

        stats_manager.update_after_pruning(5, 3, 100.0, HashMap::new());
        assert_eq!(stats_manager.total_operations(), 1);
        assert_eq!(stats_manager.total_nodes_pruned(), 5);

        // Test StrategyExecutor
        let mut executor = StrategyExecutor::new();
        assert_eq!(executor.importance_cache_size(), 0);

        // All components should work together seamlessly
        let config = PruningConfig::default();
        let system = PruningSystem::new(config);
        assert!(system.is_healthy());
    }

    #[tokio::test]
    async fn test_backward_compatibility() {
        // Test that the refactored system maintains the same API
        let config = PruningConfig::default();
        let mut system = PruningSystem::new(config);

        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        // All original methods should still work
        assert_eq!(system.stats().total_operations, 0);

        system.record_access("test".to_string());

        let stats = system.prune(&mut graph).await.unwrap();
        assert!(stats.total_operations >= 0);

        system.clear_access_tracking();

        let forced_stats = system.force_prune(&mut graph).await.unwrap();
        assert!(forced_stats.total_operations >= 0);

        let backups = system.get_backup(None);
        assert_eq!(backups.len(), 0); // No nodes to backup in empty graph
    }

    #[test]
    fn test_configuration_serialization() {
        // Test that configuration types are properly serializable
        let config = PruningConfig::default();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PruningConfig = serde_json::from_str(&json)?;

        assert_eq!(config.auto_pruning, deserialized.auto_pruning);
        assert_eq!(
            config.pruning_interval_hours,
            deserialized.pruning_interval_hours
        );
        assert_eq!(config.strategies.len(), deserialized.strategies.len());
    }

    #[test]
    fn test_line_count_compliance() {
        // Verify that each module is under 750 lines as required

        // This is a meta-test to ensure our refactoring meets the line count requirement
        // In a real implementation, you might use a build script or CI check for this

        let modules = vec![
            ("types.rs", include_str!("types.rs")),
            ("access_tracking.rs", include_str!("access_tracking.rs")),
            ("backup.rs", include_str!("backup.rs")),
            ("stats.rs", include_str!("stats.rs")),
            ("strategies.rs", include_str!("strategies.rs")),
            ("core.rs", include_str!("core.rs")),
            ("mod.rs", include_str!("mod.rs")),
        ];

        for (name, content) in modules {
            let lines = content.lines().count();
            println!("{}: {} lines", name, lines);
            assert!(
                lines <= 750,
                "{} has {} lines, exceeds 750 line limit",
                name,
                lines
            );
        }
    }
}
