//! Pruning strategy implementations
//!
//! This module contains implementations for different pruning strategies
//! that can be applied to the knowledge graph.

use super::access_tracking::AccessTracker;
use super::types::{ImportanceMethod, PruningStrategy, RemovalPriority};
use crate::error::KnowledgeGraphResult;
use crate::graph::{KnowledgeGraph, Node, NodeType};
use chrono::{Duration, Utc};
use std::collections::{HashMap, HashSet};

/// Strategy executor for different pruning approaches
pub struct StrategyExecutor {
    /// Importance scores cache
    importance_cache: HashMap<String, f64>,
}

impl StrategyExecutor {
    /// Create a new strategy executor
    pub fn new() -> Self {
        Self {
            importance_cache: HashMap::new(),
        }
    }

    /// Execute a pruning strategy and return nodes to remove
    pub async fn execute_strategy(
        &mut self,
        strategy: &PruningStrategy,
        graph: &KnowledgeGraph,
        access_tracker: &AccessTracker,
    ) -> KnowledgeGraphResult<HashSet<String>> {
        let mut nodes_to_remove = HashSet::new();

        match strategy {
            PruningStrategy::TimeBased {
                max_age_hours,
                node_types,
            } => {
                self.apply_time_based_pruning(
                    graph,
                    *max_age_hours,
                    node_types,
                    &mut nodes_to_remove,
                )
                .await?;
            }
            PruningStrategy::SizeBased {
                max_nodes,
                max_edges,
                priority,
            } => {
                self.apply_size_based_pruning(
                    graph,
                    *max_nodes,
                    *max_edges,
                    priority,
                    &mut nodes_to_remove,
                    access_tracker,
                )
                .await?;
            }
            PruningStrategy::ImportanceBased {
                min_importance,
                method,
            } => {
                self.apply_importance_based_pruning(
                    graph,
                    *min_importance,
                    method,
                    &mut nodes_to_remove,
                    access_tracker,
                )
                .await?;
            }
            PruningStrategy::UsageBased {
                min_access_count,
                time_window_hours,
            } => {
                self.apply_usage_based_pruning(
                    *min_access_count,
                    *time_window_hours,
                    &mut nodes_to_remove,
                    access_tracker,
                )
                .await?;
            }
            PruningStrategy::RedundancyBased {
                similarity_threshold: _,
                max_similar: _,
            } => {
                self.apply_redundancy_based_pruning(graph, &mut nodes_to_remove)
                    .await?;
            }
        }

        Ok(nodes_to_remove)
    }

    /// Apply time-based pruning
    async fn apply_time_based_pruning(
        &self,
        graph: &KnowledgeGraph,
        max_age_hours: i64,
        node_types: &[NodeType],
        nodes_to_remove: &mut HashSet<String>,
    ) -> KnowledgeGraphResult<()> {
        let cutoff = Utc::now() - Duration::hours(max_age_hours);

        for node_type in node_types {
            let nodes = graph.get_nodes_by_type(node_type);
            for node in nodes {
                if node.created_at < cutoff {
                    nodes_to_remove.insert(node.id.clone());
                }
            }
        }

        Ok(())
    }

    /// Apply size-based pruning
    async fn apply_size_based_pruning(
        &self,
        graph: &KnowledgeGraph,
        max_nodes: usize,
        max_edges: usize,
        priority: &RemovalPriority,
        nodes_to_remove: &mut HashSet<String>,
        access_tracker: &AccessTracker,
    ) -> KnowledgeGraphResult<()> {
        let stats = graph.stats();

        if stats.node_count <= max_nodes && stats.edge_count <= max_edges {
            return Ok(());
        }

        let excess_nodes = stats.node_count.saturating_sub(max_nodes);

        if excess_nodes > 0 {
            // Get nodes sorted by removal priority
            let mut candidates = Vec::new();

            for node_type in [NodeType::Memory, NodeType::Pattern, NodeType::Evolution] {
                let nodes = graph.get_nodes_by_type(&node_type);
                candidates.extend(nodes);
            }

            // Sort by priority
            match priority {
                RemovalPriority::OldestFirst => {
                    candidates.sort_by_key(|node| node.created_at);
                }
                RemovalPriority::LowestDegreeFirst => {
                    candidates.sort_by_key(|node| {
                        graph.get_outgoing_edges(&node.id).len()
                            + graph.get_incoming_edges(&node.id).len()
                    });
                }
                RemovalPriority::LowestImportanceFirst => {
                    // Use cached importance scores
                    candidates.sort_by(|a, b| {
                        let a_score = self.importance_cache.get(&a.id).unwrap_or(&0.0);
                        let b_score = self.importance_cache.get(&b.id).unwrap_or(&0.0);
                        a_score.partial_cmp(b_score).unwrap()
                    });
                }
                RemovalPriority::LeastAccessedFirst => {
                    candidates.sort_by_key(|node| {
                        access_tracker
                            .get_access(&node.id)
                            .map(|access| access.access_count)
                            .unwrap_or(0)
                    });
                }
            }

            // Mark for removal
            for node in candidates.iter().take(excess_nodes) {
                nodes_to_remove.insert(node.id.clone());
            }
        }

        Ok(())
    }

    /// Apply importance-based pruning
    async fn apply_importance_based_pruning(
        &mut self,
        graph: &KnowledgeGraph,
        min_importance: f64,
        method: &ImportanceMethod,
        nodes_to_remove: &mut HashSet<String>,
        access_tracker: &AccessTracker,
    ) -> KnowledgeGraphResult<()> {
        // Update importance cache
        self.update_importance_cache(graph, method, access_tracker)
            .await?;

        // Find nodes below importance threshold
        for (node_id, &importance) in &self.importance_cache {
            if importance < min_importance {
                nodes_to_remove.insert(node_id.clone());
            }
        }

        Ok(())
    }

    /// Apply usage-based pruning
    async fn apply_usage_based_pruning(
        &self,
        min_access_count: u32,
        time_window_hours: i64,
        nodes_to_remove: &mut HashSet<String>,
        access_tracker: &AccessTracker,
    ) -> KnowledgeGraphResult<()> {
        let underaccessed =
            access_tracker.get_underaccessed_entities(min_access_count, time_window_hours);

        for node_id in underaccessed {
            nodes_to_remove.insert(node_id);
        }

        Ok(())
    }

    /// Apply redundancy-based pruning
    async fn apply_redundancy_based_pruning(
        &self,
        graph: &KnowledgeGraph,
        nodes_to_remove: &mut HashSet<String>,
    ) -> KnowledgeGraphResult<()> {
        // Simplified redundancy detection - remove nodes with identical properties
        let mut property_groups: HashMap<String, Vec<String>> = HashMap::new();

        for node_type in [NodeType::Memory, NodeType::Pattern] {
            let nodes = graph.get_nodes_by_type(&node_type);

            for node in nodes {
                let properties_key = serde_json::to_string(&node.properties)
                    .unwrap_or_else(|_| "unknown".to_string());

                property_groups
                    .entry(properties_key)
                    .or_insert_with(Vec::new)
                    .push(node.id.clone());
            }
        }

        // Mark duplicates for removal (keep the first one)
        for (_, node_ids) in property_groups {
            if node_ids.len() > 1 {
                for node_id in node_ids.into_iter().skip(1) {
                    nodes_to_remove.insert(node_id);
                }
            }
        }

        Ok(())
    }

    /// Update importance cache
    async fn update_importance_cache(
        &mut self,
        graph: &KnowledgeGraph,
        method: &ImportanceMethod,
        access_tracker: &AccessTracker,
    ) -> KnowledgeGraphResult<()> {
        self.importance_cache.clear();

        match method {
            ImportanceMethod::DegreeCentrality => {
                for node_type in [NodeType::Agent, NodeType::Goal, NodeType::Concept] {
                    let nodes = graph.get_nodes_by_type(&node_type);

                    for node in nodes {
                        let degree = graph.get_outgoing_edges(&node.id).len()
                            + graph.get_incoming_edges(&node.id).len();
                        let importance = degree as f64 / 100.0; // Normalize
                        self.importance_cache.insert(node.id.clone(), importance);
                    }
                }
            }
            ImportanceMethod::AccessFrequency => {
                for (node_id, access) in access_tracker.get_all_accesses() {
                    let importance = access.access_count as f64 / 1000.0; // Normalize
                    self.importance_cache.insert(node_id.clone(), importance);
                }
            }
            ImportanceMethod::Composite => {
                // Combine degree and access frequency
                for node_type in [NodeType::Agent, NodeType::Goal, NodeType::Concept] {
                    let nodes = graph.get_nodes_by_type(&node_type);

                    for node in nodes {
                        let degree = graph.get_outgoing_edges(&node.id).len()
                            + graph.get_incoming_edges(&node.id).len();
                        let degree_score = degree as f64 / 100.0;

                        let access_score = access_tracker
                            .get_access(&node.id)
                            .map(|a| a.access_count as f64 / 1000.0)
                            .unwrap_or(0.0);

                        let importance = (degree_score + access_score) / 2.0;
                        self.importance_cache.insert(node.id.clone(), importance);
                    }
                }
            }
            _ => {
                // Other methods not implemented - use default score
                for node_type in [NodeType::Agent, NodeType::Goal, NodeType::Concept] {
                    let nodes = graph.get_nodes_by_type(&node_type);
                    for node in nodes {
                        self.importance_cache.insert(node.id.clone(), 0.5);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get cached importance score for a node
    pub fn get_importance(&self, node_id: &str) -> Option<f64> {
        self.importance_cache.get(node_id).copied()
    }

    /// Clear importance cache
    pub fn clear_importance_cache(&mut self) {
        self.importance_cache.clear();
    }

    /// Get number of cached importance scores
    pub fn importance_cache_size(&self) -> usize {
        self.importance_cache.len()
    }
}

impl Default for StrategyExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, EdgeType, KnowledgeGraphConfig};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_time_based_strategy() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;
        let mut executor = StrategyExecutor::new();
        let tracker = AccessTracker::new();

        // Add old and new nodes
        let mut old_node = Node::new(NodeType::Memory, HashMap::new());
        old_node.created_at = Utc::now() - Duration::hours(48);
        let old_id = old_node.id.clone();
        graph.add_node(old_node).unwrap();

        let new_node = Node::new(NodeType::Memory, HashMap::new());
        let new_id = new_node.id.clone();
        graph.add_node(new_node).unwrap();

        let strategy = PruningStrategy::TimeBased {
            max_age_hours: 24,
            node_types: vec![NodeType::Memory],
        };

        let to_remove = executor
            .execute_strategy(&strategy, &graph, &tracker)
            .await
            .unwrap();

        assert!(to_remove.contains(&old_id));
        assert!(!to_remove.contains(&new_id));
    }

    #[tokio::test]
    async fn test_size_based_strategy() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;
        let mut executor = StrategyExecutor::new();
        let tracker = AccessTracker::new();

        // Add more nodes than the limit
        for _ in 0..10 {
            let node = Node::new(NodeType::Memory, HashMap::new());
            graph.add_node(node).unwrap();
        }

        let strategy = PruningStrategy::SizeBased {
            max_nodes: 5,
            max_edges: 100,
            priority: RemovalPriority::OldestFirst,
        };

        let to_remove = executor
            .execute_strategy(&strategy, &graph, &tracker)
            .await
            .unwrap();

        assert_eq!(to_remove.len(), 5); // Should remove 5 excess nodes
    }

    #[tokio::test]
    async fn test_usage_based_strategy() {
        let mut executor = StrategyExecutor::new();
        let mut tracker = AccessTracker::new();

        // Add access records
        tracker.record_access("good_node".to_string());
        tracker.record_access("good_node".to_string());
        tracker.record_access("good_node".to_string());

        tracker.record_access("bad_node".to_string());

        let strategy = PruningStrategy::UsageBased {
            min_access_count: 2,
            time_window_hours: 24,
        };

        // Create dummy graph (usage strategy doesn't use it)
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let graph = KnowledgeGraph::new(graph_config).await.unwrap();

        let to_remove = executor
            .execute_strategy(&strategy, &graph, &tracker)
            .await
            .unwrap();

        assert!(to_remove.contains("bad_node"));
        assert!(!to_remove.contains("good_node"));
    }

    #[tokio::test]
    async fn test_redundancy_based_strategy() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;
        let mut executor = StrategyExecutor::new();
        let tracker = AccessTracker::new();

        // Create nodes with identical properties
        let properties = HashMap::from([(
            "type".to_string(),
            serde_json::Value::String("duplicate".to_string()),
        )]);

        let mut node_ids = Vec::new();
        for _ in 0..3 {
            let node = Node::new(NodeType::Memory, properties.clone());
            node_ids.push(node.id.clone());
            graph.add_node(node).unwrap();
        }

        let strategy = PruningStrategy::RedundancyBased {
            similarity_threshold: 0.9,
            max_similar: 1,
        };

        let to_remove = executor
            .execute_strategy(&strategy, &graph, &tracker)
            .await
            .unwrap();

        // Should remove 2 out of 3 identical nodes
        assert_eq!(to_remove.len(), 2);
        // First node should remain
        assert!(!to_remove.contains(&node_ids[0]));
    }

    #[tokio::test]
    async fn test_importance_based_strategy() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;
        let mut executor = StrategyExecutor::new();
        let tracker = AccessTracker::new();

        // Create hub node with connections
        let hub = Node::new(NodeType::Agent, HashMap::new());
        let hub_id = hub.id.clone();
        graph.add_node(hub).unwrap();

        // Create leaf node with no connections
        let leaf = Node::new(NodeType::Agent, HashMap::new());
        let leaf_id = leaf.id.clone();
        graph.add_node(leaf).unwrap();

        // Add connections to hub
        for _ in 0..3 {
            let target = Node::new(NodeType::Goal, HashMap::new());
            let target_id = target.id.clone();
            graph.add_node(target).unwrap();

            let edge = Edge::new(hub_id.clone(), target_id, EdgeType::Has, 1.0);
            graph.add_edge(edge).unwrap();
        }

        let strategy = PruningStrategy::ImportanceBased {
            min_importance: 0.02,
            method: ImportanceMethod::DegreeCentrality,
        };

        let to_remove = executor
            .execute_strategy(&strategy, &graph, &tracker)
            .await
            .unwrap();

        // Hub should remain (high importance), leaf should be removed
        assert!(!to_remove.contains(&hub_id));
        assert!(to_remove.contains(&leaf_id));
    }

    #[test]
    fn test_importance_cache() {
        let mut executor = StrategyExecutor::new();

        assert_eq!(executor.importance_cache_size(), 0);

        executor.importance_cache.insert("node1".to_string(), 0.5);
        executor.importance_cache.insert("node2".to_string(), 0.8);

        assert_eq!(executor.importance_cache_size(), 2);
        assert_eq!(executor.get_importance("node1"), Some(0.5));
        assert_eq!(executor.get_importance("node2"), Some(0.8));
        assert_eq!(executor.get_importance("node3"), None);

        executor.clear_importance_cache();
        assert_eq!(executor.importance_cache_size(), 0);
    }
}
