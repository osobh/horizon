//! Query engine with CUDA kernels for high-performance graph queries
//!
//! This module has been refactored using TDD methodology to improve maintainability
//! while preserving all original functionality. The implementation is now split
//! across multiple focused modules while maintaining backward compatibility.

// Import all functionality from the new modular structure
mod gpu_kernels;
mod query_engine;
mod query_execution;
mod query_types;

// Re-export all public APIs to maintain backward compatibility
pub use gpu_kernels::{GpuKernelManager, KernelExecutionContext, KernelStats};
pub use query_engine::{QueryEngine, QueryStats};
pub use query_execution::{ExecutionStats, QueryExecutor};
pub use query_types::{
    GraphPattern, PatternEdge, PatternNode, Query, QueryResult, QueryType, RankingAlgorithm,
};

// Preserve the original test module for compatibility
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::KnowledgeGraphConfig;
    use crate::graph::{Edge, EdgeType, Node, NodeType};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_query_engine_creation() {
        let engine = QueryEngine::new(false).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_find_nodes_query() {
        let mut graph = KnowledgeGraph::new(KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        })
        .await?;

        // Add test nodes
        let mut node1 = Node::new(NodeType::Agent, HashMap::new());
        node1.update_property(
            "name".to_string(),
            serde_json::Value::String("agent1".to_string()),
        );
        let node1_id = node1.id.clone();
        graph.add_node(node1).unwrap();

        let mut node2 = Node::new(NodeType::Agent, HashMap::new());
        node2.update_property(
            "name".to_string(),
            serde_json::Value::String("agent2".to_string()),
        );
        graph.add_node(node2).unwrap();

        let mut engine = QueryEngine::new(false).await.unwrap();

        // Query for nodes with specific property
        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            serde_json::Value::String("agent1".to_string()),
        );

        let query = Query {
            query_type: QueryType::FindNodes {
                node_type: Some(NodeType::Agent),
                properties,
            },
            timeout_ms: None,
            limit: None,
            offset: None,
            use_gpu: false,
        };

        let result = engine.execute(&graph, query).await.unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].id, node1_id);
    }

    #[tokio::test]
    async fn test_path_finding_query() {
        let mut graph = KnowledgeGraph::new(KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        })
        .await?;

        // Create a simple path: A -> B -> C
        let node_a = Node::new(NodeType::Agent, HashMap::new());
        let node_a_id = node_a.id.clone();
        graph.add_node(node_a).unwrap();

        let node_b = Node::new(NodeType::Goal, HashMap::new());
        let node_b_id = node_b.id.clone();
        graph.add_node(node_b).unwrap();

        let node_c = Node::new(NodeType::Concept, HashMap::new());
        let node_c_id = node_c.id.clone();
        graph.add_node(node_c).unwrap();

        // Add edges
        let edge_ab = Edge::new(node_a_id.clone(), node_b_id.clone(), EdgeType::Has, 1.0);
        graph.add_edge(edge_ab).unwrap();

        let edge_bc = Edge::new(
            node_b_id.clone(),
            node_c_id.clone(),
            EdgeType::Contains,
            1.0,
        );
        graph.add_edge(edge_bc).unwrap();

        let mut engine = QueryEngine::new(false).await.unwrap();

        let query = Query {
            query_type: QueryType::FindPath {
                source_id: node_a_id.clone(),
                target_id: node_c_id.clone(),
                max_length: 5,
            },
            timeout_ms: None,
            limit: None,
            offset: None,
            use_gpu: false,
        };

        let result = engine.execute(&graph, query).await.unwrap();
        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].len(), 3);
        assert_eq!(result.paths[0][0], node_a_id);
        assert_eq!(result.paths[0][1], node_b_id);
        assert_eq!(result.paths[0][2], node_c_id);
    }

    #[tokio::test]
    async fn test_neighborhood_query() {
        let mut graph = KnowledgeGraph::new(KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        })
        .await?;

        // Create a star topology: center -> node1, node2, node3
        let center = Node::new(NodeType::Agent, HashMap::new());
        let center_id = center.id.clone();
        graph.add_node(center).unwrap();

        let mut neighbors = Vec::new();
        for i in 0..3 {
            let node = Node::new(NodeType::Goal, HashMap::new());
            let node_id = node.id.clone();
            graph.add_node(node).unwrap();

            let edge = Edge::new(center_id.clone(), node_id.clone(), EdgeType::Has, 1.0);
            graph.add_edge(edge).unwrap();

            neighbors.push(node_id);
        }

        let mut engine = QueryEngine::new(false).await.unwrap();

        let query = Query {
            query_type: QueryType::Neighborhood {
                node_id: center_id.clone(),
                radius: 1,
                edge_types: None,
            },
            timeout_ms: None,
            limit: None,
            offset: None,
            use_gpu: false,
        };

        let result = engine.execute(&graph, query).await.unwrap();
        assert_eq!(result.nodes.len(), 4); // center + 3 neighbors
        assert_eq!(result.edges.len(), 3);
    }

    #[tokio::test]
    async fn test_degree_centrality_ranking() {
        let mut graph = KnowledgeGraph::new(KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        })
        .await?;

        // Create nodes with different degrees
        let hub = Node::new(NodeType::Agent, HashMap::new());
        let hub_id = hub.id.clone();
        graph.add_node(hub).unwrap();

        let leaf = Node::new(NodeType::Agent, HashMap::new());
        let leaf_id = leaf.id.clone();
        graph.add_node(leaf).unwrap();

        // Hub has more connections
        for _ in 0..3 {
            let node = Node::new(NodeType::Goal, HashMap::new());
            let node_id = node.id.clone();
            graph.add_node(node).unwrap();

            let edge = Edge::new(hub_id.clone(), node_id, EdgeType::Has, 1.0);
            graph.add_edge(edge).unwrap();
        }

        // Leaf has one connection
        let node = Node::new(NodeType::Goal, HashMap::new());
        let node_id = node.id.clone();
        graph.add_node(node).unwrap();

        let edge = Edge::new(leaf_id.clone(), node_id, EdgeType::Has, 1.0);
        graph.add_edge(edge).unwrap();

        let mut engine = QueryEngine::new(false).await.unwrap();

        let query = Query {
            query_type: QueryType::Ranking {
                algorithm: RankingAlgorithm::DegreeCentrality,
                node_type: Some(NodeType::Agent),
            },
            timeout_ms: None,
            limit: None,
            offset: None,
            use_gpu: false,
        };

        let result = engine.execute(&graph, query).await.unwrap();

        let hub_score = result.scores.get(&hub_id).unwrap_or(&0.0);
        let leaf_score = result.scores.get(&leaf_id).unwrap_or(&0.0);

        assert!(hub_score > leaf_score);
    }
}
