//! Query execution implementations
//!
//! This module contains the specific implementations for executing different
//! types of queries against the knowledge graph.

use super::gpu_kernels::GpuKernelManager;
use super::query_types::*;
use crate::error::KnowledgeGraphResult;
use crate::graph::{Edge, EdgeType, KnowledgeGraph, Node, NodeType};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

/// Query execution engine with algorithm implementations
pub struct QueryExecutor {
    /// GPU kernel manager for accelerated operations
    gpu_kernels: Option<GpuKernelManager>,
    /// Execution statistics
    stats: ExecutionStats,
}

/// Execution statistics for query performance tracking
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Number of GPU-accelerated operations
    pub gpu_operations: u64,
    /// Number of CPU fallback operations
    pub cpu_operations: u64,
    /// Average execution times by query type
    pub avg_times: HashMap<String, f64>,
}

impl QueryExecutor {
    /// Create a new query executor
    pub async fn new(gpu_enabled: bool) -> KnowledgeGraphResult<Self> {
        let gpu_kernels = if gpu_enabled {
            Some(GpuKernelManager::new(true).await?)
        } else {
            None
        };

        Ok(Self {
            gpu_kernels,
            stats: ExecutionStats::default(),
        })
    }

    /// Execute find nodes query
    pub async fn execute_find_nodes(
        &mut self,
        graph: &KnowledgeGraph,
        node_type: Option<NodeType>,
        properties: HashMap<String, serde_json::Value>,
        query: &Query,
    ) -> KnowledgeGraphResult<QueryResult> {
        self.stats.cpu_operations += 1;

        let mut matching_nodes = Vec::new();

        // Get candidate nodes by type
        let candidates = if let Some(ref nt) = node_type {
            graph.get_nodes_by_type(nt)
        } else {
            // Get all nodes - this would need to be implemented in KnowledgeGraph
            vec![]
        };

        // Filter by properties (parallelized with Rayon)
        matching_nodes = candidates
            .par_iter()
            .filter(|node| self.node_matches_properties(node, &properties))
            .map(|node| (*node).clone())
            .collect();

        // Apply limit and offset
        matching_nodes = self.apply_pagination(matching_nodes, query.limit, query.offset);

        Ok(QueryResult {
            nodes: matching_nodes,
            edges: vec![],
            paths: vec![],
            scores: HashMap::new(),
            execution_time_ms: 0,
            gpu_accelerated: false,
            metadata: HashMap::new(),
        })
    }

    /// Execute find edges query
    pub async fn execute_find_edges(
        &mut self,
        _graph: &KnowledgeGraph,
        _edge_type: Option<EdgeType>,
        _weight_range: Option<(f64, f64)>,
        _query: &Query,
    ) -> KnowledgeGraphResult<QueryResult> {
        self.stats.cpu_operations += 1;

        // Current implementation returns empty result to maintain compatibility
        Ok(QueryResult {
            nodes: vec![],
            edges: vec![],
            paths: vec![],
            scores: HashMap::new(),
            execution_time_ms: 0,
            gpu_accelerated: false,
            metadata: HashMap::new(),
        })
    }

    /// Execute path finding query
    pub async fn execute_find_path(
        &mut self,
        graph: &KnowledgeGraph,
        source_id: &str,
        target_id: &str,
        max_length: usize,
        query: &Query,
    ) -> KnowledgeGraphResult<QueryResult> {
        // Use GPU kernel if available and requested
        if query.use_gpu && self.gpu_kernels.is_some() {
            return self
                .execute_path_gpu(graph, source_id, target_id, max_length)
                .await;
        }

        // CPU implementation using BFS
        let path = self.find_path_bfs(graph, source_id, target_id, max_length)?;
        self.stats.cpu_operations += 1;

        Ok(QueryResult {
            nodes: vec![],
            edges: vec![],
            paths: if path.is_empty() { vec![] } else { vec![path] },
            scores: HashMap::new(),
            execution_time_ms: 0,
            gpu_accelerated: false,
            metadata: HashMap::new(),
        })
    }

    /// Execute neighborhood query
    pub async fn execute_neighborhood(
        &mut self,
        graph: &KnowledgeGraph,
        node_id: &str,
        radius: usize,
        edge_types: Option<Vec<EdgeType>>,
        _query: &Query,
    ) -> KnowledgeGraphResult<QueryResult> {
        self.stats.cpu_operations += 1;

        let mut result_nodes = Vec::new();
        let mut result_edges = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start with the center node
        if let Ok(center_node) = graph.get_node(node_id) {
            result_nodes.push(center_node.clone());
            visited.insert(node_id.to_string());
            queue.push_back((node_id.to_string(), 0));
        }

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= radius {
                continue;
            }

            let outgoing_edges = graph.get_outgoing_edges(&current_id);
            for edge in outgoing_edges {
                // Filter by edge type if specified
                if let Some(ref allowed_types) = edge_types {
                    if !allowed_types.contains(&edge.edge_type) {
                        continue;
                    }
                }

                result_edges.push(edge.clone());

                if !visited.contains(&edge.target_id) {
                    visited.insert(edge.target_id.clone());
                    if let Ok(target_node) = graph.get_node(&edge.target_id) {
                        result_nodes.push(target_node.clone());
                    }
                    queue.push_back((edge.target_id.clone(), depth + 1));
                }
            }
        }

        Ok(QueryResult {
            nodes: result_nodes,
            edges: result_edges,
            paths: vec![],
            scores: HashMap::new(),
            execution_time_ms: 0,
            gpu_accelerated: false,
            metadata: HashMap::new(),
        })
    }

    /// Execute subgraph extraction
    pub async fn execute_subgraph(
        &mut self,
        graph: &KnowledgeGraph,
        seed_nodes: Vec<String>,
        max_depth: usize,
        query: &Query,
    ) -> KnowledgeGraphResult<QueryResult> {
        let mut result_nodes = Vec::new();
        let mut result_edges = Vec::new();
        let mut visited = HashSet::new();

        // Expand from each seed node
        for seed_id in seed_nodes {
            let neighborhood = self
                .execute_neighborhood(
                    graph,
                    &seed_id,
                    max_depth,
                    None,
                    &Query::new(QueryType::Neighborhood {
                        node_id: seed_id.clone(),
                        radius: max_depth,
                        edge_types: None,
                    }),
                )
                .await?;

            // Merge results
            for node in neighborhood.nodes {
                if !visited.contains(&node.id) {
                    visited.insert(node.id.clone());
                    result_nodes.push(node);
                }
            }

            result_edges.extend(neighborhood.edges);
        }

        Ok(QueryResult {
            nodes: result_nodes,
            edges: result_edges,
            paths: vec![],
            scores: HashMap::new(),
            execution_time_ms: 0,
            gpu_accelerated: false,
            metadata: HashMap::new(),
        })
    }

    /// Execute pattern matching
    pub async fn execute_pattern_match(
        &mut self,
        _graph: &KnowledgeGraph,
        _pattern: GraphPattern,
        _query: &Query,
    ) -> KnowledgeGraphResult<QueryResult> {
        self.stats.cpu_operations += 1;

        // Pattern matching implementation is complex and not implemented yet
        // Return empty result to maintain compatibility
        Ok(QueryResult {
            nodes: vec![],
            edges: vec![],
            paths: vec![],
            scores: HashMap::new(),
            execution_time_ms: 0,
            gpu_accelerated: false,
            metadata: HashMap::new(),
        })
    }

    /// Execute ranking query
    pub async fn execute_ranking(
        &mut self,
        graph: &KnowledgeGraph,
        algorithm: RankingAlgorithm,
        node_type: Option<NodeType>,
        _query: &Query,
    ) -> KnowledgeGraphResult<QueryResult> {
        self.stats.cpu_operations += 1;

        let mut scores = HashMap::new();

        match algorithm {
            RankingAlgorithm::DegreeCentrality => {
                scores = self.compute_degree_centrality(graph, node_type)?;
            }
            RankingAlgorithm::PageRank {
                damping,
                iterations,
            } => {
                scores = self.compute_pagerank(graph, node_type, damping, iterations)?;
            }
            RankingAlgorithm::BetweennessCentrality => {
                scores = self.compute_betweenness_centrality(graph, node_type)?;
            }
            RankingAlgorithm::ClosenessCentrality => {
                scores = self.compute_closeness_centrality(graph, node_type)?;
            }
            RankingAlgorithm::Custom(_) => {
                // Custom algorithms not implemented
            }
        }

        Ok(QueryResult {
            nodes: vec![],
            edges: vec![],
            paths: vec![],
            scores,
            execution_time_ms: 0,
            gpu_accelerated: false,
            metadata: HashMap::new(),
        })
    }

    /// GPU-accelerated path finding
    async fn execute_path_gpu(
        &mut self,
        _graph: &KnowledgeGraph,
        _source_id: &str,
        _target_id: &str,
        _max_length: usize,
    ) -> KnowledgeGraphResult<QueryResult> {
        self.stats.gpu_operations += 1;

        // Mock GPU implementation - returns empty paths
        Ok(QueryResult {
            nodes: vec![],
            edges: vec![],
            paths: vec![],
            scores: HashMap::new(),
            execution_time_ms: 0,
            gpu_accelerated: true,
            metadata: HashMap::new(),
        })
    }

    /// CPU-based BFS path finding
    fn find_path_bfs(
        &self,
        graph: &KnowledgeGraph,
        source_id: &str,
        target_id: &str,
        max_length: usize,
    ) -> KnowledgeGraphResult<Vec<String>> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent: HashMap<String, String> = HashMap::new();

        queue.push_back((source_id.to_string(), 0));
        visited.insert(source_id.to_string());

        while let Some((current_id, depth)) = queue.pop_front() {
            if current_id == target_id {
                // Reconstruct path
                let mut path = vec![current_id.clone()];
                let mut node_id = current_id;

                while let Some(parent_id) = parent.get(&node_id) {
                    path.push(parent_id.clone());
                    node_id = parent_id.clone();
                }

                path.reverse();
                return Ok(path);
            }

            if depth >= max_length {
                continue;
            }

            // Get outgoing edges
            let outgoing_edges = graph.get_outgoing_edges(&current_id);
            for edge in outgoing_edges {
                if !visited.contains(&edge.target_id) {
                    visited.insert(edge.target_id.clone());
                    parent.insert(edge.target_id.clone(), current_id.clone());
                    queue.push_back((edge.target_id.clone(), depth + 1));
                }
            }
        }

        Ok(vec![]) // No path found
    }

    /// Check if node matches property filters
    fn node_matches_properties(
        &self,
        node: &Node,
        properties: &HashMap<String, serde_json::Value>,
    ) -> bool {
        for (key, expected_value) in properties {
            if let Some(actual_value) = node.get_property(key) {
                if actual_value != expected_value {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    /// Apply pagination to results
    fn apply_pagination<T>(
        &self,
        mut items: Vec<T>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Vec<T> {
        if let Some(offset) = offset {
            if offset < items.len() {
                items = items.into_iter().skip(offset).collect();
            } else {
                items.clear();
            }
        }

        if let Some(limit) = limit {
            items.truncate(limit);
        }

        items
    }

    /// Compute degree centrality for nodes
    fn compute_degree_centrality(
        &self,
        graph: &KnowledgeGraph,
        node_type: Option<NodeType>,
    ) -> KnowledgeGraphResult<HashMap<String, f64>> {
        let mut scores = HashMap::new();

        let candidates = if let Some(ref nt) = node_type {
            graph.get_nodes_by_type(nt)
        } else {
            vec![] // Would need all nodes
        };

        // Parallelized degree centrality computation with Rayon
        let parallel_scores: Vec<_> = candidates
            .par_iter()
            .map(|node| {
                let outgoing = graph.get_outgoing_edges(&node.id).len();
                let incoming = graph.get_incoming_edges(&node.id).len();
                (node.id.clone(), (outgoing + incoming) as f64)
            })
            .collect();

        for (id, score) in parallel_scores {
            scores.insert(id, score);
        }

        Ok(scores)
    }

    /// Compute PageRank scores (placeholder implementation)
    fn compute_pagerank(
        &self,
        _graph: &KnowledgeGraph,
        _node_type: Option<NodeType>,
        _damping: f64,
        _iterations: usize,
    ) -> KnowledgeGraphResult<HashMap<String, f64>> {
        // PageRank implementation is complex and not implemented yet
        // Return empty scores to maintain current behavior
        Ok(HashMap::new())
    }

    /// Compute betweenness centrality (placeholder implementation)
    fn compute_betweenness_centrality(
        &self,
        _graph: &KnowledgeGraph,
        _node_type: Option<NodeType>,
    ) -> KnowledgeGraphResult<HashMap<String, f64>> {
        // Betweenness centrality implementation is complex
        // Return empty scores to maintain current behavior
        Ok(HashMap::new())
    }

    /// Compute closeness centrality (placeholder implementation)
    fn compute_closeness_centrality(
        &self,
        _graph: &KnowledgeGraph,
        _node_type: Option<NodeType>,
    ) -> KnowledgeGraphResult<HashMap<String, f64>> {
        // Closeness centrality implementation is complex
        // Return empty scores to maintain current behavior
        Ok(HashMap::new())
    }

    /// Get execution statistics
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Reset execution statistics
    pub fn reset_stats(&mut self) {
        self.stats = ExecutionStats::default();
    }
}

impl ExecutionStats {
    /// Get total operations count
    pub fn total_operations(&self) -> u64 {
        self.gpu_operations + self.cpu_operations
    }

    /// Get GPU utilization ratio
    pub fn gpu_utilization(&self) -> f64 {
        if self.total_operations() == 0 {
            0.0
        } else {
            self.gpu_operations as f64 / self.total_operations() as f64
        }
    }

    /// Update average execution time for a query type
    pub fn update_avg_time(&mut self, query_type: &str, execution_time_ms: f64) {
        let current_avg = self.avg_times.get(query_type).copied().unwrap_or(0.0);
        let count = if current_avg == 0.0 { 1.0 } else { 2.0 }; // Simplified averaging
        let new_avg = (current_avg + execution_time_ms) / count;
        self.avg_times.insert(query_type.to_string(), new_avg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::KnowledgeGraphConfig;

    #[tokio::test]
    async fn test_query_executor_creation() {
        let cpu_executor = QueryExecutor::new(false).await;
        assert!(cpu_executor.is_ok());

        let gpu_executor = QueryExecutor::new(true).await;
        assert!(gpu_executor.is_ok());
    }

    #[tokio::test]
    async fn test_find_nodes_execution() {
        let mut executor = QueryExecutor::new(false).await.unwrap();
        let mut graph = KnowledgeGraph::new(KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        })
        .await
        ?;

        // Add test node
        let mut node = Node::new(NodeType::Agent, HashMap::new());
        node.update_property("name".to_string(), serde_json::json!("test"));
        graph.add_node(node).unwrap();

        let query = Query::new(QueryType::FindNodes {
            node_type: Some(NodeType::Agent),
            properties: HashMap::new(),
        });

        let result = executor
            .execute_find_nodes(&graph, Some(NodeType::Agent), HashMap::new(), &query)
            .await
            .unwrap();

        assert_eq!(result.nodes.len(), 1);
        assert!(!result.gpu_accelerated);
    }

    #[tokio::test]
    async fn test_path_finding_execution() {
        let mut executor = QueryExecutor::new(false).await.unwrap();
        let mut graph = KnowledgeGraph::new(KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        })
        .await
        ?;

        // Create simple path: A -> B
        let node_a = Node::new(NodeType::Agent, HashMap::new());
        let node_a_id = node_a.id.clone();
        graph.add_node(node_a).unwrap();

        let node_b = Node::new(NodeType::Goal, HashMap::new());
        let node_b_id = node_b.id.clone();
        graph.add_node(node_b).unwrap();

        let edge = Edge::new(node_a_id.clone(), node_b_id.clone(), EdgeType::Has, 1.0);
        graph.add_edge(edge).unwrap();

        let query = Query::new(QueryType::FindPath {
            source_id: node_a_id.clone(),
            target_id: node_b_id.clone(),
            max_length: 5,
        });

        let result = executor
            .execute_find_path(&graph, &node_a_id, &node_b_id, 5, &query)
            .await
            .unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].len(), 2);
        assert!(!result.gpu_accelerated);
    }

    #[tokio::test]
    async fn test_neighborhood_execution() {
        let mut executor = QueryExecutor::new(false).await.unwrap();
        let mut graph = KnowledgeGraph::new(KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        })
        .await
        ?;

        // Create center node with neighbor
        let center = Node::new(NodeType::Agent, HashMap::new());
        let center_id = center.id.clone();
        graph.add_node(center).unwrap();

        let neighbor = Node::new(NodeType::Goal, HashMap::new());
        let neighbor_id = neighbor.id.clone();
        graph.add_node(neighbor).unwrap();

        let edge = Edge::new(center_id.clone(), neighbor_id, EdgeType::Has, 1.0);
        graph.add_edge(edge).unwrap();

        let query = Query::new(QueryType::Neighborhood {
            node_id: center_id.clone(),
            radius: 1,
            edge_types: None,
        });

        let result = executor
            .execute_neighborhood(&graph, &center_id, 1, None, &query)
            .await
            .unwrap();

        assert_eq!(result.nodes.len(), 2); // center + neighbor
        assert_eq!(result.edges.len(), 1);
    }

    #[tokio::test]
    async fn test_degree_centrality() {
        let mut executor = QueryExecutor::new(false).await.unwrap();
        let mut graph = KnowledgeGraph::new(KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        })
        .await
        ?;

        // Create hub node
        let hub = Node::new(NodeType::Agent, HashMap::new());
        let hub_id = hub.id.clone();
        graph.add_node(hub).unwrap();

        // Connect to multiple nodes
        for _ in 0..3 {
            let node = Node::new(NodeType::Goal, HashMap::new());
            let node_id = node.id.clone();
            graph.add_node(node).unwrap();

            let edge = Edge::new(hub_id.clone(), node_id, EdgeType::Has, 1.0);
            graph.add_edge(edge).unwrap();
        }

        let query = Query::new(QueryType::Ranking {
            algorithm: RankingAlgorithm::DegreeCentrality,
            node_type: Some(NodeType::Agent),
        });

        let result = executor
            .execute_ranking(
                &graph,
                RankingAlgorithm::DegreeCentrality,
                Some(NodeType::Agent),
                &query,
            )
            .await
            .unwrap();

        assert!(result.scores.contains_key(&hub_id));
        assert_eq!(result.scores.get(&hub_id), Some(&3.0));
    }

    #[test]
    fn test_execution_stats() {
        let mut stats = ExecutionStats::default();

        assert_eq!(stats.total_operations(), 0);
        assert_eq!(stats.gpu_utilization(), 0.0);

        stats.gpu_operations = 3;
        stats.cpu_operations = 7;

        assert_eq!(stats.total_operations(), 10);
        assert_eq!(stats.gpu_utilization(), 0.3);

        stats.update_avg_time("find_nodes", 100.0);
        stats.update_avg_time("find_nodes", 200.0);

        assert_eq!(stats.avg_times.get("find_nodes"), Some(&150.0));
    }

    #[test]
    fn test_property_matching() {
        let executor = QueryExecutor {
            gpu_kernels: None,
            stats: ExecutionStats::default(),
        };

        let mut node = Node::new(NodeType::Agent, HashMap::new());
        node.update_property("name".to_string(), serde_json::json!("test"));
        node.update_property("active".to_string(), serde_json::json!(true));

        let mut properties = HashMap::new();
        properties.insert("name".to_string(), serde_json::json!("test"));

        assert!(executor.node_matches_properties(&node, &properties));

        properties.insert("active".to_string(), serde_json::json!(false));
        assert!(!executor.node_matches_properties(&node, &properties));

        properties.insert("nonexistent".to_string(), serde_json::json!("value"));
        assert!(!executor.node_matches_properties(&node, &properties));
    }

    #[test]
    fn test_pagination() {
        let executor = QueryExecutor {
            gpu_kernels: None,
            stats: ExecutionStats::default(),
        };

        let items = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // Test limit only
        let result = executor.apply_pagination(items.clone(), Some(5), None);
        assert_eq!(result, vec![1, 2, 3, 4, 5]);

        // Test offset only
        let result = executor.apply_pagination(items.clone(), None, Some(3));
        assert_eq!(result, vec![4, 5, 6, 7, 8, 9, 10]);

        // Test both limit and offset
        let result = executor.apply_pagination(items.clone(), Some(3), Some(2));
        assert_eq!(result, vec![3, 4, 5]);

        // Test offset beyond length
        let result = executor.apply_pagination(items.clone(), None, Some(15));
        assert!(result.is_empty());
    }
}
