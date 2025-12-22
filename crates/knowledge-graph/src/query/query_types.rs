//! Query type definitions and data structures
//!
//! This module contains all the data types used to represent different kinds
//! of queries and their results in the knowledge graph system.

use crate::graph::{Edge, EdgeType, Node, NodeType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Query types supported by the engine
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryType {
    /// Find nodes by property
    FindNodes {
        /// Node type filter
        node_type: Option<NodeType>,
        /// Property filters
        properties: HashMap<String, serde_json::Value>,
    },
    /// Find edges by criteria
    FindEdges {
        /// Edge type filter
        edge_type: Option<EdgeType>,
        /// Weight range filter
        weight_range: Option<(f64, f64)>,
    },
    /// Path finding between nodes
    FindPath {
        /// Source node ID
        source_id: String,
        /// Target node ID
        target_id: String,
        /// Maximum path length
        max_length: usize,
    },
    /// Neighborhood query
    Neighborhood {
        /// Center node ID
        node_id: String,
        /// Radius (number of hops)
        radius: usize,
        /// Edge type filter
        edge_types: Option<Vec<EdgeType>>,
    },
    /// Subgraph extraction
    Subgraph {
        /// Seed node IDs
        seed_nodes: Vec<String>,
        /// Maximum expansion depth
        max_depth: usize,
    },
    /// Pattern matching
    PatternMatch {
        /// Pattern specification
        pattern: GraphPattern,
    },
    /// Ranking query
    Ranking {
        /// Ranking algorithm
        algorithm: RankingAlgorithm,
        /// Node type filter
        node_type: Option<NodeType>,
    },
}

/// Graph pattern for pattern matching
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphPattern {
    /// Pattern nodes
    pub nodes: Vec<PatternNode>,
    /// Pattern edges
    pub edges: Vec<PatternEdge>,
}

/// Pattern node specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternNode {
    /// Pattern node ID
    pub id: String,
    /// Required node type
    pub node_type: Option<NodeType>,
    /// Required properties
    pub properties: HashMap<String, serde_json::Value>,
}

/// Pattern edge specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternEdge {
    /// Source pattern node ID
    pub source_id: String,
    /// Target pattern node ID
    pub target_id: String,
    /// Required edge type
    pub edge_type: Option<EdgeType>,
    /// Weight constraints
    pub weight_range: Option<(f64, f64)>,
}

/// Ranking algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RankingAlgorithm {
    /// PageRank algorithm
    PageRank {
        /// Damping factor
        damping: f64,
        /// Number of iterations
        iterations: usize,
    },
    /// Degree centrality
    DegreeCentrality,
    /// Betweenness centrality
    BetweennessCentrality,
    /// Closeness centrality
    ClosenessCentrality,
    /// Custom ranking
    Custom(String),
}

/// Query specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// Query type
    pub query_type: QueryType,
    /// Query timeout in milliseconds
    pub timeout_ms: Option<u64>,
    /// Result limit
    pub limit: Option<usize>,
    /// Result offset
    pub offset: Option<usize>,
    /// Enable GPU acceleration
    pub use_gpu: bool,
}

/// Query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Result nodes
    pub nodes: Vec<Node>,
    /// Result edges
    pub edges: Vec<Edge>,
    /// Paths (for path queries)
    pub paths: Vec<Vec<String>>,
    /// Scores (for ranking queries)
    pub scores: HashMap<String, f64>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Whether GPU was used
    pub gpu_accelerated: bool,
    /// Query metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Query {
    /// Create a new query with default settings
    pub fn new(query_type: QueryType) -> Self {
        Self {
            query_type,
            timeout_ms: None,
            limit: None,
            offset: None,
            use_gpu: false,
        }
    }

    /// Set timeout for the query
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set result offset
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Enable GPU acceleration
    pub fn with_gpu(mut self) -> Self {
        self.use_gpu = true;
        self
    }
}

impl QueryResult {
    /// Create a new empty query result
    pub fn empty() -> Self {
        Self {
            nodes: vec![],
            edges: vec![],
            paths: vec![],
            scores: HashMap::new(),
            execution_time_ms: 0,
            gpu_accelerated: false,
            metadata: HashMap::new(),
        }
    }

    /// Create a result with nodes
    pub fn with_nodes(nodes: Vec<Node>) -> Self {
        let mut result = Self::empty();
        result.nodes = nodes;
        result
    }

    /// Create a result with edges
    pub fn with_edges(edges: Vec<Edge>) -> Self {
        let mut result = Self::empty();
        result.edges = edges;
        result
    }

    /// Create a result with paths
    pub fn with_paths(paths: Vec<Vec<String>>) -> Self {
        let mut result = Self::empty();
        result.paths = paths;
        result
    }

    /// Create a result with scores
    pub fn with_scores(scores: HashMap<String, f64>) -> Self {
        let mut result = Self::empty();
        result.scores = scores;
        result
    }

    /// Set execution time
    pub fn with_execution_time(mut self, execution_time_ms: u64) -> Self {
        self.execution_time_ms = execution_time_ms;
        self
    }

    /// Mark as GPU accelerated
    pub fn with_gpu_acceleration(mut self) -> Self {
        self.gpu_accelerated = true;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
            && self.edges.is_empty()
            && self.paths.is_empty()
            && self.scores.is_empty()
    }

    /// Get total result count
    pub fn total_results(&self) -> usize {
        self.nodes.len() + self.edges.len() + self.paths.len() + self.scores.len()
    }
}

impl GraphPattern {
    /// Create a new empty pattern
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            edges: vec![],
        }
    }

    /// Add a pattern node
    pub fn add_node(mut self, node: PatternNode) -> Self {
        self.nodes.push(node);
        self
    }

    /// Add a pattern edge
    pub fn add_edge(mut self, edge: PatternEdge) -> Self {
        self.edges.push(edge);
        self
    }

    /// Check if pattern is valid
    pub fn is_valid(&self) -> bool {
        // Check that all edge references point to existing nodes
        let node_ids: std::collections::HashSet<_> = self.nodes.iter().map(|n| &n.id).collect();

        for edge in &self.edges {
            if !node_ids.contains(&edge.source_id) || !node_ids.contains(&edge.target_id) {
                return false;
            }
        }

        true
    }
}

impl Default for GraphPattern {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternNode {
    /// Create a new pattern node
    pub fn new(id: String) -> Self {
        Self {
            id,
            node_type: None,
            properties: HashMap::new(),
        }
    }

    /// Set node type constraint
    pub fn with_type(mut self, node_type: NodeType) -> Self {
        self.node_type = Some(node_type);
        self
    }

    /// Add property constraint
    pub fn with_property(mut self, key: String, value: serde_json::Value) -> Self {
        self.properties.insert(key, value);
        self
    }
}

impl PatternEdge {
    /// Create a new pattern edge
    pub fn new(source_id: String, target_id: String) -> Self {
        Self {
            source_id,
            target_id,
            edge_type: None,
            weight_range: None,
        }
    }

    /// Set edge type constraint
    pub fn with_type(mut self, edge_type: EdgeType) -> Self {
        self.edge_type = Some(edge_type);
        self
    }

    /// Set weight range constraint
    pub fn with_weight_range(mut self, min: f64, max: f64) -> Self {
        self.weight_range = Some((min, max));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_query_builder() {
        let query = Query::new(QueryType::FindNodes {
            node_type: Some(NodeType::Agent),
            properties: HashMap::new(),
        })
        .with_timeout(5000)
        .with_limit(10)
        .with_gpu();

        assert_eq!(query.timeout_ms, Some(5000));
        assert_eq!(query.limit, Some(10));
        assert!(query.use_gpu);
    }

    #[test]
    fn test_query_result_builder() {
        let result = QueryResult::empty()
            .with_execution_time(100)
            .with_gpu_acceleration()
            .with_metadata("key".to_string(), json!("value"));

        assert_eq!(result.execution_time_ms, 100);
        assert!(result.gpu_accelerated);
        assert_eq!(result.metadata.get("key"), Some(&json!("value")));
        assert!(result.is_empty());
    }

    #[test]
    fn test_graph_pattern_validation() {
        let pattern = GraphPattern::new()
            .add_node(PatternNode::new("a".to_string()).with_type(NodeType::Agent))
            .add_node(PatternNode::new("b".to_string()).with_type(NodeType::Goal))
            .add_edge(PatternEdge::new("a".to_string(), "b".to_string()).with_type(EdgeType::Has));

        assert!(pattern.is_valid());

        // Invalid pattern with edge referencing non-existent node
        let invalid_pattern = GraphPattern::new()
            .add_node(PatternNode::new("a".to_string()))
            .add_edge(PatternEdge::new("a".to_string(), "nonexistent".to_string()));

        assert!(!invalid_pattern.is_valid());
    }

    #[test]
    fn test_pattern_node_builder() {
        let node = PatternNode::new("test".to_string())
            .with_type(NodeType::Agent)
            .with_property("name".to_string(), json!("test_agent"));

        assert_eq!(node.id, "test");
        assert_eq!(node.node_type, Some(NodeType::Agent));
        assert_eq!(node.properties.get("name"), Some(&json!("test_agent")));
    }

    #[test]
    fn test_pattern_edge_builder() {
        let edge = PatternEdge::new("a".to_string(), "b".to_string())
            .with_type(EdgeType::Has)
            .with_weight_range(0.5, 1.0);

        assert_eq!(edge.source_id, "a");
        assert_eq!(edge.target_id, "b");
        assert_eq!(edge.edge_type, Some(EdgeType::Has));
        assert_eq!(edge.weight_range, Some((0.5, 1.0)));
    }

    #[test]
    fn test_query_result_metrics() {
        let nodes = vec![Node::new(NodeType::Agent, HashMap::new())];
        let result = QueryResult::with_nodes(nodes);

        assert!(!result.is_empty());
        assert_eq!(result.total_results(), 1);
    }

    #[test]
    fn test_ranking_algorithm_serialization() {
        let pagerank = RankingAlgorithm::PageRank {
            damping: 0.85,
            iterations: 10,
        };

        let serialized = serde_json::to_string(&pagerank)?;
        let deserialized: RankingAlgorithm = serde_json::from_str(&serialized)?;

        assert_eq!(pagerank, deserialized);
    }
}
