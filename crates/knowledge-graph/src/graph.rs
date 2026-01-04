//! Core knowledge graph data structures and operations

use crate::error::{KnowledgeGraphError, KnowledgeGraphResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Node types in the knowledge graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Hash, Eq)]
pub enum NodeType {
    /// Agent node
    Agent,
    /// Goal node
    Goal,
    /// Concept node
    Concept,
    /// Memory node
    Memory,
    /// Pattern node
    Pattern,
    /// Evolution state
    Evolution,
    /// Kernel node
    Kernel,
    /// Custom node type
    Custom(String),
}

/// Edge types in the knowledge graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Hash, Eq)]
pub enum EdgeType {
    /// Has relationship
    Has,
    /// Contains relationship
    Contains,
    /// Relates to
    RelatesTo,
    /// Derives from
    DerivesFrom,
    /// Evolves to
    EvolvesTo,
    /// Uses relationship
    Uses,
    /// Produces relationship
    Produces,
    /// Custom edge type
    Custom(String),
}

/// Knowledge graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique node identifier
    pub id: String,
    /// Node type
    pub node_type: NodeType,
    /// Node properties
    pub properties: HashMap<String, serde_json::Value>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Node embedding vector for semantic search
    pub embedding: Option<Vec<f32>>,
}

impl Node {
    /// Create a new node
    pub fn new(node_type: NodeType, properties: HashMap<String, serde_json::Value>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            node_type,
            properties,
            created_at: now,
            updated_at: now,
            embedding: None,
        }
    }

    /// Update node properties
    pub fn update_property(&mut self, key: String, value: serde_json::Value) {
        self.properties.insert(key, value);
        self.updated_at = Utc::now();
    }

    /// Get property value
    pub fn get_property(&self, key: &str) -> Option<&serde_json::Value> {
        self.properties.get(key)
    }

    /// Set embedding vector
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
        self.updated_at = Utc::now();
    }
}

/// Knowledge graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Unique edge identifier
    pub id: String,
    /// Source node ID
    pub source_id: String,
    /// Target node ID
    pub target_id: String,
    /// Edge type
    pub edge_type: EdgeType,
    /// Edge properties
    pub properties: HashMap<String, serde_json::Value>,
    /// Edge weight
    pub weight: f64,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl Edge {
    /// Create a new edge
    pub fn new(source_id: String, target_id: String, edge_type: EdgeType, weight: f64) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            source_id,
            target_id,
            edge_type,
            properties: HashMap::new(),
            weight,
            created_at: now,
            updated_at: now,
        }
    }

    /// Update edge weight
    pub fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
        self.updated_at = Utc::now();
    }

    /// Update edge property
    pub fn update_property(&mut self, key: String, value: serde_json::Value) {
        self.properties.insert(key, value);
        self.updated_at = Utc::now();
    }
}

/// Knowledge graph configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphConfig {
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// Maximum number of edges
    pub max_edges: usize,
    /// Enable GPU acceleration
    pub gpu_enabled: bool,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Automatic pruning threshold
    pub pruning_threshold: f64,
    /// Enable evolution tracking
    pub evolution_tracking: bool,
}

impl Default for KnowledgeGraphConfig {
    fn default() -> Self {
        Self {
            max_nodes: 1_000_000,
            max_edges: 10_000_000,
            gpu_enabled: true,
            embedding_dim: 512,
            pruning_threshold: 0.1,
            evolution_tracking: true,
        }
    }
}

/// Main knowledge graph structure
pub struct KnowledgeGraph {
    /// Graph configuration
    config: KnowledgeGraphConfig,
    /// Nodes storage
    nodes: HashMap<String, Node>,
    /// Edges storage
    edges: HashMap<String, Edge>,
    /// Index from source node to edges
    outgoing_edges: HashMap<String, Vec<String>>,
    /// Index from target node to edges
    incoming_edges: HashMap<String, Vec<String>>,
    /// Node type index
    node_type_index: HashMap<NodeType, Vec<String>>,
    /// GPU context for accelerated operations
    gpu_context: Option<stratoswarm_cuda::Context>,
}

impl KnowledgeGraph {
    /// Create a new knowledge graph
    pub async fn new(config: KnowledgeGraphConfig) -> KnowledgeGraphResult<Self> {
        let gpu_context = if config.gpu_enabled {
            Some(stratoswarm_cuda::Context::new(
                0,
                stratoswarm_cuda::ContextFlags::default(),
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            outgoing_edges: HashMap::new(),
            incoming_edges: HashMap::new(),
            node_type_index: HashMap::new(),
            gpu_context,
        })
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: Node) -> KnowledgeGraphResult<String> {
        if self.nodes.len() >= self.config.max_nodes {
            return Err(KnowledgeGraphError::Other(
                "Maximum number of nodes reached".to_string(),
            ));
        }

        let node_id = node.id.clone();
        let node_type = node.node_type.clone();

        // Update node type index
        self.node_type_index
            .entry(node_type)
            .or_default()
            .push(node_id.clone());

        // Initialize edge indices
        self.outgoing_edges.insert(node_id.clone(), Vec::new());
        self.incoming_edges.insert(node_id.clone(), Vec::new());

        self.nodes.insert(node_id.clone(), node);
        Ok(node_id)
    }

    /// Get a node by ID
    pub fn get_node(&self, node_id: &str) -> KnowledgeGraphResult<&Node> {
        self.nodes
            .get(node_id)
            .ok_or_else(|| KnowledgeGraphError::NodeNotFound {
                node_id: node_id.to_string(),
            })
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, node_id: &str) -> KnowledgeGraphResult<&mut Node> {
        self.nodes
            .get_mut(node_id)
            .ok_or_else(|| KnowledgeGraphError::NodeNotFound {
                node_id: node_id.to_string(),
            })
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: Edge) -> KnowledgeGraphResult<String> {
        if self.edges.len() >= self.config.max_edges {
            return Err(KnowledgeGraphError::Other(
                "Maximum number of edges reached".to_string(),
            ));
        }

        // Verify source and target nodes exist
        if !self.nodes.contains_key(&edge.source_id) {
            return Err(KnowledgeGraphError::NodeNotFound {
                node_id: edge.source_id.clone(),
            });
        }
        if !self.nodes.contains_key(&edge.target_id) {
            return Err(KnowledgeGraphError::NodeNotFound {
                node_id: edge.target_id.clone(),
            });
        }

        let edge_id = edge.id.clone();

        // Update edge indices
        self.outgoing_edges
            .get_mut(&edge.source_id)
            .unwrap()
            .push(edge_id.clone());
        self.incoming_edges
            .get_mut(&edge.target_id)
            .unwrap()
            .push(edge_id.clone());

        self.edges.insert(edge_id.clone(), edge);
        Ok(edge_id)
    }

    /// Get an edge by ID
    pub fn get_edge(&self, edge_id: &str) -> KnowledgeGraphResult<&Edge> {
        self.edges
            .get(edge_id)
            .ok_or_else(|| KnowledgeGraphError::EdgeNotFound {
                edge_id: edge_id.to_string(),
            })
    }

    /// Get outgoing edges for a node
    pub fn get_outgoing_edges(&self, node_id: &str) -> Vec<&Edge> {
        self.outgoing_edges
            .get(node_id)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|id| self.edges.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get incoming edges for a node
    pub fn get_incoming_edges(&self, node_id: &str) -> Vec<&Edge> {
        self.incoming_edges
            .get(node_id)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|id| self.edges.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get nodes by type
    pub fn get_nodes_by_type(&self, node_type: &NodeType) -> Vec<&Node> {
        self.node_type_index
            .get(node_type)
            .map(|node_ids| {
                node_ids
                    .iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all nodes in the graph
    pub fn get_all_nodes(&self) -> KnowledgeGraphResult<Vec<&Node>> {
        Ok(self.nodes.values().collect())
    }

    /// Get all edges in the graph
    pub fn get_all_edges(&self) -> KnowledgeGraphResult<Vec<&Edge>> {
        Ok(self.edges.values().collect())
    }

    /// Remove a node and all its edges
    pub fn remove_node(&mut self, node_id: &str) -> KnowledgeGraphResult<Node> {
        let node = self
            .nodes
            .remove(node_id)
            .ok_or_else(|| KnowledgeGraphError::NodeNotFound {
                node_id: node_id.to_string(),
            })?;

        // Remove from type index
        if let Some(node_ids) = self.node_type_index.get_mut(&node.node_type) {
            node_ids.retain(|id| id != node_id);
        }

        // Remove all edges connected to this node
        let outgoing = self.outgoing_edges.remove(node_id).unwrap_or_default();
        let incoming = self.incoming_edges.remove(node_id).unwrap_or_default();

        for edge_id in outgoing.into_iter().chain(incoming.into_iter()) {
            if let Some(edge) = self.edges.remove(&edge_id) {
                // Clean up edge indices
                if let Some(source_edges) = self.outgoing_edges.get_mut(&edge.source_id) {
                    source_edges.retain(|id| id != &edge_id);
                }
                if let Some(target_edges) = self.incoming_edges.get_mut(&edge.target_id) {
                    target_edges.retain(|id| id != &edge_id);
                }
            }
        }

        Ok(node)
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        GraphStats {
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            node_type_distribution: self
                .node_type_index
                .iter()
                .map(|(node_type, ids)| (node_type.clone(), ids.len()))
                .collect(),
        }
    }

    /// Check if GPU is available
    pub fn is_gpu_enabled(&self) -> bool {
        self.gpu_context.is_some()
    }

    /// Get all edge IDs
    pub fn get_all_edge_ids(&self) -> Vec<String> {
        self.edges.keys().cloned().collect()
    }

    /// Get all node IDs
    pub fn get_all_node_ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// Distribution of node types
    pub node_type_distribution: HashMap<NodeType, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let properties = HashMap::new();
        let node = Node::new(NodeType::Agent, properties);

        assert_eq!(node.node_type, NodeType::Agent);
        assert!(node.embedding.is_none());
        assert!(!node.id.is_empty());
    }

    #[test]
    fn test_node_property_operations() {
        let mut node = Node::new(NodeType::Goal, HashMap::new());

        node.update_property(
            "name".to_string(),
            serde_json::Value::String("test_goal".to_string()),
        );
        assert_eq!(
            node.get_property("name"),
            Some(&serde_json::Value::String("test_goal".to_string()))
        );

        node.set_embedding(vec![0.1, 0.2, 0.3]);
        assert_eq!(node.embedding, Some(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_edge_creation() {
        let edge = Edge::new("node1".to_string(), "node2".to_string(), EdgeType::Has, 1.0);

        assert_eq!(edge.source_id, "node1");
        assert_eq!(edge.target_id, "node2");
        assert_eq!(edge.edge_type, EdgeType::Has);
        assert_eq!(edge.weight, 1.0);
    }

    #[tokio::test]
    async fn test_knowledge_graph_creation() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let graph = KnowledgeGraph::new(config).await;
        assert!(graph.is_ok());
    }

    #[tokio::test]
    async fn test_add_and_get_node() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(config).await?;

        let node = Node::new(NodeType::Agent, HashMap::new());
        let node_id = node.id.clone();

        let added_id = graph.add_node(node)?;
        assert_eq!(added_id, node_id);

        let retrieved = graph.get_node(&node_id).unwrap();
        assert_eq!(retrieved.id, node_id);
        assert_eq!(retrieved.node_type, NodeType::Agent);
    }

    #[tokio::test]
    async fn test_add_and_get_edge() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(config).await?;

        // Add two nodes
        let node1 = Node::new(NodeType::Agent, HashMap::new());
        let node1_id = node1.id.clone();
        graph.add_node(node1)?;

        let node2 = Node::new(NodeType::Goal, HashMap::new());
        let node2_id = node2.id.clone();
        graph.add_node(node2).unwrap();

        // Add edge
        let edge = Edge::new(node1_id.clone(), node2_id.clone(), EdgeType::Has, 0.8);
        let edge_id = edge.id.clone();
        graph.add_edge(edge).unwrap();

        // Verify edge
        let retrieved = graph.get_edge(&edge_id).unwrap();
        assert_eq!(retrieved.source_id, node1_id);
        assert_eq!(retrieved.target_id, node2_id);
        assert_eq!(retrieved.edge_type, EdgeType::Has);

        // Test edge navigation
        let outgoing = graph.get_outgoing_edges(&node1_id);
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].id, edge_id);

        let incoming = graph.get_incoming_edges(&node2_id);
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].id, edge_id);
    }

    #[tokio::test]
    async fn test_nodes_by_type() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(config).await?;

        // Add nodes of different types
        graph.add_node(Node::new(NodeType::Agent, HashMap::new()))?;
        graph
            .add_node(Node::new(NodeType::Agent, HashMap::new()))
            .unwrap();
        graph
            .add_node(Node::new(NodeType::Goal, HashMap::new()))
            .unwrap();

        let agents = graph.get_nodes_by_type(&NodeType::Agent);
        assert_eq!(agents.len(), 2);

        let goals = graph.get_nodes_by_type(&NodeType::Goal);
        assert_eq!(goals.len(), 1);
    }

    #[tokio::test]
    async fn test_node_removal() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(config).await?;

        // Add nodes and edge
        let node1 = Node::new(NodeType::Agent, HashMap::new());
        let node1_id = node1.id.clone();
        graph.add_node(node1)?;

        let node2 = Node::new(NodeType::Goal, HashMap::new());
        let node2_id = node2.id.clone();
        graph.add_node(node2).unwrap();

        let edge = Edge::new(node1_id.clone(), node2_id.clone(), EdgeType::Has, 1.0);
        graph.add_edge(edge).unwrap();

        // Remove node1
        let removed = graph.remove_node(&node1_id).unwrap();
        assert_eq!(removed.id, node1_id);

        // Verify node and edges are gone
        assert!(graph.get_node(&node1_id).is_err());
        assert_eq!(graph.get_outgoing_edges(&node2_id).len(), 0);
        assert_eq!(graph.stats().edge_count, 0);
    }

    #[tokio::test]
    async fn test_graph_stats() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(config).await?;

        graph.add_node(Node::new(NodeType::Agent, HashMap::new()))?;
        graph
            .add_node(Node::new(NodeType::Goal, HashMap::new()))
            .unwrap();

        let stats = graph.stats();
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.node_type_distribution.get(&NodeType::Agent), Some(&1));
        assert_eq!(stats.node_type_distribution.get(&NodeType::Goal), Some(&1));
    }

    #[test]
    fn test_node_type_equality() {
        assert_eq!(NodeType::Agent, NodeType::Agent);
        assert_ne!(NodeType::Agent, NodeType::Goal);

        let custom1 = NodeType::Custom("TypeA".to_string());
        let custom2 = NodeType::Custom("TypeA".to_string());
        let custom3 = NodeType::Custom("TypeB".to_string());

        assert_eq!(custom1, custom2);
        assert_ne!(custom1, custom3);
    }

    #[test]
    fn test_edge_type_equality() {
        assert_eq!(EdgeType::Has, EdgeType::Has);
        assert_ne!(EdgeType::Has, EdgeType::Contains);

        let custom1 = EdgeType::Custom("RelA".to_string());
        let custom2 = EdgeType::Custom("RelA".to_string());
        let custom3 = EdgeType::Custom("RelB".to_string());

        assert_eq!(custom1, custom2);
        assert_ne!(custom1, custom3);
    }

    #[test]
    fn test_node_serialization() {
        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            serde_json::Value::String("test".to_string()),
        );
        properties.insert("value".to_string(), serde_json::Value::Number(42.into()));

        let node = Node::new(NodeType::Concept, properties);

        let json = serde_json::to_string(&node)?;
        let deserialized: Node = serde_json::from_str(&json).unwrap();

        assert_eq!(node.id, deserialized.id);
        assert_eq!(node.node_type, deserialized.node_type);
        assert_eq!(node.properties, deserialized.properties);
    }

    #[test]
    fn test_edge_serialization() {
        let edge = Edge::new(
            "src".to_string(),
            "tgt".to_string(),
            EdgeType::EvolvesTo,
            0.75,
        );

        let json = serde_json::to_string(&edge)?;
        let deserialized: Edge = serde_json::from_str(&json)?;

        assert_eq!(edge.id, deserialized.id);
        assert_eq!(edge.source_id, deserialized.source_id);
        assert_eq!(edge.target_id, deserialized.target_id);
        assert_eq!(edge.edge_type, deserialized.edge_type);
        assert_eq!(edge.weight, deserialized.weight);
    }

    #[test]
    fn test_node_property_types() {
        let mut node = Node::new(NodeType::Memory, HashMap::new());

        // Test different property types
        node.update_property(
            "string".to_string(),
            serde_json::Value::String("text".to_string()),
        );
        node.update_property("number".to_string(), serde_json::Value::Number(123.into()));
        node.update_property("bool".to_string(), serde_json::Value::Bool(true));
        node.update_property("null".to_string(), serde_json::Value::Null);
        node.update_property("array".to_string(), serde_json::json!([1, 2, 3]));
        node.update_property("object".to_string(), serde_json::json!({"nested": "value"}));

        assert_eq!(node.properties.len(), 6);
        assert_eq!(node.get_property("string").unwrap().as_str(), Some("text"));
        assert_eq!(node.get_property("number").unwrap().as_i64(), Some(123));
        assert_eq!(node.get_property("bool").unwrap().as_bool(), Some(true));
        assert!(node.get_property("null").unwrap().is_null());
    }

    #[test]
    fn test_node_timestamps() {
        let node = Node::new(NodeType::Agent, HashMap::new());
        let created = node.created_at;
        let updated = node.updated_at;

        assert_eq!(created, updated);

        // Wait a bit and update
        std::thread::sleep(std::time::Duration::from_millis(10));
        let mut node = node;
        node.update_property(
            "key".to_string(),
            serde_json::Value::String("value".to_string()),
        );

        assert_eq!(node.created_at, created);
        assert!(node.updated_at > created);
    }

    #[tokio::test]
    async fn test_edge_not_found_error() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let graph = KnowledgeGraph::new(config).await?;

        let result = graph.get_edge("non-existent");
        assert!(result.is_err());

        match result {
            Err(KnowledgeGraphError::EdgeNotFound { edge_id }) => {
                assert_eq!(edge_id, "non-existent");
            }
            _ => panic!("Expected EdgeNotFound error"),
        }
    }

    #[tokio::test]
    async fn test_node_not_found_error() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let graph = KnowledgeGraph::new(config).await?;

        let result = graph.get_node("non-existent");
        assert!(result.is_err());

        match result {
            Err(KnowledgeGraphError::NodeNotFound { node_id }) => {
                assert_eq!(node_id, "non-existent");
            }
            _ => panic!("Expected NodeNotFound error"),
        }
    }

    #[tokio::test]
    async fn test_duplicate_node_error() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(config).await?;

        let mut node = Node::new(NodeType::Agent, HashMap::new());
        node.id = "fixed-id".to_string();

        graph.add_node(node.clone())?;
        let result = graph.add_node(node);

        assert!(result.is_err());
        match result {
            Err(KnowledgeGraphError::DuplicateNode { node_id }) => {
                assert_eq!(node_id, "fixed-id");
            }
            _ => panic!("Expected DuplicateNode error"),
        }
    }

    #[tokio::test]
    async fn test_invalid_edge_error() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(config).await?;

        let node = Node::new(NodeType::Agent, HashMap::new());
        let node_id = node.id.clone();
        graph.add_node(node)?;

        // Try to add edge with non-existent target
        let edge = Edge::new(node_id, "non-existent".to_string(), EdgeType::Has, 1.0);
        let result = graph.add_edge(edge);

        assert!(result.is_err());
        match result {
            Err(KnowledgeGraphError::InvalidEdge { message }) => {
                assert!(message.contains("non-existent"));
            }
            _ => panic!("Expected InvalidEdge error"),
        }
    }

    #[tokio::test]
    async fn test_multiple_edges_between_nodes() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(config).await?;

        let node1 = Node::new(NodeType::Agent, HashMap::new());
        let node1_id = node1.id.clone();
        graph.add_node(node1)?;

        let node2 = Node::new(NodeType::Goal, HashMap::new());
        let node2_id = node2.id.clone();
        graph.add_node(node2).unwrap();

        // Add multiple edges between same nodes
        let edge1 = Edge::new(node1_id.clone(), node2_id.clone(), EdgeType::Has, 0.5);
        let edge2 = Edge::new(node1_id.clone(), node2_id.clone(), EdgeType::Uses, 0.8);
        let edge3 = Edge::new(node2_id.clone(), node1_id.clone(), EdgeType::Produces, 0.3);

        graph.add_edge(edge1).unwrap();
        graph.add_edge(edge2).unwrap();
        graph.add_edge(edge3).unwrap();

        assert_eq!(graph.get_outgoing_edges(&node1_id).len(), 2);
        assert_eq!(graph.get_incoming_edges(&node1_id).len(), 1);
        assert_eq!(graph.get_outgoing_edges(&node2_id).len(), 1);
        assert_eq!(graph.get_incoming_edges(&node2_id).len(), 2);
    }

    #[tokio::test]
    async fn test_self_referential_edge() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(config).await?;

        let node = Node::new(NodeType::Pattern, HashMap::new());
        let node_id = node.id.clone();
        graph.add_node(node)?;

        // Add self-referential edge
        let edge = Edge::new(node_id.clone(), node_id.clone(), EdgeType::EvolvesTo, 1.0);
        graph.add_edge(edge).unwrap();

        // Both outgoing and incoming should contain the edge
        assert_eq!(graph.get_outgoing_edges(&node_id).len(), 1);
        assert_eq!(graph.get_incoming_edges(&node_id).len(), 1);
    }

    #[test]
    fn test_edge_weight_validation() {
        let edge1 = Edge::new("a".to_string(), "b".to_string(), EdgeType::Has, 0.0);
        assert_eq!(edge1.weight, 0.0);

        let edge2 = Edge::new("a".to_string(), "b".to_string(), EdgeType::Has, 1.0);
        assert_eq!(edge2.weight, 1.0);

        let edge3 = Edge::new("a".to_string(), "b".to_string(), EdgeType::Has, -0.5);
        assert_eq!(edge3.weight, -0.5);

        let edge4 = Edge::new("a".to_string(), "b".to_string(), EdgeType::Has, 2.5);
        assert_eq!(edge4.weight, 2.5);
    }

    #[tokio::test]
    async fn test_large_graph_operations() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(config).await?;

        // Add many nodes
        let mut node_ids = Vec::new();
        for i in 0..100 {
            let mut properties = HashMap::new();
            properties.insert("index".to_string(), serde_json::Value::Number(i.into()));
            let node = Node::new(NodeType::Agent, properties);
            node_ids.push(node.id.clone());
            graph.add_node(node).unwrap();
        }

        // Add edges in a chain
        for i in 0..99 {
            let edge = Edge::new(
                node_ids[i].clone(),
                node_ids[i + 1].clone(),
                EdgeType::RelatesTo,
                1.0,
            );
            graph.add_edge(edge).unwrap();
        }

        let stats = graph.stats();
        assert_eq!(stats.node_count, 100);
        assert_eq!(stats.edge_count, 99);
    }

    #[test]
    fn test_node_embedding_operations() {
        let mut node = Node::new(NodeType::Concept, HashMap::new());
        assert!(node.embedding.is_none());

        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        node.set_embedding(embedding.clone());
        assert_eq!(node.embedding, Some(embedding));

        // Test with different size embedding
        let large_embedding = vec![0.0; 768];
        node.set_embedding(large_embedding.clone());
        assert_eq!(node.embedding, Some(large_embedding));
    }

    #[tokio::test]
    async fn test_gpu_configuration() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: true,
            max_nodes: 1_000_000,
            max_edges: 10_000_000,
            cache_size: 1024,
        };

        let graph = KnowledgeGraph::new(config).await?;
        assert!(!graph.is_gpu_enabled()); // Mock mode doesn't actually enable GPU
    }

    #[tokio::test]
    async fn test_empty_graph_operations() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let graph = KnowledgeGraph::new(config).await?;

        assert_eq!(graph.get_nodes_by_type(&NodeType::Agent).len(), 0);
        assert_eq!(graph.get_outgoing_edges("any-id").len(), 0);
        assert_eq!(graph.get_incoming_edges("any-id").len(), 0);

        let stats = graph.stats();
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert!(stats.node_type_distribution.is_empty());
    }

    #[tokio::test]
    async fn test_remove_node_cascading_effects() {
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(config).await?;

        // Create a hub node with many connections
        let hub = Node::new(NodeType::Agent, HashMap::new());
        let hub_id = hub.id.clone();
        graph.add_node(hub)?;

        let mut connected_nodes = Vec::new();
        for i in 0..5 {
            let node = Node::new(NodeType::Goal, HashMap::new());
            let node_id = node.id.clone();
            connected_nodes.push(node_id.clone());
            graph.add_node(node).unwrap();

            // Add bidirectional edges
            let edge1 = Edge::new(hub_id.clone(), node_id.clone(), EdgeType::Has, 1.0);
            let edge2 = Edge::new(node_id, hub_id.clone(), EdgeType::Produces, 1.0);
            graph.add_edge(edge1).unwrap();
            graph.add_edge(edge2).unwrap();
        }

        assert_eq!(graph.stats().edge_count, 10);

        // Remove hub node
        graph.remove_node(&hub_id).unwrap();

        // All edges should be gone
        assert_eq!(graph.stats().edge_count, 0);
        for node_id in &connected_nodes {
            assert_eq!(graph.get_outgoing_edges(node_id).len(), 0);
            assert_eq!(graph.get_incoming_edges(node_id).len(), 0);
        }
    }

    #[test]
    fn test_graph_stats_serialization() {
        let mut distribution = HashMap::new();
        distribution.insert(NodeType::Agent, 10);
        distribution.insert(NodeType::Goal, 5);
        distribution.insert(NodeType::Custom("Special".to_string()), 3);

        let stats = GraphStats {
            node_count: 18,
            edge_count: 25,
            node_type_distribution: distribution,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: GraphStats = serde_json::from_str(&json).unwrap();

        assert_eq!(stats.node_count, deserialized.node_count);
        assert_eq!(stats.edge_count, deserialized.edge_count);
        assert_eq!(
            stats.node_type_distribution.len(),
            deserialized.node_type_distribution.len()
        );
    }

    #[test]
    fn test_custom_node_and_edge_types() {
        let custom_node_type = NodeType::Custom("SpecialAgent".to_string());
        let node = Node::new(custom_node_type.clone(), HashMap::new());

        match &node.node_type {
            NodeType::Custom(name) => assert_eq!(name, "SpecialAgent"),
            _ => panic!("Expected custom node type"),
        }

        let custom_edge_type = EdgeType::Custom("Influences".to_string());
        let edge = Edge::new(
            "a".to_string(),
            "b".to_string(),
            custom_edge_type.clone(),
            0.7,
        );

        match &edge.edge_type {
            EdgeType::Custom(name) => assert_eq!(name, "Influences"),
            _ => panic!("Expected custom edge type"),
        }
    }

    #[tokio::test]
    async fn test_concurrent_graph_operations() {
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let graph = Arc::new(Mutex::new(KnowledgeGraph::new(config).await?));

        let mut handles = vec![];

        // Spawn tasks to add nodes concurrently
        for i in 0..10 {
            let graph_clone = Arc::clone(&graph);
            let handle = tokio::spawn(async move {
                let mut g = graph_clone.lock().await;
                let mut properties = HashMap::new();
                properties.insert("task_id".to_string(), serde_json::Value::Number(i.into()));
                let node = Node::new(NodeType::Agent, properties);
                g.add_node(node).unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let g = graph.lock().await;
        assert_eq!(g.stats().node_count, 10);
    }

    #[test]
    fn test_node_property_edge_cases() {
        let mut node = Node::new(NodeType::Memory, HashMap::new());

        // Empty string key
        node.update_property(
            "".to_string(),
            serde_json::Value::String("empty_key".to_string()),
        );
        assert!(node.get_property("").is_some());

        // Very long key
        let long_key = "a".repeat(1000);
        node.update_property(
            long_key.clone(),
            serde_json::Value::String("long_key".to_string()),
        );
        assert!(node.get_property(&long_key).is_some());

        // Unicode key
        node.update_property(
            "🔑".to_string(),
            serde_json::Value::String("unicode_key".to_string()),
        );
        assert!(node.get_property("🔑").is_some());

        // Large nested object
        let nested = serde_json::json!({
            "level1": {
                "level2": {
                    "level3": {
                        "data": vec![0; 100]
                    }
                }
            }
        });
        node.update_property("nested".to_string(), nested);
        assert!(node.get_property("nested").is_some());
    }

    #[test]
    fn test_edge_id_uniqueness() {
        let mut edge_ids = std::collections::HashSet::new();

        for _ in 0..1000 {
            let edge = Edge::new("a".to_string(), "b".to_string(), EdgeType::Has, 1.0);
            assert!(edge_ids.insert(edge.id));
        }
    }

    #[test]
    fn test_knowledge_graph_config_default() {
        let config = KnowledgeGraphConfig::default();
        assert!(!config.gpu_enabled);
        assert_eq!(config.max_nodes, 10_000_000);
        assert_eq!(config.max_edges, 100_000_000);
        assert_eq!(config.cache_size, 1000);
    }
}
