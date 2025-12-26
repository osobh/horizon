//! Main graph management with cross-region coordination

use crate::error::{GlobalKnowledgeGraphError, GlobalKnowledgeGraphResult};
use crate::replication::ReplicationManager;
use dashmap::DashMap;
use parking_lot::RwLock;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

/// Graph configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Maximum nodes per region
    pub max_nodes_per_region: usize,
    /// Maximum edges per node
    pub max_edges_per_node: usize,
    /// Enable cross-region replication
    pub enable_replication: bool,
    /// Default region
    pub default_region: String,
    /// Supported regions
    pub supported_regions: Vec<String>,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_nodes_per_region: 1_000_000,
            max_edges_per_node: 1000,
            enable_replication: true,
            default_region: "us-east-1".to_string(),
            supported_regions: vec![
                "us-east-1".to_string(),
                "us-west-2".to_string(),
                "eu-west-1".to_string(),
                "ap-southeast-1".to_string(),
            ],
        }
    }
}

/// Node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Node {
    /// Unique node ID
    pub id: String,
    /// Node type
    pub node_type: String,
    /// Node properties
    pub properties: HashMap<String, serde_json::Value>,
    /// Region where node is stored
    pub region: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Version for optimistic locking
    pub version: u64,
}

/// Edge in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Edge {
    /// Unique edge ID
    pub id: String,
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge type
    pub edge_type: String,
    /// Edge properties
    pub properties: HashMap<String, serde_json::Value>,
    /// Weight for traversal algorithms
    pub weight: f64,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Graph traversal options
#[derive(Debug, Clone)]
pub struct TraversalOptions {
    /// Maximum depth
    pub max_depth: usize,
    /// Filter by node types
    pub node_types: Option<HashSet<String>>,
    /// Filter by edge types
    pub edge_types: Option<HashSet<String>>,
    /// Maximum nodes to visit
    pub max_nodes: usize,
}

impl Default for TraversalOptions {
    fn default() -> Self {
        Self {
            max_depth: 10,
            node_types: None,
            edge_types: None,
            max_nodes: 1000,
        }
    }
}

/// Main graph manager with cross-region coordination
pub struct GraphManager {
    config: Arc<GraphConfig>,
    graphs: Arc<DashMap<String, Arc<RwLock<DiGraph<Node, Edge>>>>>,
    node_indices: Arc<DashMap<String, HashMap<String, NodeIndex>>>,
    replication_manager: Option<Arc<ReplicationManager>>,
}

impl GraphManager {
    /// Create new graph manager
    pub fn new(config: GraphConfig) -> GlobalKnowledgeGraphResult<Self> {
        if config.supported_regions.is_empty() {
            return Err(GlobalKnowledgeGraphError::ConfigurationError {
                parameter: "supported_regions".to_string(),
                reason: "At least one region must be configured".to_string(),
            });
        }

        let graphs = Arc::new(DashMap::new());
        let node_indices = Arc::new(DashMap::new());

        // Initialize graph for each region
        for region in &config.supported_regions {
            graphs.insert(region.clone(), Arc::new(RwLock::new(DiGraph::new())));
            node_indices.insert(region.clone(), HashMap::new());
        }

        Ok(Self {
            config: Arc::new(config),
            graphs,
            node_indices,
            replication_manager: None,
        })
    }

    /// Set replication manager
    pub fn set_replication_manager(&mut self, manager: Arc<ReplicationManager>) {
        self.replication_manager = Some(manager);
    }

    /// Add node to graph
    pub async fn add_node(&self, node: Node) -> GlobalKnowledgeGraphResult<String> {
        if !self.config.supported_regions.contains(&node.region) {
            return Err(GlobalKnowledgeGraphError::RegionUnavailable {
                region: node.region.clone(),
                reason: "Region not supported".to_string(),
            });
        }

        let graph_lock = self.graphs.get(&node.region).ok_or_else(|| {
            GlobalKnowledgeGraphError::RegionUnavailable {
                region: node.region.clone(),
                reason: "Graph not initialized".to_string(),
            }
        })?;

        let mut graph = graph_lock.write();

        // Check node limit
        if graph.node_count() >= self.config.max_nodes_per_region {
            return Err(GlobalKnowledgeGraphError::GraphOperationFailed {
                operation: "add_node".to_string(),
                reason: format!("Node limit reached for region {}", node.region),
            });
        }

        let node_id = node.id.clone();
        let node_index = graph.add_node(node.clone());

        // Update indices
        self.node_indices.get_mut(&node.region).map(|mut indices| {
            indices.insert(node_id.clone(), node_index);
        });

        // Trigger replication if enabled
        if self.config.enable_replication {
            if let Some(ref replication_manager) = self.replication_manager {
                replication_manager.replicate_node(node).await?;
            }
        }

        Ok(node_id)
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: &str, region: &str) -> GlobalKnowledgeGraphResult<Node> {
        let graph_lock = self.graphs.get(region).ok_or_else(|| {
            GlobalKnowledgeGraphError::RegionUnavailable {
                region: region.to_string(),
                reason: "Region not supported".to_string(),
            }
        })?;

        let graph = graph_lock.read();
        let indices = self.node_indices.get(region).ok_or_else(|| {
            GlobalKnowledgeGraphError::RegionUnavailable {
                region: region.to_string(),
                reason: "Node indices not found".to_string(),
            }
        })?;

        let node_index =
            indices
                .get(node_id)
                .ok_or_else(|| GlobalKnowledgeGraphError::NodeNotFound {
                    node_id: node_id.to_string(),
                    region: region.to_string(),
                })?;

        graph.node_weight(*node_index).cloned().ok_or_else(|| {
            GlobalKnowledgeGraphError::NodeNotFound {
                node_id: node_id.to_string(),
                region: region.to_string(),
            }
        })
    }

    /// Update node
    pub async fn update_node(
        &self,
        node_id: &str,
        updates: HashMap<String, serde_json::Value>,
    ) -> GlobalKnowledgeGraphResult<Node> {
        // Find node across regions
        let (region, node_index) = self.find_node_location(node_id)?;

        let graph_lock = self.graphs.get(&region)?;
        let mut graph = graph_lock.write();

        let node = graph.node_weight_mut(node_index).ok_or_else(|| {
            GlobalKnowledgeGraphError::NodeNotFound {
                node_id: node_id.to_string(),
                region: region.clone(),
            }
        })?;

        // Update properties
        for (key, value) in updates {
            node.properties.insert(key, value);
        }
        node.updated_at = chrono::Utc::now();
        node.version += 1;

        let updated_node = node.clone();

        // Trigger replication if enabled
        if self.config.enable_replication {
            if let Some(ref replication_manager) = self.replication_manager {
                replication_manager
                    .replicate_node(updated_node.clone())
                    .await?;
            }
        }

        Ok(updated_node)
    }

    /// Delete node
    pub async fn delete_node(&self, node_id: &str) -> GlobalKnowledgeGraphResult<()> {
        let (region, node_index) = self.find_node_location(node_id)?;

        let graph_lock = self.graphs.get(&region)?;
        let mut graph = graph_lock.write();

        graph.remove_node(node_index);

        // Update indices
        self.node_indices.get_mut(&region).map(|mut indices| {
            indices.remove(node_id);
        });

        // Trigger replication if enabled
        if self.config.enable_replication {
            if let Some(ref replication_manager) = self.replication_manager {
                replication_manager
                    .replicate_node_deletion(node_id, &region)
                    .await?;
            }
        }

        Ok(())
    }

    /// Add edge between nodes
    pub async fn add_edge(&self, edge: Edge) -> GlobalKnowledgeGraphResult<String> {
        let (source_region, source_index) = self.find_node_location(&edge.source)?;
        let (target_region, target_index) = self.find_node_location(&edge.target)?;

        if source_region != target_region {
            return Err(GlobalKnowledgeGraphError::GraphOperationFailed {
                operation: "add_edge".to_string(),
                reason: "Cross-region edges not supported in this version".to_string(),
            });
        }

        let graph_lock = self.graphs.get(&source_region).unwrap();
        let mut graph = graph_lock.write();

        // Check edge limit
        let edge_count = graph.edges(source_index).count();
        if edge_count >= self.config.max_edges_per_node {
            return Err(GlobalKnowledgeGraphError::GraphOperationFailed {
                operation: "add_edge".to_string(),
                reason: format!("Edge limit reached for node {}", edge.source),
            });
        }

        graph.add_edge(source_index, target_index, edge.clone());

        // Trigger replication if enabled
        if self.config.enable_replication {
            if let Some(ref replication_manager) = self.replication_manager {
                replication_manager.replicate_edge(edge.clone()).await?;
            }
        }

        Ok(edge.id)
    }

    /// Delete edge
    pub async fn delete_edge(
        &self,
        edge_id: &str,
        source: &str,
        target: &str,
    ) -> GlobalKnowledgeGraphResult<()> {
        let (source_region, source_index) = self.find_node_location(source)?;
        let (_, target_index) = self.find_node_location(target)?;

        let graph_lock = self.graphs.get(&source_region)?;
        let mut graph = graph_lock.write();

        // Find and remove edge
        let edge_index = graph
            .edges(source_index)
            .find(|edge_ref| edge_ref.target() == target_index && edge_ref.weight().id == edge_id)
            .map(|edge_ref| edge_ref.id())
            .ok_or_else(|| GlobalKnowledgeGraphError::EdgeNotFound {
                edge_id: edge_id.to_string(),
                source_node: source.to_string(),
                target_node: target.to_string(),
            })?;

        graph.remove_edge(edge_index);

        // Trigger replication if enabled
        if self.config.enable_replication {
            if let Some(ref replication_manager) = self.replication_manager {
                replication_manager
                    .replicate_edge_deletion(edge_id, source, target)
                    .await?;
            }
        }

        Ok(())
    }

    /// Traverse graph from a starting node
    pub fn traverse(
        &self,
        start_node_id: &str,
        options: TraversalOptions,
    ) -> GlobalKnowledgeGraphResult<Vec<Node>> {
        let (region, start_index) = self.find_node_location(start_node_id)?;
        let graph_lock = self.graphs.get(&region)?;
        let graph = graph_lock.read();

        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = vec![(start_index, 0)];

        while let Some((current_index, depth)) = queue.pop() {
            if depth > options.max_depth || result.len() >= options.max_nodes {
                break;
            }

            if !visited.insert(current_index) {
                continue;
            }

            if let Some(node) = graph.node_weight(current_index) {
                // Apply node type filter
                if let Some(ref node_types) = options.node_types {
                    if !node_types.contains(&node.node_type) {
                        continue;
                    }
                }

                result.push(node.clone());

                // Traverse edges
                for edge_ref in graph.edges(current_index) {
                    let edge = edge_ref.weight();

                    // Apply edge type filter
                    if let Some(ref edge_types) = options.edge_types {
                        if !edge_types.contains(&edge.edge_type) {
                            continue;
                        }
                    }

                    queue.push((edge_ref.target(), depth + 1));
                }
            }
        }

        Ok(result)
    }

    /// Extract subgraph
    pub fn extract_subgraph(
        &self,
        node_ids: &[String],
        include_edges: bool,
    ) -> GlobalKnowledgeGraphResult<(Vec<Node>, Vec<Edge>)> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_set = HashSet::new();

        // Collect nodes
        for node_id in node_ids {
            let node = self.get_node_any_region(node_id)?;
            node_set.insert(node_id.clone());
            nodes.push(node);
        }

        // Collect edges if requested
        if include_edges {
            for region in &self.config.supported_regions {
                if let Some(graph_lock) = self.graphs.get(region) {
                    let graph = graph_lock.read();

                    for edge_ref in graph.edge_references() {
                        let edge = edge_ref.weight();
                        if node_set.contains(&edge.source) && node_set.contains(&edge.target) {
                            edges.push(edge.clone());
                        }
                    }
                }
            }
        }

        Ok((nodes, edges))
    }

    /// Find node location across regions
    fn find_node_location(&self, node_id: &str) -> GlobalKnowledgeGraphResult<(String, NodeIndex)> {
        for region in &self.config.supported_regions {
            if let Some(indices) = self.node_indices.get(region) {
                if let Some(node_index) = indices.get(node_id) {
                    return Ok((region.clone(), *node_index));
                }
            }
        }

        Err(GlobalKnowledgeGraphError::NodeNotFound {
            node_id: node_id.to_string(),
            region: "any".to_string(),
        })
    }

    /// Get node from any region
    fn get_node_any_region(&self, node_id: &str) -> GlobalKnowledgeGraphResult<Node> {
        let (region, _) = self.find_node_location(node_id)?;
        self.get_node(node_id, &region)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_graph_manager_creation() {
        let config = GraphConfig::default();
        let manager = GraphManager::new(config);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_graph_manager_invalid_config() {
        let config = GraphConfig {
            supported_regions: vec![],
            ..Default::default()
        };
        let result = GraphManager::new(config);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_add_node() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();
        let node = Node {
            id: Uuid::new_v4().to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let result = manager.add_node(node.clone()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), node.id);
    }

    #[tokio::test]
    async fn test_add_node_invalid_region() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();
        let node = Node {
            id: Uuid::new_v4().to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "invalid-region".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let result = manager.add_node(node).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_node() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();
        let node = Node {
            id: Uuid::new_v4().to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        manager.add_node(node.clone()).await.unwrap();
        let retrieved = manager.get_node(&node.id, &node.region).unwrap();
        assert_eq!(retrieved.id, node.id);
    }

    #[tokio::test]
    async fn test_get_node_not_found() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();
        let result = manager.get_node("non-existent", "us-east-1");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_update_node() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();
        let node = Node {
            id: Uuid::new_v4().to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        manager.add_node(node.clone()).await.unwrap();

        let mut updates = HashMap::new();
        updates.insert(
            "name".to_string(),
            serde_json::Value::String("Updated".to_string()),
        );

        let updated = manager.update_node(&node.id, updates).await.unwrap();
        assert_eq!(updated.version, 2);
        assert_eq!(updated.properties.get("name").unwrap(), "Updated");
    }

    #[tokio::test]
    async fn test_delete_node() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();
        let node = Node {
            id: Uuid::new_v4().to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        manager.add_node(node.clone()).await.unwrap();
        let result = manager.delete_node(&node.id).await;
        assert!(result.is_ok());

        let get_result = manager.get_node(&node.id, &node.region);
        assert!(get_result.is_err());
    }

    #[tokio::test]
    async fn test_add_edge() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();

        let node1 = Node {
            id: Uuid::new_v4().to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let node2 = Node {
            id: Uuid::new_v4().to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        manager.add_node(node1.clone()).await.unwrap();
        manager.add_node(node2.clone()).await.unwrap();

        let edge = Edge {
            id: Uuid::new_v4().to_string(),
            source: node1.id.clone(),
            target: node2.id.clone(),
            edge_type: "relates_to".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        };

        let result = manager.add_edge(edge.clone()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_add_edge_cross_region_fails() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();

        let node1 = Node {
            id: Uuid::new_v4().to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let node2 = Node {
            id: Uuid::new_v4().to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "eu-west-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        manager.add_node(node1.clone()).await.unwrap();
        manager.add_node(node2.clone()).await.unwrap();

        let edge = Edge {
            id: Uuid::new_v4().to_string(),
            source: node1.id.clone(),
            target: node2.id.clone(),
            edge_type: "relates_to".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        };

        let result = manager.add_edge(edge).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_edge() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();

        let node1 = Node {
            id: Uuid::new_v4().to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let node2 = Node {
            id: Uuid::new_v4().to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        manager.add_node(node1.clone()).await.unwrap();
        manager.add_node(node2.clone()).await.unwrap();

        let edge = Edge {
            id: Uuid::new_v4().to_string(),
            source: node1.id.clone(),
            target: node2.id.clone(),
            edge_type: "relates_to".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        };

        manager.add_edge(edge.clone()).await.unwrap();
        let result = manager.delete_edge(&edge.id, &node1.id, &node2.id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_traverse_graph() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();

        // Create a simple graph: node1 -> node2 -> node3
        let node1 = Node {
            id: "node1".to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let node2 = Node {
            id: "node2".to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let node3 = Node {
            id: "node3".to_string(),
            node_type: "document".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        manager.add_node(node1.clone()).await.unwrap();
        manager.add_node(node2.clone()).await.unwrap();
        manager.add_node(node3.clone()).await.unwrap();

        let edge1 = Edge {
            id: "edge1".to_string(),
            source: "node1".to_string(),
            target: "node2".to_string(),
            edge_type: "relates_to".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        };

        let edge2 = Edge {
            id: "edge2".to_string(),
            source: "node2".to_string(),
            target: "node3".to_string(),
            edge_type: "relates_to".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        };

        manager.add_edge(edge1).await.unwrap();
        manager.add_edge(edge2).await.unwrap();

        let options = TraversalOptions::default();
        let result = manager.traverse("node1", options).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn test_traverse_with_filters() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();

        let node1 = Node {
            id: "node1".to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let node2 = Node {
            id: "node2".to_string(),
            node_type: "document".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        manager.add_node(node1.clone()).await.unwrap();
        manager.add_node(node2.clone()).await.unwrap();

        let edge = Edge {
            id: "edge1".to_string(),
            source: "node1".to_string(),
            target: "node2".to_string(),
            edge_type: "relates_to".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        };

        manager.add_edge(edge).await.unwrap();

        let mut node_types = HashSet::new();
        node_types.insert("entity".to_string());

        let options = TraversalOptions {
            node_types: Some(node_types),
            ..Default::default()
        };

        let result = manager.traverse("node1", options).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].node_type, "entity");
    }

    #[tokio::test]
    async fn test_traverse_depth_limit() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();

        // Create a chain: node1 -> node2 -> node3 -> node4
        for i in 1..=4 {
            let node = Node {
                id: format!("node{}", i),
                node_type: "entity".to_string(),
                properties: HashMap::new(),
                region: "us-east-1".to_string(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            };
            manager.add_node(node).await.unwrap();
        }

        for i in 1..4 {
            let edge = Edge {
                id: format!("edge{}", i),
                source: format!("node{}", i),
                target: format!("node{}", i + 1),
                edge_type: "relates_to".to_string(),
                properties: HashMap::new(),
                weight: 1.0,
                created_at: chrono::Utc::now(),
            };
            manager.add_edge(edge).await.unwrap();
        }

        let options = TraversalOptions {
            max_depth: 2,
            ..Default::default()
        };

        let result = manager.traverse("node1", options).unwrap();
        assert_eq!(result.len(), 3); // Should only reach node1, node2, node3
    }

    #[tokio::test]
    async fn test_extract_subgraph() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();

        // Create nodes
        for i in 1..=4 {
            let node = Node {
                id: format!("node{}", i),
                node_type: "entity".to_string(),
                properties: HashMap::new(),
                region: "us-east-1".to_string(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            };
            manager.add_node(node).await.unwrap();
        }

        // Create edges
        let edges = vec![
            ("node1", "node2"),
            ("node2", "node3"),
            ("node3", "node4"),
            ("node1", "node3"),
        ];

        for (i, (source, target)) in edges.iter().enumerate() {
            let edge = Edge {
                id: format!("edge{}", i),
                source: source.to_string(),
                target: target.to_string(),
                edge_type: "relates_to".to_string(),
                properties: HashMap::new(),
                weight: 1.0,
                created_at: chrono::Utc::now(),
            };
            manager.add_edge(edge).await.unwrap();
        }

        let node_ids = vec![
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ];
        let (nodes, edges) = manager.extract_subgraph(&node_ids, true).unwrap();

        assert_eq!(nodes.len(), 3);
        assert_eq!(edges.len(), 3); // Only edges between node1, node2, node3
    }

    #[tokio::test]
    async fn test_extract_subgraph_without_edges() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();

        for i in 1..=3 {
            let node = Node {
                id: format!("node{}", i),
                node_type: "entity".to_string(),
                properties: HashMap::new(),
                region: "us-east-1".to_string(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            };
            manager.add_node(node).await.unwrap();
        }

        let node_ids = vec!["node1".to_string(), "node2".to_string()];
        let (nodes, edges) = manager.extract_subgraph(&node_ids, false).unwrap();

        assert_eq!(nodes.len(), 2);
        assert_eq!(edges.len(), 0);
    }

    #[tokio::test]
    async fn test_node_limit_per_region() {
        let config = GraphConfig {
            max_nodes_per_region: 2,
            ..Default::default()
        };
        let manager = GraphManager::new(config)?;

        for i in 1..=2 {
            let node = Node {
                id: format!("node{}", i),
                node_type: "entity".to_string(),
                properties: HashMap::new(),
                region: "us-east-1".to_string(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            };
            manager.add_node(node).await.unwrap();
        }

        let node3 = Node {
            id: "node3".to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let result = manager.add_node(node3).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_edge_limit_per_node() {
        let config = GraphConfig {
            max_edges_per_node: 2,
            ..Default::default()
        };
        let manager = GraphManager::new(config)?;

        // Create source node
        let source = Node {
            id: "source".to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };
        manager.add_node(source).await.unwrap();

        // Create target nodes and edges
        for i in 1..=3 {
            let target = Node {
                id: format!("target{}", i),
                node_type: "entity".to_string(),
                properties: HashMap::new(),
                region: "us-east-1".to_string(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            };
            manager.add_node(target).await.unwrap();

            let edge = Edge {
                id: format!("edge{}", i),
                source: "source".to_string(),
                target: format!("target{}", i),
                edge_type: "relates_to".to_string(),
                properties: HashMap::new(),
                weight: 1.0,
                created_at: chrono::Utc::now(),
            };

            if i <= 2 {
                manager.add_edge(edge).await.unwrap();
            } else {
                let result = manager.add_edge(edge).await;
                assert!(result.is_err());
            }
        }
    }

    #[tokio::test]
    async fn test_multi_region_nodes() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();

        let regions = vec!["us-east-1", "eu-west-1", "ap-southeast-1"];

        for (i, region) in regions.iter().enumerate() {
            let node = Node {
                id: format!("node{}", i),
                node_type: "entity".to_string(),
                properties: HashMap::new(),
                region: region.to_string(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            };
            manager.add_node(node.clone()).await.unwrap();

            let retrieved = manager.get_node(&node.id, region).unwrap();
            assert_eq!(retrieved.region, *region);
        }
    }

    #[tokio::test]
    async fn test_node_properties() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();

        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            serde_json::Value::String("Test Entity".to_string()),
        );
        properties.insert("age".to_string(), serde_json::Value::Number(42.into()));
        properties.insert("active".to_string(), serde_json::Value::Bool(true));

        let node = Node {
            id: "test-node".to_string(),
            node_type: "entity".to_string(),
            properties: properties.clone(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        manager.add_node(node).await.unwrap();
        let retrieved = manager.get_node("test-node", "us-east-1").unwrap();

        assert_eq!(retrieved.properties.get("name").unwrap(), "Test Entity");
        assert_eq!(retrieved.properties.get("age").unwrap(), 42);
        assert_eq!(retrieved.properties.get("active").unwrap(), true);
    }

    #[tokio::test]
    async fn test_edge_properties() {
        let manager = GraphManager::new(GraphConfig::default()).unwrap();

        let node1 = Node {
            id: "node1".to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let node2 = Node {
            id: "node2".to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        manager.add_node(node1).await.unwrap();
        manager.add_node(node2).await.unwrap();

        let mut edge_properties = HashMap::new();
        edge_properties.insert("strength".to_string(), serde_json::json!(0.8));
        edge_properties.insert(
            "type".to_string(),
            serde_json::Value::String("strong".to_string()),
        );

        let edge = Edge {
            id: "edge1".to_string(),
            source: "node1".to_string(),
            target: "node2".to_string(),
            edge_type: "relates_to".to_string(),
            properties: edge_properties,
            weight: 0.8,
            created_at: chrono::Utc::now(),
        };

        let result = manager.add_edge(edge).await;
        assert!(result.is_ok());
    }
}
