//! Knowledge graph storage primitives

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Graph node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: Uuid,
    pub data: Vec<u8>,
    pub node_type: String,
}

/// Graph edge representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: Uuid,
    pub from_node: Uuid,
    pub to_node: Uuid,
    pub edge_type: String,
    pub weight: f32,
}

/// Knowledge graph storage interface
pub struct KnowledgeGraphStorage {
    // Implementation details
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_node_creation() {
        let id = Uuid::new_v4();
        let data = vec![1, 2, 3, 4];
        let node_type = "entity".to_string();

        let node = GraphNode {
            id,
            data: data.clone(),
            node_type: node_type.clone(),
        };

        assert_eq!(node.id, id);
        assert_eq!(node.data, data);
        assert_eq!(node.node_type, node_type);
    }

    #[test]
    fn test_graph_node_clone() {
        let node = GraphNode {
            id: Uuid::new_v4(),
            data: vec![5, 6, 7],
            node_type: "relationship".to_string(),
        };

        let cloned = node.clone();
        assert_eq!(node.id, cloned.id);
        assert_eq!(node.data, cloned.data);
        assert_eq!(node.node_type, cloned.node_type);
    }

    #[test]
    fn test_graph_node_serialization() -> serde_json::Result<()> {
        let node = GraphNode {
            id: Uuid::new_v4(),
            data: vec![10, 20, 30],
            node_type: "concept".to_string(),
        };

        let serialized = serde_json::to_string(&node)?;
        let deserialized: GraphNode = serde_json::from_str(&serialized)?;

        assert_eq!(node.id, deserialized.id);
        assert_eq!(node.data, deserialized.data);
        assert_eq!(node.node_type, deserialized.node_type);
        Ok(())
    }

    #[test]
    fn test_graph_edge_creation() {
        let id = Uuid::new_v4();
        let from_node = Uuid::new_v4();
        let to_node = Uuid::new_v4();
        let edge_type = "connects".to_string();
        let weight = 0.75;

        let edge = GraphEdge {
            id,
            from_node,
            to_node,
            edge_type: edge_type.clone(),
            weight,
        };

        assert_eq!(edge.id, id);
        assert_eq!(edge.from_node, from_node);
        assert_eq!(edge.to_node, to_node);
        assert_eq!(edge.edge_type, edge_type);
        assert_eq!(edge.weight, weight);
    }

    #[test]
    fn test_graph_edge_clone() {
        let edge = GraphEdge {
            id: Uuid::new_v4(),
            from_node: Uuid::new_v4(),
            to_node: Uuid::new_v4(),
            edge_type: "relates_to".to_string(),
            weight: 1.0,
        };

        let cloned = edge.clone();
        assert_eq!(edge.id, cloned.id);
        assert_eq!(edge.from_node, cloned.from_node);
        assert_eq!(edge.to_node, cloned.to_node);
        assert_eq!(edge.edge_type, cloned.edge_type);
        assert_eq!(edge.weight, cloned.weight);
    }

    #[test]
    fn test_graph_edge_serialization() -> serde_json::Result<()> {
        let edge = GraphEdge {
            id: Uuid::new_v4(),
            from_node: Uuid::new_v4(),
            to_node: Uuid::new_v4(),
            edge_type: "dependency".to_string(),
            weight: 0.5,
        };

        let serialized = serde_json::to_string(&edge)?;
        let deserialized: GraphEdge = serde_json::from_str(&serialized)?;

        assert_eq!(edge.id, deserialized.id);
        assert_eq!(edge.from_node, deserialized.from_node);
        assert_eq!(edge.to_node, deserialized.to_node);
        assert_eq!(edge.edge_type, deserialized.edge_type);
        assert_eq!(edge.weight, deserialized.weight);
        Ok(())
    }

    #[test]
    fn test_graph_node_debug() {
        let node = GraphNode {
            id: Uuid::new_v4(),
            data: vec![1, 2],
            node_type: "test".to_string(),
        };

        let debug_str = format!("{:?}", node);
        assert!(debug_str.contains("GraphNode"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_graph_edge_debug() {
        let edge = GraphEdge {
            id: Uuid::new_v4(),
            from_node: Uuid::new_v4(),
            to_node: Uuid::new_v4(),
            edge_type: "test".to_string(),
            weight: 0.0,
        };

        let debug_str = format!("{:?}", edge);
        assert!(debug_str.contains("GraphEdge"));
        assert!(debug_str.contains("test"));
        assert!(debug_str.contains("0"));
    }

    #[test]
    fn test_graph_node_edge_cases() {
        // Empty data
        let node1 = GraphNode {
            id: Uuid::new_v4(),
            data: vec![],
            node_type: String::new(),
        };
        assert!(node1.data.is_empty());
        assert!(node1.node_type.is_empty());

        // Large data
        let large_data = vec![42u8; 10000];
        let node2 = GraphNode {
            id: Uuid::new_v4(),
            data: large_data.clone(),
            node_type: "large_node".to_string(),
        };
        assert_eq!(node2.data.len(), 10000);
        assert_eq!(node2.data, large_data);
    }

    #[test]
    fn test_graph_edge_weight_extremes() {
        // Zero weight
        let edge1 = GraphEdge {
            id: Uuid::new_v4(),
            from_node: Uuid::new_v4(),
            to_node: Uuid::new_v4(),
            edge_type: "zero_weight".to_string(),
            weight: 0.0,
        };
        assert_eq!(edge1.weight, 0.0);

        // Negative weight
        let edge2 = GraphEdge {
            id: Uuid::new_v4(),
            from_node: Uuid::new_v4(),
            to_node: Uuid::new_v4(),
            edge_type: "negative_weight".to_string(),
            weight: -1.5,
        };
        assert_eq!(edge2.weight, -1.5);

        // Very large weight
        let edge3 = GraphEdge {
            id: Uuid::new_v4(),
            from_node: Uuid::new_v4(),
            to_node: Uuid::new_v4(),
            edge_type: "large_weight".to_string(),
            weight: f32::MAX,
        };
        assert_eq!(edge3.weight, f32::MAX);
    }

    #[test]
    fn test_uuid_uniqueness() {
        let node1 = GraphNode {
            id: Uuid::new_v4(),
            data: vec![1],
            node_type: "type1".to_string(),
        };

        let node2 = GraphNode {
            id: Uuid::new_v4(),
            data: vec![1],
            node_type: "type1".to_string(),
        };

        // Same data and type, but different UUIDs
        assert_ne!(node1.id, node2.id);
    }

    #[test]
    fn test_self_referencing_edge() {
        let node_id = Uuid::new_v4();
        let edge = GraphEdge {
            id: Uuid::new_v4(),
            from_node: node_id,
            to_node: node_id,
            edge_type: "self_reference".to_string(),
            weight: 1.0,
        };

        assert_eq!(edge.from_node, edge.to_node);
        assert_eq!(edge.edge_type, "self_reference");
    }

    #[test]
    fn test_unicode_node_types() -> serde_json::Result<()> {
        let node = GraphNode {
            id: Uuid::new_v4(),
            data: vec![1, 2, 3],
            node_type: "概念节点".to_string(), // Chinese characters
        };

        assert_eq!(node.node_type, "概念节点");

        let serialized = serde_json::to_string(&node)?;
        let deserialized: GraphNode = serde_json::from_str(&serialized)?;
        assert_eq!(deserialized.node_type, "概念节点");
        Ok(())
    }
}
