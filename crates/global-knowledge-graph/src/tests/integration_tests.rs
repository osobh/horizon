use crate::cache_layer::*;
use crate::compliance_handler::*;
use crate::consistency_manager::*;
use crate::error::*;
use crate::graph_manager::*;
use crate::query_engine::*;
use crate::region_router::*;
use crate::replication::*;
use crate::*;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

#[tokio::test]
async fn test_full_workflow_node_lifecycle() {
    let manager = GraphManager::new(GraphConfig::default()).unwrap();

    // Create node
    let node = Node {
        id: "integration_node".to_string(),
        node_type: "entity".to_string(),
        properties: {
            let mut props = HashMap::new();
            props.insert(
                "name".to_string(),
                serde_json::Value::String("Test Entity".to_string()),
            );
            props
        },
        region: "us-east-1".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        version: 1,
    };

    // Test full lifecycle
    let node_id = manager.add_node(node.clone()).await.unwrap();
    assert_eq!(node_id, node.id);

    let retrieved = manager.get_node(&node.id, &node.region).unwrap();
    assert_eq!(retrieved.id, node.id);

    let mut updates = HashMap::new();
    updates.insert(
        "status".to_string(),
        serde_json::Value::String("active".to_string()),
    );
    let updated = manager.update_node(&node.id, updates).await.unwrap();
    assert_eq!(updated.version, 2);

    manager.delete_node(&node.id).await.unwrap();
    let get_result = manager.get_node(&node.id, &node.region);
    assert!(get_result.is_err());
}

#[tokio::test]
async fn test_multi_region_graph_operations() {
    let manager = GraphManager::new(GraphConfig::default()).unwrap();
    let regions = vec!["us-east-1", "eu-west-1", "ap-southeast-1"];

    // Create nodes in different regions
    for (i, region) in regions.iter().enumerate() {
        let node = Node {
            id: format!("multi_region_node_{}", i),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: region.to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };
        manager.add_node(node).await.unwrap();
    }

    // Verify nodes exist in their respective regions
    for (i, region) in regions.iter().enumerate() {
        let node_id = format!("multi_region_node_{}", i);
        let retrieved = manager.get_node(&node_id, region).unwrap();
        assert_eq!(retrieved.region, *region);
    }
}

#[tokio::test]
async fn test_graph_traversal_integration() {
    let manager = GraphManager::new(GraphConfig::default()).unwrap();

    // Create a test graph: A -> B -> C -> D
    let nodes = vec!["A", "B", "C", "D"];
    for node_id in &nodes {
        let node = Node {
            id: node_id.to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };
        manager.add_node(node).await.unwrap();
    }

    // Create edges A->B, B->C, C->D
    for i in 0..nodes.len() - 1 {
        let edge = Edge {
            id: format!("edge_{}", i),
            source: nodes[i].to_string(),
            target: nodes[i + 1].to_string(),
            edge_type: "connects".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        };
        manager.add_edge(edge).await.unwrap();
    }

    // Test traversal
    let options = TraversalOptions::default();
    let result = manager.traverse("A", options).unwrap();
    assert_eq!(result.len(), 4); // Should find all nodes
}

#[tokio::test]
async fn test_subgraph_extraction_integration() {
    let manager = GraphManager::new(GraphConfig::default()).unwrap();

    // Create a more complex graph
    let node_ids = vec!["X", "Y", "Z", "W"];
    for node_id in &node_ids {
        let node = Node {
            id: node_id.to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };
        manager.add_node(node).await.unwrap();
    }

    // Create edges: X->Y, Y->Z, Z->W, X->Z
    let edges = vec![("X", "Y"), ("Y", "Z"), ("Z", "W"), ("X", "Z")];

    for (i, (source, target)) in edges.iter().enumerate() {
        let edge = Edge {
            id: format!("edge_{}", i),
            source: source.to_string(),
            target: target.to_string(),
            edge_type: "relates".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        };
        manager.add_edge(edge).await.unwrap();
    }

    // Extract subgraph containing X, Y, Z
    let subgraph_nodes = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
    let (nodes, edges) = manager.extract_subgraph(&subgraph_nodes, true).unwrap();

    assert_eq!(nodes.len(), 3);
    assert_eq!(edges.len(), 3); // X->Y, Y->Z, X->Z
}

#[tokio::test]
async fn test_concurrent_operations_integration() {
    let manager = Arc::new(GraphManager::new(GraphConfig::default()).unwrap());
    let mut handles = vec![];

    // Launch concurrent operations
    for i in 0..20 {
        let manager_clone = manager.clone();
        handles.push(tokio::spawn(async move {
            let node = Node {
                id: format!("concurrent_node_{}", i),
                node_type: "entity".to_string(),
                properties: HashMap::new(),
                region: "us-east-1".to_string(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            };

            let result = manager_clone.add_node(node.clone()).await;
            if result.is_ok() {
                // Try to read the node back
                let retrieved = manager_clone.get_node(&node.id, &node.region);
                retrieved.is_ok()
            } else {
                false
            }
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles).await;
    let success_count = results
        .iter()
        .filter(|r| r.as_ref().unwrap_or(&false))
        .count();

    // Most operations should succeed
    assert!(success_count >= 15);
}

#[tokio::test]
async fn test_error_handling_integration() {
    let manager = GraphManager::new(GraphConfig::default()).unwrap();

    // Test invalid region
    let invalid_node = Node {
        id: "invalid_node".to_string(),
        node_type: "entity".to_string(),
        properties: HashMap::new(),
        region: "invalid-region".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        version: 1,
    };

    let result = manager.add_node(invalid_node).await;
    assert!(result.is_err());

    // Test non-existent node access
    let result = manager.get_node("non_existent", "us-east-1");
    assert!(result.is_err());

    // Test invalid edge (non-existent nodes)
    let invalid_edge = Edge {
        id: "invalid_edge".to_string(),
        source: "non_existent_source".to_string(),
        target: "non_existent_target".to_string(),
        edge_type: "relates".to_string(),
        properties: HashMap::new(),
        weight: 1.0,
        created_at: chrono::Utc::now(),
    };

    let result = manager.add_edge(invalid_edge).await;
    assert!(result.is_err());
}
