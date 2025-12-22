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
async fn test_graph_config_serialization() {
    let config = GraphConfig::default();
    let serialized = serde_json::to_string(&config).unwrap();
    let deserialized: GraphConfig = serde_json::from_str(&serialized).unwrap();
    assert_eq!(
        config.max_nodes_per_region,
        deserialized.max_nodes_per_region
    );
}

#[tokio::test]
async fn test_node_version_increment() {
    let manager = GraphManager::new(GraphConfig::default()).unwrap();
    let mut node = Node {
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
        "test".to_string(),
        serde_json::Value::String("value".to_string()),
    );

    let updated = manager.update_node(&node.id, updates).await.unwrap();
    assert_eq!(updated.version, 2);
}

#[tokio::test]
async fn test_concurrent_node_operations() {
    let manager = Arc::new(GraphManager::new(GraphConfig::default()).unwrap());
    let mut handles = vec![];

    // Test concurrent node creation
    for i in 0..10 {
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
            manager_clone.add_node(node).await
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles).await;
    assert_eq!(results.len(), 10);
    assert!(results.iter().all(|r| r.is_ok()));
}

#[tokio::test]
async fn test_graph_statistics() {
    let manager = GraphManager::new(GraphConfig::default()).unwrap();

    // Add test data
    for i in 0..5 {
        let node = Node {
            id: format!("stats_node_{}", i),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };
        manager.add_node(node).await.unwrap();
    }

    // Test statistics collection would go here in a real implementation
    // For now, we'll verify basic counting works
    let region_graphs = manager.graphs.get("us-east-1").unwrap();
    let graph = region_graphs.read();
    assert_eq!(graph.node_count(), 5);
}
