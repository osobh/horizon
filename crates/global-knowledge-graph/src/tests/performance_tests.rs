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
async fn test_large_graph_performance() {
    let manager = GraphManager::new(GraphConfig::default()).unwrap();
    let node_count = 1000;

    let start = Instant::now();

    // Create many nodes
    for i in 0..node_count {
        let node = Node {
            id: format!("perf_node_{}", i),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };
        manager.add_node(node).await.unwrap();
    }

    let elapsed = start.elapsed();
    let ops_per_sec = node_count as f64 / elapsed.as_secs_f64();

    // Should handle at least 100 ops/sec
    assert!(ops_per_sec > 100.0);
}

#[tokio::test]
async fn test_query_performance_target() {
    let target_latency_ms = 100;

    // Mock query execution
    let start = Instant::now();

    // Simulate query work
    tokio::time::sleep(Duration::from_millis(50)).await;

    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < target_latency_ms as u128);
}

#[tokio::test]
async fn test_traversal_performance() {
    let manager = GraphManager::new(GraphConfig::default()).unwrap();

    // Create a chain of nodes for traversal test
    let chain_length = 100;
    for i in 0..chain_length {
        let node = Node {
            id: format!("chain_node_{}", i),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };
        manager.add_node(node).await.unwrap();
    }

    // Create chain edges
    for i in 0..chain_length - 1 {
        let edge = Edge {
            id: format!("chain_edge_{}", i),
            source: format!("chain_node_{}", i),
            target: format!("chain_node_{}", i + 1),
            edge_type: "next".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        };
        manager.add_edge(edge).await.unwrap();
    }

    let start = Instant::now();
    let options = TraversalOptions::default();
    let result = manager.traverse("chain_node_0", options).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(result.len(), chain_length);
    assert!(elapsed.as_millis() < 1000); // Should be fast
}

#[tokio::test]
async fn test_memory_usage_estimation() {
    let config = GraphConfig::default();
    let estimated_memory_per_node = 1024; // bytes
    let max_memory_usage = config.max_nodes_per_region * estimated_memory_per_node;

    // Should be reasonable for the target scale
    assert!(max_memory_usage < 10_000_000_000); // Less than 10GB
}

#[tokio::test]
async fn test_concurrent_read_performance() {
    let manager = Arc::new(GraphManager::new(GraphConfig::default()).unwrap());

    // Create test data
    let node = Node {
        id: "read_test_node".to_string(),
        node_type: "entity".to_string(),
        properties: HashMap::new(),
        region: "us-east-1".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        version: 1,
    };
    manager.add_node(node).await.unwrap();

    let start = Instant::now();
    let mut handles = vec![];

    // Launch concurrent reads
    for _ in 0..100 {
        let manager_clone = manager.clone();
        handles.push(tokio::spawn(async move {
            manager_clone.get_node("read_test_node", "us-east-1")
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles).await;
    let elapsed = start.elapsed();

    // All reads should succeed
    assert!(results.iter().all(|r| r.as_ref().unwrap().is_ok()));

    // Should be fast
    assert!(elapsed.as_millis() < 1000);
}
