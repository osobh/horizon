//! Tests for temporal knowledge graph functionality
//!
//! Tests time-based graph operations including:
//! - Temporal node and edge creation
//! - Time-based queries and filtering
//! - Temporal graph evolution
//! - Time window operations

use super::temporal::*;
use crate::knowledge::{KnowledgeEdge, KnowledgeNode};
use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

// Helper to create test device
fn create_test_device() -> Result<Arc<CudaDevice>> {
    Ok(CudaDevice::new(0)?)
}

// Helper to create test temporal node
fn create_temporal_node(id: u32, timestamp: i64) -> TemporalNode {
    TemporalNode {
        base_node: KnowledgeNode {
            id,
            content: format!("Node {}", id),
            node_type: "test".to_string(),
            embedding: vec![0.1 * id as f32; 128],
        },
        timestamp,
        valid_from: timestamp,
        valid_to: None,
        confidence: 0.9,
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[test]
fn test_temporal_node_creation() {
    let timestamp = 1000;
    let node = create_temporal_node(1, timestamp);

    assert_eq!(node.base_node.id, 1);
    assert_eq!(node.timestamp, timestamp);
    assert_eq!(node.valid_from, timestamp);
    assert!(node.valid_to.is_none());
    assert_eq!(node.confidence, 0.9);
}

#[test]
fn test_temporal_edge_creation() {
    let edge = TemporalEdge {
        base_edge: KnowledgeEdge {
            source_id: 1,
            target_id: 2,
            edge_type: "temporal_link".to_string(),
            weight: 0.8,
        },
        timestamp: 1500,
        duration: Some(100),
        causality_score: 0.7,
    };

    assert_eq!(edge.base_edge.source_id, 1);
    assert_eq!(edge.base_edge.target_id, 2);
    assert_eq!(edge.timestamp, 1500);
    assert_eq!(edge.duration, Some(100));
    assert_eq!(edge.causality_score, 0.7);
}

#[test]
fn test_temporal_graph_creation() -> Result<()> {
    let device = create_test_device()?;
    let graph = TemporalKnowledgeGraph::new(device, 1000)?;

    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(graph.get_time_range(), (i64::MAX, i64::MIN));

    Ok(())
}

#[test]
fn test_add_temporal_nodes() -> Result<()> {
    let device = create_test_device()?;
    let mut graph = TemporalKnowledgeGraph::new(device, 1000)?;

    // Add nodes at different timestamps
    let node1 = create_temporal_node(1, 1000);
    let node2 = create_temporal_node(2, 2000);
    let node3 = create_temporal_node(3, 1500);

    graph.add_temporal_node(node1)?;
    graph.add_temporal_node(node2)?;
    graph.add_temporal_node(node3)?;

    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.get_time_range(), (1000, 2000));

    Ok(())
}

#[test]
fn test_add_temporal_edges() -> Result<()> {
    let device = create_test_device()?;
    let mut graph = TemporalKnowledgeGraph::new(device, 1000)?;

    // Add nodes first
    graph.add_temporal_node(create_temporal_node(1, 1000))?;
    graph.add_temporal_node(create_temporal_node(2, 1100))?;

    // Add temporal edge
    let edge = TemporalEdge {
        base_edge: KnowledgeEdge {
            source_id: 1,
            target_id: 2,
            edge_type: "causes".to_string(),
            weight: 0.9,
        },
        timestamp: 1050,
        duration: Some(50),
        causality_score: 0.85,
    };

    graph.add_temporal_edge(edge)?;

    assert_eq!(graph.edge_count(), 1);

    Ok(())
}

#[test]
fn test_time_window_query() -> Result<()> {
    let device = create_test_device()?;
    let mut graph = TemporalKnowledgeGraph::new(device, 1000)?;

    // Add nodes across time range
    for i in 0..10 {
        let node = create_temporal_node(i, 1000 + i as i64 * 100);
        graph.add_temporal_node(node)?;
    }

    // Query specific time window
    let window_query = TimeWindowQuery {
        start_time: 1200,
        end_time: 1600,
        include_edges: true,
    };

    let results = graph.query_time_window(window_query)?;

    // Should return nodes with timestamps 1200, 1300, 1400, 1500, 1600
    assert_eq!(results.nodes.len(), 5);
    assert!(results
        .nodes
        .iter()
        .all(|n| n.timestamp >= 1200 && n.timestamp <= 1600));

    Ok(())
}

#[test]
fn test_temporal_path_query() -> Result<()> {
    let device = create_test_device()?;
    let mut graph = TemporalKnowledgeGraph::new(device, 1000)?;

    // Create temporal chain: 1 -> 2 -> 3 -> 4
    for i in 1..=4 {
        graph.add_temporal_node(create_temporal_node(i, 1000 + i as i64 * 100))?;
    }

    // Add edges with causality
    for i in 1..4 {
        let edge = TemporalEdge {
            base_edge: KnowledgeEdge {
                source_id: i,
                target_id: i + 1,
                edge_type: "causes".to_string(),
                weight: 0.8,
            },
            timestamp: 1000 + i as i64 * 100 + 50,
            duration: Some(50),
            causality_score: 0.9,
        };
        graph.add_temporal_edge(edge)?;
    }

    // Query temporal path
    let path_query = TemporalPathQuery {
        source_id: 1,
        target_id: 4,
        max_time_delta: 500,
        min_causality_score: 0.7,
    };

    let paths = graph.find_temporal_paths(path_query)?;

    assert!(!paths.is_empty());
    let path = &paths[0];
    assert_eq!(path.nodes.len(), 4);
    assert_eq!(path.nodes[0], 1);
    assert_eq!(path.nodes[3], 4);
    assert!(path.total_causality_score > 0.0);

    Ok(())
}

#[test]
fn test_temporal_aggregation() -> Result<()> {
    let device = create_test_device()?;
    let mut graph = TemporalKnowledgeGraph::new(device, 1000)?;

    // Add nodes with different timestamps
    for i in 0..100 {
        let timestamp = 1000 + (i % 10) as i64 * 1000; // 10 time buckets
        graph.add_temporal_node(create_temporal_node(i, timestamp))?;
    }

    // Aggregate by time bucket (1000ms intervals)
    let aggregation = TemporalAggregation {
        bucket_size_ms: 1000,
        aggregation_type: AggregationType::Count,
    };

    let results = graph.aggregate_temporal_data(aggregation)?;

    assert_eq!(results.buckets.len(), 10);
    assert!(results.buckets.iter().all(|&count| count == 10.0));

    Ok(())
}

#[test]
fn test_temporal_evolution() -> Result<()> {
    let device = create_test_device()?;
    let mut graph = TemporalKnowledgeGraph::new(device, 1000)?;

    // Add evolving node (same ID, different timestamps)
    let node_id = 1;
    for i in 0..5 {
        let mut node = create_temporal_node(node_id, 1000 + i * 100);
        node.confidence = 0.5 + i as f32 * 0.1; // Increasing confidence
        graph.add_temporal_node(node)?;
    }

    // Query evolution of node
    let evolution = graph.get_node_evolution(node_id)?;

    assert_eq!(evolution.versions.len(), 5);
    // Check confidence increases over time
    for i in 1..evolution.versions.len() {
        assert!(evolution.versions[i].confidence > evolution.versions[i - 1].confidence);
    }

    Ok(())
}

#[test]
fn test_temporal_consistency() -> Result<()> {
    let device = create_test_device()?;
    let mut graph = TemporalKnowledgeGraph::new(device, 1000)?;

    // Add node with validity period
    let mut node = create_temporal_node(1, 1000);
    node.valid_from = 1000;
    node.valid_to = Some(2000);
    graph.add_temporal_node(node)?;

    // Check node validity at different times
    assert!(graph.is_node_valid_at(1, 1500)?);
    assert!(!graph.is_node_valid_at(1, 2500)?);
    assert!(!graph.is_node_valid_at(1, 500)?);

    Ok(())
}

#[test]
fn test_temporal_graph_snapshot() -> Result<()> {
    let device = create_test_device()?;
    let mut graph = TemporalKnowledgeGraph::new(device.clone(), 1000)?;

    // Build temporal graph
    for i in 0..10 {
        let node = create_temporal_node(i, 1000 + i as i64 * 200);
        graph.add_temporal_node(node)?;
    }

    // Take snapshot at specific time
    let snapshot_time = 1500;
    let snapshot = graph.snapshot_at_time(snapshot_time)?;

    // Snapshot should only include nodes valid at that time
    assert!(snapshot.node_count() <= graph.node_count());

    Ok(())
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_temporal_knowledge_propagation() -> Result<()> {
    let device = create_test_device()?;
    let mut graph = TemporalKnowledgeGraph::new(device, 1000)?;

    // Create knowledge propagation scenario
    // Event A causes B, B causes C, with time delays

    let node_a = create_temporal_node(1, 1000);
    let node_b = create_temporal_node(2, 1100);
    let node_c = create_temporal_node(3, 1250);

    graph.add_temporal_node(node_a)?;
    graph.add_temporal_node(node_b)?;
    graph.add_temporal_node(node_c)?;

    // A -> B edge
    graph.add_temporal_edge(TemporalEdge {
        base_edge: KnowledgeEdge {
            source_id: 1,
            target_id: 2,
            edge_type: "causes".to_string(),
            weight: 0.9,
        },
        timestamp: 1050,
        duration: Some(50),
        causality_score: 0.85,
    })?;

    // B -> C edge
    graph.add_temporal_edge(TemporalEdge {
        base_edge: KnowledgeEdge {
            source_id: 2,
            target_id: 3,
            edge_type: "causes".to_string(),
            weight: 0.8,
        },
        timestamp: 1175,
        duration: Some(75),
        causality_score: 0.75,
    })?;

    // Analyze causality chain
    let causality = graph.analyze_causality_chain(1, 3)?;

    assert!(causality.is_causal);
    assert_eq!(causality.chain_length, 2);
    assert!(causality.total_delay > 0);
    assert!(causality.confidence > 0.6); // Combined causality

    Ok(())
}

#[test]
fn test_temporal_anomaly_detection() -> Result<()> {
    let device = create_test_device()?;
    let mut graph = TemporalKnowledgeGraph::new(device, 1000)?;

    // Add regular pattern
    for i in 0..20 {
        let node = create_temporal_node(i, 1000 + i as i64 * 100);
        graph.add_temporal_node(node)?;
    }

    // Add anomaly (out of pattern timing)
    let anomaly = create_temporal_node(100, 1550); // Between regular intervals
    graph.add_temporal_node(anomaly)?;

    // Detect temporal anomalies
    let anomalies = graph.detect_temporal_anomalies()?;

    assert!(!anomalies.is_empty());
    assert!(anomalies.iter().any(|a| a.node_id == 100));

    Ok(())
}

// =============================================================================
// Performance Tests
// =============================================================================

#[test]
#[ignore] // Run with --ignored for performance testing
fn test_temporal_graph_scaling() -> Result<()> {
    let device = create_test_device()?;
    let mut graph = TemporalKnowledgeGraph::new(device, 100_000)?;

    let start = std::time::Instant::now();

    // Add many temporal nodes
    for i in 0..10_000 {
        let timestamp = 1000 + (i % 1000) as i64;
        let node = create_temporal_node(i, timestamp);
        graph.add_temporal_node(node)?;
    }

    let insert_time = start.elapsed();

    // Query performance
    let query_start = std::time::Instant::now();
    let window_query = TimeWindowQuery {
        start_time: 1200,
        end_time: 1400,
        include_edges: false,
    };

    let results = graph.query_time_window(window_query)?;
    let query_time = query_start.elapsed();

    println!("Temporal graph performance:");
    println!("  Insert 10K nodes: {:?}", insert_time);
    println!("  Time window query: {:?}", query_time);
    println!("  Results found: {}", results.nodes.len());

    assert!(insert_time.as_secs() < 5);
    assert!(query_time.as_millis() < 100);

    Ok(())
}

// =============================================================================
// GPU Kernel Tests
// =============================================================================

#[test]
fn test_temporal_gpu_kernels() -> Result<()> {
    let device = create_test_device()?;
    let mut graph = TemporalKnowledgeGraph::new(device.clone(), 1000)?;

    // Add test data
    for i in 0..100 {
        let node = create_temporal_node(i, 1000 + i as i64 * 10);
        graph.add_temporal_node(node)?;
    }

    // Upload to GPU
    graph.sync_to_gpu()?;

    // Test GPU time window kernel
    let gpu_results = graph.gpu_time_window_query(1200, 1400)?;

    assert!(!gpu_results.is_empty());

    // Test GPU temporal aggregation kernel
    let gpu_aggregation = graph.gpu_temporal_aggregate(100)?; // 100ms buckets

    assert!(!gpu_aggregation.is_empty());

    Ok(())
}
