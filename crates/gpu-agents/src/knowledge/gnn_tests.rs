//! Tests for Graph Neural Network functionality
//!
//! Tests GNN operations including:
//! - Node embeddings and message passing
//! - Graph convolution operations
//! - Attention mechanisms
//! - Training and inference

use super::gnn::*;
use crate::knowledge::{KnowledgeEdge, KnowledgeNode};
use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

// Helper to create test device
fn create_test_device() -> Result<Arc<CudaDevice>> {
    Ok(CudaDevice::new(0)?)
}

// Helper to create test graph data
fn create_test_graph() -> (Vec<KnowledgeNode>, Vec<KnowledgeEdge>) {
    let nodes = vec![
        KnowledgeNode {
            id: 0,
            content: "Node A".to_string(),
            node_type: "entity".to_string(),
            embedding: vec![0.1; 64],
        },
        KnowledgeNode {
            id: 1,
            content: "Node B".to_string(),
            node_type: "entity".to_string(),
            embedding: vec![0.2; 64],
        },
        KnowledgeNode {
            id: 2,
            content: "Node C".to_string(),
            node_type: "property".to_string(),
            embedding: vec![0.3; 64],
        },
        KnowledgeNode {
            id: 3,
            content: "Node D".to_string(),
            node_type: "entity".to_string(),
            embedding: vec![0.4; 64],
        },
    ];

    let edges = vec![
        KnowledgeEdge {
            source_id: 0,
            target_id: 1,
            edge_type: "connected".to_string(),
            weight: 0.8,
        },
        KnowledgeEdge {
            source_id: 1,
            target_id: 2,
            edge_type: "has_property".to_string(),
            weight: 0.9,
        },
        KnowledgeEdge {
            source_id: 0,
            target_id: 3,
            edge_type: "related".to_string(),
            weight: 0.7,
        },
        KnowledgeEdge {
            source_id: 3,
            target_id: 2,
            edge_type: "has_property".to_string(),
            weight: 0.6,
        },
    ];

    (nodes, edges)
}

// =============================================================================
// Unit Tests
// =============================================================================

#[test]
fn test_gnn_config_creation() {
    let config = GnnConfig {
        input_dim: 64,
        hidden_dim: 128,
        output_dim: 32,
        num_layers: 3,
        dropout: 0.1,
        activation: ActivationFunction::ReLU,
        aggregation: AggregationFunction::Mean,
    };

    assert_eq!(config.input_dim, 64);
    assert_eq!(config.hidden_dim, 128);
    assert_eq!(config.output_dim, 32);
    assert_eq!(config.num_layers, 3);
}

#[test]
fn test_graph_neural_network_creation() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig::default();

    let gnn = GraphNeuralNetwork::new(device, config)?;

    assert_eq!(gnn.layer_count(), 2); // Default layers
    assert!(!gnn.is_trained());

    Ok(())
}

#[test]
fn test_load_graph_data() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig {
        input_dim: 64,
        ..Default::default()
    };

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    let (nodes, edges) = create_test_graph();

    gnn.load_graph(&nodes, &edges)?;

    assert_eq!(gnn.node_count(), 4);
    assert_eq!(gnn.edge_count(), 4);

    Ok(())
}

#[test]
fn test_forward_pass() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig {
        input_dim: 64,
        hidden_dim: 32,
        output_dim: 16,
        num_layers: 2,
        ..Default::default()
    };

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    let (nodes, edges) = create_test_graph();

    gnn.load_graph(&nodes, &edges)?;

    // Perform forward pass
    let embeddings = gnn.forward()?;

    assert_eq!(embeddings.len(), 4); // One per node
    assert_eq!(embeddings[0].len(), 16); // Output dimension

    // Embeddings should be different for different nodes
    assert_ne!(embeddings[0], embeddings[1]);

    Ok(())
}

#[test]
fn test_message_passing() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig::default();

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    let (nodes, edges) = create_test_graph();

    gnn.load_graph(&nodes, &edges)?;

    // Test message passing layer
    let messages = gnn.compute_messages(0)?; // Layer 0

    // Each node should receive messages from neighbors
    assert_eq!(messages.len(), 4);

    // Node 1 has 2 neighbors (0 and 2)
    assert!(messages[1].len() > 0);

    Ok(())
}

#[test]
fn test_graph_attention() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig {
        input_dim: 64,
        hidden_dim: 32,
        output_dim: 16,
        num_layers: 2,
        activation: ActivationFunction::ReLU,
        aggregation: AggregationFunction::Attention,
    };

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    let (nodes, edges) = create_test_graph();

    gnn.load_graph(&nodes, &edges)?;

    // Compute attention weights
    let attention_weights = gnn.compute_attention_weights()?;

    // Attention weights should sum to 1 for each node
    for node_weights in &attention_weights {
        let sum: f32 = node_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    Ok(())
}

#[test]
fn test_node_classification() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig {
        input_dim: 64,
        hidden_dim: 32,
        output_dim: 3, // 3 classes
        num_layers: 2,
        ..Default::default()
    };

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    let (nodes, edges) = create_test_graph();

    gnn.load_graph(&nodes, &edges)?;

    // Perform node classification
    let predictions = gnn.classify_nodes()?;

    assert_eq!(predictions.len(), 4);

    // Each prediction should be a valid class
    for &pred in &predictions {
        assert!(pred < 3);
    }

    Ok(())
}

#[test]
fn test_link_prediction() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig::default();

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    let (nodes, edges) = create_test_graph();

    gnn.load_graph(&nodes, &edges)?;

    // Predict link between nodes
    let score = gnn.predict_link(1, 3)?; // Not directly connected

    assert!(score >= 0.0 && score <= 1.0);

    // Existing links should have higher scores
    let existing_score = gnn.predict_link(0, 1)?;
    assert!(existing_score > 0.5);

    Ok(())
}

#[test]
fn test_graph_embedding() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig {
        input_dim: 64,
        output_dim: 32,
        ..Default::default()
    };

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    let (nodes, edges) = create_test_graph();

    gnn.load_graph(&nodes, &edges)?;

    // Get graph-level embedding
    let graph_embedding = gnn.get_graph_embedding()?;

    assert_eq!(graph_embedding.len(), 32);

    // Embedding should be non-zero
    let norm: f32 = graph_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm > 0.1);

    Ok(())
}

#[test]
fn test_subgraph_extraction() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig::default();

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    let (nodes, edges) = create_test_graph();

    gnn.load_graph(&nodes, &edges)?;

    // Extract k-hop subgraph around node 1
    let subgraph = gnn.extract_subgraph(1, 2)?;

    // Subgraph should include nodes 0, 1, 2, 3 (all reachable in 2 hops)
    assert_eq!(subgraph.node_ids.len(), 4);
    assert!(subgraph.node_ids.contains(&1));

    Ok(())
}

// =============================================================================
// Training Tests
// =============================================================================

#[test]
fn test_training_step() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig {
        input_dim: 64,
        output_dim: 3,
        ..Default::default()
    };

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    let (nodes, edges) = create_test_graph();

    gnn.load_graph(&nodes, &edges)?;

    // Create training labels
    let labels = vec![0, 1, 2, 0]; // Class labels for each node

    // Perform training step
    let loss = gnn.train_step(&labels, 0.001)?; // Learning rate 0.001

    assert!(loss > 0.0);
    assert!(loss < 10.0); // Reasonable loss value

    Ok(())
}

#[test]
fn test_batch_training() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig {
        input_dim: 64,
        output_dim: 2,
        ..Default::default()
    };

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    let (nodes, edges) = create_test_graph();

    gnn.load_graph(&nodes, &edges)?;

    // Train for multiple epochs
    let labels = vec![0, 1, 1, 0];
    let mut losses = Vec::new();

    for _ in 0..10 {
        let loss = gnn.train_step(&labels, 0.01)?;
        losses.push(loss);
    }

    // Loss should generally decrease
    let avg_early = losses[..3].iter().sum::<f32>() / 3.0;
    let avg_late = losses[7..].iter().sum::<f32>() / 3.0;

    assert!(avg_late <= avg_early * 1.1); // Allow some fluctuation

    Ok(())
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_end_to_end_node_classification() -> Result<()> {
    let device = create_test_device()?;

    // Create a larger test graph
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Create 20 nodes
    for i in 0..20 {
        nodes.push(KnowledgeNode {
            id: i,
            content: format!("Node {}", i),
            node_type: if i % 2 == 0 { "type_a" } else { "type_b" },
            embedding: vec![i as f32 * 0.05; 64],
        });
    }

    // Create edges (ring + some cross connections)
    for i in 0..20 {
        edges.push(KnowledgeEdge {
            source_id: i,
            target_id: (i + 1) % 20,
            edge_type: "next".to_string(),
            weight: 0.9,
        });

        if i % 5 == 0 {
            edges.push(KnowledgeEdge {
                source_id: i,
                target_id: (i + 10) % 20,
                edge_type: "cross".to_string(),
                weight: 0.5,
            });
        }
    }

    // Configure GNN
    let config = GnnConfig {
        input_dim: 64,
        hidden_dim: 32,
        output_dim: 2, // Binary classification
        num_layers: 3,
        dropout: 0.1,
        activation: ActivationFunction::ReLU,
        aggregation: AggregationFunction::Mean,
    };

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    gnn.load_graph(&nodes, &edges)?;

    // Create labels based on node type
    let labels: Vec<u32> = nodes
        .iter()
        .map(|n| if n.node_type == "type_a" { 0 } else { 1 })
        .collect();

    // Train
    for epoch in 0..20 {
        let loss = gnn.train_step(&labels, 0.01)?;
        if epoch % 5 == 0 {
            println!("Epoch {}: loss = {:.4}", epoch, loss);
        }
    }

    // Evaluate
    let predictions = gnn.classify_nodes()?;
    let accuracy = predictions
        .iter()
        .zip(&labels)
        .filter(|(pred, label)| pred == label)
        .count() as f32
        / labels.len() as f32;

    println!("Classification accuracy: {:.2}%", accuracy * 100.0);
    assert!(accuracy > 0.7); // Should achieve reasonable accuracy

    Ok(())
}

// =============================================================================
// Performance Tests
// =============================================================================

#[test]
#[ignore] // Run with --ignored for performance testing
fn test_gnn_scaling() -> Result<()> {
    let device = create_test_device()?;

    // Test with different graph sizes
    let sizes = vec![100, 500, 1000, 5000];

    for size in sizes {
        // Create random graph
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for i in 0..size {
            nodes.push(KnowledgeNode {
                id: i as u32,
                content: format!("Node {}", i),
                node_type: "test".to_string(),
                embedding: vec![0.1; 64],
            });
        }

        // Random edges (average degree ~10)
        for i in 0..size {
            for _ in 0..10 {
                let target = rand::random::<u32>() % size as u32;
                if target != i as u32 {
                    edges.push(KnowledgeEdge {
                        source_id: i as u32,
                        target_id: target,
                        edge_type: "edge".to_string(),
                        weight: rand::random::<f32>(),
                    });
                }
            }
        }

        let config = GnnConfig {
            input_dim: 64,
            hidden_dim: 128,
            output_dim: 32,
            num_layers: 3,
            ..Default::default()
        };

        let mut gnn = GraphNeuralNetwork::new(device.clone(), config)?;

        let load_start = std::time::Instant::now();
        gnn.load_graph(&nodes, &edges)?;
        let load_time = load_start.elapsed();

        let forward_start = std::time::Instant::now();
        let _ = gnn.forward()?;
        let forward_time = forward_start.elapsed();

        println!(
            "Graph size {}: load={:?}, forward={:?}",
            size, load_time, forward_time
        );
    }

    Ok(())
}

// =============================================================================
// GPU Kernel Tests
// =============================================================================

#[test]
fn test_gpu_message_passing_kernel() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig::default();

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    let (nodes, edges) = create_test_graph();

    gnn.load_graph(&nodes, &edges)?;

    // Upload to GPU
    gnn.sync_to_gpu()?;

    // Test GPU message passing
    let gpu_messages = gnn.gpu_message_passing(0)?;

    assert!(!gpu_messages.is_empty());

    Ok(())
}

#[test]
fn test_gpu_aggregation_kernel() -> Result<()> {
    let device = create_test_device()?;
    let config = GnnConfig {
        aggregation: AggregationFunction::Max,
        ..Default::default()
    };

    let mut gnn = GraphNeuralNetwork::new(device, config)?;
    let (nodes, edges) = create_test_graph();

    gnn.load_graph(&nodes, &edges)?;
    gnn.sync_to_gpu()?;

    // Test GPU aggregation
    let aggregated = gnn.gpu_aggregate_messages()?;

    assert_eq!(aggregated.len(), 4); // One per node

    Ok(())
}

// Helper for random numbers (simplified)
mod rand {
    pub fn random<T>() -> T
    where
        T: Default,
    {
        T::default()
    }
}
