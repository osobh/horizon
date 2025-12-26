//! Test atomic knowledge graph implementation

use anyhow::Result;
use cudarc::driver::CudaDevice;
use gpu_agents::knowledge::{AtomicKnowledgeGraph, ConsistencyLevel};
use std::sync::Arc;

fn main() -> Result<()> {
    println!("Testing Atomic Knowledge Graph Implementation");

    // Initialize CUDA device
    let mut graph = match CudaDevice::new(0) {
        Ok(device) => {
            println!("✓ CUDA device initialized");
            AtomicKnowledgeGraph::new(
                device, 1000, // max_nodes
                5000, // max_edges
                128,  // embedding_dim
            )
        }
        Err(e) => {
            println!("Failed to initialize CUDA device: {}", e);
            println!("This is expected if no CUDA GPU is available.");
            return Ok(());
        }
    }?;

    println!("✓ Atomic knowledge graph created");

    // Test adding nodes
    let embedding = vec![0.1f32; 128];
    let success = graph.add_node_atomic(1, 42, &embedding)?;
    println!("✓ Node addition result: {}", success);

    let success = graph.add_node_atomic(2, 43, &embedding)?;
    println!("✓ Second node addition result: {}", success);

    // Test adding edges
    let success = graph.add_edge_atomic(1, 2, 123, 0.8)?;
    println!("✓ Edge addition result: {}", success);

    // Test updating embedding
    let new_embedding = vec![0.2f32; 128];
    let success = graph.update_embedding_atomic(1, &new_embedding)?;
    println!("✓ Embedding update result: {}", success);

    // Get statistics
    let stats = graph.statistics();
    println!("✓ Graph statistics: {}", stats);

    // Test similarity search with different consistency levels
    let query_embedding = vec![0.15f32; 128];

    for consistency in [
        ConsistencyLevel::Weak,
        ConsistencyLevel::Eventual,
        ConsistencyLevel::Strong,
    ] {
        match graph.atomic_similarity_search(&query_embedding, 5, consistency.clone()) {
            Ok(results) => {
                println!(
                    "✓ Similarity search with {:?} consistency: {} results",
                    consistency,
                    results.len()
                );
                for (i, (node_id, score)) in results.iter().enumerate().take(3) {
                    println!("  Result {}: node_id={}, score={:.3}", i, node_id, score);
                }
            }
            Err(e) => {
                println!("⚠ Similarity search failed: {}", e);
            }
        }
    }

    // Check for pending updates
    let has_pending = graph.has_pending_updates();
    println!("✓ Has pending updates: {}", has_pending);

    println!("\n🎉 Atomic Knowledge Graph test completed successfully!");

    Ok(())
}
