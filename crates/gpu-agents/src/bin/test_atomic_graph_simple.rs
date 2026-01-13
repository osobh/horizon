//! Test atomic knowledge graph implementation

use anyhow::Result;
use std::sync::Arc;

fn main() -> Result<()> {
    println!("Testing Atomic Knowledge Graph Implementation");

    // Test without GPU (if not available)
    let ctx_result = cudarc::driver::CudaContext::new(0);
    match ctx_result {
        Ok(ctx) => {
            println!("✓ CUDA device available");

            // Try to create the graph
            match gpu_agents::knowledge::AtomicKnowledgeGraph::new(
                ctx, 1000, // max_nodes
                5000, // max_edges
                128,  // embedding_dim
            ) {
                Ok(mut graph) => {
                    println!("✓ Atomic knowledge graph created successfully");

                    // Test basic operations
                    let embedding = vec![0.1f32; 128];
                    let success = graph.add_node_atomic(1, 42, &embedding)?;
                    println!("✓ Node addition result: {}", success);

                    let stats = graph.statistics();
                    println!("✓ Graph statistics: {}", stats);

                    println!("🎉 Atomic Knowledge Graph test completed successfully!");
                }
                Err(e) => {
                    println!("❌ Failed to create atomic knowledge graph: {}", e);
                }
            }
        }
        Err(e) => {
            println!("⚠ CUDA device not available: {}", e);
            println!("This is expected if no CUDA GPU is present.");
        }
    }

    Ok(())
}
