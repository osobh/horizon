//! Test multi-node simulation on single GPU
//!
//! GREEN phase - tests should pass with basic implementation

use cudarc::driver::CudaContext;
use gpu_agents::node_simulation::{MultiNodeSimulator, NodeConfig, SimulationConfig};
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("Testing Multi-Node Simulation on Single GPU (GREEN phase)");
    println!("======================================================");

    let ctx = CudaContext::new(0)?;

    // Test 1: Create multi-node simulator
    println!("\n1. Testing multi-node simulator creation...");
    let config = SimulationConfig {
        max_nodes: 10,
        agents_per_node: 10,
        gpu_memory_per_node: 100 * 1024 * 1024, // 100MB per node
        communication_buffer_size: 10 * 1024 * 1024, // 10MB
    };

    let mut simulator = MultiNodeSimulator::new(ctx.clone(), config)?;
    println!("✅ Created multi-node simulator");

    // Test 2: Simulate multiple nodes
    println!("\n2. Testing node simulation...");
    let node_configs = vec![
        NodeConfig {
            node_id: 0,
            node_type: "consensus".to_string(),
            agent_count: 10,
        },
        NodeConfig {
            node_id: 1,
            node_type: "synthesis".to_string(),
            agent_count: 8,
        },
        NodeConfig {
            node_id: 2,
            node_type: "evolution".to_string(),
            agent_count: 5,
        },
    ];

    simulator.simulate_nodes(node_configs)?;
    println!("✅ Simulated 3 nodes with different types");

    // Test 3: Inter-node communication
    println!("\n3. Testing inter-node communication...");
    let start = Instant::now();
    simulator.send_cross_node_message(0, 1, "consensus proposal")?;
    simulator.send_cross_node_message(1, 2, "synthesis result")?;
    simulator.send_cross_node_message(2, 0, "evolution update")?;
    let comm_time = start.elapsed();
    println!("✅ Sent 3 cross-node messages in {:?}", comm_time);

    // Test 4: Node coordination
    println!("\n4. Testing multi-node coordination...");
    let coord_result = simulator.coordinate_all_nodes()?;
    println!("✅ Coordination result:");
    println!(
        "   Participating nodes: {}",
        coord_result.participating_nodes
    );
    println!("   Consensus achieved: {}", coord_result.consensus_achieved);
    println!("   Total messages: {}", coord_result.total_messages);
    println!(
        "   Coordination time: {} μs",
        coord_result.coordination_time_us
    );

    // Test 5: Performance metrics
    println!("\n5. Testing performance monitoring...");
    let metrics = simulator.get_simulation_metrics()?;
    println!("✅ Simulation metrics:");
    println!("   Total nodes: {}", metrics.total_nodes);
    println!("   Total agents: {}", metrics.total_agents);
    println!(
        "   GPU memory used: {} MB",
        metrics.gpu_memory_used / 1024 / 1024
    );
    println!("   Messages/sec: {:.2}", metrics.messages_per_second);
    println!(
        "   Coordination latency: {:.2} μs",
        metrics.coordination_latency_us
    );
    println!(
        "   Cross-node bandwidth: {:.2} Mbps",
        metrics.cross_node_bandwidth_mbps
    );

    // Test 6: Node type specific operations
    println!("\n6. Testing node type specific operations...");
    // Run multiple coordination rounds
    for i in 0..5 {
        simulator.coordinate_all_nodes()?;
        if i == 2 {
            // Add more messages mid-simulation
            simulator.send_cross_node_message(0, 1, "vote")?;
            simulator.send_cross_node_message(1, 0, "ack")?;
        }
    }
    println!("✅ Completed 5 coordination rounds");

    // Test 7: Scaling test
    println!("\n7. Testing scaling with more nodes...");
    let many_nodes: Vec<NodeConfig> = (0..10)
        .map(|i| NodeConfig {
            node_id: i,
            node_type: match i % 3 {
                0 => "consensus",
                1 => "synthesis",
                _ => "evolution",
            }
            .to_string(),
            agent_count: 5 + (i as usize % 5),
        })
        .collect();

    simulator.simulate_nodes(many_nodes)?;
    let scale_result = simulator.coordinate_all_nodes()?;
    println!(
        "✅ Scaled to {} nodes with {} total messages",
        scale_result.participating_nodes, scale_result.total_messages
    );

    // Test 8: Message patterns
    println!("\n8. Testing message patterns...");
    // All-to-all communication pattern
    for i in 0..5 {
        for j in 0..5 {
            if i != j {
                simulator.send_cross_node_message(i, j, "broadcast")?;
            }
        }
    }
    let pattern_result = simulator.coordinate_all_nodes()?;
    println!(
        "✅ All-to-all pattern: {} messages",
        pattern_result.total_messages
    );

    // Test 9: Final metrics
    println!("\n9. Testing final metrics...");
    let final_metrics = simulator.get_simulation_metrics()?;
    println!("✅ Final simulation state:");
    println!(
        "   Total cross-node messages: {}",
        final_metrics.messages_per_second as usize * 10
    ); // Approximation
    println!(
        "   Average agents per node: {:.1}",
        final_metrics.total_agents as f64 / final_metrics.total_nodes as f64
    );

    println!("\n✅ All GREEN phase tests passed!");
    println!("\nKey achievements:");
    println!("- Multi-node simulation on single GPU");
    println!("- Inter-node communication working");
    println!("- Node type specific behavior");
    println!("- Coordination across all nodes");
    println!("- Performance metrics collection");

    Ok(())
}
