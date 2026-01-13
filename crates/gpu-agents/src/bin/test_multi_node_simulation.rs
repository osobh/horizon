//! Test multi-node simulation on single GPU
//!
//! RED phase - create failing tests

use cudarc::driver::CudaContext;
use gpu_agents::node_simulation::{MultiNodeSimulator, NodeConfig, SimulationConfig};

fn main() -> anyhow::Result<()> {
    println!("Testing Multi-Node Simulation on Single GPU (RED phase)");
    println!("====================================================");

    let ctx = CudaContext::new(0)?;

    // Test 1: Create multi-node simulator
    println!("\n1. Testing multi-node simulator creation...");
    let config = SimulationConfig {
        max_nodes: 10,
        agents_per_node: 10,
        gpu_memory_per_node: 100 * 1024 * 1024, // 100MB per node
        communication_buffer_size: 10 * 1024 * 1024, // 10MB
    };

    let mut simulator = MultiNodeSimulator::new(ctx, config)?;
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
    simulator.send_cross_node_message(0, 1, "consensus proposal")?;
    println!("✅ Sent cross-node message");

    // Test 4: Node coordination
    println!("\n4. Testing multi-node coordination...");
    let coord_result = simulator.coordinate_all_nodes()?;
    println!("✅ Coordinated {} nodes", coord_result.participating_nodes);

    // Test 5: Performance metrics
    println!("\n5. Testing performance monitoring...");
    let metrics = simulator.get_simulation_metrics()?;
    println!(
        "✅ Total nodes: {}, Total agents: {}",
        metrics.total_nodes, metrics.total_agents
    );

    Ok(())
}
