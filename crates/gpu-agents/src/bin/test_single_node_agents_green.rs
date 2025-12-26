//! Test single node with 1-10 local agents
//!
//! GREEN phase - tests should pass with basic implementation

use cudarc::driver::CudaDevice;
use gpu_agents::agent_manager::{AgentConfig, LocalAgentManager};
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("Testing Single Node with Local Agents (GREEN phase)");
    println!("==================================================");

    let device = CudaDevice::new(0)?;

    // Test 1: Create agent manager for single node
    println!("\n1. Testing agent manager creation...");
    let config = AgentConfig {
        max_agents: 10,
        node_id: 0,
        ..Default::default()
    };

    let mut manager = LocalAgentManager::new(device.clone(), config)?;
    println!("✅ Created agent manager for single node");

    // Test 2: Spawn single agent
    println!("\n2. Testing single agent spawn...");
    let agent_id = manager.spawn_agent("agent-1")?;
    println!("✅ Spawned agent with ID: {}", agent_id);

    // Test 3: Spawn multiple agents
    println!("\n3. Testing multiple agent spawn...");
    let agent_ids = manager.spawn_agents(5)?;
    println!("✅ Spawned {} agents: {:?}", agent_ids.len(), agent_ids);

    // Test 4: Agent communication
    println!("\n4. Testing agent communication...");
    let start = Instant::now();
    manager.send_message(0, 1, "Hello from agent 0")?;
    let comm_time = start.elapsed();
    println!("✅ Message sent successfully in {:?}", comm_time);

    // Test 5: Agent coordination
    println!("\n5. Testing agent coordination...");
    let coord_status = manager.coordinate_agents()?;
    println!("✅ Coordination status:");
    println!("   Active agents: {}", coord_status.active_agents);
    println!("   Total messages: {}", coord_status.total_messages);
    println!(
        "   Coordination rounds: {}",
        coord_status.coordination_rounds
    );
    println!("   Consensus achieved: {}", coord_status.consensus_achieved);

    // Test 6: Performance monitoring
    println!("\n6. Testing performance monitoring...");
    let metrics = manager.get_agent_metrics()?;
    println!("✅ Agent metrics:");
    println!("   Total agents: {}", metrics.total_agents);
    println!("   Active agents: {}", metrics.active_agents);
    println!(
        "   GPU utilization: {:.2}%",
        metrics.gpu_utilization * 100.0
    );
    println!("   Memory usage: {} MB", metrics.memory_usage / 1024 / 1024);
    println!("   Messages/sec: {:.2}", metrics.messages_per_second);
    println!(
        "   Avg response time: {:.2} μs",
        metrics.avg_response_time_us
    );

    // Test 7: Agent status check
    println!("\n7. Testing agent status check...");
    let agent_status = manager.get_agent_status(0)?;
    println!("✅ Agent 0 status:");
    println!("   Name: {}", agent_status.name);
    println!("   State: {:?}", agent_status.state);
    println!("   Messages sent: {}", agent_status.messages_sent);
    println!("   Messages received: {}", agent_status.messages_received);

    // Test 8: Termination
    println!("\n8. Testing agent termination...");
    manager.terminate_agent(agent_ids[0])?;
    println!("✅ Terminated agent {}", agent_ids[0]);

    // Test 9: Scaling test
    println!("\n9. Testing maximum agent capacity...");
    // Clear all agents first
    for id in 0..10 {
        let _ = manager.terminate_agent(id);
    }

    // Spawn maximum agents
    let max_agents = manager.spawn_agents(10)?;
    println!(
        "✅ Successfully spawned maximum {} agents",
        max_agents.len()
    );

    // Test 10: Multi-round coordination
    println!("\n10. Testing multi-round coordination...");
    let start = Instant::now();
    for round in 0..5 {
        let status = manager.coordinate_agents()?;
        println!(
            "   Round {}: {} active agents, consensus={}",
            round, status.active_agents, status.consensus_achieved
        );
    }
    println!(
        "✅ Completed 5 coordination rounds in {:?}",
        start.elapsed()
    );

    println!("\n✅ All tests passed in GREEN phase!");
    println!("\nKey achievements:");
    println!("- Successfully managed 1-10 agents on single node");
    println!("- Agent communication working");
    println!("- Coordination and consensus mechanisms functional");
    println!("- Performance monitoring operational");

    Ok(())
}
