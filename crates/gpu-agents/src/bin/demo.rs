//! Simple demo of GPU agents

use anyhow::Result;
use gpu_agents::{GpuSwarm, GpuSwarmConfig};

fn main() -> Result<()> {
    println!("GPU Agents Demo");
    println!("===============");

    // Create swarm configuration
    let config = GpuSwarmConfig::default();
    println!("Creating GPU swarm with config: {:?}", config);

    // Create GPU swarm
    let mut swarm = GpuSwarm::new(config)?;
    println!("GPU swarm created successfully!");

    // Initialize with 1000 agents
    let agent_count = 1000;
    println!("\nInitializing {} agents...", agent_count);
    swarm.initialize(agent_count)?;
    println!("Agents initialized!");

    // Run a few simulation steps
    println!("\nRunning simulation steps...");
    for i in 0..5 {
        println!("Step {}", i + 1);
        swarm.step()?;
    }

    // Get metrics
    let metrics = swarm.metrics();
    println!("\nSwarm Metrics:");
    println!("  Agent count: {}", metrics.agent_count);
    println!("  GPU memory used: {} bytes", metrics.gpu_memory_used);
    println!("  Kernel time: {:.2} ms", metrics.kernel_time_ms);

    println!("\nDemo completed successfully!");
    Ok(())
}
