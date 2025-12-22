//! Scalability benchmarks using real GPU agents

use anyhow::Result;
use std::time::Instant;

use crate::{get_gpu_device_properties, GpuSwarm, GpuSwarmConfig};

/// Run scalability benchmarks with real GPU acceleration
pub async fn run_scalability_benchmark(
    quick_mode: bool,
    stress_mode: bool,
) -> Result<ScalabilityBenchmarkResults> {
    println!("🔧 Running Scalability Benchmarks");

    // Get GPU properties
    let gpu_props = get_gpu_device_properties(0)?;
    println!(
        "   GPU: {} ({:.1} GB)",
        gpu_props.name,
        gpu_props.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Configure based on mode
    let agent_counts = if quick_mode {
        vec![1_000, 10_000, 100_000]
    } else if stress_mode {
        vec![1_000_000, 10_000_000, 50_000_000]
    } else {
        vec![1_000, 10_000, 100_000, 1_000_000, 5_000_000]
    };

    let mut max_agents = 0;
    let mut peak_agents_per_second: f64 = 0.0;
    let mut results = Vec::new();

    for &agent_count in &agent_counts {
        println!("   Testing {} agents...", agent_count);

        let result = test_agent_scalability(agent_count, quick_mode).await;

        match result {
            Ok(perf) => {
                if perf.success {
                    max_agents = max_agents.max(agent_count);
                    peak_agents_per_second = peak_agents_per_second.max(perf.agents_per_second);

                    println!("   ✅ Success: {:.1} agents/second", perf.agents_per_second);

                    results.push(perf);
                }
            }
            Err(e) => {
                println!("   ❌ Failed: {}", e);
                break;
            }
        }
    }

    // Calculate memory efficiency
    let memory_efficiency = calculate_memory_efficiency(&results);

    Ok(ScalabilityBenchmarkResults {
        max_agents,
        peak_agents_per_second,
        memory_efficiency,
        agent_results: results,
    })
}

async fn test_agent_scalability(
    agent_count: usize,
    quick_mode: bool,
) -> Result<AgentScalabilityResult> {
    // Create swarm configuration
    let config = GpuSwarmConfig {
        device_id: 0,
        max_agents: agent_count,
        block_size: 256,
        shared_memory_size: 48 * 1024, // 48KB
        evolution_interval: 100,
        enable_llm: false,
        enable_collective_intelligence: false,
        enable_collective_knowledge: false,
        enable_knowledge_graph: false,
    };

    let mut swarm = GpuSwarm::new(config)?;
    swarm.initialize(agent_count)?;

    // Run performance test
    let steps = if quick_mode { 10 } else { 100 };
    let start = Instant::now();

    for _ in 0..steps {
        swarm.step()?;
    }

    let duration = start.elapsed();
    let total_agent_steps = (agent_count * steps) as f64;
    let agents_per_second = total_agent_steps / duration.as_secs_f64();

    // Get memory usage
    let metrics = swarm.metrics();
    let memory_used_mb = metrics.gpu_memory_used as f64 / (1024.0 * 1024.0);

    Ok(AgentScalabilityResult {
        agent_count,
        agents_per_second,
        memory_used_mb,
        steps_completed: steps,
        success: true,
    })
}

fn calculate_memory_efficiency(results: &[AgentScalabilityResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }

    // Calculate average memory per agent
    let total_memory_per_agent: f64 = results
        .iter()
        .map(|r| r.memory_used_mb / r.agent_count as f64)
        .sum::<f64>()
        / results.len() as f64;

    // Ideal memory per agent (assuming 1KB per agent)
    let ideal_memory_per_agent = 0.001; // MB

    // Efficiency is inverse of memory usage ratio
    (ideal_memory_per_agent / total_memory_per_agent).min(1.0)
}

#[derive(Debug, Clone)]
pub struct ScalabilityBenchmarkResults {
    pub max_agents: usize,
    pub peak_agents_per_second: f64,
    pub memory_efficiency: f64,
    pub agent_results: Vec<AgentScalabilityResult>,
}

#[derive(Debug, Clone)]
pub struct AgentScalabilityResult {
    pub agent_count: usize,
    pub agents_per_second: f64,
    pub memory_used_mb: f64,
    pub steps_completed: usize,
    pub success: bool,
}
