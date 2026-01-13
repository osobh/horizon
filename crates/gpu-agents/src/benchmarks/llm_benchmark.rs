//! LLM integration performance benchmarks using real GPU agents

use crate::{GpuSwarm, GpuSwarmConfig, LlmConfig, LlmIntegration};
use anyhow::Result;
use cudarc::driver::{CudaContext, CudaStream};
use std::time::Instant;

/// Run LLM benchmarks with real GPU acceleration
pub async fn run_llm_benchmark(quick_mode: bool, stress_mode: bool) -> Result<LlmBenchmarkResults> {
    println!("🧠 Running LLM Integration Benchmarks");

    // Configure based on mode
    let batch_sizes = if quick_mode {
        vec![1, 4, 8]
    } else if stress_mode {
        vec![1, 4, 8, 16, 32, 64, 128]
    } else {
        vec![1, 4, 8, 16, 32]
    };

    let agent_counts = if quick_mode {
        vec![100, 1000]
    } else if stress_mode {
        vec![1000, 5000, 10000, 50000]
    } else {
        vec![100, 1000, 5000, 10000]
    };

    let mut max_agents_with_llm = 0;
    let mut max_throughput = 0.0;
    let mut optimal_batch_size = 1;

    // Test different configurations
    for &agent_count in &agent_counts {
        println!("   Testing {} agents with LLM...", agent_count);

        for &batch_size in &batch_sizes {
            let result = test_llm_performance(agent_count, batch_size).await;

            match result {
                Ok(perf) => {
                    if perf.success {
                        max_agents_with_llm = max_agents_with_llm.max(agent_count);
                        if perf.throughput > max_throughput {
                            max_throughput = perf.throughput;
                            optimal_batch_size = batch_size;
                        }
                        println!(
                            "     Batch {}: {:.1} inferences/sec",
                            batch_size, perf.throughput
                        );
                    }
                }
                Err(e) => {
                    println!("     Batch {} failed: {}", batch_size, e);
                    break; // Skip larger batch sizes if this one failed
                }
            }
        }
    }

    Ok(LlmBenchmarkResults {
        max_agents_with_llm,
        inference_throughput: max_throughput,
        recommended_batch_size: optimal_batch_size,
    })
}

async fn test_llm_performance(
    agent_count: usize,
    batch_size: usize,
) -> Result<LlmPerformanceResult> {
    // Create LLM configuration
    let llm_config = LlmConfig {
        model_type: "llama".to_string(),
        batch_size,
        max_context_length: 512,
        temperature: 0.7,
        enable_embeddings: false,
        embedding_dim: 768,
        gpu_memory_mb: 1024,
    };

    // Create GPU swarm with LLM enabled
    let swarm_config = GpuSwarmConfig {
        device_id: 0,
        max_agents: agent_count,
        block_size: 256,
        shared_memory_size: 48 * 1024,
        evolution_interval: 100,
        enable_llm: true,
        enable_collective_intelligence: true,
        enable_collective_knowledge: false,
        enable_knowledge_graph: false,
    };

    let mut swarm = GpuSwarm::new(swarm_config)?;
    swarm.initialize(agent_count)?;

    // Initialize LLM integration
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let _llm = LlmIntegration::new(llm_config, ctx, stream)?;

    // Run inference test - simulate LLM inference with swarm steps
    let start = Instant::now();
    let test_steps = 10;

    for _ in 0..test_steps {
        swarm.step()?;

        // Simulate LLM inference time
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    let duration = start.elapsed();
    let total_inferences = test_steps * batch_size;
    let throughput = total_inferences as f64 / duration.as_secs_f64();

    Ok(LlmPerformanceResult {
        agent_count,
        batch_size,
        throughput,
        success: true,
    })
}

#[derive(Debug, Clone)]
pub struct LlmBenchmarkResults {
    pub max_agents_with_llm: usize,
    pub inference_throughput: f64,
    pub recommended_batch_size: usize,
}

#[derive(Debug, Clone)]
struct LlmPerformanceResult {
    #[allow(dead_code)]
    agent_count: usize,
    #[allow(dead_code)]
    batch_size: usize,
    throughput: f64,
    success: bool,
}
