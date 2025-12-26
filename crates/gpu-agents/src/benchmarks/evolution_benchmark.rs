//! Evolution strategy performance benchmarks using real GPU agents

use crate::{GpuEvolutionConfig, GpuEvolutionEngine, GpuSwarm, GpuSwarmConfig};
use anyhow::Result;
use std::time::Instant;

/// Run evolution benchmarks with real GPU acceleration
pub async fn run_evolution_benchmark(
    quick_mode: bool,
    stress_mode: bool,
) -> Result<EvolutionBenchmarkResults> {
    println!("🧬 Running Evolution Strategy Benchmarks");

    // Configure based on mode
    let population_sizes = if quick_mode {
        vec![100, 500, 1000]
    } else if stress_mode {
        vec![1000, 5000, 10000, 50000]
    } else {
        vec![100, 500, 1000, 5000]
    };

    let generations_per_test = if quick_mode { 10 } else { 100 };

    let mut max_generations_per_second: f64 = 0.0;
    let mut results = Vec::new();

    // Test different population sizes
    for &population_size in &population_sizes {
        println!("   Testing population size: {}", population_size);

        let result = test_evolution_performance(population_size, generations_per_test).await?;
        println!(
            "   ✅ {} agents: {:.1} generations/sec",
            population_size, result.generations_per_second
        );

        max_generations_per_second = max_generations_per_second.max(result.generations_per_second);
        results.push(result);
    }

    // Determine linear scaling limit
    let linear_scaling_limit = find_linear_scaling_limit(&results);

    // Calculate parallel efficiency
    let parallel_efficiency = calculate_parallel_efficiency(&results);

    Ok(EvolutionBenchmarkResults {
        generations_per_second: max_generations_per_second,
        linear_scaling_limit,
        parallel_efficiency,
        population_results: results,
    })
}

async fn test_evolution_performance(
    population_size: usize,
    generations: usize,
) -> Result<PopulationResult> {
    // Ensure population size is multiple of 32 for GPU efficiency
    let aligned_population_size = ((population_size + 31) / 32) * 32;

    // Create evolution configuration
    let evolution_config = GpuEvolutionConfig {
        population_size: aligned_population_size,
        genome_size: 256,
        fitness_objectives: 2, // Performance and Efficiency
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        elite_percentage: 0.1,
        block_size: 256,
    };

    // Create GPU swarm for fitness evaluation
    let swarm_config = GpuSwarmConfig {
        device_id: 0,
        max_agents: population_size,
        block_size: 256,
        shared_memory_size: 48 * 1024, // 48KB
        evolution_interval: 1,
        enable_llm: false,
        enable_collective_intelligence: false,
        enable_collective_knowledge: false,
        enable_knowledge_graph: false,
    };

    let mut swarm = GpuSwarm::new(swarm_config)?;
    swarm.initialize(population_size)?;

    // Get device from swarm
    let device = swarm.get_device().clone();

    // Create evolution engine
    let mut evolution_engine = GpuEvolutionEngine::new(device, evolution_config)?;
    evolution_engine.initialize_random()?;

    // Measure evolution performance
    let start = Instant::now();

    for _generation in 0..generations {
        // Run one GPU step for fitness evaluation
        swarm.step()?;

        // Evolve population using the swarm
        // Evaluate fitness
        evolution_engine.evaluate_fitness()?;

        // Evolve to next generation
        evolution_engine.evolve_generation()?;
    }

    let duration = start.elapsed();
    let generations_per_second = generations as f64 / duration.as_secs_f64();

    // Get memory usage from swarm metrics
    let swarm_metrics = swarm.metrics();
    let memory_usage_mb = swarm_metrics.gpu_memory_used as f64 / (1024.0 * 1024.0);

    Ok(PopulationResult {
        population_size,
        generations_completed: generations,
        generations_per_second,
        memory_usage_mb,
    })
}

fn find_linear_scaling_limit(results: &[PopulationResult]) -> usize {
    // Find where efficiency starts dropping significantly
    let mut prev_efficiency = f64::INFINITY;

    for result in results {
        let efficiency = result.generations_per_second / result.population_size as f64;
        if efficiency < prev_efficiency * 0.9 {
            return result.population_size;
        }
        prev_efficiency = efficiency;
    }

    results.last().map(|r| r.population_size).unwrap_or(0)
}

fn calculate_parallel_efficiency(results: &[PopulationResult]) -> f64 {
    if results.len() < 2 {
        return 1.0;
    }

    let first = &results[0];
    let last = &results[results.len() - 1];

    let speedup = last.generations_per_second / first.generations_per_second;
    let ideal_speedup = (last.population_size as f64 / first.population_size as f64).sqrt();

    (speedup / ideal_speedup).min(1.0)
}

#[derive(Debug, Clone)]
pub struct EvolutionBenchmarkResults {
    pub generations_per_second: f64,
    pub linear_scaling_limit: usize,
    pub parallel_efficiency: f64,
    pub population_results: Vec<PopulationResult>,
}

#[derive(Debug, Clone)]
pub struct PopulationResult {
    pub population_size: usize,
    pub generations_completed: usize,
    pub generations_per_second: f64,
    pub memory_usage_mb: f64,
}
