//! Simplified evolution benchmark that bypasses GpuSwarm infrastructure
//! This directly tests our fixed GpuEvolutionEngine without the complex integration

use crate::evolution::{GpuEvolutionConfig, GpuEvolutionEngine};
use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::time::Instant;

/// Evolution benchmark results
#[derive(Debug, Clone)]
pub struct SimplifiedEvolutionResults {
    pub generations_per_second: f64,
    pub population_results: Vec<PopulationResult>,
    pub linear_scaling_limit: usize,
    pub parallel_efficiency: f64,
}

/// Population-specific results
#[derive(Debug, Clone)]
pub struct PopulationResult {
    pub population_size: usize,
    pub genome_size: usize,
    pub generations: usize,
    pub total_time_ms: f64,
    pub avg_generation_time_ms: f64,
    pub final_best_fitness: f64,
    pub final_avg_fitness: f64,
    pub generations_per_second: f64,
}

/// Run simplified evolution benchmarks without GpuSwarm dependency
pub async fn run_simplified_evolution_benchmark(
    quick_mode: bool,
    stress_mode: bool,
) -> Result<SimplifiedEvolutionResults> {
    println!("🧬 Running Simplified Evolution Strategy Benchmarks");
    println!("   (Bypassing GpuSwarm - Direct GpuEvolutionEngine testing)");

    // Configure based on mode
    let population_sizes = if quick_mode {
        vec![1024, 2048, 4096]
    } else if stress_mode {
        vec![4096, 8192, 16384, 32768]
    } else {
        vec![1024, 2048, 4096, 8192]
    };

    let generations_per_test = if quick_mode { 10 } else { 50 };

    let mut max_generations_per_second: f64 = 0.0;
    let mut results = Vec::new();

    // Initialize CUDA device once
    let device = Arc::new(CudaDevice::new(0)?);
    println!("   ✅ CUDA device initialized");

    // Test different population sizes
    for &population_size in &population_sizes {
        println!("   Testing population size: {}", population_size);

        let result =
            test_direct_evolution_performance(&device, population_size, generations_per_test)
                .await?;
        println!(
            "   ✅ {} agents: {:.1} generations/sec, best_fitness: {:.4}",
            population_size, result.generations_per_second, result.final_best_fitness
        );

        max_generations_per_second = max_generations_per_second.max(result.generations_per_second);
        results.push(result);
    }

    // Determine linear scaling limit
    let linear_scaling_limit = find_linear_scaling_limit(&results);

    // Calculate parallel efficiency
    let parallel_efficiency = calculate_parallel_efficiency(&results);

    println!(
        "   📊 Peak performance: {:.1} generations/sec",
        max_generations_per_second
    );
    println!(
        "   🔄 Linear scaling limit: {} agents",
        linear_scaling_limit
    );
    println!(
        "   ⚡ Parallel efficiency: {:.1}%",
        parallel_efficiency * 100.0
    );

    Ok(SimplifiedEvolutionResults {
        generations_per_second: max_generations_per_second,
        population_results: results,
        linear_scaling_limit,
        parallel_efficiency,
    })
}

async fn test_direct_evolution_performance(
    device: &Arc<CudaDevice>,
    population_size: usize,
    generations: usize,
) -> Result<PopulationResult> {
    // Ensure population size is multiple of 32 for GPU efficiency
    let aligned_population_size = ((population_size + 31) / 32) * 32;
    let genome_size = 256; // Fixed genome size for consistency

    // Create evolution configuration
    let config = GpuEvolutionConfig {
        population_size: aligned_population_size,
        genome_size,
        fitness_objectives: 1,
        mutation_rate: 0.01,
        crossover_rate: 0.7,
        elite_percentage: 0.1,
        block_size: 256,
    };

    // Create evolution engine directly (no GpuSwarm dependency)
    let mut evolution_engine = GpuEvolutionEngine::new(device.clone(), config)?;
    evolution_engine.initialize_random()?;

    // Measure evolution performance
    let start = Instant::now();

    for _generation in 0..generations {
        // Direct evolution without swarm integration
        evolution_engine.evolve_generation()?;
    }

    let total_time = start.elapsed();
    let stats = evolution_engine.statistics();

    Ok(PopulationResult {
        population_size: aligned_population_size,
        genome_size,
        generations,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        avg_generation_time_ms: total_time.as_secs_f64() * 1000.0 / generations as f64,
        final_best_fitness: stats.best_fitness,
        final_avg_fitness: stats.average_fitness,
        generations_per_second: generations as f64 / total_time.as_secs_f64(),
    })
}

fn find_linear_scaling_limit(results: &[PopulationResult]) -> usize {
    // Find the population size where scaling efficiency drops below 50%
    if results.len() < 2 {
        return results.first().map(|r| r.population_size).unwrap_or(0);
    }

    let baseline = &results[0];
    let baseline_throughput_per_agent =
        baseline.generations_per_second / baseline.population_size as f64;

    for result in results.iter().skip(1) {
        let current_throughput_per_agent =
            result.generations_per_second / result.population_size as f64;
        let efficiency = current_throughput_per_agent / baseline_throughput_per_agent;

        if efficiency < 0.5 {
            return results[results
                .iter()
                .position(|r| r.population_size == result.population_size)
                .unwrap()
                - 1]
            .population_size;
        }
    }

    // If we never drop below 50%, return the largest tested size
    results.last()?.population_size
}

fn calculate_parallel_efficiency(results: &[PopulationResult]) -> f64 {
    if results.len() < 2 {
        return 1.0;
    }

    let baseline = &results[0];
    let largest = results.last()?;

    let theoretical_speedup = largest.population_size as f64 / baseline.population_size as f64;
    let actual_speedup = largest.generations_per_second / baseline.generations_per_second;

    actual_speedup / theoretical_speedup
}
