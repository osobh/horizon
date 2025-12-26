//! Working evolution benchmark that properly evaluates fitness
use cudarc::driver::CudaDevice;
use gpu_agents::evolution::{GpuEvolutionConfig, GpuEvolutionEngine};
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("🧬 GPU Evolution Working Benchmark");
    println!("=================================");

    // Initialize CUDA device
    println!("Initializing CUDA device...");
    let device = Arc::new(CudaDevice::new(0)?);
    println!("✅ CUDA device initialized");

    // Test different population sizes
    let test_sizes = vec![32, 64, 128, 256, 512, 1024, 2048];
    let generations = 10;

    for &population_size in &test_sizes {
        println!("\n🔬 Testing population size: {}", population_size);

        // Create test configuration
        let config = GpuEvolutionConfig {
            population_size,
            genome_size: 64,
            fitness_objectives: 1,
            mutation_rate: 0.01,
            crossover_rate: 0.7,
            elite_percentage: 0.1,
            block_size: 256,
        };

        match test_evolution_performance(&device, config, generations) {
            Ok((gens_per_sec, best_fitness)) => {
                println!("  ✅ Performance: {:.2} generations/sec", gens_per_sec);
                println!("  📊 Best fitness: {:.4}", best_fitness);
            }
            Err(e) => {
                println!("  ❌ Failed: {}", e);
                break; // Stop at first failure
            }
        }
    }

    println!("\n✅ Evolution benchmark completed!");
    Ok(())
}

fn test_evolution_performance(
    device: &Arc<CudaDevice>,
    config: GpuEvolutionConfig,
    generations: usize,
) -> anyhow::Result<(f64, f64)> {
    // Create evolution engine
    let mut engine = GpuEvolutionEngine::new(device.clone(), config)?;

    // Initialize random population
    engine.initialize_random()?;

    // IMPORTANT: Evaluate fitness before getting statistics
    engine.evaluate_fitness()?;

    // Now we can safely get initial statistics
    let initial_stats = engine.statistics();
    println!(
        "    Initial: gen={}, best={:.4}, avg={:.4}",
        initial_stats.generation, initial_stats.best_fitness, initial_stats.average_fitness
    );

    // Run generations
    let start = Instant::now();

    for gen in 0..generations {
        engine.evolve_generation()?;

        // Log progress every few generations
        if gen % 5 == 0 || gen == generations - 1 {
            let stats = engine.statistics();
            println!(
                "    Gen {}: best={:.4}, avg={:.4}",
                stats.generation, stats.best_fitness, stats.average_fitness
            );
        }
    }

    let duration = start.elapsed();
    let generations_per_second = generations as f64 / duration.as_secs_f64();

    let final_stats = engine.statistics();

    Ok((generations_per_second, final_stats.best_fitness))
}
