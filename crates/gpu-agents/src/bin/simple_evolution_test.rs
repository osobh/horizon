//! Simple evolution test to validate CUDA fix
use cudarc::driver::CudaContext;
use gpu_agents::evolution::{GpuEvolutionConfig, GpuEvolutionEngine};

fn main() -> anyhow::Result<()> {
    println!("🧬 Simple GPU Evolution Test");
    println!("============================");

    // Initialize CUDA device
    println!("Initializing CUDA device...");
    let ctx = CudaContext::new(0)?;
    println!("✅ CUDA device initialized");

    // Create small test configuration
    let config = GpuEvolutionConfig {
        population_size: 1024, // Small test population
        genome_size: 64,       // Small genome
        fitness_objectives: 1,
        mutation_rate: 0.01,
        crossover_rate: 0.7,
        elite_percentage: 0.1,
        block_size: 256,
    };

    println!("Creating GPU evolution engine...");
    let mut engine = GpuEvolutionEngine::new(ctx, config)?;
    println!("✅ GPU evolution engine created");

    println!("Initializing random population...");
    engine.initialize_random()?;
    println!("✅ Random population initialized");

    println!("Running evolution for 5 generations...");
    for generation in 0..5 {
        engine.evolve_generation()?;
        let stats = engine.statistics();
        println!(
            "  Gen {}: best={:.4}, avg={:.4}, diversity={:.4}",
            generation, stats.best_fitness, stats.average_fitness, stats.diversity_index
        );
    }

    println!("✅ Evolution test completed successfully!");
    Ok(())
}
