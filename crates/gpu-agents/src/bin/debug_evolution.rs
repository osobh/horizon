//! Debug evolution step by step
use cudarc::driver::CudaContext;
use gpu_agents::evolution::{GpuEvolutionConfig, GpuEvolutionEngine};

fn main() -> anyhow::Result<()> {
    println!("🔍 Debug GPU Evolution Step by Step");
    println!("=====================================");

    // Initialize CUDA device
    println!("1. Initializing CUDA device...");
    let ctx = CudaContext::new(0)?;
    println!("✅ CUDA device initialized");

    // Create test configuration
    println!("2. Creating evolution config...");
    let config = GpuEvolutionConfig {
        population_size: 1024,
        genome_size: 64,
        fitness_objectives: 1,
        mutation_rate: 0.01,
        crossover_rate: 0.7,
        elite_percentage: 0.1,
        block_size: 256,
    };
    println!("✅ Config created");

    // Create evolution engine
    println!("3. Creating GpuEvolutionEngine...");
    let mut engine = GpuEvolutionEngine::new(ctx, config)?;
    println!("✅ GpuEvolutionEngine created");

    // Initialize random population
    println!("4. Initializing random population...");
    engine.initialize_random()?;
    println!("✅ Random population initialized");

    // Get initial statistics
    println!("5. Getting initial statistics...");
    let stats = engine.statistics();
    println!(
        "✅ Initial stats: gen={}, pop={}, best={:.4}, avg={:.4}, diversity={:.4}",
        stats.generation,
        stats.population_size,
        stats.best_fitness,
        stats.average_fitness,
        stats.diversity_index
    );

    // Try one evolution step
    println!("6. Running one evolution generation...");
    engine.evolve_generation()?;
    println!("✅ Evolution generation completed");

    // Get final statistics
    println!("7. Getting final statistics...");
    let final_stats = engine.statistics();
    println!(
        "✅ Final stats: gen={}, pop={}, best={:.4}, avg={:.4}, diversity={:.4}",
        final_stats.generation,
        final_stats.population_size,
        final_stats.best_fitness,
        final_stats.average_fitness,
        final_stats.diversity_index
    );

    println!("\n✅ All debugging steps completed successfully!");
    Ok(())
}
