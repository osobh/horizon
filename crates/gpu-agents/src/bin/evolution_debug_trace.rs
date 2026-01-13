//! Debug trace to find exact hang location
use cudarc::driver::CudaContext;
use gpu_agents::evolution::{GpuEvolutionConfig, GpuEvolutionEngine};

fn main() -> anyhow::Result<()> {
    println!("🧬 Evolution Debug Trace");
    println!("=======================");

    // Initialize CUDA device
    println!("1. Creating CUDA device...");
    let ctx = CudaContext::new(0)?;
    println!("   ✅ Device created");

    // Small test config
    let config = GpuEvolutionConfig {
        population_size: 32,
        genome_size: 8,
        fitness_objectives: 1,
        mutation_rate: 0.01,
        crossover_rate: 0.7,
        elite_percentage: 0.1,
        block_size: 32,
    };

    println!("2. Creating evolution engine...");
    let mut engine = GpuEvolutionEngine::new(ctx, config)?;
    println!("   ✅ Engine created");

    println!("3. Initializing random population...");
    engine.initialize_random()?;
    println!("   ✅ Population initialized");

    println!("4. Getting initial statistics...");
    let stats = engine.statistics();
    println!(
        "   ✅ Stats: gen={}, pop={}",
        stats.generation, stats.population_size
    );

    println!("5. Running evolve_generation()...");
    println!("   5a. Checking has_fitness...");
    // The evolve_generation method checks has_fitness first
    // Let's manually check it

    println!("   5b. About to call evolve_generation...");
    match engine.evolve_generation() {
        Ok(_) => {
            println!("   ✅ Evolution completed!");
            let new_stats = engine.statistics();
            println!(
                "   📊 New stats: gen={}, best={:.4}",
                new_stats.generation, new_stats.best_fitness
            );
        }
        Err(e) => {
            println!("   ❌ Evolution failed: {}", e);
            if e.to_string().contains("RNG states not initialized") {
                println!("   ⚠️  RNG states issue detected");
            }
        }
    }

    println!("\n✅ Debug trace completed");
    Ok(())
}
