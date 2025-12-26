//! Trace individual statistics method calls
use cudarc::driver::CudaDevice;
use gpu_agents::evolution::{GpuEvolutionConfig, GpuEvolutionEngine};

fn main() -> anyhow::Result<()> {
    println!("🧬 Evolution Statistics Trace");
    println!("============================");

    // Initialize CUDA device
    let device = CudaDevice::new(0)?;

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

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;

    // Now let's trace each statistics component
    println!("\n📊 Testing individual statistics components:");

    // We can't directly access the population, but we can test through public methods
    println!("1. Testing if we can run evolve_generation first...");
    match engine.evolve_generation() {
        Ok(_) => println!("   ✅ Evolution step completed"),
        Err(e) => println!("   ❌ Evolution failed: {}", e),
    }

    println!("\n2. Now testing statistics after evolution...");
    let stats = engine.statistics();
    println!(
        "   ✅ Stats retrieved: gen={}, pop={}, best={:.4}",
        stats.generation, stats.population_size, stats.best_fitness
    );

    Ok(())
}
