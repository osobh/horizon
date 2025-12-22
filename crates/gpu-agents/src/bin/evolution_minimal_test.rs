//! Minimal evolution test without statistics
use cudarc::driver::CudaDevice;
use gpu_agents::evolution::{GpuEvolutionConfig, GpuEvolutionEngine};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("🧬 Minimal Evolution Test");
    println!("========================");

    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    println!("✅ CUDA device initialized");

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

    println!("\n📊 Creating evolution engine...");
    let mut engine = GpuEvolutionEngine::new(device, config)?;
    println!("✅ Engine created");

    println!("\n🎲 Initializing random population...");
    engine.initialize_random()?;
    println!("✅ Population initialized");

    println!("\n🧮 Evaluating fitness...");
    engine.evaluate_fitness()?;
    println!("✅ Fitness evaluated");

    println!("\n🔄 Running single evolution generation...");
    let start = Instant::now();
    engine.evolve_generation()?;
    let duration = start.elapsed();
    println!(
        "✅ Generation completed in {:.2} ms",
        duration.as_secs_f64() * 1000.0
    );

    println!("\n🎯 Running 10 more generations...");
    let start = Instant::now();
    for i in 1..=10 {
        engine.evolve_generation()?;
        println!("  Gen {}: ✅", i);
    }
    let duration = start.elapsed();
    let gens_per_sec = 10.0 / duration.as_secs_f64();
    println!("✅ Performance: {:.2} generations/second", gens_per_sec);

    println!("\n✅ Minimal evolution test completed successfully!");
    Ok(())
}
