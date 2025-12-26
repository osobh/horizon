//! Test evolution at different scales to find where it hangs
use cudarc::driver::CudaDevice;
use gpu_agents::evolution::{GpuEvolutionConfig, GpuEvolutionEngine};
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("🧬 Evolution Scaling Test");
    println!("========================");

    // Initialize CUDA device
    let device = CudaDevice::new(0)?;

    // Test progressively larger population sizes
    let test_sizes = vec![32, 64, 128, 256, 512, 1024];

    for &population_size in &test_sizes {
        println!("\n📊 Testing population size: {}", population_size);

        let config = GpuEvolutionConfig {
            population_size,
            genome_size: 32, // Smaller genome for testing
            fitness_objectives: 1,
            mutation_rate: 0.01,
            crossover_rate: 0.7,
            elite_percentage: 0.1,
            block_size: 32, // Match smaller populations
        };

        match test_single_generation(device.clone(), config) {
            Ok(duration_ms) => {
                println!("  ✅ Success: {:.2} ms", duration_ms);
            }
            Err(e) => {
                println!("  ❌ Failed at size {}: {}", population_size, e);
                break;
            }
        }
    }

    Ok(())
}

fn test_single_generation(
    device: Arc<CudaDevice>,
    config: GpuEvolutionConfig,
) -> anyhow::Result<f64> {
    // Create engine
    let mut engine = GpuEvolutionEngine::new(device, config)?;

    // Initialize
    println!("  🔧 Initializing population...");
    engine.initialize_random()?;

    // Test single generation
    println!("  🧬 Running evolution...");
    let start = Instant::now();
    engine.evolve_generation()?;
    let duration = start.elapsed();

    // Get stats
    let stats = engine.statistics();
    println!(
        "  📈 Stats: gen={}, best={:.4}",
        stats.generation, stats.best_fitness
    );

    Ok(duration.as_secs_f64() * 1000.0)
}
