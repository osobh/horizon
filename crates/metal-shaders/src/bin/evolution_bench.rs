//! Evolution Kernel Benchmark for Metal GPU
//!
//! Benchmarks the embedded ML evolution kernels:
//! - Fitness evaluation throughput
//! - Full evolution step performance
//! - Scaling from 1K to 1M agents
//!
//! Run with: cargo run --release -p stratoswarm-metal-shaders --bin evolution_bench

use std::time::{Duration, Instant};
use stratoswarm_metal_shaders::core::backend::MetalBackend;
use stratoswarm_metal_shaders::core::buffer::MetalBuffer;
use stratoswarm_metal_shaders::core::command::{
    MetalCommandBuffer, MetalCommandQueue, MetalComputeEncoder,
};
use stratoswarm_metal_shaders::core::metal3::Metal3Backend;
use stratoswarm_metal_shaders::{combine_shaders, common, evolution};

/// Evolution parameters matching the shader's EvolutionParams struct
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EvolutionParams {
    population_size: u32,
    genome_length: u32,
    mutation_rate: f32,
    mutation_strength: f32,
    crossover_rate: f32,
    tournament_size: u32,
    elitism_count: u32,
    generation: u32,
}

/// Fitness network weights (simplified for benchmark)
#[repr(C)]
#[derive(Clone, Copy)]
struct FitnessNetworkWeights {
    layer1_weights: [f32; 64 * 32],
    layer1_bias: [f32; 32],
    layer2_weights: [f32; 32 * 16],
    layer2_bias: [f32; 16],
    layer3_weights: [f32; 16],
    layer3_bias: f32,
    _padding: [f32; 15], // Align to 16 bytes
}

// Manually implement Pod since we can't derive it for large arrays
unsafe impl bytemuck::Pod for FitnessNetworkWeights {}
unsafe impl bytemuck::Zeroable for FitnessNetworkWeights {}

fn main() -> anyhow::Result<()> {
    println!("Metal Evolution Kernel Benchmark (M4)");
    println!("{}", "=".repeat(70));
    println!();

    // Initialize Metal backend
    let backend = Metal3Backend::new()?;
    let device = backend.device();
    let info = device.info();

    println!("Device: {}", info.name);
    println!();

    // Combine shaders
    let shader_source = combine_shaders(&[common::RNG, common::ATOMICS, evolution::EVOLUTION]);

    // Compile evolution kernels
    println!("Compiling evolution shaders...");
    let fitness_pipeline = backend.create_compute_pipeline(&shader_source, "evaluate_fitness")?;
    let evolution_pipeline = backend.create_compute_pipeline(&shader_source, "evolution_step")?;
    println!("Compilation successful!");
    println!();

    let queue = backend.create_command_queue()?;

    // ========================================================================
    // Benchmark Configurations
    // ========================================================================
    struct BenchConfig {
        name: &'static str,
        population_size: u32,
        genome_length: u32,
    }

    let configs = [
        BenchConfig {
            name: "Small",
            population_size: 1_024,
            genome_length: 64,
        },
        BenchConfig {
            name: "Medium",
            population_size: 10_000,
            genome_length: 64,
        },
        BenchConfig {
            name: "Large",
            population_size: 100_000,
            genome_length: 64,
        },
        BenchConfig {
            name: "XL",
            population_size: 1_000_000,
            genome_length: 64,
        },
        BenchConfig {
            name: "Wide",
            population_size: 10_000,
            genome_length: 256,
        },
    ];

    // ========================================================================
    // Benchmark 1: Fitness Evaluation Throughput
    // ========================================================================
    println!("1. Fitness Evaluation Throughput");
    println!("{}", "-".repeat(70));
    println!(
        "{:>12} {:>12} {:>12} {:>12} {:>15} {:>12}",
        "Config", "Population", "Genome", "Avg Time", "Agents/sec", "GFLOPS"
    );

    for config in &configs {
        let pop_size = config.population_size as usize;
        let genome_len = config.genome_length as usize;

        // Create buffers
        let genomes = backend.create_buffer::<f32>(pop_size * genome_len)?;
        let fitness = backend.create_buffer::<f32>(pop_size)?;

        // Initialize genomes with random data
        {
            let mut genomes_mut = backend.create_buffer::<f32>(pop_size * genome_len)?;
            let data = genomes_mut.contents_mut::<f32>();
            for (i, val) in data.iter_mut().enumerate() {
                *val = ((i as f32) * 0.001).sin();
            }
        }

        // Create params
        let params = EvolutionParams {
            population_size: config.population_size,
            genome_length: config.genome_length,
            mutation_rate: 0.1,
            mutation_strength: 0.1,
            crossover_rate: 0.8,
            tournament_size: 3,
            elitism_count: 10,
            generation: 0,
        };
        let params_buffer = backend.create_buffer_with_data(bytemuck::bytes_of(&params))?;

        // Create network weights (random initialization)
        let weights = create_random_weights();
        let weights_buffer = backend.create_buffer_with_data(bytemuck::bytes_of(&weights))?;

        // Warmup
        for _ in 0..5 {
            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&fitness_pipeline)?;
                encoder.set_buffer(0, &genomes, 0)?;
                encoder.set_buffer(1, &fitness, 0)?;
                encoder.set_bytes(2, bytemuck::bytes_of(&params))?;
                encoder.set_bytes(3, bytemuck::bytes_of(&weights))?;
                encoder.dispatch_threads(pop_size as u64)?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;
        }

        // Benchmark
        let iterations = if pop_size < 100_000 { 100 } else { 50 };
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();

            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&fitness_pipeline)?;
                encoder.set_buffer(0, &genomes, 0)?;
                encoder.set_buffer(1, &fitness, 0)?;
                encoder.set_bytes(2, bytemuck::bytes_of(&params))?;
                encoder.set_bytes(3, bytemuck::bytes_of(&weights))?;
                encoder.dispatch_threads(pop_size as u64)?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;

            times.push(start.elapsed());
        }

        times.sort();
        let avg = times.iter().sum::<Duration>() / times.len() as u32;
        let agents_per_sec = (pop_size as f64) / avg.as_secs_f64();

        // Estimate FLOPS: per agent = genome_len*32 + 32*32 + 32*16 + 16*16 + 16 MACs
        // Each MAC = 2 FLOPS
        let flops_per_agent =
            ((genome_len * 32) + (32 * 32) + (32 * 16) + (16 * 16) + 16) as f64 * 2.0;
        let gflops = agents_per_sec * flops_per_agent / 1e9;

        println!(
            "{:>12} {:>12} {:>12} {:>10.2?} {:>13.0} {:>10.2}",
            config.name, config.population_size, config.genome_length, avg, agents_per_sec, gflops
        );
    }
    println!();

    // ========================================================================
    // Benchmark 2: Full Evolution Step
    // ========================================================================
    println!("2. Full Evolution Step (Selection + Crossover + Mutation + Fitness)");
    println!("{}", "-".repeat(70));
    println!(
        "{:>12} {:>12} {:>12} {:>15} {:>15}",
        "Config", "Population", "Avg Time", "Gen/sec", "Agents/sec"
    );

    for config in &configs {
        let pop_size = config.population_size as usize;
        let genome_len = config.genome_length as usize;

        // Create buffers
        let genomes = backend.create_buffer::<f32>(pop_size * genome_len)?;
        let new_genomes = backend.create_buffer::<f32>(pop_size * genome_len)?;
        let fitness = backend.create_buffer::<f32>(pop_size)?;
        let rng_state = backend.create_buffer::<[u32; 4]>(pop_size)?;

        // Initialize RNG state
        {
            let mut rng_mut = backend.create_buffer::<u32>(pop_size * 4)?;
            let data = rng_mut.contents_mut::<u32>();
            for (i, val) in data.iter_mut().enumerate() {
                *val = i as u32;
            }
        }

        // Initialize genomes
        {
            let mut genomes_mut = backend.create_buffer::<f32>(pop_size * genome_len)?;
            let data = genomes_mut.contents_mut::<f32>();
            for (i, val) in data.iter_mut().enumerate() {
                *val = ((i as f32) * 0.001).sin();
            }
        }

        // Initialize fitness
        {
            let mut fitness_mut = backend.create_buffer::<f32>(pop_size)?;
            let data = fitness_mut.contents_mut::<f32>();
            for (i, val) in data.iter_mut().enumerate() {
                *val = (i as f32) / (pop_size as f32);
            }
        }

        let params = EvolutionParams {
            population_size: config.population_size,
            genome_length: config.genome_length,
            mutation_rate: 0.1,
            mutation_strength: 0.1,
            crossover_rate: 0.8,
            tournament_size: 3,
            elitism_count: 10,
            generation: 0,
        };

        let weights = create_random_weights();

        // Warmup
        for _ in 0..5 {
            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&evolution_pipeline)?;
                encoder.set_buffer(0, &genomes, 0)?;
                encoder.set_buffer(1, &new_genomes, 0)?;
                encoder.set_buffer(2, &fitness, 0)?;
                encoder.set_buffer(3, &rng_state, 0)?;
                encoder.set_bytes(4, bytemuck::bytes_of(&params))?;
                encoder.set_bytes(5, bytemuck::bytes_of(&weights))?;
                encoder.dispatch_threads(pop_size as u64)?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;
        }

        // Benchmark
        let iterations = if pop_size < 100_000 { 50 } else { 20 };
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();

            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&evolution_pipeline)?;
                encoder.set_buffer(0, &genomes, 0)?;
                encoder.set_buffer(1, &new_genomes, 0)?;
                encoder.set_buffer(2, &fitness, 0)?;
                encoder.set_buffer(3, &rng_state, 0)?;
                encoder.set_bytes(4, bytemuck::bytes_of(&params))?;
                encoder.set_bytes(5, bytemuck::bytes_of(&weights))?;
                encoder.dispatch_threads(pop_size as u64)?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;

            times.push(start.elapsed());
        }

        times.sort();
        let avg = times.iter().sum::<Duration>() / times.len() as u32;
        let gens_per_sec = 1.0 / avg.as_secs_f64();
        let agents_per_sec = (pop_size as f64) / avg.as_secs_f64();

        println!(
            "{:>12} {:>12} {:>12.2?} {:>13.1} {:>13.0}",
            config.name, config.population_size, avg, gens_per_sec, agents_per_sec
        );
    }
    println!();

    // ========================================================================
    // Benchmark 3: Scaling Analysis
    // ========================================================================
    println!("3. Scaling Analysis (Fixed Genome Length = 64)");
    println!("{}", "-".repeat(70));
    println!(
        "{:>15} {:>12} {:>15} {:>15}",
        "Population", "Time", "Agents/sec", "Efficiency"
    );

    let scaling_sizes: [u32; 8] = [
        1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000,
    ];
    let mut base_throughput: Option<f64> = None;

    for pop_size in scaling_sizes {
        let genome_len = 64usize;

        let genomes = backend.create_buffer::<f32>(pop_size as usize * genome_len)?;
        let new_genomes = backend.create_buffer::<f32>(pop_size as usize * genome_len)?;
        let fitness = backend.create_buffer::<f32>(pop_size as usize)?;
        let rng_state = backend.create_buffer::<[u32; 4]>(pop_size as usize)?;

        let params = EvolutionParams {
            population_size: pop_size,
            genome_length: 64,
            mutation_rate: 0.1,
            mutation_strength: 0.1,
            crossover_rate: 0.8,
            tournament_size: 3,
            elitism_count: 10,
            generation: 0,
        };

        let weights = create_random_weights();

        // Warmup
        for _ in 0..3 {
            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&evolution_pipeline)?;
                encoder.set_buffer(0, &genomes, 0)?;
                encoder.set_buffer(1, &new_genomes, 0)?;
                encoder.set_buffer(2, &fitness, 0)?;
                encoder.set_buffer(3, &rng_state, 0)?;
                encoder.set_bytes(4, bytemuck::bytes_of(&params))?;
                encoder.set_bytes(5, bytemuck::bytes_of(&weights))?;
                encoder.dispatch_threads(pop_size as u64)?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;
        }

        // Benchmark
        let iterations = 20;
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();

            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&evolution_pipeline)?;
                encoder.set_buffer(0, &genomes, 0)?;
                encoder.set_buffer(1, &new_genomes, 0)?;
                encoder.set_buffer(2, &fitness, 0)?;
                encoder.set_buffer(3, &rng_state, 0)?;
                encoder.set_bytes(4, bytemuck::bytes_of(&params))?;
                encoder.set_bytes(5, bytemuck::bytes_of(&weights))?;
                encoder.dispatch_threads(pop_size as u64)?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;

            times.push(start.elapsed());
        }

        times.sort();
        let avg = times.iter().sum::<Duration>() / times.len() as u32;
        let throughput = (pop_size as f64) / avg.as_secs_f64();

        let efficiency = if let Some(base) = base_throughput {
            (throughput / base) * 100.0
        } else {
            base_throughput = Some(throughput);
            100.0
        };

        println!(
            "{:>15} {:>10.2?} {:>13.0} {:>13.1}%",
            format_count(pop_size as u64),
            avg,
            throughput,
            efficiency
        );
    }
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("{}", "=".repeat(70));
    println!("Summary");
    println!("{}", "-".repeat(70));
    println!("Target: >500K agents/second for embedded ML evolution");
    println!("Benchmark complete!");
    println!();

    Ok(())
}

fn create_random_weights() -> FitnessNetworkWeights {
    let mut weights = FitnessNetworkWeights {
        layer1_weights: [0.0; 64 * 32],
        layer1_bias: [0.0; 32],
        layer2_weights: [0.0; 32 * 16],
        layer2_bias: [0.0; 16],
        layer3_weights: [0.0; 16],
        layer3_bias: 0.0,
        _padding: [0.0; 15],
    };

    // Initialize with simple pattern (Xavier-like)
    for (i, w) in weights.layer1_weights.iter_mut().enumerate() {
        *w = ((i as f32) * 0.01).sin() * 0.1;
    }
    for (i, w) in weights.layer2_weights.iter_mut().enumerate() {
        *w = ((i as f32) * 0.02).sin() * 0.1;
    }
    for (i, w) in weights.layer3_weights.iter_mut().enumerate() {
        *w = ((i as f32) * 0.05).sin() * 0.1;
    }

    weights
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}
