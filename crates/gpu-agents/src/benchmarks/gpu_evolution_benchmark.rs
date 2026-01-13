//! GPU Evolution benchmarks

use anyhow::Result;
use cudarc::driver::CudaContext;
use std::sync::Arc;
use std::time::Instant;

use crate::evolution::selection::SelectionMethod;
use crate::evolution::{GpuEvolutionConfig, GpuEvolutionEngine};

/// Evolution benchmark configuration
#[derive(Debug, Clone)]
pub struct EvolutionBenchmarkConfig {
    /// Population sizes to test
    pub population_sizes: Vec<usize>,
    /// Genome sizes to test
    pub genome_sizes: Vec<usize>,
    /// Number of generations
    pub generations: u64,
    /// Mutation rates to test
    pub mutation_rates: Vec<f32>,
    /// Selection strategies to test
    pub selection_methods: Vec<SelectionMethod>,
}

impl Default for EvolutionBenchmarkConfig {
    fn default() -> Self {
        Self {
            population_sizes: vec![1024, 10_240, 102_400, 1_024_000],
            genome_sizes: vec![64, 256, 1024],
            generations: 100,
            mutation_rates: vec![0.001, 0.01, 0.1],
            selection_methods: vec![
                SelectionMethod::Tournament { size: 3 },
                SelectionMethod::Elite,
            ],
        }
    }
}

/// Evolution benchmark results
#[derive(Debug, Clone)]
pub struct EvolutionBenchmarkResults {
    pub population_size: usize,
    pub genome_size: usize,
    pub generations: u64,
    pub total_time_ms: f64,
    pub avg_generation_time_ms: f64,
    pub mutations_per_second: f64,
    pub final_best_fitness: f64,
    pub final_avg_fitness: f64,
    pub memory_usage_mb: f64,
}

/// GPU Evolution benchmark suite
pub struct GpuEvolutionBenchmark {
    ctx: Arc<CudaContext>,
    config: EvolutionBenchmarkConfig,
}

impl GpuEvolutionBenchmark {
    /// Create new benchmark suite
    pub fn new(device_id: i32, config: EvolutionBenchmarkConfig) -> Result<Self> {
        let ctx = CudaContext::new(device_id as usize)?;
        Ok(Self { ctx, config })
    }

    /// Run all benchmarks
    pub async fn run_all(&mut self) -> Result<Vec<EvolutionBenchmarkResults>> {
        let mut results = Vec::new();

        for &population_size in &self.config.population_sizes {
            for &genome_size in &self.config.genome_sizes {
                println!(
                    "\nBenchmarking: population={}, genome={}",
                    population_size, genome_size
                );

                match self.benchmark_evolution(population_size, genome_size).await {
                    Ok(result) => {
                        println!("  Total time: {:.2}ms", result.total_time_ms);
                        println!("  Avg generation: {:.2}ms", result.avg_generation_time_ms);
                        println!(
                            "  Mutations/sec: {:.2}M",
                            result.mutations_per_second / 1_000_000.0
                        );
                        println!("  Best fitness: {:.4}", result.final_best_fitness);
                        println!("  Memory usage: {:.2}MB", result.memory_usage_mb);
                        results.push(result);
                    }
                    Err(e) => {
                        eprintln!("  Benchmark failed: {}", e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Benchmark evolution with specific parameters
    async fn benchmark_evolution(
        &self,
        population_size: usize,
        genome_size: usize,
    ) -> Result<EvolutionBenchmarkResults> {
        // Round population size to multiple of 32 for GPU efficiency
        let population_size = ((population_size + 31) / 32) * 32;

        let config = GpuEvolutionConfig {
            population_size,
            genome_size,
            fitness_objectives: 1,
            mutation_rate: 0.01,
            crossover_rate: 0.7,
            elite_percentage: 0.1,
            block_size: 256,
        };

        let mut engine = GpuEvolutionEngine::new(self.ctx.clone(), config)?;

        // Initialize population
        engine.initialize_random()?;

        let start = Instant::now();

        // Run evolution
        for generation in 0..self.config.generations {
            engine.evolve_generation()?;

            // Log progress every 10 generations
            if generation % 10 == 0 {
                let stats = engine.statistics();
                println!(
                    "    Gen {}: best={:.4}, avg={:.4}",
                    generation, stats.best_fitness, stats.average_fitness
                );
            }
        }

        let total_time = start.elapsed();
        let stats = engine.statistics();

        // Calculate memory usage
        let memory_usage_mb = (population_size * genome_size * std::mem::size_of::<u8>()
            + population_size * std::mem::size_of::<f32>()) as f64
            / (1024.0 * 1024.0);

        Ok(EvolutionBenchmarkResults {
            population_size,
            genome_size,
            generations: self.config.generations,
            total_time_ms: total_time.as_secs_f64() * 1000.0,
            avg_generation_time_ms: total_time.as_secs_f64() * 1000.0
                / self.config.generations as f64,
            mutations_per_second: stats.mutations_per_second,
            final_best_fitness: stats.best_fitness,
            final_avg_fitness: stats.average_fitness,
            memory_usage_mb,
        })
    }

    /// Benchmark mutation strategies
    pub async fn benchmark_mutation_strategies(&self) -> Result<()> {
        println!("\n=== Mutation Strategy Benchmarks ===");

        let population_size = 100_000;
        let genome_size = 256;

        for &mutation_rate in &self.config.mutation_rates {
            println!("\nMutation rate: {}", mutation_rate);

            let config = GpuEvolutionConfig {
                population_size,
                genome_size,
                mutation_rate,
                ..Default::default()
            };

            let mut engine = GpuEvolutionEngine::new(self.ctx.clone(), config)?;
            engine.initialize_random()?;

            let start = Instant::now();

            // Just run mutations
            for _ in 0..10 {
                engine.evolve_generation()?;
            }

            let elapsed = start.elapsed();
            let mutations_per_second = (population_size * 10) as f64 / elapsed.as_secs_f64();

            println!(
                "  Time: {:.2}ms, Mutations/sec: {:.2}M",
                elapsed.as_millis(),
                mutations_per_second / 1_000_000.0
            );
        }

        Ok(())
    }

    /// Benchmark selection methods
    pub async fn benchmark_selection_methods(&self) -> Result<()> {
        println!("\n=== Selection Method Benchmarks ===");

        let population_size = 100_000;
        let genome_size = 256;

        for method in &self.config.selection_methods {
            println!("\nSelection method: {:?}", method);

            let config = GpuEvolutionConfig {
                population_size,
                genome_size,
                ..Default::default()
            };

            let mut engine = GpuEvolutionEngine::new(self.ctx.clone(), config)?;
            engine.initialize_random()?;

            // Set selection method
            // Note: This would need to be added to the engine API

            let start = Instant::now();

            for _ in 0..10 {
                engine.evolve_generation()?;
            }

            let elapsed = start.elapsed();
            println!("  Time for 10 generations: {:.2}ms", elapsed.as_millis());
        }

        Ok(())
    }

    /// Benchmark scaling with population size
    pub async fn benchmark_scaling(&self) -> Result<()> {
        println!("\n=== Population Scaling Benchmarks ===");

        let genome_size = 256;
        let sizes = vec![1_000, 10_000, 100_000, 1_000_000, 10_000_000];

        for size in sizes {
            let population_size = ((size + 31) / 32) * 32; // Round to multiple of 32
            println!("\nPopulation size: {}", population_size);

            let config = GpuEvolutionConfig {
                population_size,
                genome_size,
                ..Default::default()
            };

            match GpuEvolutionEngine::new(self.ctx.clone(), config) {
                Ok(mut engine) => {
                    engine.initialize_random()?;

                    let start = Instant::now();
                    engine.evolve_generation()?;
                    let elapsed = start.elapsed();

                    let throughput = population_size as f64 / elapsed.as_secs_f64();
                    println!(
                        "  Single generation: {:.2}ms, Throughput: {:.2}M agents/sec",
                        elapsed.as_millis(),
                        throughput / 1_000_000.0
                    );
                }
                Err(e) => {
                    println!("  Failed to create engine: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }
}

/// Run evolution benchmarks
pub async fn run_evolution_benchmarks() -> Result<()> {
    println!("=== GPU Evolution Benchmarks ===");

    let config = EvolutionBenchmarkConfig::default();
    let mut benchmark = GpuEvolutionBenchmark::new(0, config)?;

    // Run comprehensive benchmarks
    let results = benchmark.run_all().await?;

    // Print summary
    println!("\n=== Summary ===");
    for result in &results {
        println!(
            "Pop: {}, Genome: {}, Time: {:.2}ms, Best: {:.4}",
            result.population_size,
            result.genome_size,
            result.total_time_ms,
            result.final_best_fitness
        );
    }

    // Run specialized benchmarks
    benchmark.benchmark_mutation_strategies().await?;
    benchmark.benchmark_selection_methods().await?;
    benchmark.benchmark_scaling().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_evolution_benchmark_small() {
        let config = EvolutionBenchmarkConfig {
            population_sizes: vec![1024],
            genome_sizes: vec![64],
            generations: 10,
            ..Default::default()
        };

        if let Ok(mut benchmark) = GpuEvolutionBenchmark::new(0, config) {
            let results = benchmark.run_all().await.unwrap();
            assert!(!results.is_empty());
            assert!(results[0].total_time_ms > 0.0);
        }
    }
}
