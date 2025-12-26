//! Evolution performance benchmarking infrastructure

use std::time::{Duration, Instant};

use crate::{EvolutionEngine, GeneticEvolutionEngine, Individual, Population};

/// Evolution benchmark configuration
#[derive(Debug, Clone)]
pub struct EvolutionBenchmarkConfig {
    pub population_size: usize,
    pub num_generations: usize,
    pub genome_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elite_count: usize,
    pub warmup_generations: usize,
}

impl Default for EvolutionBenchmarkConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            num_generations: 50,
            genome_size: 1000,
            mutation_rate: 0.01,
            crossover_rate: 0.8,
            elite_count: 10,
            warmup_generations: 5,
        }
    }
}

/// Evolution benchmark results
#[derive(Debug, Clone)]
pub struct EvolutionBenchmarkResults {
    pub total_duration: Duration,
    pub generations_per_second: f64,
    pub avg_generation_time_ms: f64,
    pub min_generation_time_ms: f64,
    pub max_generation_time_ms: f64,
    pub p95_generation_time_ms: f64,
    pub p99_generation_time_ms: f64,
    pub fitness_improvement_rate: f64,
    pub convergence_generation: Option<usize>,
}

impl EvolutionBenchmarkResults {
    pub fn new(
        generation_durations: &[Duration],
        fitness_scores: &[f64],
        total_duration: Duration,
    ) -> Self {
        let mut sorted_durations = generation_durations.to_vec();
        sorted_durations.sort();

        let total_generations = generation_durations.len();
        let generations_per_second = total_generations as f64 / total_duration.as_secs_f64();

        let avg_generation_time_ms = generation_durations
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / total_generations as f64;

        let min_generation_time_ms = sorted_durations[0].as_secs_f64() * 1000.0;
        let max_generation_time_ms = sorted_durations[total_generations - 1].as_secs_f64() * 1000.0;

        let p95_idx = (total_generations as f64 * 0.95) as usize;
        let p99_idx = (total_generations as f64 * 0.99) as usize;
        let p95_generation_time_ms =
            sorted_durations[p95_idx.min(total_generations - 1)].as_secs_f64() * 1000.0;
        let p99_generation_time_ms =
            sorted_durations[p99_idx.min(total_generations - 1)].as_secs_f64() * 1000.0;

        // Calculate fitness improvement rate
        let fitness_improvement_rate = if fitness_scores.len() >= 2 {
            let first_fitness = fitness_scores[0];
            let last_fitness = fitness_scores[fitness_scores.len() - 1];
            if first_fitness > 0.0 {
                (last_fitness - first_fitness) / first_fitness
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Find convergence generation (when fitness improvement drops below threshold)
        let convergence_generation = Self::find_convergence_generation(fitness_scores, 0.001);

        Self {
            total_duration,
            generations_per_second,
            avg_generation_time_ms,
            min_generation_time_ms,
            max_generation_time_ms,
            p95_generation_time_ms,
            p99_generation_time_ms,
            fitness_improvement_rate,
            convergence_generation,
        }
    }

    fn find_convergence_generation(fitness_scores: &[f64], threshold: f64) -> Option<usize> {
        if fitness_scores.len() < 10 {
            return None;
        }

        for i in 10..fitness_scores.len() {
            let window = &fitness_scores[i - 10..i];
            let mut improvements = 0;

            for j in 1..window.len() {
                if (window[j] - window[j - 1]).abs() > threshold {
                    improvements += 1;
                }
            }

            if improvements < 2 {
                // Less than 2 significant improvements in last 10 generations
                return Some(i);
            }
        }

        None
    }
}

/// Evolution benchmark suite
pub struct EvolutionBenchmark {
    config: EvolutionBenchmarkConfig,
}

impl EvolutionBenchmark {
    pub fn new(config: EvolutionBenchmarkConfig) -> Self {
        Self { config }
    }

    /// Benchmark population initialization performance
    pub async fn benchmark_population_init(
        &self,
        engine: &GeneticEvolutionEngine,
    ) -> EvolutionBenchmarkResults {
        let mut durations = Vec::new();
        let mut fitness_scores = Vec::new();

        // Warmup
        for _i in 0..self.config.warmup_generations {
            let _population = engine
                .initialize_population(self.config.population_size)
                .await
                .unwrap();
        }

        // Actual benchmark
        let start_time = Instant::now();

        for _i in 0..self.config.num_generations {
            let init_start = Instant::now();
            let population = engine
                .initialize_population(self.config.population_size)
                .await
                .unwrap();
            durations.push(init_start.elapsed());

            // Calculate average fitness
            let avg_fitness = population
                .individuals
                .iter()
                .filter_map(|ind| ind.fitness())
                .sum::<f64>()
                / population.individuals.len() as f64;
            fitness_scores.push(avg_fitness);
        }

        let total_duration = start_time.elapsed();
        EvolutionBenchmarkResults::new(&durations, &fitness_scores, total_duration)
    }

    /// Benchmark fitness evaluation performance
    pub async fn benchmark_fitness_evaluation(
        &self,
        engine: &GeneticEvolutionEngine,
    ) -> EvolutionBenchmarkResults {
        let mut durations = Vec::new();
        let mut fitness_scores = Vec::new();

        // Create test population without evaluating fitness
        let population = Population::random(self.config.population_size, self.config.genome_size);

        // Warmup
        for _i in 0..self.config.warmup_generations {
            let _fitness_scores = engine.evaluate_fitness(&population).await.unwrap();
        }

        // Actual benchmark
        let start_time = Instant::now();

        for _gen in 0..self.config.num_generations {
            let eval_start = Instant::now();
            let fitness_scores_result = engine.evaluate_fitness(&population).await.unwrap();
            durations.push(eval_start.elapsed());

            let avg_fitness = fitness_scores_result
                .iter()
                .map(|score| score.value)
                .sum::<f64>()
                / fitness_scores_result.len() as f64;
            fitness_scores.push(avg_fitness);
        }

        let total_duration = start_time.elapsed();
        EvolutionBenchmarkResults::new(&durations, &fitness_scores, total_duration)
    }

    /// Benchmark evolution generation performance
    pub async fn benchmark_evolution_generation(
        &self,
        engine: &GeneticEvolutionEngine,
    ) -> EvolutionBenchmarkResults {
        let mut durations = Vec::new();
        let mut fitness_scores = Vec::new();

        // Create initial population
        let mut population = engine
            .initialize_population(self.config.population_size)
            .await
            .unwrap();

        // Warmup
        for _i in 0..self.config.warmup_generations {
            engine.evolve_generation(&mut population).await.unwrap();
        }

        // Actual benchmark
        let start_time = Instant::now();

        for _gen in 0..self.config.num_generations {
            let generation_start = Instant::now();
            engine.evolve_generation(&mut population).await.unwrap();
            durations.push(generation_start.elapsed());

            // Calculate average fitness
            let avg_fitness = population
                .individuals
                .iter()
                .filter_map(|ind| ind.fitness())
                .sum::<f64>()
                / population.individuals.len() as f64;
            fitness_scores.push(avg_fitness);
        }

        let total_duration = start_time.elapsed();
        EvolutionBenchmarkResults::new(&durations, &fitness_scores, total_duration)
    }

    /// Benchmark individual crossover performance
    pub async fn benchmark_crossover(&self) -> EvolutionBenchmarkResults {
        let mut durations = Vec::new();
        let mut fitness_scores = Vec::new();

        // Create test individuals
        let parent1 = Individual::new(self.config.genome_size);
        let parent2 = Individual::new(self.config.genome_size);

        // Warmup
        for _i in 0..self.config.warmup_generations {
            let _offspring = parent1.crossover(&parent2, self.config.crossover_rate);
        }

        // Actual benchmark
        let start_time = Instant::now();

        for _gen in 0..self.config.num_generations {
            let crossover_start = Instant::now();
            let offspring = parent1.crossover(&parent2, self.config.crossover_rate);
            durations.push(crossover_start.elapsed());

            // Simple fitness calculation for consistency
            let fitness = offspring.genome.iter().map(|&b| b as f64).sum::<f64>()
                / (255.0 * offspring.genome.len() as f64);
            fitness_scores.push(fitness);
        }

        let total_duration = start_time.elapsed();
        EvolutionBenchmarkResults::new(&durations, &fitness_scores, total_duration)
    }

    /// Benchmark individual mutation performance
    pub async fn benchmark_mutation(&self) -> EvolutionBenchmarkResults {
        let mut durations = Vec::new();
        let mut fitness_scores = Vec::new();

        // Create test individual
        let original_individual = Individual::new(self.config.genome_size);

        // Warmup
        for _i in 0..self.config.warmup_generations {
            let mut test_individual = original_individual.clone();
            test_individual.mutate(self.config.mutation_rate);
        }

        // Actual benchmark
        let start_time = Instant::now();

        for _gen in 0..self.config.num_generations {
            let mut test_individual = original_individual.clone();

            let mutate_start = Instant::now();
            test_individual.mutate(self.config.mutation_rate);
            durations.push(mutate_start.elapsed());

            // Simple fitness calculation for consistency
            let fitness = test_individual
                .genome
                .iter()
                .map(|&b| b as f64)
                .sum::<f64>()
                / (255.0 * test_individual.genome.len() as f64);
            fitness_scores.push(fitness);
        }

        let total_duration = start_time.elapsed();
        EvolutionBenchmarkResults::new(&durations, &fitness_scores, total_duration)
    }

    /// Benchmark complete multi-generation evolution
    pub async fn benchmark_multi_generation_evolution(
        &self,
        engine: &GeneticEvolutionEngine,
    ) -> EvolutionBenchmarkResults {
        let mut durations = Vec::new();
        let mut fitness_scores = Vec::new();

        // Create initial population
        let mut population = engine
            .initialize_population(self.config.population_size)
            .await
            .unwrap();

        // Warmup - run a few generations
        for _i in 0..self.config.warmup_generations {
            engine.evolve_generation(&mut population).await.unwrap();
        }

        // Reset population for actual benchmark
        population = engine
            .initialize_population(self.config.population_size)
            .await
            .unwrap();

        // Actual benchmark - time each generation
        let start_time = Instant::now();

        for _gen in 0..self.config.num_generations {
            let generation_start = Instant::now();
            engine.evolve_generation(&mut population).await.unwrap();
            durations.push(generation_start.elapsed());

            // Calculate average fitness
            let avg_fitness = population
                .individuals
                .iter()
                .filter_map(|ind| ind.fitness())
                .sum::<f64>()
                / population.individuals.len() as f64;
            fitness_scores.push(avg_fitness);
        }

        let total_duration = start_time.elapsed();
        EvolutionBenchmarkResults::new(&durations, &fitness_scores, total_duration)
    }

    /// Run comprehensive evolution benchmark suite
    pub async fn run_full_suite(
        &self,
        engine: &GeneticEvolutionEngine,
    ) -> FullEvolutionBenchmarkResults {
        println!("Running evolution benchmark suite...");

        println!("  Running population initialization benchmark...");
        let init_results = self.benchmark_population_init(engine).await;

        println!("  Running fitness evaluation benchmark...");
        let fitness_results = self.benchmark_fitness_evaluation(engine).await;

        println!("  Running evolution generation benchmark...");
        let generation_results = self.benchmark_evolution_generation(engine).await;

        println!("  Running crossover benchmark...");
        let crossover_results = self.benchmark_crossover().await;

        println!("  Running mutation benchmark...");
        let mutation_results = self.benchmark_mutation().await;

        println!("  Running multi-generation evolution benchmark...");
        let evolution_results = self.benchmark_multi_generation_evolution(engine).await;

        FullEvolutionBenchmarkResults {
            init_results,
            fitness_results,
            generation_results,
            crossover_results,
            mutation_results,
            evolution_results,
            config: self.config.clone(),
        }
    }
}

/// Complete evolution benchmark results
#[derive(Debug, Clone)]
pub struct FullEvolutionBenchmarkResults {
    pub init_results: EvolutionBenchmarkResults,
    pub fitness_results: EvolutionBenchmarkResults,
    pub generation_results: EvolutionBenchmarkResults,
    pub crossover_results: EvolutionBenchmarkResults,
    pub mutation_results: EvolutionBenchmarkResults,
    pub evolution_results: EvolutionBenchmarkResults,
    pub config: EvolutionBenchmarkConfig,
}

impl FullEvolutionBenchmarkResults {
    /// Print formatted benchmark results
    pub fn print_results(&self) {
        println!("\n========== Evolution Benchmark Results ==========");
        println!("Configuration:");
        println!("  Population size: {}", self.config.population_size);
        println!("  Generations: {}", self.config.num_generations);
        println!("  Genome size: {}", self.config.genome_size);
        println!("  Mutation rate: {:.3}", self.config.mutation_rate);
        println!("  Crossover rate: {:.3}", self.config.crossover_rate);
        println!();

        self.print_single_result("Population Initialization", &self.init_results);
        self.print_single_result("Fitness Evaluation", &self.fitness_results);
        self.print_single_result("Evolution Generation", &self.generation_results);
        self.print_single_result("Crossover", &self.crossover_results);
        self.print_single_result("Mutation", &self.mutation_results);
        self.print_single_result("Multi-Generation Evolution", &self.evolution_results);

        println!("================================================\n");
    }

    fn print_single_result(&self, name: &str, results: &EvolutionBenchmarkResults) {
        println!("{}:", name);
        println!("  Generations/sec: {:.2}", results.generations_per_second);
        println!(
            "  Avg time: {:.2}ms, P95: {:.2}ms, P99: {:.2}ms",
            results.avg_generation_time_ms,
            results.p95_generation_time_ms,
            results.p99_generation_time_ms
        );
        println!(
            "  Fitness improvement: {:.2}%",
            results.fitness_improvement_rate * 100.0
        );
        if let Some(conv_gen) = results.convergence_generation {
            println!("  Convergence at generation: {}", conv_gen);
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_evolution_benchmark_config_default() {
        let config = EvolutionBenchmarkConfig::default();
        assert_eq!(config.population_size, 100);
        assert_eq!(config.num_generations, 50);
        assert_eq!(config.genome_size, 1000);
        assert_eq!(config.mutation_rate, 0.01);
        assert_eq!(config.crossover_rate, 0.8);
        assert_eq!(config.elite_count, 10);
        assert_eq!(config.warmup_generations, 5);
    }

    #[tokio::test]
    async fn test_evolution_benchmark_results_calculation() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(15),
            Duration::from_millis(12),
            Duration::from_millis(18),
            Duration::from_millis(14),
        ];
        let fitness_scores = vec![0.1, 0.2, 0.25, 0.3, 0.32];
        let total_duration = Duration::from_millis(100);

        let results = EvolutionBenchmarkResults::new(&durations, &fitness_scores, total_duration);

        assert!(results.generations_per_second > 0.0);
        assert!(results.avg_generation_time_ms > 0.0);
        assert!(results.min_generation_time_ms <= results.max_generation_time_ms);
        assert!(results.p95_generation_time_ms >= results.avg_generation_time_ms);
        assert!(results.fitness_improvement_rate > 0.0); // Should show improvement
    }

    #[tokio::test]
    async fn test_convergence_detection() {
        // Test with clearly converging fitness scores - very stable at the end
        let converging_scores = vec![
            0.1, 0.2, 0.25, 0.3, 0.32, 0.33, 0.34, 0.35, 0.351, 0.351, 0.351, 0.351, 0.351, 0.351,
            0.351, 0.351, 0.351, 0.351,
        ];
        let convergence =
            EvolutionBenchmarkResults::find_convergence_generation(&converging_scores, 0.01);
        assert!(convergence.is_some());

        // Test with non-converging fitness scores
        let non_converging_scores = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let no_convergence =
            EvolutionBenchmarkResults::find_convergence_generation(&non_converging_scores, 0.001);
        assert!(no_convergence.is_none());

        // Test with too short sequence
        let short_scores = vec![0.1, 0.2, 0.3];
        let no_convergence_short =
            EvolutionBenchmarkResults::find_convergence_generation(&short_scores, 0.001);
        assert!(no_convergence_short.is_none());
    }

    #[tokio::test]
    async fn test_population_init_benchmark() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);
        let config = EvolutionBenchmarkConfig {
            population_size: 10,
            num_generations: 5,
            genome_size: 100,
            warmup_generations: 1,
            ..Default::default()
        };

        let benchmark = EvolutionBenchmark::new(config);
        let results = benchmark.benchmark_population_init(&engine).await;

        assert!(results.generations_per_second > 0.0);
        assert!(results.avg_generation_time_ms >= 0.0);
        assert!(results.min_generation_time_ms <= results.max_generation_time_ms);
    }

    #[tokio::test]
    async fn test_fitness_evaluation_benchmark() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);
        let config = EvolutionBenchmarkConfig {
            population_size: 10,
            num_generations: 5,
            genome_size: 100,
            warmup_generations: 1,
            ..Default::default()
        };

        let benchmark = EvolutionBenchmark::new(config);
        let results = benchmark.benchmark_fitness_evaluation(&engine).await;

        assert!(results.generations_per_second > 0.0);
        assert!(results.avg_generation_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_evolution_generation_benchmark() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);
        let config = EvolutionBenchmarkConfig {
            population_size: 10,
            num_generations: 5,
            genome_size: 100,
            warmup_generations: 1,
            ..Default::default()
        };

        let benchmark = EvolutionBenchmark::new(config);
        let results = benchmark.benchmark_evolution_generation(&engine).await;

        assert!(results.generations_per_second > 0.0);
        assert!(results.avg_generation_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_crossover_benchmark() {
        let config = EvolutionBenchmarkConfig {
            population_size: 10,
            num_generations: 5,
            genome_size: 100,
            warmup_generations: 1,
            ..Default::default()
        };

        let benchmark = EvolutionBenchmark::new(config);
        let results = benchmark.benchmark_crossover().await;

        assert!(results.generations_per_second > 0.0);
        assert!(results.avg_generation_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_mutation_benchmark() {
        let config = EvolutionBenchmarkConfig {
            population_size: 10,
            num_generations: 5,
            genome_size: 100,
            warmup_generations: 1,
            ..Default::default()
        };

        let benchmark = EvolutionBenchmark::new(config);
        let results = benchmark.benchmark_mutation().await;

        assert!(results.generations_per_second > 0.0);
        assert!(results.avg_generation_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_multi_generation_evolution_benchmark() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);
        let config = EvolutionBenchmarkConfig {
            population_size: 10,
            num_generations: 5,
            genome_size: 50,
            warmup_generations: 1,
            ..Default::default()
        };

        let benchmark = EvolutionBenchmark::new(config);
        let results = benchmark
            .benchmark_multi_generation_evolution(&engine)
            .await;

        assert!(results.generations_per_second > 0.0);
        assert!(results.avg_generation_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_full_benchmark_suite() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);
        let config = EvolutionBenchmarkConfig {
            population_size: 5,
            num_generations: 3,
            genome_size: 50,
            warmup_generations: 1,
            ..Default::default()
        };

        let benchmark = EvolutionBenchmark::new(config);
        let results = benchmark.run_full_suite(&engine).await;

        assert!(results.init_results.generations_per_second > 0.0);
        assert!(results.fitness_results.generations_per_second > 0.0);
        assert!(results.generation_results.generations_per_second > 0.0);
        assert!(results.crossover_results.generations_per_second > 0.0);
        assert!(results.mutation_results.generations_per_second > 0.0);
        assert!(results.evolution_results.generations_per_second > 0.0);

        // Test print functionality (just ensure it doesn't panic)
        results.print_results();
    }

    #[tokio::test]
    async fn test_performance_comparison() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);

        let small_config = EvolutionBenchmarkConfig {
            population_size: 5,
            num_generations: 3,
            genome_size: 10,
            warmup_generations: 1,
            ..Default::default()
        };

        let large_config = EvolutionBenchmarkConfig {
            population_size: 20,
            num_generations: 3,
            genome_size: 100,
            warmup_generations: 1,
            ..Default::default()
        };

        let small_benchmark = EvolutionBenchmark::new(small_config);
        let large_benchmark = EvolutionBenchmark::new(large_config);

        let small_results = small_benchmark.benchmark_population_init(&engine).await;
        let large_results = large_benchmark.benchmark_population_init(&engine).await;

        // Both should have valid results
        assert!(small_results.generations_per_second > 0.0);
        assert!(large_results.generations_per_second > 0.0);

        // Smaller problems should generally be faster per generation
        println!(
            "Small population: {:.2} gen/sec, {:.2}ms avg",
            small_results.generations_per_second, small_results.avg_generation_time_ms
        );
        println!(
            "Large population: {:.2} gen/sec, {:.2}ms avg",
            large_results.generations_per_second, large_results.avg_generation_time_ms
        );
    }

    #[tokio::test]
    async fn test_fitness_improvement_calculation() {
        let durations = vec![Duration::from_millis(10); 5];

        // Test improvement scenario
        let improving_scores = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let improving_results = EvolutionBenchmarkResults::new(
            &durations,
            &improving_scores,
            Duration::from_millis(50),
        );
        assert!(improving_results.fitness_improvement_rate > 0.0);

        // Test declining scenario
        let declining_scores = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        let declining_results = EvolutionBenchmarkResults::new(
            &durations,
            &declining_scores,
            Duration::from_millis(50),
        );
        assert!(declining_results.fitness_improvement_rate < 0.0);

        // Test stagnant scenario
        let stagnant_scores = vec![0.3, 0.3, 0.3, 0.3, 0.3];
        let stagnant_results =
            EvolutionBenchmarkResults::new(&durations, &stagnant_scores, Duration::from_millis(50));
        assert_eq!(stagnant_results.fitness_improvement_rate, 0.0);
    }
}
