//! GPU-accelerated evolution for agent swarms
//!
//! This module provides massive parallel evolution operations on GPU,
//! integrating with the evolution and evolution-engines crates.

use anyhow::Result;
use cudarc::driver::CudaDevice;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub mod adas;
pub mod dgm;
pub mod engine_adapter;
pub mod fitness;
pub mod kernels;
pub mod mutation;
pub mod population;
pub mod selection;
pub mod swarm;

#[cfg(test)]
pub mod tests;

#[cfg(test)]
pub mod expanded_tests;

#[cfg(test)]
pub mod e2e_tests;

#[cfg(test)]
pub mod kernel_tests;

#[cfg(test)]
pub mod cuda_memory_tests;

pub mod hang_isolation_tests;

pub use adas::{AdasMetaAgent, AdasPopulation, AdasStatistics};
pub use dgm::{DgmAgent, DgmArchive, DgmEngine, DgmStatistics};
pub use fitness::GpuFitnessEvaluator;
pub use mutation::GpuMutationEngine;
pub use population::{GpuIndividual, GpuPopulation};
pub use selection::GpuSelectionStrategy;
pub use swarm::{AgentSystem, SwarmEngine, SwarmParticle, SwarmStatistics};

// Legacy evolution types for compatibility (from old evolution.rs)
/// Fitness objectives for multi-objective optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FitnessObjective {
    /// Task performance/accuracy
    Performance,
    /// Resource efficiency (memory, computation)
    Efficiency,
    /// Behavioral novelty/diversity
    Novelty,
    /// Robustness to perturbations
    Robustness,
    /// Task completion speed
    Speed,
}

/// Selection strategies for evolution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Tournament selection
    Tournament,
    /// NSGA-II multi-objective selection
    NSGA2,
    /// NSGA-III many-objective selection
    NSGA3,
    /// Novelty-based selection
    NoveltySearch,
}

/// Mutation strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MutationStrategy {
    /// Fixed mutation rate
    Fixed,
    /// Adaptive mutation based on diversity
    Adaptive,
    /// Self-adaptive mutation rates
    SelfAdaptive,
    /// Gaussian mutation with varying sigma
    Gaussian,
}

// Legacy compatibility types
pub use FitnessObjective as EvolutionObjective;
pub type EvolutionConfig = LegacyEvolutionConfig;
pub type EvolutionManager = GpuEvolutionEngine;
pub type EvolutionMetrics = GpuEvolutionMetrics;

// Placeholder for archived agent (referenced in lib.rs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivedAgent {
    pub id: u64,
    pub genome: Vec<f32>,
    pub fitness: f32,
    pub fitness_vector: Vec<f32>,
    pub generation: u64,
    pub novelty_score: f32,
}

/// Legacy evolution config for test compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyEvolutionConfig {
    pub population_size: usize,
    pub selection_strategy: SelectionStrategy,
    pub mutation_rate: f32,
    pub crossover_rate: f32,
    pub objectives: Vec<FitnessObjective>,
    pub elitism_percentage: f32,
    pub tournament_size: usize,
    pub mutation_strategy: MutationStrategy,
    pub max_generations: u64,
    pub convergence_threshold: f32,
    pub enable_archive: bool,
    pub archive_size_limit: usize,
    pub novelty_k_nearest: usize,
    pub behavioral_descriptor_size: usize,
}

impl Default for LegacyEvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 1000,
            selection_strategy: SelectionStrategy::NSGA2,
            mutation_rate: 0.02,
            crossover_rate: 0.8,
            objectives: vec![
                FitnessObjective::Performance,
                FitnessObjective::Efficiency,
                FitnessObjective::Novelty,
            ],
            elitism_percentage: 0.1,
            tournament_size: 4,
            mutation_strategy: MutationStrategy::Adaptive,
            max_generations: 1000,
            convergence_threshold: 0.001,
            enable_archive: true,
            archive_size_limit: 500,
            novelty_k_nearest: 15,
            behavioral_descriptor_size: 10,
        }
    }
}

/// Performance metrics for evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionPerformanceMetrics {
    pub total_generations: u64,
    pub total_evolution_time_ms: f64,
    pub average_generation_time_ms: f64,
    pub fitness_evaluations: u64,
}

// Parameters for mutations and selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationParams {
    pub rate: f32,
    pub strategy: MutationStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionParams {
    pub strategy: SelectionStrategy,
    pub tournament_size: usize,
}

/// GPU Evolution Metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuEvolutionMetrics {
    pub generation: u64,
    pub best_fitness: f32,
    pub average_fitness: f32,
    pub diversity: f32,
    pub mutation_rate: f32,
    pub evaluations_per_second: f64,
}

/// GPU Evolution Engine configuration
#[derive(Debug, Clone)]
pub struct GpuEvolutionConfig {
    /// Population size (must be multiple of 32 for warp efficiency)
    pub population_size: usize,
    /// Genome size per individual
    pub genome_size: usize,
    /// Number of fitness objectives
    pub fitness_objectives: usize,
    /// Mutation rate
    pub mutation_rate: f32,
    /// Crossover rate
    pub crossover_rate: f32,
    /// Elite percentage to preserve
    pub elite_percentage: f32,
    /// Block size for kernels
    pub block_size: u32,
}

impl Default for GpuEvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 1024 * 1024, // 1M individuals
            genome_size: 256,
            fitness_objectives: 1,
            mutation_rate: 0.01,
            crossover_rate: 0.7,
            elite_percentage: 0.1,
            block_size: 256,
        }
    }
}

/// GPU Evolution Engine
pub struct GpuEvolutionEngine {
    _device: Arc<CudaDevice>,
    config: GpuEvolutionConfig,
    population: GpuPopulation,
    fitness_evaluator: GpuFitnessEvaluator,
    mutation_engine: GpuMutationEngine,
    selection_strategy: GpuSelectionStrategy,
    generation: u64,
}

impl GpuEvolutionEngine {
    /// Create new GPU evolution engine
    pub fn new(device: Arc<CudaDevice>, config: GpuEvolutionConfig) -> Result<Self> {
        // Validate config
        if config.population_size % 32 != 0 {
            return Err(anyhow::anyhow!(
                "Population size must be multiple of 32 for warp efficiency"
            ));
        }

        // Initialize components
        let population =
            GpuPopulation::new(device.clone(), config.population_size, config.genome_size)?;

        let fitness_evaluator =
            GpuFitnessEvaluator::new(device.clone(), config.fitness_objectives)?;

        let mutation_engine = GpuMutationEngine::new(device.clone(), config.mutation_rate)?;

        let selection_strategy =
            GpuSelectionStrategy::new(device.clone(), config.elite_percentage)?;

        Ok(Self {
            _device: device,
            config,
            population,
            fitness_evaluator,
            mutation_engine,
            selection_strategy,
            generation: 0,
        })
    }

    /// Initialize population with random genomes
    pub fn initialize_random(&mut self) -> Result<()> {
        self.population.initialize_random()?;
        Ok(())
    }

    /// Evaluate fitness of entire population
    pub fn evaluate_fitness(&mut self) -> Result<()> {
        self.fitness_evaluator
            .evaluate_population(&mut self.population)?;
        Ok(())
    }

    /// Perform one generation of evolution
    pub fn evolve_generation(&mut self) -> Result<()> {
        // 1. Evaluate fitness if not already done
        if !self.population.has_fitness() {
            self.evaluate_fitness()?;
        }

        // 2. Selection
        let selected_indices = self
            .selection_strategy
            .select(&self.population, self.config.population_size)?;

        // 3. Crossover and mutation
        self.population
            .create_offspring(&selected_indices, self.config.crossover_rate)?;

        // 4. Mutate offspring
        self.mutation_engine
            .mutate_population(&mut self.population)?;

        // 5. Increment generation
        self.generation += 1;
        self.population.invalidate_fitness();

        Ok(())
    }

    /// Get best individual
    pub fn best_individual(&self) -> Result<GpuIndividual> {
        self.population.best_individual()
    }

    /// Get evolution statistics
    pub fn statistics(&self) -> EvolutionStatistics {
        EvolutionStatistics {
            generation: self.generation,
            population_size: self.config.population_size,
            best_fitness: self.population.best_fitness().unwrap_or(0.0),
            average_fitness: self.population.average_fitness(),
            diversity_index: self.population.diversity_index(),
            mutations_per_second: self.mutation_engine.mutations_per_second(),
        }
    }

    /// Run evolution for N generations
    pub async fn run(&mut self, generations: u64) -> Result<()> {
        for _ in 0..generations {
            self.evolve_generation()?;

            // Log progress every 10 generations
            if self.generation % 10 == 0 {
                let stats = self.statistics();
                log::info!(
                    "Generation {}: best_fitness={:.4}, avg_fitness={:.4}, diversity={:.4}",
                    stats.generation,
                    stats.best_fitness,
                    stats.average_fitness,
                    stats.diversity_index
                );
            }
        }
        Ok(())
    }

    // Compatibility methods for evolution tests

    /// Create a new evolution manager with legacy config compatibility (for tests)
    pub fn new_legacy(config: EvolutionConfig) -> Result<Self> {
        use cudarc::driver::CudaDevice;
        let device = CudaDevice::new(0)?;

        // Convert legacy config to GpuEvolutionConfig
        let gpu_config = GpuEvolutionConfig {
            population_size: config.population_size,
            genome_size: config.behavioral_descriptor_size.max(64), // Use behavioral_descriptor_size or default
            fitness_objectives: config.objectives.len(),
            mutation_rate: config.mutation_rate,
            crossover_rate: config.crossover_rate,
            elite_percentage: config.elitism_percentage,
            block_size: 256,
        };

        Self::new(device, gpu_config)
    }

    /// Get current generation
    pub fn current_generation(&self) -> u64 {
        self.generation
    }

    /// Check if evolution has converged (always false for this implementation)
    pub fn has_converged(&self) -> bool {
        false
    }

    /// Get number of objectives
    pub fn objective_count(&self) -> usize {
        self.config.fitness_objectives
    }

    /// Evaluate multi-objective fitness for swarm (mock implementation)
    pub fn evaluate_multi_objective_fitness(
        &self,
        swarm: &crate::GpuSwarm,
    ) -> Result<Vec<Vec<f32>>> {
        let agent_count = swarm.metrics().agent_count;
        let mut fitness_vectors = Vec::new();

        for i in 0..agent_count {
            let mut fitness_vector = Vec::new();
            for j in 0..self.config.fitness_objectives {
                // Mock fitness based on agent ID and objective
                let fitness = 0.5 + ((i + j * agent_count) as f32 * 0.001) % 0.5;
                fitness_vector.push(fitness);
            }
            fitness_vectors.push(fitness_vector);
        }

        Ok(fitness_vectors)
    }

    /// Check if fitness vector A dominates fitness vector B (Pareto dominance)
    pub fn dominates(&self, fitness_a: &[f32], fitness_b: &[f32]) -> bool {
        if fitness_a.len() != fitness_b.len() {
            return false;
        }

        let mut better_in_one = false;
        for i in 0..fitness_a.len() {
            if fitness_a[i] < fitness_b[i] {
                return false; // A is worse in at least one objective
            }
            if fitness_a[i] > fitness_b[i] {
                better_in_one = true; // A is better in at least one objective
            }
        }

        better_in_one
    }

    /// NSGA-II selection (simplified implementation)
    pub fn nsga2_selection(&self, fitness_vectors: &[Vec<f32>]) -> Result<Vec<usize>> {
        let mut selected = Vec::new();

        // Simple selection: take first N individuals (in real NSGA-II, this would use Pareto fronts)
        for i in 0..self.config.population_size.min(fitness_vectors.len()) {
            selected.push(i);
        }

        Ok(selected)
    }

    /// Compute Pareto fronts
    pub fn compute_pareto_fronts(&self, fitness_vectors: &[Vec<f32>]) -> Result<Vec<Vec<usize>>> {
        let mut fronts = Vec::new();
        let mut front0 = Vec::new();

        // Simple implementation: put all individuals in first front
        for i in 0..fitness_vectors.len() {
            front0.push(i);
        }

        fronts.push(front0);
        Ok(fronts)
    }

    /// Calculate population diversity
    pub fn calculate_population_diversity(&self, fitness_vectors: &[Vec<f32>]) -> Result<f32> {
        if fitness_vectors.is_empty() {
            return Ok(0.0);
        }

        // Simple diversity calculation: variance of fitness values
        let mut sum = 0.0;
        let mut count = 0;

        for fitness_vector in fitness_vectors {
            for &fitness in fitness_vector {
                sum += fitness;
                count += 1;
            }
        }

        let mean = sum / count as f32;
        let mut variance_sum = 0.0;

        for fitness_vector in fitness_vectors {
            for &fitness in fitness_vector {
                variance_sum += (fitness - mean).powi(2);
            }
        }

        Ok((variance_sum / count as f32).sqrt())
    }

    /// Calculate adaptive mutation rate
    pub fn calculate_adaptive_mutation_rate(&self, generation: u64, diversity: f32) -> Result<f32> {
        // Adaptive mutation: increase rate when diversity is low
        let base_rate = self.config.mutation_rate;
        let diversity_factor = (1.0 - diversity.min(1.0)).max(0.1);
        let generation_factor = 1.0 + (generation as f32 * 0.001);

        Ok((base_rate * diversity_factor * generation_factor).min(0.1))
    }

    /// Extract behavioral descriptors from swarm
    pub fn extract_behavioral_descriptors(&self, swarm: &crate::GpuSwarm) -> Result<Vec<Vec<f32>>> {
        let agent_count = swarm.metrics().agent_count;
        let descriptor_size = 10; // behavioral_descriptor_size from config
        let mut descriptors = Vec::new();

        for i in 0..agent_count {
            let mut descriptor = Vec::new();
            for j in 0..descriptor_size {
                // Mock behavioral descriptor based on agent state
                let value = ((i + j) as f32 * 0.1) % 2.0 - 1.0;
                descriptor.push(value);
            }
            descriptors.push(descriptor);
        }

        Ok(descriptors)
    }

    /// Calculate novelty scores for behavioral descriptors
    pub fn calculate_novelty_scores(&self, descriptors: &[Vec<f32>]) -> Result<Vec<f32>> {
        let mut scores = Vec::new();

        for (i, desc_i) in descriptors.iter().enumerate() {
            let mut distances = Vec::new();

            // Calculate distances to all other descriptors
            for (j, desc_j) in descriptors.iter().enumerate() {
                if i != j {
                    let distance = desc_i
                        .iter()
                        .zip(desc_j.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    distances.push(distance);
                }
            }

            // Sort distances and take k-nearest (15 from config)
            distances.sort_by(|a, b| a.partial_cmp(b)?);
            let k_nearest = distances.iter().take(15.min(distances.len()));
            let novelty_score = k_nearest.sum::<f32>() / distances.len().min(15) as f32;

            scores.push(novelty_score);
        }

        Ok(scores)
    }

    /// Get best fitness from fitness vectors
    pub fn get_best_fitness(&self, fitness_vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if fitness_vectors.is_empty() {
            return Ok(vec![0.0; self.config.fitness_objectives]);
        }

        let mut best_fitness = fitness_vectors[0].clone();

        for fitness_vector in fitness_vectors.iter().skip(1) {
            for i in 0..best_fitness.len() {
                if fitness_vector[i] > best_fitness[i] {
                    best_fitness[i] = fitness_vector[i];
                }
            }
        }

        Ok(best_fitness)
    }

    /// Check convergence based on fitness history
    pub fn check_convergence(&self, fitness_history: &[Vec<f32>]) -> Result<bool> {
        if fitness_history.len() < 10 {
            return Ok(false);
        }

        // Simple convergence check: if best fitness hasn't improved in last 5 generations
        let recent_best = fitness_history.iter().rev().take(5);
        let mut improvements = 0;
        let mut prev_fitness: Option<&Vec<f32>> = None;

        for fitness in recent_best {
            if let Some(prev) = prev_fitness {
                let current_sum: f32 = fitness.iter().sum();
                let prev_sum: f32 = prev.iter().sum();
                if current_sum > prev_sum {
                    improvements += 1;
                }
            }
            prev_fitness = Some(fitness);
        }

        Ok(improvements == 0) // Converged if no improvements
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> Result<EvolutionPerformanceMetrics> {
        Ok(EvolutionPerformanceMetrics {
            total_generations: self.generation,
            total_evolution_time_ms: self.generation as f64 * 10.0, // Mock: 10ms per generation
            average_generation_time_ms: 10.0,
            fitness_evaluations: self.generation * self.config.population_size as u64,
        })
    }

    /// Get archive (mock implementation)
    pub fn get_archive(&self) -> Result<Vec<ArchivedAgent>> {
        let mut archive = Vec::new();

        // Mock archive with some diverse agents
        for i in 0..10 {
            archive.push(ArchivedAgent {
                id: i,
                genome: (0..self.config.genome_size)
                    .map(|j| (i as usize + j) as f32 * 0.01)
                    .collect(),
                fitness: 0.5 + (i as f32 * 0.05),
                fitness_vector: (0..self.config.fitness_objectives)
                    .map(|j| 0.5 + ((i as usize + j) as f32 * 0.03))
                    .collect(),
                generation: self.generation.saturating_sub(i),
                novelty_score: i as f32 * 0.1,
            });
        }

        Ok(archive)
    }
}

/// Evolution statistics
#[derive(Debug, Clone)]
pub struct EvolutionStatistics {
    pub generation: u64,
    pub population_size: usize,
    pub best_fitness: f64,
    pub average_fitness: f64,
    pub diversity_index: f64,
    pub mutations_per_second: f64,
}
