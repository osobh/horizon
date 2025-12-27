//! Main ADAS engine implementation  

use super::config::AdasConfig;
use crate::{
    adas_meta_agent::{DiscoveredWorkflow, MetaAgent},
    error::{EvolutionEngineError, EvolutionEngineResult},
    metrics::{EvolutionMetrics, MetricsCollector},
    population::{Individual, Population},
    traits::{
        AgentGenome, ArchitectureGenes, BehaviorGenes, EngineConfig, EvolutionEngine, Evolvable,
        EvolvableAgent,
    },
};
use async_trait::async_trait;
use futures::future::join_all;
use rayon::prelude::*;
use stratoswarm_agent_core::{Agent, AgentConfig, Goal, GoalPriority};
use stratoswarm_synthesis::interpreter::GoalInterpreter;
use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

/// ADAS evolution engine
pub struct AdasEngine {
    pub(crate) config: AdasConfig,
    pub(crate) metrics_collector: MetricsCollector,
    pub(crate) rng: Arc<RwLock<StdRng>>,
    pub(crate) goal_interpreter: Arc<GoalInterpreter>,
    pub(crate) architecture_performance: Arc<RwLock<Vec<(ArchitectureGenes, f64)>>>,
    pub(crate) meta_agent: MetaAgent,
    pub(crate) discovered_workflows: Vec<DiscoveredWorkflow>,
}

impl AdasEngine {
    /// Create new ADAS engine
    pub fn new(config: AdasConfig) -> EvolutionEngineResult<Self> {
        config.validate()?;

        let rng = if let Some(seed) = config.base.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        Ok(Self {
            config: config.clone(),
            metrics_collector: MetricsCollector::new(),
            rng: Arc::new(RwLock::new(rng)),
            goal_interpreter: Arc::new(GoalInterpreter::new(
                stratoswarm_synthesis::interpreter::InterpreterConfig::default(),
            )),
            architecture_performance: Arc::new(RwLock::new(Vec::new())),
            meta_agent: MetaAgent::new(config.base.max_generations as usize),
            discovered_workflows: Vec::new(),
        })
    }

    /// Generate random architecture genes
    pub fn random_architecture(&self) -> ArchitectureGenes {
        let mut rng = self.rng.write();
        let space = &self.config.architecture_space;

        let depth = rng.gen_range(space.network_depth_range.0..=space.network_depth_range.1);
        let mut topology = Vec::with_capacity(depth);

        for _ in 0..depth {
            topology.push(rng.gen_range(space.network_width_range.0..=space.network_width_range.1));
        }

        ArchitectureGenes {
            memory_capacity: rng
                .gen_range(space.memory_capacity_range.0..=space.memory_capacity_range.1),
            processing_units: rng
                .gen_range(space.processing_units_range.0..=space.processing_units_range.1),
            network_topology: topology,
        }
    }

    /// Generate random behavior genes
    pub fn random_behavior(&self) -> BehaviorGenes {
        let mut rng = self.rng.write();
        let space = &self.config.behavior_space;

        BehaviorGenes {
            exploration_rate: rng
                .gen_range(space.exploration_rate_range.0..=space.exploration_rate_range.1),
            learning_rate: rng.gen_range(space.learning_rate_range.0..=space.learning_rate_range.1),
            risk_tolerance: rng
                .gen_range(space.risk_tolerance_range.0..=space.risk_tolerance_range.1),
        }
    }

    /// Get all discovered workflows from the meta agent search
    pub fn get_discovered_workflows(&self) -> &[DiscoveredWorkflow] {
        &self.discovered_workflows
    }

    /// Get the current meta agent archive statistics
    pub fn get_meta_agent_stats(&self) -> (usize, usize, f64) {
        let archive = &self.meta_agent.archive;
        let total_workflows = archive.workflows.len();
        let current_iteration = self.meta_agent.current_iteration;
        let best_performance = archive
            .get_best_workflow()
            .map(|w| w.performance_metrics.success_rate)
            .unwrap_or(0.0);

        (total_workflows, current_iteration, best_performance)
    }

    /// Generate agentic system code from discovered workflow
    pub fn generate_agentic_system_code(&self, workflow: &DiscoveredWorkflow) -> String {
        format!(
            "# ADAS-Generated Agentic System: {}\n\n{}",
            workflow.name, workflow.code_implementation
        )
    }

    /// Mutate architecture genes
    pub fn mutate_architecture(&self, genes: &ArchitectureGenes) -> ArchitectureGenes {
        let mut rng = self.rng.write();
        let mut mutated = genes.clone();
        let space = &self.config.architecture_space;

        // Mutate memory capacity
        if rng.gen_bool(0.3) {
            let delta = rng.gen_range(-0.2..=0.2);
            let new_capacity = (mutated.memory_capacity as f64 * (1.0 + delta)) as usize;
            mutated.memory_capacity =
                new_capacity.clamp(space.memory_capacity_range.0, space.memory_capacity_range.1);
        }

        // Mutate processing units
        if rng.gen_bool(0.3) {
            let delta = rng.gen_range(-2..=2);
            mutated.processing_units = (mutated.processing_units as i32 + delta).clamp(
                space.processing_units_range.0 as i32,
                space.processing_units_range.1 as i32,
            ) as u32;
        }

        // Mutate network topology
        if rng.gen_bool(0.3) {
            // Add or remove layer
            if rng.gen_bool(0.5) && mutated.network_topology.len() < space.network_depth_range.1 {
                let width =
                    rng.gen_range(space.network_width_range.0..=space.network_width_range.1);
                let pos = rng.gen_range(0..=mutated.network_topology.len());
                mutated.network_topology.insert(pos, width);
            } else if mutated.network_topology.len() > space.network_depth_range.0 {
                let pos = rng.gen_range(0..mutated.network_topology.len());
                mutated.network_topology.remove(pos);
            }
        }

        // Mutate layer widths
        for width in &mut mutated.network_topology {
            if rng.gen_bool(0.2) {
                let delta = rng.gen_range(-0.2..=0.2);
                let new_width = (*width as f64 * (1.0 + delta)) as u32;
                *width = new_width.clamp(space.network_width_range.0, space.network_width_range.1);
            }
        }

        mutated
    }

    /// Mutate behavior genes
    pub fn mutate_behavior(&self, genes: &BehaviorGenes) -> BehaviorGenes {
        let mut rng = self.rng.write();
        let mut mutated = genes.clone();
        let space = &self.config.behavior_space;

        // Mutate exploration rate
        if rng.gen_bool(0.5) {
            let delta = rng.gen_range(-0.1..=0.1) * self.config.meta_learning_rate;
            mutated.exploration_rate = (mutated.exploration_rate + delta).clamp(
                space.exploration_rate_range.0,
                space.exploration_rate_range.1,
            );
        }

        // Mutate learning rate
        if rng.gen_bool(0.5) {
            let delta = rng.gen_range(-0.5..=0.5) * self.config.meta_learning_rate;
            mutated.learning_rate = (mutated.learning_rate * (1.0 + delta))
                .clamp(space.learning_rate_range.0, space.learning_rate_range.1);
        }

        // Mutate risk tolerance
        if rng.gen_bool(0.5) {
            let delta = rng.gen_range(-0.2..=0.2) * self.config.meta_learning_rate;
            mutated.risk_tolerance = (mutated.risk_tolerance + delta)
                .clamp(space.risk_tolerance_range.0, space.risk_tolerance_range.1);
        }

        mutated
    }

    /// Calculate population diversity using parallel pairwise comparisons
    pub(crate) fn calculate_diversity(&self, population: &[EvolvableAgent]) -> f64 {
        if population.len() < 2 {
            return 0.0;
        }

        // Generate all pairs for parallel processing
        let pairs: Vec<(usize, usize)> = (0..population.len())
            .flat_map(|i| ((i + 1)..population.len()).map(move |j| (i, j)))
            .collect();

        let comparisons = pairs.len();
        if comparisons == 0 {
            return 0.0;
        }

        // Parallel diversity calculation using rayon
        let total_diversity: f64 = pairs
            .par_iter()
            .map(|&(i, j)| {
                let genome1 = &population[i].genome;
                let genome2 = &population[j].genome;

                // Architecture diversity
                let arch_div =
                    self.architecture_distance(&genome1.architecture, &genome2.architecture);

                // Behavior diversity
                let behav_div = self.behavior_distance(&genome1.behavior, &genome2.behavior);

                (arch_div + behav_div) / 2.0
            })
            .sum();

        total_diversity / comparisons as f64
    }

    /// Calculate distance between architectures
    fn architecture_distance(&self, arch1: &ArchitectureGenes, arch2: &ArchitectureGenes) -> f64 {
        let memory_diff = (arch1.memory_capacity as f64 - arch2.memory_capacity as f64).abs()
            / self.config.architecture_space.memory_capacity_range.1 as f64;

        let units_diff = (arch1.processing_units as f64 - arch2.processing_units as f64).abs()
            / self.config.architecture_space.processing_units_range.1 as f64;

        let topology_diff = if arch1.network_topology.len() != arch2.network_topology.len() {
            0.5
        } else {
            let mut diff = 0.0;
            for (w1, w2) in arch1.network_topology.iter().zip(&arch2.network_topology) {
                diff += (*w1 as f64 - *w2 as f64).abs()
                    / self.config.architecture_space.network_width_range.1 as f64;
            }
            diff / arch1.network_topology.len() as f64
        };

        (memory_diff + units_diff + topology_diff) / 3.0
    }

    /// Calculate distance between behaviors
    fn behavior_distance(&self, behav1: &BehaviorGenes, behav2: &BehaviorGenes) -> f64 {
        let exploration_diff = (behav1.exploration_rate - behav2.exploration_rate).abs();
        let learning_diff = (behav1.learning_rate - behav2.learning_rate).abs() * 10.0; // Scale
        let risk_diff = (behav1.risk_tolerance - behav2.risk_tolerance).abs();

        (exploration_diff + learning_diff + risk_diff) / 3.0
    }
}

#[async_trait]
impl EvolutionEngine for AdasEngine {
    type Entity = EvolvableAgent;
    type Config = AdasConfig;

    async fn initialize(config: Self::Config) -> EvolutionEngineResult<Self> {
        AdasEngine::new(config)
    }

    async fn evolve_step(
        &mut self,
        mut population: Population<Self::Entity>,
    ) -> EvolutionEngineResult<Population<Self::Entity>> {
        self.metrics_collector.start_generation();

        // Evaluate fitness for all individuals in parallel using async concurrency
        // This leverages multiple CPU cores for fitness evaluation
        let fitness_futures: Vec<_> = population
            .individuals
            .iter()
            .map(|individual| individual.entity.evaluate_fitness())
            .collect();

        let fitness_results = join_all(fitness_futures).await;

        // Process results and update individuals
        let mut total_fitness = 0.0;
        let mut best_fitness = 0.0;
        let mut arch_performance_batch = Vec::with_capacity(population.individuals.len());

        for (individual, result) in population.individuals.iter_mut().zip(fitness_results) {
            let fitness = result?;
            total_fitness += fitness;
            best_fitness = f64::max(best_fitness, fitness);
            individual.fitness = Some(fitness);

            // Collect architecture performance for batch update
            arch_performance_batch.push((individual.entity.genome.architecture.clone(), fitness));
        }

        // Batch update architecture performance (single lock acquisition)
        self.architecture_performance
            .write()
            .extend(arch_performance_batch);

        // Sort by fitness
        population.individuals.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create next generation
        let mut next_generation_individuals = Vec::new();
        let elite_count = (self.config.base.population_size / 10).max(1); // 10% elitism, at least 1

        // Keep elite
        for i in 0..elite_count.min(population.individuals.len()) {
            next_generation_individuals
                .push(Individual::new(population.individuals[i].entity.clone()));
        }

        // Generate offspring
        while next_generation_individuals.len() < self.config.base.population_size {
            // Tournament selection
            let (parent1_idx, parent2_idx) = {
                let mut rng = self.rng.write();
                let p1 = rng.gen_range(0..population.individuals.len().min(elite_count * 2).max(1));
                let p2 = rng.gen_range(0..population.individuals.len().min(elite_count * 2).max(1));
                (p1, p2)
            };

            let parent1 = &population.individuals[parent1_idx].entity;
            let parent2 = &population.individuals[parent2_idx].entity;

            // Crossover
            let (child1, child2) = parent1.crossover(parent2).await?;

            // Mutate
            let mutated1 = child1.mutate(self.config.base.mutation_rate).await?;
            let mutated2 = child2.mutate(self.config.base.mutation_rate).await?;

            next_generation_individuals.push(Individual::new(mutated1));
            if next_generation_individuals.len() < self.config.base.population_size {
                next_generation_individuals.push(Individual::new(mutated2));
            }
        }

        // Update metrics
        let average_fitness = total_fitness / population.individuals.len() as f64;
        let entities: Vec<_> = next_generation_individuals
            .iter()
            .map(|i| i.entity.clone())
            .collect();
        let diversity = self.calculate_diversity(&entities);

        self.metrics_collector.end_generation(
            best_fitness,
            average_fitness,
            diversity,
            population.individuals.len() as u64,
        );

        let mut new_population = Population::from_individuals(next_generation_individuals);
        new_population.generation = population.generation + 1;

        Ok(new_population)
    }

    async fn generate_initial_population(
        &self,
        size: usize,
    ) -> EvolutionEngineResult<Population<Self::Entity>> {
        let mut individuals = Vec::new();

        for i in 0..size {
            let genome = AgentGenome {
                goal: Goal::new("Meta-evolve optimal agent".to_string(), GoalPriority::High),
                architecture: self.random_architecture(),
                behavior: self.random_behavior(),
            };

            let config = AgentConfig {
                name: format!("adas_agent_{i}"),
                agent_type: "adas".to_string(),
                max_memory: genome.architecture.memory_capacity,
                max_gpu_memory: genome.architecture.memory_capacity / 4,
                priority: 1,
                metadata: serde_json::Value::Null,
            };

            let agent =
                Agent::new(config).map_err(|e| EvolutionEngineError::InitializationError {
                    message: format!("Failed to create agent: {e}"),
                })?;
            let evolvable = EvolvableAgent { agent, genome };

            individuals.push(Individual::new(evolvable));
        }

        Ok(Population::from_individuals(individuals))
    }

    async fn should_terminate(&self, metrics: &EvolutionMetrics) -> bool {
        // Check generation limit
        if metrics.generation >= self.config.base.max_generations {
            return true;
        }

        // Check target fitness
        if let Some(target) = self.config.base.target_fitness {
            if metrics.best_fitness >= target {
                return true;
            }
        }

        // Check convergence
        if metrics.convergence_rate < 0.001 && metrics.generation > 20 {
            return true;
        }

        false
    }

    fn metrics(&self) -> &EvolutionMetrics {
        self.metrics_collector.metrics()
    }

    async fn adapt_parameters(&mut self, metrics: &EvolutionMetrics) -> EvolutionEngineResult<()> {
        if !self.config.base.adaptive_parameters {
            return Ok(());
        }

        // Adapt meta-learning rate based on convergence
        if metrics.convergence_rate < 0.01 {
            self.config.meta_learning_rate *= 1.1; // Increase exploration
        } else if metrics.convergence_rate > 0.1 {
            self.config.meta_learning_rate *= 0.9; // Decrease exploration
        }

        self.config.meta_learning_rate = self.config.meta_learning_rate.clamp(0.001, 0.1);

        Ok(())
    }
}
