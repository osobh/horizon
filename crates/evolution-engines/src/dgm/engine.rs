//! DGM evolution engine implementation

use crate::{
    dgm_self_assessment::{
        AssessmentReport, BenchmarkResults, DgmSelfAssessment, ModificationType, SelfModification,
    },
    error::{EvolutionEngineError, EvolutionEngineResult},
    metrics::{EvolutionMetrics, MetricsCollector},
    population::{Individual, Population},
    traits::{
        AgentGenome, ArchitectureGenes, BehaviorGenes, EngineConfig, EvolutionEngine, Evolvable,
        EvolvableAgent,
    },
};
use async_trait::async_trait;
use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::sync::Arc;

use super::{
    config::DgmConfig,
    improvement::{GrowthHistory, GrowthPattern},
    patterns::PatternDiscovery,
};

/// DGM evolution engine
pub struct DgmEngine {
    /// Configuration
    config: DgmConfig,
    /// Metrics collector
    metrics_collector: MetricsCollector,
    /// Random number generator
    rng: Arc<RwLock<StdRng>>,
    /// Discovered growth patterns
    pub growth_patterns: Arc<RwLock<HashMap<String, GrowthPattern>>>,
    /// Pattern application history
    pub pattern_history: Arc<RwLock<Vec<(String, f64)>>>,
    /// Current best genome
    pub best_genome: Arc<RwLock<Option<AgentGenome>>>,
    /// Improvement velocity
    pub improvement_velocity: Arc<RwLock<f64>>,
    /// Self-assessment system
    self_assessment: DgmSelfAssessment,
    /// Pattern discovery system
    pattern_discovery: PatternDiscovery,
    /// Growth history tracker
    growth_history: Arc<RwLock<GrowthHistory>>,
}

impl DgmEngine {
    /// Create new DGM engine
    pub fn new(config: DgmConfig) -> EvolutionEngineResult<Self> {
        config.validate()?;

        let seed = config.base.seed.unwrap_or_else(|| rand::random());
        let rng = StdRng::seed_from_u64(seed);

        let pattern_discovery = PatternDiscovery::new(
            config.growth_patterns.similarity_threshold,
            config.growth_patterns.max_history,
            config.growth_patterns.consolidation_interval,
        );

        Ok(Self {
            config: config.clone(),
            metrics_collector: MetricsCollector::new(),
            rng: Arc::new(RwLock::new(rng)),
            growth_patterns: Arc::new(RwLock::new(HashMap::new())),
            pattern_history: Arc::new(RwLock::new(Vec::new())),
            best_genome: Arc::new(RwLock::new(None)),
            improvement_velocity: Arc::new(RwLock::new(0.0)),
            self_assessment: DgmSelfAssessment::new(config.self_assessment),
            pattern_discovery,
            growth_history: Arc::new(RwLock::new(GrowthHistory::default())),
        })
    }

    /// Discover new growth patterns
    pub fn discover_patterns(&self, population: &Population<EvolvableAgent>) {
        let mut patterns = self.growth_patterns.write();
        let mut rng = self.rng.write();

        // Look for successful mutations
        for i in 0..population.individuals.len() {
            if let Some(fitness) = population.individuals[i].fitness {
                // Check if this individual represents an improvement
                if fitness > 0.5 && rng.gen_bool(self.config.discovery_rate) {
                    // Create a growth pattern from this success
                    let pattern_id = format!("pattern_{}", uuid::Uuid::new_v4());
                    let source = self.generate_base_genome();
                    let target = population.individuals[i].entity.genome.clone();

                    let pattern = GrowthPattern::new(
                        pattern_id.clone(),
                        source,
                        target,
                        fitness,
                        population.generation as u32,
                    );

                    patterns.insert(pattern_id.clone(), pattern);

                    // Record discovery in history
                    self.growth_history
                        .write()
                        .add_discovery(population.generation as u32, pattern_id);
                }
            }
        }

        // Prune old patterns
        if patterns.len() > self.config.growth_patterns.max_history {
            let current_gen = population.generation as u32;
            patterns.retain(|_, pattern| {
                pattern.is_relevant(current_gen, 50)
                    && pattern.success_rate() > self.config.pattern_retention_threshold
            });
        }
    }

    /// Apply discovered patterns
    pub async fn apply_patterns(
        &self,
        agent: &EvolvableAgent,
        generation: u32,
    ) -> EvolutionEngineResult<(EvolvableAgent, Option<String>)> {
        let (should_mutate, selected_pattern_id) = {
            let patterns = self.growth_patterns.read();
            let mut rng = self.rng.write();

            if patterns.is_empty() || rng.gen_bool(1.0 - self.config.growth_momentum) {
                // No patterns or exploration mode - do random mutation
                (true, None)
            } else {
                // Select a pattern based on success rate
                let pattern_scores: Vec<_> = patterns
                    .values()
                    .map(|p| (p.id.clone(), p.success_rate() * p.fitness_delta))
                    .collect();

                if pattern_scores.is_empty() {
                    (true, None)
                } else {
                    // Select pattern probabilistically
                    let total_score: f64 = pattern_scores.iter().map(|(_, s)| *s).sum();
                    let mut selection = rng.gen_range(0.0..total_score);

                    let selected_pattern_id = pattern_scores
                        .iter()
                        .find(|(_, score)| {
                            selection -= score;
                            selection <= 0.0
                        })
                        .map(|(id, _)| id.clone())
                        .unwrap_or_else(|| pattern_scores[0].0.clone());

                    (false, Some(selected_pattern_id))
                }
            }
        };

        if should_mutate {
            let mutated = agent.mutate(self.config.base.mutation_rate).await?;
            return Ok((mutated, None));
        }

        // Apply the selected pattern
        if let Some(pattern_id) = selected_pattern_id {
            let pattern = self.growth_patterns.read().get(&pattern_id).cloned();

            if let Some(pattern) = pattern {
                let new_genome = self.pattern_discovery.apply_pattern(
                    &agent.genome,
                    &pattern.source,
                    &pattern.target,
                )?;

                let result = EvolvableAgent::from_genome(new_genome).await?;

                // Record pattern application in history
                self.growth_history.write().add_application(
                    pattern_id.clone(),
                    pattern.fitness_delta,
                    generation,
                );

                Ok((result, Some(pattern_id)))
            } else {
                let mutated = agent.mutate(self.config.base.mutation_rate).await?;
                Ok((mutated, None))
            }
        } else {
            let mutated = agent.mutate(self.config.base.mutation_rate).await?;
            Ok((mutated, None))
        }
    }

    /// Generate base genome
    pub fn generate_base_genome(&self) -> AgentGenome {
        let mut rng = self.rng.write();

        AgentGenome {
            goal: stratoswarm_agent_core::Goal::new(
                "Self-improvement".to_string(),
                stratoswarm_agent_core::GoalPriority::High,
            ),
            architecture: ArchitectureGenes {
                memory_capacity: rng.gen_range(1024..100_000),
                processing_units: rng.gen_range(1..4),
                network_topology: vec![rng.gen_range(10..50), rng.gen_range(10..50)],
            },
            behavior: BehaviorGenes {
                exploration_rate: rng.gen_range(0.1..0.5),
                learning_rate: rng.gen_range(0.001..0.05),
                risk_tolerance: rng.gen_range(0.2..0.8),
            },
        }
    }

    /// Update improvement velocity
    pub fn update_velocity(&self, old_best: f64, new_best: f64) {
        let delta = new_best - old_best;
        let mut velocity = self.improvement_velocity.write();
        *velocity = *velocity * self.config.improvement_params.learning_decay + delta;
    }

    /// Get current self-assessment report
    pub fn get_self_assessment_report(&self) -> Option<AssessmentReport> {
        self.self_assessment.get_current_assessment()
    }

    /// Perform immediate self-assessment
    pub async fn perform_self_assessment(
        &self,
        generation: u32,
    ) -> EvolutionEngineResult<AssessmentReport> {
        self.self_assessment.perform_assessment(generation).await
    }

    /// Update pattern success/failure statistics
    fn update_pattern_statistics(&self, best_fitness: f64, generation: u32) {
        let mut patterns = self.growth_patterns.write();
        let history = self.pattern_history.read();

        for (pattern_id, old_fitness) in history.iter() {
            if let Some(pattern) = patterns.get_mut(pattern_id) {
                // Check if pattern improved fitness
                let improved = best_fitness > *old_fitness;
                if improved {
                    pattern.success_count += 1;
                } else {
                    pattern.failure_count += 1;
                }
                pattern.last_used = generation;

                // Update self-assessment pattern effectiveness
                self.self_assessment.update_pattern_effectiveness(
                    pattern_id,
                    improved,
                    best_fitness - *old_fitness,
                    generation,
                );
            }
        }
    }

    /// Record benchmark results for self-assessment
    fn record_benchmark(&self, population: &Population<EvolvableAgent>, average_fitness: f64) {
        self.self_assessment.record_benchmark(BenchmarkResults {
            name: "DGM Evolution".to_string(),
            tasks_attempted: population.individuals.len() as u32,
            tasks_succeeded: population
                .individuals
                .iter()
                .filter(|i| i.fitness.unwrap_or(0.0) > 0.5)
                .count() as u32,
            avg_completion_time: 1.0, // Placeholder - would be actual timing
            code_quality_score: average_fitness,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        });
    }
}

#[async_trait]
impl EvolutionEngine for DgmEngine {
    type Entity = EvolvableAgent;
    type Config = DgmConfig;

    async fn initialize(config: Self::Config) -> EvolutionEngineResult<Self> {
        Self::new(config)
    }

    async fn evolve_step(
        &mut self,
        mut population: Population<Self::Entity>,
    ) -> EvolutionEngineResult<Population<Self::Entity>> {
        self.metrics_collector.start_generation();

        // Evaluate fitness
        let mut total_fitness = 0.0;
        let mut best_fitness = 0.0;
        let old_best = self
            .best_genome
            .read()
            .is_some()
            .then(|| 0.5) // Simplified - would normally track actual best fitness
            .unwrap_or(0.0);

        for individual in &mut population.individuals {
            let fitness = individual.entity.evaluate_fitness().await?;
            individual.fitness = Some(fitness);
            total_fitness += fitness;

            if fitness > best_fitness {
                best_fitness = fitness;
                *self.best_genome.write() = Some(individual.entity.genome.clone());
            }
        }

        // Update velocity and history
        self.update_velocity(old_best, best_fitness);
        self.growth_history
            .write()
            .add_fitness(population.generation as u32, best_fitness);

        // Discover new patterns
        self.discover_patterns(&population);

        // Sort by fitness
        population.individuals.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create next generation with self-improvement focus
        let mut new_individuals = Vec::new();
        let elite_count = (self.config.base.population_size / 20).max(1); // 5% elitism

        // Keep elite
        for i in 0..elite_count.min(population.individuals.len()) {
            new_individuals.push(Individual::new(population.individuals[i].entity.clone()));
        }

        // Generate rest through pattern application and mutation
        while new_individuals.len() < self.config.base.population_size {
            // Select parent with tournament selection
            let tournament_size = 3;
            let mut best_idx = 0;
            let mut best_fit = 0.0;

            {
                let mut rng = self.rng.write();
                for _ in 0..tournament_size {
                    let idx = rng.gen_range(0..population.individuals.len());
                    if let Some(fitness) = population.individuals[idx].fitness {
                        if fitness > best_fit {
                            best_fit = fitness;
                            best_idx = idx;
                        }
                    }
                }
            }

            let parent = &population.individuals[best_idx].entity;
            let parent_fitness = population.individuals[best_idx].fitness.unwrap_or(0.0);

            // Apply growth patterns or mutate
            let (child, pattern_id) = self
                .apply_patterns(parent, population.generation as u32)
                .await?;

            // Record modification for self-assessment
            let modification_id = uuid::Uuid::new_v4().to_string();
            let modification = SelfModification {
                id: modification_id.clone(),
                generation: population.generation as u32,
                parent_id: format!("agent_{}", best_idx),
                child_id: format!("agent_{}_{}", population.generation, new_individuals.len()),
                modification_type: if let Some(pid) = pattern_id.clone() {
                    ModificationType::PatternApplication(pid)
                } else {
                    ModificationType::RandomMutation
                },
                description: pattern_id
                    .clone()
                    .unwrap_or_else(|| "Random mutation".to_string()),
                performance_before: parent_fitness,
                performance_after: None, // Will be updated after evaluation
                successful: None,
            };

            self.self_assessment.record_modification(modification);

            new_individuals.push(Individual::new(child));
        }

        // Update pattern statistics
        if population.generation > 0 {
            self.update_pattern_statistics(best_fitness, population.generation as u32);
        }

        // Update metrics
        let average_fitness = total_fitness / population.individuals.len() as f64;
        let velocity = *self.improvement_velocity.read();

        // Record benchmark results
        self.record_benchmark(&population, average_fitness);

        // Perform self-assessment if needed
        if self
            .self_assessment
            .should_assess(population.generation as u32)
        {
            if let Ok(report) = self
                .self_assessment
                .perform_assessment(population.generation as u32)
                .await
            {
                // Log recommendations
                for recommendation in &report.recommendations {
                    tracing::info!("DGM Self-Assessment: {}", recommendation);
                }
            }
        }

        self.metrics_collector.end_generation(
            best_fitness,
            average_fitness,
            velocity.abs(), // Use velocity as diversity proxy
            population.individuals.len() as u64,
        );

        let mut new_population = Population::from_individuals(new_individuals);
        new_population.generation = population.generation + 1;

        Ok(new_population)
    }

    async fn generate_initial_population(
        &self,
        size: usize,
    ) -> EvolutionEngineResult<Population<Self::Entity>> {
        let mut individuals = Vec::new();

        for i in 0..size {
            let genome = self.generate_base_genome();

            let config = stratoswarm_agent_core::AgentConfig {
                name: format!("dgm_agent_{i}"),
                agent_type: "dgm".to_string(),
                max_memory: genome.architecture.memory_capacity,
                max_gpu_memory: genome.architecture.memory_capacity / 4,
                priority: 1,
                metadata: serde_json::Value::Null,
            };

            let agent = stratoswarm_agent_core::Agent::new(config).map_err(|e| {
                EvolutionEngineError::InitializationError {
                    message: format!("Failed to create agent: {e}"),
                }
            })?;

            individuals.push(Individual::new(EvolvableAgent { agent, genome }));
        }

        Ok(Population::from_individuals(individuals))
    }

    async fn should_terminate(&self, metrics: &EvolutionMetrics) -> bool {
        // Check generation limit
        if metrics.generation >= self.config.base.max_generations {
            return true;
        }

        // Check fitness target
        if let Some(target) = self.config.base.target_fitness {
            if metrics.best_fitness >= target {
                return true;
            }
        }

        // Check if improvement has stagnated
        let velocity = *self.improvement_velocity.read();
        if metrics.generation > 50
            && velocity.abs() < self.config.improvement_params.improvement_threshold
        {
            return true;
        }

        false
    }

    fn metrics(&self) -> &EvolutionMetrics {
        self.metrics_collector.metrics()
    }

    async fn adapt_parameters(&mut self, _metrics: &EvolutionMetrics) -> EvolutionEngineResult<()> {
        if self.config.base.adaptive_parameters {
            let velocity = *self.improvement_velocity.read();

            // Adapt discovery rate based on improvement velocity
            if velocity < self.config.improvement_params.improvement_threshold {
                // Increase exploration
                self.config.discovery_rate *= self.config.improvement_params.exploration_bonus;
                self.config.growth_momentum *= self.config.improvement_params.exploitation_penalty;
            } else {
                // Increase exploitation
                self.config.discovery_rate *= self.config.improvement_params.exploitation_penalty;
                self.config.growth_momentum *= self.config.improvement_params.exploration_bonus;
            }

            // Clamp values
            self.config.discovery_rate = self.config.discovery_rate.clamp(0.05, 0.5);
            self.config.growth_momentum = self.config.growth_momentum.clamp(0.5, 0.95);

            // Adapt pattern retention based on pattern success
            let patterns = self.growth_patterns.read();
            if !patterns.is_empty() {
                let avg_success_rate: f64 =
                    patterns.values().map(|p| p.success_rate()).sum::<f64>()
                        / patterns.len() as f64;

                if avg_success_rate < 0.5 {
                    self.config.pattern_retention_threshold *= 1.1;
                } else {
                    self.config.pattern_retention_threshold *= 0.95;
                }

                self.config.pattern_retention_threshold =
                    self.config.pattern_retention_threshold.clamp(0.3, 0.9);
            }
        }

        Ok(())
    }
}
