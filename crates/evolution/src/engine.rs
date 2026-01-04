//! Genetic Algorithm Evolution Engine Implementation

use crate::{
    channels::SharedEvolutionChannelBridge, AgentFitnessScore, EvolutionEngine, EvolutionError,
    EvolutionStats, FitnessFunction, FitnessScore, Population, XPFitnessFunction,
};
use std::sync::{Arc, Mutex};
use stratoswarm_agent_core::agent::{Agent, AgentId, EvolutionMetrics, EvolutionResult};

/// Simple evolution engine using genetic algorithms
pub struct GeneticEvolutionEngine {
    fitness_function: Arc<dyn FitnessFunction>,
    mutation_rate: f64,
    crossover_rate: f64,
    elite_size: usize,
    tournament_size: usize,
    stats: Arc<Mutex<EvolutionStats>>,
    /// HPC-Channels event bridge for publishing evolution events
    event_bridge: SharedEvolutionChannelBridge,
}

impl GeneticEvolutionEngine {
    /// Create new genetic evolution engine
    pub fn new(
        fitness_function: Arc<dyn FitnessFunction>,
        mutation_rate: f64,
        crossover_rate: f64,
        elite_size: usize,
        tournament_size: usize,
    ) -> Self {
        Self {
            fitness_function,
            mutation_rate,
            crossover_rate,
            elite_size,
            tournament_size,
            stats: Arc::new(Mutex::new(EvolutionStats {
                generation: 0,
                population_size: 0,
                best_fitness: 0.0,
                average_fitness: 0.0,
                mutations_per_second: 0.0,
                diversity_index: 0.0,
            })),
            event_bridge: crate::channels::shared_channel_bridge(),
        }
    }

    /// Create with default parameters
    pub fn with_defaults(fitness_function: Arc<dyn FitnessFunction>) -> Self {
        Self::new(
            fitness_function,
            0.01, // 1% mutation rate
            0.8,  // 80% crossover rate
            5,    // Top 5 elites
            3,    // Tournament size 3
        )
    }

    /// Get the event bridge for subscribing to evolution events
    pub fn event_bridge(&self) -> &SharedEvolutionChannelBridge {
        &self.event_bridge
    }

    /// Evaluate fitness for all individuals in population
    async fn evaluate_population_fitness(
        &self,
        population: &mut Population,
    ) -> Result<(), EvolutionError> {
        for individual in &mut population.individuals {
            let fitness = self.fitness_function.evaluate(&individual.genome);
            individual.set_fitness(fitness);
        }
        Ok(())
    }

    /// Create next generation using genetic algorithm
    async fn create_next_generation(
        &self,
        population: &Population,
    ) -> Result<Population, EvolutionError> {
        if population.is_empty() {
            return Err(EvolutionError::PopulationEmpty);
        }

        let mut new_population = Population::new();
        new_population.generation = population.generation + 1;

        // Keep elites (best individuals)
        let mut sorted_pop = population.clone();
        sorted_pop.sort_by_fitness();

        for i in 0..self.elite_size.min(sorted_pop.size()) {
            new_population.add_individual(sorted_pop.individuals[i].clone());
        }

        // Fill rest with offspring
        let target_size = population.size();
        while new_population.size() < target_size {
            // Tournament selection for parents
            let parent1 = population.tournament_selection(self.tournament_size)?;
            let parent2 = population.tournament_selection(self.tournament_size)?;

            // Create offspring through crossover
            let mut offspring = parent1.crossover(parent2, self.crossover_rate);

            // Apply mutation
            offspring.mutate(self.mutation_rate);

            new_population.add_individual(offspring);
        }

        Ok(new_population)
    }

    /// Update evolution statistics
    fn update_stats(&self, population: &Population) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.generation = population.generation;
            stats.population_size = population.size();
            stats.average_fitness = population.average_fitness();
            stats.diversity_index = population.diversity_index();

            if let Ok(best) = population.best_individual() {
                if let Some(fitness) = best.fitness() {
                    stats.best_fitness = fitness;
                }
            }

            // Simulate mutations per second (would be calculated from actual timing)
            stats.mutations_per_second = (population.size() as f64) * self.mutation_rate * 1000.0;
        }
    }
}

#[async_trait::async_trait]
impl EvolutionEngine for GeneticEvolutionEngine {
    async fn initialize_population(&self, size: usize) -> Result<Population, EvolutionError> {
        if size == 0 {
            return Err(EvolutionError::PopulationEmpty);
        }

        let mut population = Population::random(size, 32); // 32 bytes genome by default

        // Evaluate initial population
        self.evaluate_population_fitness(&mut population).await?;

        // Update stats
        self.update_stats(&population);

        // Publish population initialized event to hpc-channels
        self.event_bridge.publish_population_initialized(size);

        Ok(population)
    }

    async fn evolve_generation(&self, population: &mut Population) -> Result<(), EvolutionError> {
        // Create next generation
        let mut next_gen = self.create_next_generation(population).await?;

        // Evaluate fitness for new individuals
        self.evaluate_population_fitness(&mut next_gen).await?;

        // Replace current population
        *population = next_gen;

        // Update stats
        self.update_stats(population);

        // Publish generation complete event to hpc-channels
        if let Ok(stats) = self.stats.lock() {
            self.event_bridge
                .publish_generation_complete(stats.generation, &stats);
        }

        Ok(())
    }

    async fn evaluate_fitness(
        &self,
        population: &Population,
    ) -> Result<Vec<FitnessScore>, EvolutionError> {
        let mut scores = Vec::new();

        for individual in &population.individuals {
            let fitness = self.fitness_function.evaluate(&individual.genome);
            scores.push(FitnessScore {
                value: fitness,
                individual_id: individual.id,
            });
        }

        Ok(scores)
    }

    async fn stats(&self) -> Result<EvolutionStats, EvolutionError> {
        self.stats.lock().map(|stats| stats.clone()).map_err(|_| {
            EvolutionError::FitnessEvaluationFailed {
                reason: "Failed to acquire stats lock".to_string(),
            }
        })
    }
}

/// Simple fitness function for testing
pub struct SimpleFitnessFunction;

impl FitnessFunction for SimpleFitnessFunction {
    fn evaluate(&self, individual: &[u8]) -> f64 {
        // Simple fitness: count of 1s in the genome
        let ones = individual.iter().filter(|&&b| b == 1).count();
        ones as f64 / individual.len() as f64
    }
}

/// Maximization fitness function for testing
pub struct MaximizationFitnessFunction;

impl FitnessFunction for MaximizationFitnessFunction {
    fn evaluate(&self, individual: &[u8]) -> f64 {
        // Fitness based on sum of bytes (maximization)
        individual.iter().map(|&b| b as f64).sum::<f64>() / (255.0 * individual.len() as f64)
    }
}

/// Target matching fitness function
pub struct TargetMatchingFitnessFunction {
    target: Vec<u8>,
}

impl TargetMatchingFitnessFunction {
    pub fn new(target: Vec<u8>) -> Self {
        Self { target }
    }
}

impl FitnessFunction for TargetMatchingFitnessFunction {
    fn evaluate(&self, individual: &[u8]) -> f64 {
        let min_len = individual.len().min(self.target.len());
        let matches = (0..min_len)
            .filter(|&i| individual[i] == self.target[i])
            .count();

        // Penalty for length difference
        let length_penalty = (individual.len() as i32 - self.target.len() as i32).abs() as f64;
        let max_len = individual.len().max(self.target.len()) as f64;

        (matches as f64 - length_penalty * 0.5) / max_len
    }
}

/// XP-aware evolution engine trait
#[async_trait::async_trait]
pub trait XPEvolutionEngine: Send + Sync {
    /// Evolve a collection of agents based on their XP and performance
    async fn evolve_agent_population(
        &self,
        agents: &mut Vec<Agent>,
    ) -> Result<Vec<EvolutionResult>, EvolutionError>;

    /// Evaluate agent fitness using XP-based metrics
    async fn evaluate_agent_fitness(
        &self,
        agent: &Agent,
    ) -> Result<AgentFitnessScore, EvolutionError>;

    /// Select agents for evolution based on XP thresholds and performance
    async fn select_evolution_candidates(
        &self,
        agents: &[Agent],
    ) -> Result<Vec<AgentId>, EvolutionError>;

    /// Award XP based on evolution outcomes
    async fn award_evolution_xp(
        &self,
        agent: &Agent,
        evolution_result: &EvolutionResult,
    ) -> Result<(), EvolutionError>;

    /// Get evolution statistics for XP-enabled agents
    async fn get_xp_evolution_stats(&self) -> Result<XPEvolutionStats, EvolutionError>;
}

/// Statistics for XP-based evolution
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct XPEvolutionStats {
    pub total_agents_evolved: u64,
    pub average_level_improvement: f64,
    pub total_xp_awarded: u64,
    pub evolution_success_rate: f64,
    pub average_fitness_improvement: f64,
    pub generations_completed: u64,
}

/// Agent-based evolution engine using XP system
pub struct AgentEvolutionEngine<F: XPFitnessFunction> {
    fitness_function: F,
    evolution_threshold_multiplier: f64,
    max_agents_per_generation: usize,
    xp_bonus_factor: f64,
    stats: Arc<Mutex<XPEvolutionStats>>,
}

impl<F: XPFitnessFunction> AgentEvolutionEngine<F> {
    pub fn new(
        fitness_function: F,
        evolution_threshold_multiplier: f64,
        max_agents_per_generation: usize,
        xp_bonus_factor: f64,
    ) -> Self {
        Self {
            fitness_function,
            evolution_threshold_multiplier,
            max_agents_per_generation,
            xp_bonus_factor,
            stats: Arc::new(Mutex::new(XPEvolutionStats::default())),
        }
    }

    pub fn with_defaults(fitness_function: F) -> Self {
        Self::new(
            fitness_function,
            1.0, // 100% of ready agents
            100, // Max 100 agents per generation
            1.5, // 1.5x XP bonus factor
        )
    }
}

#[async_trait::async_trait]
impl<F: XPFitnessFunction> XPEvolutionEngine for AgentEvolutionEngine<F> {
    async fn evolve_agent_population(
        &self,
        agents: &mut Vec<Agent>,
    ) -> Result<Vec<EvolutionResult>, EvolutionError> {
        let mut evolution_results = Vec::new();

        // Select candidates for evolution
        let candidate_ids = self.select_evolution_candidates(agents).await?;

        let mut evolved_count = 0;

        for agent in agents.iter() {
            if candidate_ids.contains(&agent.id()) && evolved_count < self.max_agents_per_generation
            {
                // Check if agent is ready to evolve
                if self.fitness_function.should_evolve(agent).await {
                    // Get fitness before evolution
                    let pre_fitness = self.fitness_function.evaluate_agent_fitness(agent).await;

                    // Trigger evolution
                    match agent.trigger_evolution().await {
                        Ok(evolution_result) => {
                            // Get fitness after evolution
                            let post_fitness =
                                self.fitness_function.evaluate_agent_fitness(agent).await;

                            // Calculate fitness improvement
                            let fitness_improvement = post_fitness - pre_fitness;

                            // Award XP based on improvement
                            self.award_evolution_xp(agent, &evolution_result).await?;

                            // Award bonus XP for significant improvement
                            if fitness_improvement > 0.1 {
                                let bonus_xp =
                                    (fitness_improvement * 100.0 * self.xp_bonus_factor) as u64;
                                agent
                                    .award_xp(
                                        bonus_xp,
                                        "Fitness improvement bonus".to_string(),
                                        "evolution_bonus".to_string(),
                                    )
                                    .await
                                    .map_err(|e| EvolutionError::FitnessEvaluationFailed {
                                        reason: format!("Failed to award bonus XP: {}", e),
                                    })?;
                            }

                            evolution_results.push(evolution_result);
                            evolved_count += 1;
                        }
                        Err(e) => {
                            // Log error but continue with other agents
                            tracing::warn!("Failed to evolve agent {}: {}", agent.id(), e);
                        }
                    }
                }
            }
        }

        // Update statistics
        self.update_evolution_stats(&evolution_results).await;

        Ok(evolution_results)
    }

    async fn evaluate_agent_fitness(
        &self,
        agent: &Agent,
    ) -> Result<AgentFitnessScore, EvolutionError> {
        let stats = agent.stats().await;
        let fitness = self.fitness_function.evaluate_agent_fitness(agent).await;
        let performance_metrics = self.calculate_performance_metrics(agent).await;

        Ok(AgentFitnessScore {
            fitness,
            agent_id: agent.id(),
            xp_contribution: stats.current_xp,
            level: stats.level,
            performance_metrics,
        })
    }

    async fn select_evolution_candidates(
        &self,
        agents: &[Agent],
    ) -> Result<Vec<AgentId>, EvolutionError> {
        let mut candidates = Vec::new();

        for agent in agents {
            if self.fitness_function.should_evolve(agent).await {
                let fitness = self.fitness_function.evaluate_agent_fitness(agent).await;

                // Apply threshold multiplier for selection
                if fitness >= 0.3 * self.evolution_threshold_multiplier {
                    candidates.push(agent.id());
                }
            }
        }

        // Limit to max agents per generation
        if candidates.len() > self.max_agents_per_generation {
            candidates.truncate(self.max_agents_per_generation);
        }

        Ok(candidates)
    }

    async fn award_evolution_xp(
        &self,
        agent: &Agent,
        evolution_result: &EvolutionResult,
    ) -> Result<(), EvolutionError> {
        let fitness_improvement = evolution_result.new_metrics.processing_speed
            - evolution_result.previous_metrics.processing_speed;

        let xp_reward = self
            .fitness_function
            .calculate_xp_reward(fitness_improvement, &evolution_result.new_metrics);

        agent
            .award_xp(
                xp_reward,
                format!(
                    "Evolution from level {} to {}",
                    evolution_result.previous_level, evolution_result.new_level
                ),
                "evolution_outcome".to_string(),
            )
            .await
            .map_err(|e| EvolutionError::FitnessEvaluationFailed {
                reason: format!("Failed to award evolution XP: {}", e),
            })?;

        Ok(())
    }

    async fn get_xp_evolution_stats(&self) -> Result<XPEvolutionStats, EvolutionError> {
        self.stats.lock().map(|stats| stats.clone()).map_err(|_| {
            EvolutionError::FitnessEvaluationFailed {
                reason: "Failed to acquire stats lock".to_string(),
            }
        })
    }
}

impl<F: XPFitnessFunction> AgentEvolutionEngine<F> {
    async fn update_evolution_stats(&self, evolution_results: &[EvolutionResult]) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_agents_evolved += evolution_results.len() as u64;
            stats.generations_completed += 1;

            if !evolution_results.is_empty() {
                let level_improvements: Vec<u32> = evolution_results
                    .iter()
                    .map(|r| r.new_level.saturating_sub(r.previous_level))
                    .collect();

                let avg_level_improvement =
                    level_improvements.iter().sum::<u32>() as f64 / level_improvements.len() as f64;

                stats.average_level_improvement =
                    (stats.average_level_improvement + avg_level_improvement) / 2.0;

                // Calculate fitness improvements
                let fitness_improvements: Vec<f64> = evolution_results
                    .iter()
                    .map(|r| r.new_metrics.processing_speed - r.previous_metrics.processing_speed)
                    .collect();

                let avg_fitness_improvement =
                    fitness_improvements.iter().sum::<f64>() / fitness_improvements.len() as f64;

                stats.average_fitness_improvement =
                    (stats.average_fitness_improvement + avg_fitness_improvement) / 2.0;

                stats.total_xp_awarded += evolution_results.len() as u64 * 100; // Approximate

                // Simple success rate (all evolutions that completed are successful)
                stats.evolution_success_rate = 1.0; // Could be refined based on actual metrics
            }
        }
    }

    async fn calculate_performance_metrics(&self, agent: &Agent) -> EvolutionMetrics {
        let stats = agent.stats().await;

        let avg_completion_time = if stats.goals_processed > 0 {
            std::time::Duration::from_secs(
                stats.total_execution_time.as_secs() / stats.goals_processed,
            )
        } else {
            std::time::Duration::from_secs(60)
        };

        let success_rate = if stats.goals_processed > 0 {
            stats.goals_succeeded as f64 / stats.goals_processed as f64
        } else {
            0.0
        };

        EvolutionMetrics {
            avg_completion_time,
            success_rate,
            memory_efficiency: 0.8, // Default value
            processing_speed: 1.0 + (stats.level as f64 * 0.1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_genetic_engine_creation() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);

        let stats = engine.stats().await.expect("Should get stats");
        assert_eq!(stats.generation, 0);
        assert_eq!(stats.population_size, 0);
    }

    #[tokio::test]
    async fn test_population_initialization() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);

        let population = engine
            .initialize_population(10)
            .await
            .expect("Should initialize population");
        assert_eq!(population.size(), 10);
        assert_eq!(population.generation, 0);

        // All individuals should have fitness scores
        for individual in &population.individuals {
            assert!(individual.fitness().is_some());
        }

        let stats = engine.stats().await.expect("Should get stats");
        assert_eq!(stats.population_size, 10);
        assert!(stats.average_fitness >= 0.0);
    }

    #[tokio::test]
    async fn test_evolution_generation() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);

        let mut population = engine
            .initialize_population(20)
            .await
            .expect("Should initialize population");
        let initial_generation = population.generation;

        engine
            .evolve_generation(&mut population)
            .await
            .expect("Should evolve generation");

        assert_eq!(population.generation, initial_generation + 1);
        assert_eq!(population.size(), 20); // Population size should remain the same

        // All individuals should still have fitness scores
        for individual in &population.individuals {
            assert!(individual.fitness().is_some());
        }

        let stats = engine.stats().await.expect("Should get stats");
        assert_eq!(stats.generation, 1);
    }

    #[tokio::test]
    async fn test_fitness_evaluation() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);

        let population = engine
            .initialize_population(5)
            .await
            .expect("Should initialize population");
        let scores = engine
            .evaluate_fitness(&population)
            .await
            .expect("Should evaluate fitness");

        assert_eq!(scores.len(), 5);
        for score in scores {
            assert!(score.value >= 0.0 && score.value <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_multiple_generations() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);

        let mut population = engine
            .initialize_population(10)
            .await
            .expect("Should initialize population");

        let initial_best = population
            .best_individual()
            .expect("Should have best individual")
            .fitness()
            .unwrap();

        // Evolve for several generations
        for _ in 0..5 {
            engine
                .evolve_generation(&mut population)
                .await
                .expect("Should evolve generation");
        }

        assert_eq!(population.generation, 5);

        let final_stats = engine.stats().await.expect("Should get final stats");
        assert_eq!(final_stats.generation, 5);
        assert!(final_stats.mutations_per_second > 0.0);
    }

    #[tokio::test]
    async fn test_empty_population_error() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);

        let result = engine.initialize_population(0).await;
        assert!(matches!(result, Err(EvolutionError::PopulationEmpty)));
    }

    #[test]
    fn test_simple_fitness_function() {
        let fitness_fn = SimpleFitnessFunction;

        assert_eq!(fitness_fn.evaluate(&[0, 0, 0, 0]), 0.0);
        assert_eq!(fitness_fn.evaluate(&[1, 1, 1, 1]), 1.0);
        assert_eq!(fitness_fn.evaluate(&[1, 0, 1, 0]), 0.5);
    }

    #[test]
    fn test_maximization_fitness_function() {
        let fitness_fn = MaximizationFitnessFunction;

        assert_eq!(fitness_fn.evaluate(&[0, 0, 0, 0]), 0.0);
        assert_eq!(fitness_fn.evaluate(&[255, 255, 255, 255]), 1.0);

        let result = fitness_fn.evaluate(&[128, 128, 128, 128]);
        assert!((result - 0.5).abs() < 0.01); // Should be approximately 0.5
    }

    #[test]
    fn test_target_matching_fitness_function() {
        let target = vec![1, 2, 3, 4];
        let fitness_fn = TargetMatchingFitnessFunction::new(target.clone());

        assert_eq!(fitness_fn.evaluate(&target), 1.0);
        assert_eq!(fitness_fn.evaluate(&[5, 6, 7, 8]), 0.0);

        let partial_match = fitness_fn.evaluate(&[1, 2, 7, 8]);
        assert!(partial_match > 0.0 && partial_match < 1.0);
    }

    #[tokio::test]
    async fn test_engine_with_target_matching() {
        let target = vec![1, 1, 1, 1, 1];
        let fitness_fn = Arc::new(TargetMatchingFitnessFunction::new(target));
        let engine = GeneticEvolutionEngine::new(fitness_fn, 0.1, 0.9, 2, 3);

        let mut population = engine
            .initialize_population(20)
            .await
            .expect("Should initialize population");

        // Evolve for multiple generations
        for _ in 0..10 {
            engine
                .evolve_generation(&mut population)
                .await
                .expect("Should evolve generation");
        }

        // Should show improvement over generations
        let stats = engine.stats().await.expect("Should get stats");
        assert_eq!(stats.generation, 10);
        // Fitness can be negative due to length penalties, so just check it's a valid fitness score
        assert!(stats.best_fitness >= -1.0 && stats.best_fitness <= 1.0);
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);

        let mut population = engine
            .initialize_population(15)
            .await
            .expect("Should initialize population");

        let initial_stats = engine.stats().await.expect("Should get initial stats");
        assert_eq!(initial_stats.generation, 0);
        assert_eq!(initial_stats.population_size, 15);

        engine
            .evolve_generation(&mut population)
            .await
            .expect("Should evolve generation");

        let evolved_stats = engine.stats().await.expect("Should get evolved stats");
        assert_eq!(evolved_stats.generation, 1);
        assert_eq!(evolved_stats.population_size, 15);
        assert!(evolved_stats.mutations_per_second > 0.0);
        assert!(evolved_stats.diversity_index >= 0.0);
    }

    #[tokio::test]
    async fn test_create_next_generation_empty_population() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);

        let empty_population = Population::new();
        let result = engine.create_next_generation(&empty_population).await;

        assert!(matches!(result, Err(EvolutionError::PopulationEmpty)));
    }

    #[tokio::test]
    async fn test_evaluate_empty_population() {
        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine::with_defaults(fitness_fn);

        let mut empty_population = Population::new();

        // Evaluating empty population should succeed (no individuals to evaluate)
        let result = engine
            .evaluate_population_fitness(&mut empty_population)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_mutex_poisoning_stats() {
        use crate::test_helpers::tests::create_poisoned_mutex;

        let fitness_fn = Arc::new(SimpleFitnessFunction);
        let engine = GeneticEvolutionEngine {
            fitness_function: fitness_fn,
            mutation_rate: 0.01,
            crossover_rate: 0.8,
            elite_size: 2,
            tournament_size: 3,
            stats: create_poisoned_mutex(),
            event_bridge: crate::channels::shared_channel_bridge(),
        };

        let result = engine.stats().await;
        assert!(result.is_err());

        match result {
            Err(EvolutionError::FitnessEvaluationFailed { reason }) => {
                assert!(reason.contains("Failed to acquire stats lock"));
            }
            _ => panic!("Expected FitnessEvaluationFailed error with lock failure"),
        }
    }
}
