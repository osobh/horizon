//! Managed population implementation

use super::types::*;
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use crate::traits::EvolvableAgent;
use rand::prelude::*;
use std::collections::HashMap;

/// A managed population of evolvable agents
#[derive(Debug, Clone)]
pub struct ManagedPopulation {
    /// Population ID
    id: String,
    /// Agents in the population
    agents: Vec<EvolvableAgent>,
    /// Current generation
    generation: u32,
    /// Best fitness achieved
    best_fitness: f64,
    /// Average fitness
    average_fitness: f64,
    /// Population diversity
    diversity: f64,
    /// Whether population has converged
    converged: bool,
    /// Fitness history
    fitness_history: Vec<f64>,
}

impl ManagedPopulation {
    /// Create new managed population
    pub fn new(id: String, agents: Vec<EvolvableAgent>) -> EvolutionEngineResult<Self> {
        let mut population = Self {
            id,
            agents,
            generation: 0,
            best_fitness: 0.0,
            average_fitness: 0.0,
            diversity: 0.0,
            converged: false,
            fitness_history: Vec::new(),
        };

        if !population.agents.is_empty() {
            population.update_metrics()?;
        }

        Ok(population)
    }

    /// Get population ID
    #[inline]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get population size
    #[inline]
    pub fn size(&self) -> usize {
        self.agents.len()
    }

    /// Get current generation
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Check if population has converged
    #[inline]
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Get agents
    #[inline]
    pub fn get_agents(&self) -> &[EvolvableAgent] {
        &self.agents
    }

    /// Get best fitness
    #[inline]
    pub fn get_best_fitness(&self) -> f64 {
        self.best_fitness
    }

    /// Get average fitness
    #[inline]
    pub fn get_average_fitness(&self) -> f64 {
        self.average_fitness
    }

    /// Calculate population diversity
    pub fn calculate_diversity(&self) -> f64 {
        if self.agents.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut comparisons = 0;

        for i in 0..self.agents.len() {
            for j in (i + 1)..self.agents.len() {
                let distance = self.calculate_agent_distance(&self.agents[i], &self.agents[j]);
                total_distance += distance;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_distance / comparisons as f64
        } else {
            0.0
        }
    }

    /// Evolve population for one generation
    pub fn evolve_generation(
        &mut self,
        selection_strategy: &SelectionStrategy,
        crossover_strategy: &CrossoverStrategy,
    ) -> EvolutionEngineResult<()> {
        if self.agents.is_empty() {
            return Ok(());
        }

        // Select parents
        let parents = self.select_parents(selection_strategy, self.agents.len())?;

        // Generate offspring through crossover and mutation
        let mut offspring = Vec::new();
        for i in (0..parents.len()).step_by(2) {
            let parent1 = &parents[i];
            let parent2 = if i + 1 < parents.len() {
                &parents[i + 1]
            } else {
                &parents[0] // Wrap around if odd number
            };

            let children = self.crossover(parent1, parent2, crossover_strategy)?;
            offspring.extend(children);
        }

        // Replace population with offspring (plus some elites)
        let elite_count = self.agents.len() / 10; // Keep top 10%
        let mut new_population = Vec::new();

        // Add elites
        let mut sorted_agents = self.agents.clone();
        sorted_agents.sort_by(|a, b| {
            let fitness_a = self.get_agent_fitness(a);
            let fitness_b = self.get_agent_fitness(b);
            fitness_b
                .partial_cmp(&fitness_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for i in 0..elite_count.min(sorted_agents.len()) {
            new_population.push(sorted_agents[i].clone());
        }

        // Add offspring
        for i in 0..(self.agents.len() - elite_count).min(offspring.len()) {
            new_population.push(offspring[i].clone());
        }

        // Fill remaining slots if needed
        while new_population.len() < self.agents.len() {
            let idx = rand::thread_rng().gen_range(0..offspring.len());
            new_population.push(offspring[idx].clone());
        }

        self.agents = new_population;
        self.generation += 1;
        self.update_metrics()?;

        Ok(())
    }

    /// Select parents for reproduction
    pub fn select_parents(
        &self,
        strategy: &SelectionStrategy,
        count: usize,
    ) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        if self.agents.is_empty() {
            return Ok(Vec::new());
        }

        match strategy {
            SelectionStrategy::Tournament { size } => self.tournament_selection(*size, count),
            SelectionStrategy::Elite { count: elite_count } => self.elite_selection(*elite_count),
            SelectionStrategy::RouletteWheel => self.roulette_wheel_selection(count),
            SelectionStrategy::Rank => self.rank_selection(count),
        }
    }

    /// Perform crossover between two parents
    pub fn crossover(
        &self,
        parent1: &EvolvableAgent,
        parent2: &EvolvableAgent,
        strategy: &CrossoverStrategy,
    ) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        match strategy {
            CrossoverStrategy::Uniform { rate } => self.uniform_crossover(parent1, parent2, *rate),
            CrossoverStrategy::SinglePoint => self.single_point_crossover(parent1, parent2),
            CrossoverStrategy::TwoPoint => self.two_point_crossover(parent1, parent2),
            CrossoverStrategy::Blend { alpha } => self.blend_crossover(parent1, parent2, *alpha),
        }
    }

    /// Check if population has converged
    pub fn check_convergence(&self, threshold: f64) -> bool {
        if self.agents.len() < 2 {
            return false;
        }

        let diversity = self.calculate_diversity();
        diversity < threshold
    }

    // Helper methods

    fn update_metrics(&mut self) -> EvolutionEngineResult<()> {
        if self.agents.is_empty() {
            return Ok(());
        }

        let mut total_fitness = 0.0;
        let mut max_fitness = 0.0;

        for agent in &self.agents {
            let fitness = self.get_agent_fitness(agent);
            total_fitness += fitness;
            if fitness > max_fitness {
                max_fitness = fitness;
            }
        }

        self.best_fitness = max_fitness;
        self.average_fitness = total_fitness / self.agents.len() as f64;
        self.diversity = self.calculate_diversity();
        self.fitness_history.push(self.average_fitness);

        // Simple convergence check
        self.converged = self.check_convergence(0.1);

        Ok(())
    }

    fn get_agent_fitness(&self, agent: &EvolvableAgent) -> f64 {
        // Use exploration rate as fitness proxy
        agent.genome.behavior.exploration_rate
    }

    fn calculate_agent_distance(&self, agent1: &EvolvableAgent, agent2: &EvolvableAgent) -> f64 {
        let behavior1 = &agent1.genome.behavior;
        let behavior2 = &agent2.genome.behavior;

        let exploration_diff = (behavior1.exploration_rate - behavior2.exploration_rate).abs();
        let learning_diff = (behavior1.learning_rate - behavior2.learning_rate).abs();
        let risk_diff = (behavior1.risk_tolerance - behavior2.risk_tolerance).abs();

        (exploration_diff + learning_diff + risk_diff) / 3.0
    }

    fn tournament_selection(
        &self,
        tournament_size: usize,
        count: usize,
    ) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        let mut selected = Vec::new();
        let mut rng = rand::thread_rng();

        for _ in 0..count {
            let mut best_agent = &self.agents[0];
            let mut best_fitness = self.get_agent_fitness(best_agent);

            for _ in 0..tournament_size {
                let idx = rng.gen_range(0..self.agents.len());
                let candidate = &self.agents[idx];
                let fitness = self.get_agent_fitness(candidate);

                if fitness > best_fitness {
                    best_agent = candidate;
                    best_fitness = fitness;
                }
            }

            selected.push(best_agent.clone());
        }

        Ok(selected)
    }

    fn elite_selection(&self, count: usize) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        let mut sorted_agents = self.agents.clone();
        sorted_agents.sort_by(|a, b| {
            let fitness_a = self.get_agent_fitness(a);
            let fitness_b = self.get_agent_fitness(b);
            fitness_b
                .partial_cmp(&fitness_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(sorted_agents.into_iter().take(count).collect())
    }

    fn roulette_wheel_selection(&self, count: usize) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        let mut selected = Vec::new();
        let total_fitness: f64 = self
            .agents
            .iter()
            .map(|agent| self.get_agent_fitness(agent))
            .sum();

        if total_fitness <= 0.0 {
            // Fallback to random selection
            let mut rng = rand::thread_rng();
            for _ in 0..count {
                let idx = rng.gen_range(0..self.agents.len());
                selected.push(self.agents[idx].clone());
            }
            return Ok(selected);
        }

        let mut rng = rand::thread_rng();
        for _ in 0..count {
            let mut spin = rng.gen::<f64>() * total_fitness;
            for agent in &self.agents {
                spin -= self.get_agent_fitness(agent);
                if spin <= 0.0 {
                    selected.push(agent.clone());
                    break;
                }
            }
        }

        Ok(selected)
    }

    fn rank_selection(&self, count: usize) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        let mut indexed_agents: Vec<(usize, f64)> = self
            .agents
            .iter()
            .enumerate()
            .map(|(i, agent)| (i, self.get_agent_fitness(agent)))
            .collect();

        indexed_agents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected = Vec::new();
        let mut rng = rand::thread_rng();

        for _ in 0..count {
            // Higher ranks have higher probability
            let rank_sum = (self.agents.len() * (self.agents.len() + 1)) / 2;
            let mut spin = rng.gen_range(1..=rank_sum);

            for (rank, (idx, _)) in indexed_agents.iter().enumerate() {
                let rank_value = self.agents.len() - rank;
                if spin <= rank_value {
                    selected.push(self.agents[*idx].clone());
                    break;
                }
                spin -= rank_value;
            }
        }

        Ok(selected)
    }

    fn uniform_crossover(
        &self,
        parent1: &EvolvableAgent,
        parent2: &EvolvableAgent,
        rate: f64,
    ) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        let mut rng = rand::thread_rng();

        let child1_genome = if rng.gen::<f64>() < rate {
            parent1.genome.clone()
        } else {
            parent2.genome.clone()
        };

        let child2_genome = if rng.gen::<f64>() < rate {
            parent2.genome.clone()
        } else {
            parent1.genome.clone()
        };

        let child1 = self.create_child_agent(child1_genome)?;
        let child2 = self.create_child_agent(child2_genome)?;

        Ok(vec![child1, child2])
    }

    fn single_point_crossover(
        &self,
        parent1: &EvolvableAgent,
        parent2: &EvolvableAgent,
    ) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        // Simplified single point crossover - blend behavior parameters
        let mut child1_genome = parent1.genome.clone();
        let mut child2_genome = parent2.genome.clone();

        child1_genome.behavior.learning_rate = parent2.genome.behavior.learning_rate;
        child2_genome.behavior.learning_rate = parent1.genome.behavior.learning_rate;

        let child1 = self.create_child_agent(child1_genome)?;
        let child2 = self.create_child_agent(child2_genome)?;

        Ok(vec![child1, child2])
    }

    fn two_point_crossover(
        &self,
        parent1: &EvolvableAgent,
        parent2: &EvolvableAgent,
    ) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        // Simplified two point crossover
        let mut child1_genome = parent1.genome.clone();
        let mut child2_genome = parent2.genome.clone();

        // Swap middle section (learning_rate)
        child1_genome.behavior.learning_rate = parent2.genome.behavior.learning_rate;
        child2_genome.behavior.learning_rate = parent1.genome.behavior.learning_rate;

        let child1 = self.create_child_agent(child1_genome)?;
        let child2 = self.create_child_agent(child2_genome)?;

        Ok(vec![child1, child2])
    }

    fn blend_crossover(
        &self,
        parent1: &EvolvableAgent,
        parent2: &EvolvableAgent,
        alpha: f64,
    ) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        let mut rng = rand::thread_rng();

        let p1_exploration = parent1.genome.behavior.exploration_rate;
        let p2_exploration = parent2.genome.behavior.exploration_rate;
        let range = (p2_exploration - p1_exploration).abs() * alpha;

        let child1_exploration = p1_exploration + rng.gen::<f64>() * range * 2.0 - range;
        let child2_exploration = p2_exploration + rng.gen::<f64>() * range * 2.0 - range;

        let mut child1_genome = parent1.genome.clone();
        let mut child2_genome = parent2.genome.clone();

        child1_genome.behavior.exploration_rate = child1_exploration.clamp(0.0, 1.0);
        child2_genome.behavior.exploration_rate = child2_exploration.clamp(0.0, 1.0);

        let child1 = self.create_child_agent(child1_genome)?;
        let child2 = self.create_child_agent(child2_genome)?;

        Ok(vec![child1, child2])
    }

    fn create_child_agent(
        &self,
        genome: crate::traits::AgentGenome,
    ) -> EvolutionEngineResult<EvolvableAgent> {
        use stratoswarm_agent_core::{Agent, AgentConfig};

        let config = AgentConfig {
            name: format!("child_{}", uuid::Uuid::new_v4()),
            agent_type: "evolved".to_string(),
            max_memory: genome.architecture.memory_capacity,
            max_gpu_memory: genome.architecture.memory_capacity / 4,
            priority: 1,
            metadata: serde_json::Value::Null,
        };

        let agent = Agent::new(config).map_err(|e| {
            EvolutionEngineError::Other(format!("Failed to create child agent: {}", e))
        })?;

        Ok(EvolvableAgent { agent, genome })
    }
}
