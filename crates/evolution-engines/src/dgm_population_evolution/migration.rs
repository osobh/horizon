//! Migration controller for population evolution

use super::population::ManagedPopulation;
use super::types::*;
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use crate::traits::EvolvableAgent;
use rand::prelude::*;

/// Controls migration between populations
pub struct MigrationController {
    /// Migration policy
    policy: MigrationPolicy,
    /// Migration history
    migration_history: Vec<PopulationMigration>,
    /// Migration count
    migration_count: usize,
}

impl MigrationController {
    /// Create new migration controller
    pub fn new(policy: MigrationPolicy) -> EvolutionEngineResult<Self> {
        Ok(Self {
            policy,
            migration_history: Vec::new(),
            migration_count: 0,
        })
    }

    /// Get migration count
    #[inline]
    pub fn get_migration_count(&self) -> usize {
        self.migration_count
    }

    /// Get migration history
    #[inline]
    pub fn get_migration_history(&self) -> &[PopulationMigration] {
        &self.migration_history
    }

    /// Plan migration between two populations
    pub fn plan_migration(
        &self,
        source: &ManagedPopulation,
        target: &ManagedPopulation,
        generation: u32,
    ) -> EvolutionEngineResult<PopulationMigration> {
        let agents = match &self.policy {
            MigrationPolicy::BestAgent { rate } => self.select_best_agents(source, *rate),
            MigrationPolicy::Random { rate } => self.select_random_agents(source, *rate),
            MigrationPolicy::DiversityBased { rate } => self.select_diverse_agents(source, *rate),
            MigrationPolicy::RingTopology { rate } => {
                self.select_ring_topology_agents(source, *rate)
            }
            MigrationPolicy::IslandModel { rate } => self.select_island_model_agents(source, *rate),
        }?;

        Ok(PopulationMigration {
            source_population: source.id().to_string(),
            target_population: target.id().to_string(),
            agents,
            generation,
            reason: format!("Migration using policy: {:?}", self.policy),
        })
    }

    /// Record a migration event
    pub fn record_migration(
        &mut self,
        migration: PopulationMigration,
    ) -> EvolutionEngineResult<()> {
        self.migration_history.push(migration);
        self.migration_count += 1;
        Ok(())
    }

    // Selection strategies

    fn select_best_agents(
        &self,
        population: &ManagedPopulation,
        rate: f64,
    ) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        let agents = population.get_agents();
        if agents.is_empty() {
            return Ok(Vec::new());
        }

        let migration_count = ((agents.len() as f64 * rate).ceil() as usize).max(1);

        // Sort by fitness (using exploration rate as proxy)
        let mut sorted_agents = agents.to_vec();
        sorted_agents.sort_by(|a, b| {
            let fitness_a = a.genome.behavior.exploration_rate;
            let fitness_b = b.genome.behavior.exploration_rate;
            fitness_b
                .partial_cmp(&fitness_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(sorted_agents.into_iter().take(migration_count).collect())
    }

    fn select_random_agents(
        &self,
        population: &ManagedPopulation,
        rate: f64,
    ) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        let agents = population.get_agents();
        if agents.is_empty() {
            return Ok(Vec::new());
        }

        let migration_count = ((agents.len() as f64 * rate).ceil() as usize).max(1);
        let mut rng = rand::thread_rng();

        let mut selected = Vec::new();
        let mut indices: Vec<usize> = (0..agents.len()).collect();
        indices.shuffle(&mut rng);

        for i in 0..migration_count.min(indices.len()) {
            selected.push(agents[indices[i]].clone());
        }

        Ok(selected)
    }

    fn select_diverse_agents(
        &self,
        population: &ManagedPopulation,
        rate: f64,
    ) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        let agents = population.get_agents();
        if agents.is_empty() {
            return Ok(Vec::new());
        }

        let migration_count = ((agents.len() as f64 * rate).ceil() as usize).max(1);
        let mut selected = Vec::new();

        // Start with a random agent
        let mut rng = rand::thread_rng();
        let start_idx = rng.gen_range(0..agents.len());
        selected.push(agents[start_idx].clone());

        // Select agents that are maximally different from already selected
        for _ in 1..migration_count {
            let mut best_candidate = &agents[0];
            let mut max_distance = 0.0;

            for candidate in agents {
                let mut min_distance_to_selected = f64::INFINITY;

                for selected_agent in &selected {
                    let distance = self.calculate_agent_distance(candidate, selected_agent);
                    if distance < min_distance_to_selected {
                        min_distance_to_selected = distance;
                    }
                }

                if min_distance_to_selected > max_distance {
                    max_distance = min_distance_to_selected;
                    best_candidate = candidate;
                }
            }

            selected.push(best_candidate.clone());
        }

        Ok(selected)
    }

    fn select_ring_topology_agents(
        &self,
        population: &ManagedPopulation,
        rate: f64,
    ) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        // For ring topology, select based on position in sorted fitness
        self.select_best_agents(population, rate)
    }

    fn select_island_model_agents(
        &self,
        population: &ManagedPopulation,
        rate: f64,
    ) -> EvolutionEngineResult<Vec<EvolvableAgent>> {
        // For island model, mix best and diverse agents
        let agents = population.get_agents();
        if agents.is_empty() {
            return Ok(Vec::new());
        }

        let migration_count = ((agents.len() as f64 * rate).ceil() as usize).max(1);
        let best_count = migration_count / 2;
        let diverse_count = migration_count - best_count;

        let mut selected = Vec::new();

        // Add best agents
        let best_agents =
            self.select_best_agents(population, best_count as f64 / agents.len() as f64)?;
        selected.extend(best_agents);

        // Add diverse agents
        let diverse_agents =
            self.select_diverse_agents(population, diverse_count as f64 / agents.len() as f64)?;
        selected.extend(diverse_agents);

        Ok(selected)
    }

    // Helper methods

    fn calculate_agent_distance(&self, agent1: &EvolvableAgent, agent2: &EvolvableAgent) -> f64 {
        let behavior1 = &agent1.genome.behavior;
        let behavior2 = &agent2.genome.behavior;

        let exploration_diff = (behavior1.exploration_rate - behavior2.exploration_rate).abs();
        let learning_diff = (behavior1.learning_rate - behavior2.learning_rate).abs();
        let risk_diff = (behavior1.risk_tolerance - behavior2.risk_tolerance).abs();

        (exploration_diff + learning_diff + risk_diff) / 3.0
    }
}
