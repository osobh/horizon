//! Snapshot Management
//!
//! Handles creation and management of evolution snapshots

use crate::time_travel::evolution_debugger::{AgentGenome, EvolutionState};
use std::time::SystemTime;

/// Evolution snapshot containing complete state information
#[derive(Debug, Clone)]
pub struct EvolutionSnapshot {
    pub id: String,
    pub generation: usize,
    pub timestamp: SystemTime,
    pub state: EvolutionState,
    pub fitness_metrics: FitnessMetrics,
    pub population: Vec<AgentGenome>,
}

/// Fitness metrics for snapshot
#[derive(Debug, Clone)]
pub struct FitnessMetrics {
    pub best_fitness: f64,
    pub average_fitness: f64,
    pub worst_fitness: f64,
    pub fitness_std_dev: f64,
}

impl EvolutionSnapshot {
    /// Create new evolution snapshot
    pub fn new(
        id: String,
        generation: usize,
        state: EvolutionState,
        population: Vec<AgentGenome>,
    ) -> Self {
        let fitness_metrics = FitnessMetrics::calculate_from_population(&population);

        Self {
            id,
            generation,
            timestamp: SystemTime::now(),
            state,
            fitness_metrics,
            population,
        }
    }

    /// Get snapshot size in bytes (estimated)
    pub fn estimated_size(&self) -> usize {
        // Simplified size calculation for GREEN phase
        self.population.len() * 1024 + 512 // Rough estimate
    }
}

impl FitnessMetrics {
    /// Calculate fitness metrics from population
    pub fn calculate_from_population(population: &[AgentGenome]) -> Self {
        if population.is_empty() {
            return Self {
                best_fitness: 0.0,
                average_fitness: 0.0,
                worst_fitness: 0.0,
                fitness_std_dev: 0.0,
            };
        }

        let fitnesses: Vec<f64> = population.iter().map(|agent| agent.fitness).collect();
        let best_fitness = fitnesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let worst_fitness = fitnesses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let average_fitness = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;

        // Calculate standard deviation
        let variance = fitnesses
            .iter()
            .map(|&x| (x - average_fitness).powi(2))
            .sum::<f64>()
            / fitnesses.len() as f64;
        let fitness_std_dev = variance.sqrt();

        Self {
            best_fitness,
            average_fitness,
            worst_fitness,
            fitness_std_dev,
        }
    }
}
