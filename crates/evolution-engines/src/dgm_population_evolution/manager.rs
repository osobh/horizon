//! Population evolution manager implementation

use super::migration::MigrationController;
use super::population::ManagedPopulation;
use super::types::*;
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use crate::traits::EvolvableAgent;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Manages multiple populations for parallel evolution
pub struct PopulationEvolutionManager {
    /// Configuration
    config: PopulationConfig,
    /// Managed populations
    populations: Arc<DashMap<String, ManagedPopulation>>,
    /// Migration controller
    migration_controller: MigrationController,
    /// Current generation across all populations
    current_generation: u32,
    /// Global best fitness
    global_best_fitness: f64,
    /// Start time for resource tracking
    start_time: Instant,
}

impl PopulationEvolutionManager {
    /// Create new population evolution manager
    pub fn new(config: PopulationConfig) -> EvolutionEngineResult<Self> {
        let migration_controller = MigrationController::new(config.migration_policy.clone())?;
        let populations = Arc::new(DashMap::new());

        let mut manager = Self {
            config,
            populations,
            migration_controller,
            current_generation: 0,
            global_best_fitness: 0.0,
            start_time: Instant::now(),
        };

        // Initialize populations
        manager.initialize_populations()?;

        Ok(manager)
    }

    /// Get population count
    pub fn get_population_count(&self) -> usize {
        self.populations.len()
    }

    /// Get configuration
    pub fn get_config(&self) -> &PopulationConfig {
        &self.config
    }

    /// Get a specific population
    pub fn get_population(&self, id: &str) -> Option<ManagedPopulation> {
        self.populations.get(id).map(|r| r.clone())
    }

    /// Initialize populations with empty agents (to be filled later)
    fn initialize_populations(&mut self) -> EvolutionEngineResult<()> {
        for i in 0..self.config.num_populations {
            let pop_id = format!("population_{}", i);
            let empty_agents = Vec::new(); // Will be initialized later
            let population = ManagedPopulation::new(pop_id.clone(), empty_agents)?;
            self.populations.insert(pop_id, population);
        }

        Ok(())
    }

    /// Initialize a specific population with agents
    pub fn initialize_population(
        &mut self,
        id: &str,
        agents: Vec<EvolvableAgent>,
    ) -> EvolutionEngineResult<()> {
        let population = ManagedPopulation::new(id.to_string(), agents)?;
        self.populations.insert(id.to_string(), population);
        Ok(())
    }

    /// Evolve all populations for one generation
    pub fn evolve_generation(&mut self) -> EvolutionEngineResult<()> {
        for mut entry in self.populations.iter_mut() {
            let population = entry.value_mut();
            if !population.is_converged() {
                population.evolve_generation(
                    &self.config.selection_strategy,
                    &self.config.crossover_strategy,
                )?;
            }
        }

        self.current_generation += 1;

        // Update global best
        let mut best_fitness = self.global_best_fitness;
        for entry in self.populations.iter() {
            let pop_best = entry.value().get_best_fitness();
            if pop_best > best_fitness {
                best_fitness = pop_best;
            }
        }
        self.global_best_fitness = best_fitness;

        Ok(())
    }

    /// Perform migrations between populations
    pub fn perform_migrations(&mut self, generation: u32) -> EvolutionEngineResult<()> {
        if generation % self.config.migration_frequency != 0 {
            return Ok(());
        }

        let pop_ids: Vec<String> = self.populations.iter().map(|e| e.key().clone()).collect();

        // Plan migrations between population pairs
        for i in 0..pop_ids.len() {
            for j in 0..pop_ids.len() {
                if i != j {
                    let source = self.populations.get(&pop_ids[i]);
                    let target = self.populations.get(&pop_ids[j]);

                    if let (Some(src), Some(tgt)) = (source, target) {
                        let migration = self
                            .migration_controller
                            .plan_migration(&src, &tgt, generation)?;
                        if !migration.agents.is_empty() {
                            self.migration_controller.record_migration(migration)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Calculate comprehensive metrics
    pub fn calculate_metrics(&self) -> EvolutionEngineResult<PopulationMetrics> {
        let total_populations = self.populations.len();
        let mut active_populations = 0;
        let mut global_best_fitness = 0.0;
        let mut total_fitness = 0.0;
        let mut fitness_count = 0;
        let mut population_diversities = HashMap::new();
        let mut convergence_status = HashMap::new();

        for entry in self.populations.iter() {
            let id = entry.key();
            let population = entry.value();

            if !population.is_converged() {
                active_populations += 1;
            }

            let best_fitness = population.get_best_fitness();
            if best_fitness > global_best_fitness {
                global_best_fitness = best_fitness;
            }

            total_fitness += population.get_average_fitness();
            fitness_count += 1;

            population_diversities.insert(id.clone(), population.calculate_diversity());
            convergence_status.insert(id.clone(), population.is_converged());
        }

        let average_fitness = if fitness_count > 0 {
            total_fitness / fitness_count as f64
        } else {
            0.0
        };

        let migration_stats = MigrationStats {
            total_migrations: self.migration_controller.get_migration_count(),
            successful_migrations: self.migration_controller.get_migration_count(), // Simplified
            average_improvement: 0.1,                                               // Simplified
            migration_frequency: self.config.migration_frequency as f64,
        };

        let resource_utilization = self.calculate_resource_utilization();

        Ok(PopulationMetrics {
            total_populations,
            active_populations,
            global_best_fitness,
            average_fitness,
            population_diversities,
            migration_stats,
            convergence_status,
            resource_utilization,
        })
    }

    /// Assess population health
    pub fn assess_population_health(
        &self,
        population_id: &str,
    ) -> EvolutionEngineResult<PopulationHealthReport> {
        let population = self.populations.get(population_id).ok_or_else(|| {
            EvolutionEngineError::Other(format!("Population {} not found", population_id))
        })?;

        let diversity_score = population.calculate_diversity();
        let is_healthy =
            diversity_score >= self.config.diversity_threshold && !population.is_converged();

        let mut recommendations = Vec::new();
        if diversity_score < self.config.diversity_threshold {
            recommendations.push("Increase mutation rate to improve diversity".to_string());
            recommendations
                .push("Consider introducing new agents from other populations".to_string());
        }
        if population.is_converged() {
            recommendations
                .push("Population has converged - consider restarting or migration".to_string());
        }

        Ok(PopulationHealthReport {
            population_id: population_id.to_string(),
            is_healthy,
            diversity_score,
            convergence_status: population.is_converged(),
            generation: population.generation(),
            recommendations,
        })
    }

    // Helper methods

    fn calculate_resource_utilization(&self) -> ResourceUtilization {
        let _elapsed = self.start_time.elapsed();
        let cpu_usage = 50.0; // Simplified simulation
        let memory_usage = 100_000_000; // 100MB simulated
        let evaluation_time = Duration::from_millis(100);
        let parallel_efficiency = 0.8; // 80% efficiency

        ResourceUtilization {
            cpu_usage,
            memory_usage,
            evaluation_time,
            parallel_efficiency,
        }
    }
}

/// Population health assessment report
#[derive(Debug, Clone)]
pub struct PopulationHealthReport {
    /// Population ID
    pub population_id: String,
    /// Whether population is healthy
    pub is_healthy: bool,
    /// Diversity score
    pub diversity_score: f64,
    /// Convergence status
    pub convergence_status: bool,
    /// Current generation
    pub generation: u32,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}
