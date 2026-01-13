//! Hybrid evolution strategy coordination
//! 
//! This module coordinates multiple evolution strategies (ADAS, DGM, Swarm)
//! to leverage their complementary strengths through:
//! - Dynamic resource allocation
//! - Population migration
//! - Phase-based strategy selection
//! - Performance-based adaptation

use super::{adas::*, dgm::*, swarm::*, FitnessObjective};
use anyhow::{Result, anyhow};
use cudarc::driver::CudaContext;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;

/// Resource allocation strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResourceAllocation {
    /// Fixed allocation based on initial weights
    Fixed,
    /// Dynamic allocation based on performance
    Dynamic,
    /// Adaptive allocation based on evolution phase
    Adaptive,
}

/// Evolution phase
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvolutionPhase {
    /// Early exploration phase
    Exploration,
    /// Balanced exploration/exploitation
    Balanced,
    /// Late exploitation phase
    Exploitation,
}

/// Evolution strategy type
#[derive(Debug, Clone)]
pub enum EvolutionStrategy {
    Adas(AdasConfig),
    Dgm(DgmConfig),
    Swarm(SwarmConfig),
}

/// ADAS configuration
#[derive(Debug, Clone)]
pub struct AdasConfig {
    pub population_size: usize,
    pub max_code_size: usize,
    pub mutation_rate: f32,
}

impl Default for AdasConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_code_size: 512,
            mutation_rate: 0.05,
        }
    }
}

/// DGM configuration
#[derive(Debug, Clone)]
pub struct DgmConfig {
    pub population_size: usize,
    pub max_code_size: usize,
    pub archive_size: usize,
    pub self_modification_rate: f32,
}

impl Default for DgmConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_code_size: 512,
            archive_size: 20,
            self_modification_rate: 0.1,
        }
    }
}

/// Swarm configuration
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    pub population_size: usize,
    pub dimensions: usize,
    pub params: SwarmParams,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            dimensions: 64,
            params: SwarmParams::default(),
        }
    }
}

/// Hybrid coordinator configuration
#[derive(Debug, Clone)]
pub struct HybridConfig {
    pub population_size: usize,
    pub genome_size: usize,
    pub max_strategies: usize,
    pub adaptation_interval: usize,
    pub resource_allocation: ResourceAllocation,
    pub performance_window: usize,
    pub strategy_weights: Vec<f32>,
    pub enable_migration: bool,
    pub migration_rate: f32,
    pub migration_interval: usize,
    pub convergence_threshold: f32,
    pub convergence_window: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            population_size: 300,
            genome_size: 128,
            max_strategies: 3,
            adaptation_interval: 20,
            resource_allocation: ResourceAllocation::Dynamic,
            performance_window: 50,
            strategy_weights: vec![0.33, 0.33, 0.34],
            enable_migration: true,
            migration_rate: 0.05,
            migration_interval: 10,
            convergence_threshold: 0.001,
            convergence_window: 10,
        }
    }
}

/// Evolution metrics
#[derive(Debug, Clone)]
pub struct EvolutionMetrics {
    pub generation: usize,
    pub best_fitness: f64,
    pub average_fitness: f64,
    pub strategy_performances: HashMap<String, f64>,
    pub migrations_performed: usize,
    pub objective_scores: Vec<f64>,
    pub pareto_front_size: usize,
}

/// Coordinator status
#[derive(Debug, Clone)]
pub struct CoordinatorStatus {
    pub strategies_initialized: bool,
    pub total_population: usize,
    pub active_strategies: Vec<String>,
    pub current_phase: EvolutionPhase,
}

/// Strategy instance
struct StrategyInstance {
    strategy_type: EvolutionStrategy,
    population_share: f32,
    performance_history: Vec<f64>,
    current_performance: f64,
    adas: Option<AdasPopulation>,
    dgm: Option<DgmEngine>,
    swarm: Option<SwarmEngine>,
}

/// Hybrid evolution coordinator
pub struct HybridCoordinator {
    device: Arc<CudaContext>,
    config: HybridConfig,
    strategies: Arc<DashMap<String, StrategyInstance>>,
    generation: usize,
    phase: EvolutionPhase,
    fitness_objectives: Vec<FitnessObjective>,
    convergence_history: Vec<f64>,
    is_running: bool,
}

impl HybridCoordinator {
    /// Create new hybrid coordinator
    pub fn new(device: Arc<CudaContext>, config: HybridConfig) -> Result<Self> {
        Ok(Self {
            device,
            config,
            strategies: Arc::new(DashMap::new()),
            generation: 0,
            phase: EvolutionPhase::Exploration,
            fitness_objectives: vec![FitnessObjective::Performance],
            convergence_history: Vec::new(),
            is_running: false,
        })
    }

    /// Add evolution strategy
    pub fn add_strategy(&mut self, name: &str, strategy: EvolutionStrategy) -> Result<()> {
        if self.strategies.len() >= self.config.max_strategies {
            return Err(anyhow!("Maximum number of strategies reached"));
        }

        // Validate population size
        let strategy_pop_size = match &strategy {
            EvolutionStrategy::Adas(config) => config.population_size,
            EvolutionStrategy::Dgm(config) => config.population_size,
            EvolutionStrategy::Swarm(config) => config.population_size,
        };

        if strategy_pop_size > self.config.population_size {
            return Err(anyhow!("Strategy population exceeds total population limit"));
        }

        let population_share = if self.strategies.is_empty() {
            1.0
        } else {
            1.0 / (self.strategies.len() + 1) as f32
        };

        // Create strategy instance
        let instance = StrategyInstance {
            strategy_type: strategy.clone(),
            population_share,
            performance_history: Vec::new(),
            current_performance: 0.0,
            adas: None,
            dgm: None,
            swarm: None,
        };

        self.strategies.insert(name.to_string(), instance);

        // Rebalance population shares
        self.rebalance_population_shares()?;

        Ok(())
    }

    /// Initialize all strategies
    pub fn initialize_all(&mut self) -> Result<()> {
        for mut entry in self.strategies.iter_mut() {
            let instance = entry.value_mut();
            match &instance.strategy_type {
                EvolutionStrategy::Adas(config) => {
                    let mut adas = AdasPopulation::new(
                        Arc::clone(&self.device),
                        config.population_size,
                        config.max_code_size,
                    )?;
                    adas.initialize()?;
                    instance.adas = Some(adas);
                }
                EvolutionStrategy::Dgm(config) => {
                    let mut dgm = DgmEngine::new(
                        Arc::clone(&self.device),
                        config.population_size,
                        config.max_code_size,
                        config.archive_size,
                    )?;
                    dgm.initialize()?;
                    instance.dgm = Some(dgm);
                }
                EvolutionStrategy::Swarm(config) => {
                    let mut swarm = SwarmEngine::new(
                        Arc::clone(&self.device),
                        config.population_size,
                        config.dimensions,
                        config.params.clone(),
                    )?;
                    swarm.initialize()?;
                    instance.swarm = Some(swarm);
                }
            }
        }

        self.is_running = true;
        Ok(())
    }

    /// Get number of active strategies
    pub fn active_strategies(&self) -> usize {
        self.strategies.len()
    }

    /// Check if coordinator is running
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Get coordinator status
    pub fn get_status(&self) -> Result<CoordinatorStatus> {
        let strategies_initialized = self.strategies.iter().all(|entry| {
            let s = entry.value();
            s.adas.is_some() || s.dgm.is_some() || s.swarm.is_some()
        });

        let total_population = self.strategies.iter()
            .map(|entry| match &entry.value().strategy_type {
                EvolutionStrategy::Adas(c) => c.population_size,
                EvolutionStrategy::Dgm(c) => c.population_size,
                EvolutionStrategy::Swarm(c) => c.population_size,
            })
            .sum();

        Ok(CoordinatorStatus {
            strategies_initialized,
            total_population,
            active_strategies: self.strategies.iter().map(|e| e.key().clone()).collect(),
            current_phase: self.phase,
        })
    }

    /// Evolve one generation
    pub fn evolve_step(&mut self) -> Result<EvolutionMetrics> {
        if !self.is_running {
            return Err(anyhow!("Coordinator not initialized"));
        }

        let mut strategy_performances = HashMap::new();
        let mut all_fitness_values = Vec::new();

        // Evolve each strategy
        for mut entry in self.strategies.iter_mut() {
            let name = entry.key().clone();
            let instance = entry.value_mut();
            let performance = match &instance.strategy_type {
                EvolutionStrategy::Adas(_) => {
                    if let Some(adas) = &mut instance.adas {
                        adas.evolve()?;
                        let stats = adas.get_statistics()?;
                        all_fitness_values.push(stats.best_performance as f64);
                        stats.best_performance as f64
                    } else { 0.0 }
                }
                EvolutionStrategy::Dgm(_) => {
                    if let Some(dgm) = &mut instance.dgm {
                        dgm.evolve()?;
                        let stats = dgm.get_statistics()?;
                        all_fitness_values.push(stats.best_benchmark_score as f64);
                        stats.best_benchmark_score as f64
                    } else { 0.0 }
                }
                EvolutionStrategy::Swarm(_) => {
                    if let Some(swarm) = &mut instance.swarm {
                        swarm.step()?;
                        let stats = swarm.get_statistics()?;
                        all_fitness_values.push(stats.best_fitness as f64);
                        stats.best_fitness as f64
                    } else { 0.0 }
                }
            };

            instance.current_performance = performance;
            instance.performance_history.push(performance);
            strategy_performances.insert(name, performance);
        }

        self.generation += 1;

        // Perform migration if enabled
        let migrations_performed = if self.config.enable_migration &&
                                     self.generation % self.config.migration_interval == 0 {
            self.perform_migration()?
        } else {
            0
        };

        // Adapt resources if needed
        if self.generation % self.config.adaptation_interval == 0 {
            self.adapt_resources()?;
        }

        // Calculate metrics
        let best_fitness = all_fitness_values.iter().cloned().fold(0.0, f64::max);
        let average_fitness = if !all_fitness_values.is_empty() {
            all_fitness_values.iter().sum::<f64>() / all_fitness_values.len() as f64
        } else {
            0.0
        };

        self.convergence_history.push(best_fitness);

        Ok(EvolutionMetrics {
            generation: self.generation,
            best_fitness,
            average_fitness,
            strategy_performances,
            migrations_performed,
            objective_scores: vec![best_fitness], // Simplified for now
            pareto_front_size: 1, // Simplified
        })
    }

    /// Set fitness objectives
    pub fn set_fitness_objectives(&mut self, objectives: Vec<FitnessObjective>) -> Result<()> {
        self.fitness_objectives = objectives;
        Ok(())
    }

    /// Set evolution phase
    pub fn set_evolution_phase(&mut self, phase: EvolutionPhase) -> Result<()> {
        self.phase = phase;

        // Adjust strategy parameters based on phase
        for mut entry in self.strategies.iter_mut() {
            let instance = entry.value_mut();
            match (&instance.strategy_type, phase) {
                (EvolutionStrategy::Adas(_), EvolutionPhase::Exploration) => {
                    // Increase mutation rate for exploration
                    if let Some(adas) = &mut instance.adas {
                        adas.set_mutation_rate(0.15)?;
                    }
                }
                (EvolutionStrategy::Swarm(_), EvolutionPhase::Exploitation) => {
                    // Decrease inertia for exploitation
                    if let Some(swarm) = &mut instance.swarm {
                        swarm.set_inertia_weight(0.4)?;
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Get resource allocations
    pub fn get_resource_allocations(&self) -> Result<HashMap<String, f32>> {
        let mut allocations = HashMap::new();

        for entry in self.strategies.iter() {
            allocations.insert(entry.key().clone(), entry.value().population_share);
        }

        Ok(allocations)
    }

    /// Set strategy performance (for testing)
    pub fn set_strategy_performance(&mut self, name: &str, performance: f64) -> Result<()> {
        if let Some(mut instance) = self.strategies.get_mut(name) {
            instance.current_performance = performance;
            instance.performance_history.push(performance);
            Ok(())
        } else {
            Err(anyhow!("Strategy not found"))
        }
    }

    /// Check if evolution has converged
    pub fn has_converged(&self) -> Result<bool> {
        if self.convergence_history.len() < self.config.convergence_window {
            return Ok(false);
        }

        let window = &self.convergence_history[self.convergence_history.len() - self.config.convergence_window..];
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let variance = window.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / window.len() as f64;

        Ok(variance.sqrt() < self.config.convergence_threshold)
    }

    /// Rebalance population shares
    fn rebalance_population_shares(&self) -> Result<()> {
        let num_strategies = self.strategies.len() as f32;

        if num_strategies == 0.0 {
            return Ok(());
        }

        let equal_share = 1.0 / num_strategies;

        for mut entry in self.strategies.iter_mut() {
            entry.value_mut().population_share = equal_share;
        }

        Ok(())
    }

    /// Perform migration between populations
    fn perform_migration(&self) -> Result<usize> {
        let migration_count = (self.config.migration_rate * self.config.population_size as f32) as usize;

        // Simple ring migration for now
        let strategy_names: Vec<String> = self.strategies.iter().map(|e| e.key().clone()).collect();

        if strategy_names.len() < 2 {
            return Ok(0);
        }

        // Migrate best individuals in a ring
        for i in 0..strategy_names.len() {
            let _source = &strategy_names[i];
            let _target = &strategy_names[(i + 1) % strategy_names.len()];

            // Get best individuals from source
            // (Simplified - in real implementation would transfer actual individuals)
            // Mark migration occurred
            // In real implementation, would transfer genetic material
        }

        Ok(migration_count)
    }

    /// Adapt resource allocation based on performance
    fn adapt_resources(&self) -> Result<()> {
        match self.config.resource_allocation {
            ResourceAllocation::Fixed => Ok(()),
            ResourceAllocation::Dynamic => {
                // Calculate performance-based weights
                let total_performance: f64 = self.strategies.iter()
                    .map(|e| e.value().current_performance.max(0.1)) // Avoid zero
                    .sum();

                if total_performance > 0.0 {
                    for mut entry in self.strategies.iter_mut() {
                        let instance = entry.value_mut();
                        instance.population_share =
                            (instance.current_performance.max(0.1) / total_performance) as f32;
                    }
                }

                Ok(())
            }
            ResourceAllocation::Adaptive => {
                // Adapt based on evolution phase
                match self.phase {
                    EvolutionPhase::Exploration => {
                        // Favor ADAS for exploration
                        for mut entry in self.strategies.iter_mut() {
                            let instance = entry.value_mut();
                            match instance.strategy_type {
                                EvolutionStrategy::Adas(_) => instance.population_share = 0.5,
                                _ => instance.population_share = 0.25,
                            }
                        }
                    }
                    EvolutionPhase::Exploitation => {
                        // Favor Swarm for exploitation
                        for mut entry in self.strategies.iter_mut() {
                            let instance = entry.value_mut();
                            match instance.strategy_type {
                                EvolutionStrategy::Swarm(_) => instance.population_share = 0.5,
                                _ => instance.population_share = 0.25,
                            }
                        }
                    }
                    EvolutionPhase::Balanced => {
                        // Equal allocation
                        self.rebalance_population_shares()?;
                    }
                }

                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_config_default() {
        let config = HybridConfig::default();
        assert_eq!(config.population_size, 300);
        assert_eq!(config.max_strategies, 3);
        assert!(config.enable_migration);
    }

    #[test]
    fn test_evolution_strategy_creation() {
        let adas = EvolutionStrategy::Adas(AdasConfig::default());
        match adas {
            EvolutionStrategy::Adas(config) => {
                assert_eq!(config.population_size, 100);
                assert_eq!(config.mutation_rate, 0.05);
            }
            _ => panic!("Wrong strategy type"),
        }
    }
}