//! Type definitions for population evolution system

use crate::traits::EvolvableAgent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Configuration for population evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationConfig {
    /// Number of parallel populations
    pub num_populations: usize,
    /// Size of each population
    pub population_size: usize,
    /// Maximum generations per population
    pub max_generations: u32,
    /// Migration frequency (every N generations)
    pub migration_frequency: u32,
    /// Selection strategy for evolution
    pub selection_strategy: SelectionStrategy,
    /// Crossover strategy for breeding
    pub crossover_strategy: CrossoverStrategy,
    /// Migration policy between populations
    pub migration_policy: MigrationPolicy,
    /// Diversity threshold for population health
    pub diversity_threshold: f64,
    /// Performance convergence threshold
    pub convergence_threshold: f64,
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            num_populations: 4,
            population_size: 50,
            max_generations: 100,
            migration_frequency: 10,
            selection_strategy: SelectionStrategy::Tournament { size: 3 },
            crossover_strategy: CrossoverStrategy::Uniform { rate: 0.7 },
            migration_policy: MigrationPolicy::BestAgent { rate: 0.1 },
            diversity_threshold: 0.3,
            convergence_threshold: 0.01,
        }
    }
}

/// Selection strategies for evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Tournament selection with given size
    Tournament { size: usize },
    /// Roulette wheel selection
    RouletteWheel,
    /// Rank-based selection
    Rank,
    /// Elite selection (top N)
    Elite { count: usize },
}

/// Crossover strategies for breeding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossoverStrategy {
    /// Single point crossover
    SinglePoint,
    /// Two point crossover
    TwoPoint,
    /// Uniform crossover with rate
    Uniform { rate: f64 },
    /// Blend crossover for continuous values
    Blend { alpha: f64 },
}

/// Migration policies between populations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationPolicy {
    /// Migrate best agents
    BestAgent { rate: f64 },
    /// Random migration
    Random { rate: f64 },
    /// Diversity-based migration
    DiversityBased { rate: f64 },
    /// Ring topology migration
    RingTopology { rate: f64 },
    /// Island model migration
    IslandModel { rate: f64 },
}

/// Managed population with evolution state
#[derive(Debug, Clone)]
pub struct EvolutionPopulation {
    /// Population ID
    pub id: String,
    /// Current agents in population
    pub agents: Vec<EvolvableAgent>,
    /// Current generation number
    pub generation: u32,
    /// Best fitness achieved
    pub best_fitness: f64,
    /// Average fitness
    pub average_fitness: f64,
    /// Population diversity metric
    pub diversity: f64,
    /// Convergence status
    pub converged: bool,
    /// Evolution history
    pub fitness_history: Vec<f64>,
}

/// Migration event between populations
#[derive(Debug, Clone)]
pub struct PopulationMigration {
    /// Source population ID
    pub source_population: String,
    /// Target population ID
    pub target_population: String,
    /// Migrated agents
    pub agents: Vec<EvolvableAgent>,
    /// Migration generation
    pub generation: u32,
    /// Migration reason
    pub reason: String,
}

/// Metrics for population evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationMetrics {
    /// Total populations
    pub total_populations: usize,
    /// Active populations (not converged)
    pub active_populations: usize,
    /// Overall best fitness
    pub global_best_fitness: f64,
    /// Average fitness across populations
    pub average_fitness: f64,
    /// Population diversity scores
    pub population_diversities: HashMap<String, f64>,
    /// Migration statistics
    pub migration_stats: MigrationStats,
    /// Convergence status per population
    pub convergence_status: HashMap<String, bool>,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Migration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStats {
    /// Total migrations performed
    pub total_migrations: usize,
    /// Successful migrations (improved target)
    pub successful_migrations: usize,
    /// Average improvement from migration
    pub average_improvement: f64,
    /// Migration frequency
    pub migration_frequency: f64,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Evaluation time per generation
    pub evaluation_time: Duration,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
}
