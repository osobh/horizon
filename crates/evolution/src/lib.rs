//! ExoRust Evolution Primitives
//!
//! This crate provides evolution primitives for the ExoRust
//! self-evolving agent system.

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod benchmarks;
pub mod engine;
pub mod error;
pub mod fitness;
pub mod mutation;
pub mod population;
pub mod xp_rewards;
pub mod performance_xp_bridge;
pub mod agent_xp_sharing;

#[cfg(test)]
pub mod integration_tests;

// Production self-evolution modules
pub mod kernel_hot_swap;
pub mod performance_profiler;
pub mod marketplace;

#[cfg(test)]
mod test_helpers;

pub use benchmarks::{
    EvolutionBenchmark, EvolutionBenchmarkConfig, EvolutionBenchmarkResults,
    FullEvolutionBenchmarkResults,
};
pub use engine::{
    GeneticEvolutionEngine, MaximizationFitnessFunction, SimpleFitnessFunction,
    TargetMatchingFitnessFunction, XPEvolutionEngine, AgentEvolutionEngine,
    XPEvolutionStats,
};
pub use error::EvolutionError;
pub use fitness::{
    FitnessFunction, FitnessScore, XPFitnessFunction, AgentFitnessScore,
    AgentPerformanceFitnessFunction, LevelBasedFitnessFunction,
};
pub use mutation::{Mutation, MutationType};
pub use population::{Individual, Population};

// Production self-evolution exports
pub use kernel_hot_swap::{
    KernelHotSwap, KernelMetadata, HotSwapEvent, ProductionWorkloadSimulator,
};
pub use performance_profiler::{
    GpuPerformanceProfiler, GpuPerformanceMetrics, ProfilerConfig, PerformanceTrend,
    OptimizationAction, OptimizationCommand, AutonomousOptimizer, PerformanceFeedbackLoop,
    PerformanceStats,
};
pub use marketplace::{
    EvolutionMarketplace, EvolutionPackage, BenchmarkResults, CompatibilityMatrix,
    UsageStatistics, DistributedConsensus, ValidationResult, EvolutionParameters,
    MarketplaceStats, ClusterInfo, SyncCommand, KnowledgeTransferCoordinator,
    TransferRequest, TransferPriority, SecurityManager,
};
pub use xp_rewards::{
    EvolutionXPRewardCalculator, XPRewardCategory, XPRewardBreakdown,
};
pub use performance_xp_bridge::{
    PerformanceXPConverter, PerformanceMetricType, PerformanceMeasurement,
    PerformanceXPBreakdown, MetricXPConfig,
};
pub use agent_xp_sharing::{
    AgentXPSharingCoordinator, KnowledgeType, KnowledgePackage, LearningResult,
    XPSharingConfig, LearningStats, XPSharingResult,
};

/// Evolution engine interface
#[async_trait::async_trait]
pub trait EvolutionEngine: Send + Sync {
    /// Create initial population
    async fn initialize_population(&self, size: usize) -> Result<Population, EvolutionError>;

    /// Evolve population for one generation
    async fn evolve_generation(&self, population: &mut Population) -> Result<(), EvolutionError>;

    /// Evaluate fitness of population
    async fn evaluate_fitness(
        &self,
        population: &Population,
    ) -> Result<Vec<FitnessScore>, EvolutionError>;

    /// Get evolution statistics
    async fn stats(&self) -> Result<EvolutionStats, EvolutionError>;
}

/// Evolution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvolutionStats {
    pub generation: u64,
    pub population_size: usize,
    pub best_fitness: f64,
    pub average_fitness: f64,
    pub mutations_per_second: f64,
    pub diversity_index: f64,
}
