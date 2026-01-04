//! ExoRust Evolution Primitives
//!
//! This crate provides evolution primitives for the ExoRust
//! self-evolving agent system.

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod agent_xp_sharing;
pub mod benchmarks;
pub mod channels;
pub mod engine;
pub mod error;
pub mod fitness;
pub mod mutation;
pub mod performance_xp_bridge;
pub mod population;
pub mod xp_rewards;

#[cfg(test)]
pub mod integration_tests;

// Production self-evolution modules (require CUDA)
#[cfg(feature = "cuda")]
pub mod kernel_hot_swap;
#[cfg(feature = "cuda")]
pub mod marketplace;
#[cfg(feature = "cuda")]
pub mod performance_profiler;

#[cfg(test)]
mod test_helpers;

pub use benchmarks::{
    EvolutionBenchmark, EvolutionBenchmarkConfig, EvolutionBenchmarkResults,
    FullEvolutionBenchmarkResults,
};
pub use engine::{
    AgentEvolutionEngine, GeneticEvolutionEngine, MaximizationFitnessFunction,
    SimpleFitnessFunction, TargetMatchingFitnessFunction, XPEvolutionEngine, XPEvolutionStats,
};
pub use error::EvolutionError;
pub use fitness::{
    AgentFitnessScore, AgentPerformanceFitnessFunction, FitnessFunction, FitnessScore,
    LevelBasedFitnessFunction, XPFitnessFunction,
};
pub use mutation::{Mutation, MutationType};
pub use population::{Individual, Population};

// Production self-evolution exports (require CUDA)
pub use agent_xp_sharing::{
    AgentXPSharingCoordinator, KnowledgePackage, KnowledgeType, LearningResult, LearningStats,
    XPSharingConfig, XPSharingResult,
};
pub use channels::{
    shared_channel_bridge, CapabilityUpdate, EvolutionChannelBridge, EvolutionEvent,
    SharedEvolutionChannelBridge,
};
#[cfg(feature = "cuda")]
pub use kernel_hot_swap::{
    HotSwapEvent, KernelHotSwap, KernelMetadata, ProductionWorkloadSimulator,
};
#[cfg(feature = "cuda")]
pub use marketplace::{
    BenchmarkResults, ClusterInfo, CompatibilityMatrix, DistributedConsensus, EvolutionMarketplace,
    EvolutionPackage, EvolutionParameters, KnowledgeTransferCoordinator, MarketplaceStats,
    SecurityManager, SyncCommand, TransferPriority, TransferRequest, UsageStatistics,
    ValidationResult,
};
#[cfg(feature = "cuda")]
pub use performance_profiler::{
    AutonomousOptimizer, GpuPerformanceMetrics, GpuPerformanceProfiler, OptimizationAction,
    OptimizationCommand, PerformanceFeedbackLoop, PerformanceStats, PerformanceTrend,
    ProfilerConfig,
};
pub use performance_xp_bridge::{
    MetricXPConfig, PerformanceMeasurement, PerformanceMetricType, PerformanceXPBreakdown,
    PerformanceXPConverter,
};
pub use xp_rewards::{EvolutionXPRewardCalculator, XPRewardBreakdown, XPRewardCategory};

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
