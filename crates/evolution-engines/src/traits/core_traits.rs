//! Core trait definitions for evolution engines

use crate::error::EvolutionEngineResult;
use crate::metrics::EvolutionMetrics;
use crate::population::Population;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Configuration trait for evolution engines
pub trait EngineConfig: Debug + Send + Sync {
    /// Validate the configuration
    fn validate(&self) -> EvolutionEngineResult<()>;

    /// Get the engine name
    fn engine_name(&self) -> &str;
}

/// Trait for entities that can be evolved
#[async_trait]
pub trait Evolvable: Clone + Send + Sync + Debug {
    /// The genome type for this evolvable entity
    type Genome: Clone + Send + Sync + Debug + Serialize + for<'de> Deserialize<'de>;

    /// The fitness type
    type Fitness: Clone + Send + Sync + Debug + PartialOrd;

    /// Extract genome from the entity
    fn genome(&self) -> &Self::Genome;

    /// Create entity from genome
    async fn from_genome(genome: Self::Genome) -> EvolutionEngineResult<Self>;

    /// Evaluate fitness
    async fn evaluate_fitness(&self) -> EvolutionEngineResult<Self::Fitness>;

    /// Mutate the entity
    async fn mutate(&self, mutation_rate: f64) -> EvolutionEngineResult<Self>;

    /// Crossover with another entity
    async fn crossover(&self, other: &Self) -> EvolutionEngineResult<(Self, Self)>;
}

/// Base trait for evolution engines
#[async_trait]
pub trait EvolutionEngine: Send + Sync {
    /// The evolvable type this engine works with
    type Entity: Evolvable;

    /// The configuration type
    type Config: EngineConfig;

    /// Initialize the engine
    async fn initialize(config: Self::Config) -> EvolutionEngineResult<Self>
    where
        Self: Sized;

    /// Run a single evolution step
    async fn evolve_step(
        &mut self,
        population: Population<Self::Entity>,
    ) -> EvolutionEngineResult<Population<Self::Entity>>;

    /// Generate initial population
    async fn generate_initial_population(
        &self,
        size: usize,
    ) -> EvolutionEngineResult<Population<Self::Entity>>;

    /// Check if evolution should terminate
    async fn should_terminate(&self, metrics: &EvolutionMetrics) -> bool;

    /// Get current metrics
    fn metrics(&self) -> &EvolutionMetrics;

    /// Adapt parameters based on performance
    async fn adapt_parameters(&mut self, metrics: &EvolutionMetrics) -> EvolutionEngineResult<()>;
}
