//! Population Evolution Manager for Darwin Gödel Machine
//!
//! This module manages parallel evolution of multiple agent populations,
//! enabling diverse exploration paths and cross-population knowledge transfer.

pub mod manager;
pub mod migration;
pub mod population;
pub mod types;

pub use manager::PopulationEvolutionManager;
pub use migration::MigrationController;
pub use population::ManagedPopulation;
pub use types::{
    CrossoverStrategy, EvolutionPopulation, MigrationPolicy, PopulationConfig, PopulationMetrics,
    PopulationMigration, SelectionStrategy,
};

#[cfg(test)]
mod tests;
