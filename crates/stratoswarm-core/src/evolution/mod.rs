//! Evolution service module for channel-based evolution engine communication.
//!
//! This module provides an evolutionary algorithm service that processes commands
//! through channels, enabling distributed evolution with GPU acceleration.
//!
//! # Architecture
//!
//! - **EvolutionService**: Main service that processes evolution commands from a channel
//! - **Population**: Structure of Arrays (SoA) layout for GPU-efficient population storage
//! - **FitnessFunction**: Trait for defining fitness evaluation functions
//! - **SelectionStrategy**: Various selection strategies (tournament, roulette, rank, elite)
//! - **MutationOperator**: Various mutation operators (Gaussian, uniform, polynomial)
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use stratoswarm_core::evolution::{EvolutionService, EvolutionConfig};
//! use stratoswarm_core::channels::ChannelRegistry;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create channel registry
//!     let registry = ChannelRegistry::new();
//!
//!     // Create evolution service
//!     let config = EvolutionConfig::default();
//!     let service = EvolutionService::new(config);
//!
//!     // Get channels
//!     let evolution_rx = registry.subscribe_evolution();
//!     let events_tx = registry.event_sender();
//!
//!     // Spawn the evolution service
//!     tokio::spawn(async move {
//!         service.run(evolution_rx, events_tx).await
//!     });
//!
//!     // Send evolution commands via the registry
//!     let evolution_tx = registry.evolution_sender();
//!     // ... send commands ...
//! }
//! ```

pub mod fitness;
pub mod mutation;
pub mod population;
pub mod selection;
pub mod service;

// Re-export main types
pub use fitness::{FitnessFunction, RastriginFunction, RosenbrockFunction, SphereFunction};
pub use mutation::{
    GaussianMutation, MutationOperator, PolynomialMutation, UniformMutation,
};
pub use population::{Individual, Population, PopulationId, PopulationStats};
pub use selection::{
    EliteSelection, RankSelection, RouletteSelection, SelectionStrategy, TournamentSelection,
};
pub use service::{EvolutionConfig, EvolutionMetrics, EvolutionService};
