//! Evolution engines for advanced agent synthesis
//!
//! This crate provides three evolution engines:
//! - ADAS: Automated Design of Agentic Systems
//! - SwarmAgentic: Population-based optimization
//! - DGM: Discovered Agent Growth Mode

#![warn(missing_docs)]

pub mod adas;
pub mod adas_meta_agent;
pub mod config;
pub mod dgm;
pub mod dgm_code_modification;
pub mod dgm_darwin_archive;
pub mod dgm_darwin_integration;
pub mod dgm_empirical_validation;
pub mod dgm_population_evolution;
pub mod dgm_self_assessment;
pub mod error;
pub mod hybrid;
pub mod metrics;
pub mod population;
pub mod swarm;
pub mod swarm_distributed;
pub mod swarm_fault_tolerance;
pub mod swarm_network;
pub mod swarm_particle;
pub mod swarm_topology;
pub mod swarm_velocity;
pub mod swarm_xp_bridge;
pub mod traits;

pub use config::EvolutionEngineConfig;
pub use error::{EvolutionEngineError, EvolutionEngineResult};
pub use metrics::EvolutionMetrics;
pub use swarm_xp_bridge::{SwarmXPEngine, SwarmXPFitnessFunction, SwarmXPStats};
pub use traits::{EngineConfig, EvolutionEngine, Evolvable};

/// Initialize the evolution engines subsystem
pub async fn init() -> EvolutionEngineResult<()> {
    tracing::info!("Initializing evolution engines subsystem");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_evolution_engines_init() {
        assert!(init().await.is_ok());
    }
}
