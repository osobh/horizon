//! Base traits for evolution engines

mod agent_genome;
mod core_traits;
mod evolvable_agent;
mod mock_agent;

#[cfg(test)]
mod tests;

pub use agent_genome::{AgentGenome, ArchitectureGenes, BehaviorGenes};
pub use core_traits::{EngineConfig, EvolutionEngine, Evolvable};
pub use evolvable_agent::EvolvableAgent;
pub use mock_agent::MockEvolvableAgent;
