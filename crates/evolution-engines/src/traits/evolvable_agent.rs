//! EvolvableAgent implementation

use crate::error::EvolutionEngineResult;
use crate::traits::{AgentGenome, Evolvable};
use async_trait::async_trait;
use exorust_agent_core::{Agent, AgentConfig};

/// Agent-specific evolvable implementation
pub struct EvolvableAgent {
    /// The agent
    pub agent: Agent,
    /// The agent's genome representation
    pub genome: AgentGenome,
}

impl std::fmt::Debug for EvolvableAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvolvableAgent")
            .field("agent_id", &self.agent.id())
            .field("genome", &self.genome)
            .finish()
    }
}

impl Clone for EvolvableAgent {
    fn clone(&self) -> Self {
        // Clone by creating a new agent with the same genome
        let config = AgentConfig {
            name: format!("cloned_agent_{}", uuid::Uuid::new_v4()),
            agent_type: "evolved".to_string(),
            max_memory: self.genome.architecture.memory_capacity,
            max_gpu_memory: self.genome.architecture.memory_capacity / 4,
            priority: 1,
            metadata: serde_json::Value::Null,
        };
        let agent = Agent::new(config).expect("Failed to clone agent");
        Self {
            agent,
            genome: self.genome.clone(),
        }
    }
}

// Implementation of Evolvable for EvolvableAgent
#[async_trait]
impl Evolvable for EvolvableAgent {
    type Genome = AgentGenome;
    type Fitness = f64;

    fn genome(&self) -> &Self::Genome {
        &self.genome
    }

    async fn from_genome(genome: Self::Genome) -> EvolutionEngineResult<Self> {
        let config = AgentConfig {
            name: format!("evolved_agent_{}", uuid::Uuid::new_v4()),
            agent_type: "evolved".to_string(),
            max_memory: genome.architecture.memory_capacity,
            max_gpu_memory: genome.architecture.memory_capacity / 4,
            priority: 1,
            metadata: serde_json::Value::Null,
        };
        let agent = Agent::new(config).map_err(|e| {
            crate::error::EvolutionEngineError::InitializationError {
                message: format!("Failed to create agent: {e}"),
            }
        })?;
        Ok(Self { agent, genome })
    }

    async fn evaluate_fitness(&self) -> EvolutionEngineResult<Self::Fitness> {
        // Mock fitness evaluation based on genome characteristics
        let arch_score = (self.genome.architecture.memory_capacity as f64 / 1_000_000.0).min(1.0)
            * 0.3
            + (self.genome.architecture.processing_units as f64 / 16.0).min(1.0) * 0.2
            + (self.genome.architecture.network_topology.len() as f64 / 10.0).min(1.0) * 0.2;

        let behav_score = (1.0 - (self.genome.behavior.exploration_rate - 0.2).abs() * 2.0) * 0.1
            + (1.0 - (self.genome.behavior.learning_rate - 0.01).abs() * 10.0) * 0.1
            + (1.0 - (self.genome.behavior.risk_tolerance - 0.5).abs()) * 0.1;

        Ok((arch_score + behav_score).max(0.0).min(1.0))
    }

    async fn mutate(&self, mutation_rate: f64) -> EvolutionEngineResult<Self> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::from_entropy();
        let mut mutated_genome = self.genome.clone();

        // Mutate architecture with probability
        if rng.gen_bool(mutation_rate.min(1.0)) {
            mutated_genome.architecture.memory_capacity =
                (mutated_genome.architecture.memory_capacity as f64 * rng.gen_range(0.8..1.2))
                    as usize;
            mutated_genome.architecture.processing_units =
                (mutated_genome.architecture.processing_units as f64 * rng.gen_range(0.8..1.2))
                    as u32;
        }

        // Mutate behavior
        if rng.gen_bool(mutation_rate.min(1.0)) {
            mutated_genome.behavior.exploration_rate = (mutated_genome.behavior.exploration_rate
                + rng.gen_range(-0.1..0.1))
            .clamp(0.0, 1.0);
            mutated_genome.behavior.learning_rate = (mutated_genome.behavior.learning_rate
                * rng.gen_range(0.5..2.0))
            .clamp(0.0001, 0.1);
            mutated_genome.behavior.risk_tolerance =
                (mutated_genome.behavior.risk_tolerance + rng.gen_range(-0.2..0.2)).clamp(0.0, 1.0);
        }

        Self::from_genome(mutated_genome).await
    }

    async fn crossover(&self, other: &Self) -> EvolutionEngineResult<(Self, Self)> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::from_entropy();

        // Create two children with mixed genes
        let mut child1_genome = self.genome.clone();
        let mut child2_genome = other.genome.clone();

        // Crossover architecture
        if rng.gen_bool(0.5) {
            std::mem::swap(
                &mut child1_genome.architecture,
                &mut child2_genome.architecture,
            );
        }

        // Crossover behavior parameters individually
        if rng.gen_bool(0.5) {
            std::mem::swap(
                &mut child1_genome.behavior.exploration_rate,
                &mut child2_genome.behavior.exploration_rate,
            );
        }
        if rng.gen_bool(0.5) {
            std::mem::swap(
                &mut child1_genome.behavior.learning_rate,
                &mut child2_genome.behavior.learning_rate,
            );
        }
        if rng.gen_bool(0.5) {
            std::mem::swap(
                &mut child1_genome.behavior.risk_tolerance,
                &mut child2_genome.behavior.risk_tolerance,
            );
        }

        let child1 = Self::from_genome(child1_genome).await?;
        let child2 = Self::from_genome(child2_genome).await?;

        Ok((child1, child2))
    }
}
