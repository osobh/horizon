//! Mock implementation for testing

use crate::error::EvolutionEngineResult;
use crate::traits::Evolvable;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Mock implementation of EvolvableAgent for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockEvolvableAgent {
    pub id: String,
    pub fitness: f64,
}

impl MockEvolvableAgent {
    /// Create a new mock agent
    pub fn new(id: String) -> Self {
        Self {
            id,
            fitness: 0.5, // Default fitness
        }
    }

    /// Get the agent's fitness
    pub fn get_fitness(&self) -> f64 {
        self.fitness
    }

    /// Set the agent's fitness
    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness.clamp(0.0, 1.0);
    }
}

#[async_trait]
impl Evolvable for MockEvolvableAgent {
    type Genome = String;
    type Fitness = f64;

    fn genome(&self) -> &Self::Genome {
        &self.id
    }

    async fn from_genome(genome: Self::Genome) -> EvolutionEngineResult<Self> {
        Ok(Self::new(genome))
    }

    async fn evaluate_fitness(&self) -> EvolutionEngineResult<Self::Fitness> {
        Ok(self.fitness)
    }

    async fn mutate(&self, _mutation_rate: f64) -> EvolutionEngineResult<Self> {
        let mut mutated = self.clone();
        mutated.fitness = (mutated.fitness + 0.01).clamp(0.0, 1.0);
        Ok(mutated)
    }

    async fn crossover(&self, other: &Self) -> EvolutionEngineResult<(Self, Self)> {
        let child1_fitness = (self.fitness + other.fitness) / 2.0;
        let child2_fitness = (self.fitness * 0.7 + other.fitness * 0.3).clamp(0.0, 1.0);

        let child1 = Self {
            id: format!("{}_x_{}", self.id, other.id),
            fitness: child1_fitness,
        };

        let child2 = Self {
            id: format!("{}_x_{}_alt", self.id, other.id),
            fitness: child2_fitness,
        };

        Ok((child1, child2))
    }
}
