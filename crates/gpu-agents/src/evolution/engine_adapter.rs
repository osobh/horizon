//! Evolution engines adapter for gpu-agents integration
//!
//! This module provides adapters to connect the evolution-engines crate
//! with gpu-agents, enabling ADAS, DGM, and SwarmAgentic algorithms.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;

// Import from exorust-evolution-engines crate
use stratoswarm_evolution_engines::{
    adas::{AdasConfig, AdasEngine},
    dgm::{DgmConfig, DgmEngine},
    population::{Individual, Population},
    swarm::{SwarmConfig, SwarmEngine},
    traits::{AgentGenome, ArchitectureGenes, BehaviorGenes, EvolvableAgent},
    EngineConfig, EvolutionEngine, EvolutionEngineError, EvolutionEngineResult, EvolutionMetrics,
    Evolvable,
};

// Import from exorust-agent-core
use stratoswarm_agent_core::{Agent, AgentConfig, AgentId, Goal, GoalPriority};

// Import local types
use crate::consensus_synthesis::integration::ConsensusSynthesisEngine;

/// Adapter for evolution engines integration with consensus-synthesis
pub struct EvolutionEngineAdapter {
    /// ADAS evolution engine
    adas_engine: Option<AdasEngine>,
    /// DGM evolution engine  
    dgm_engine: Option<DgmEngine>,
    /// SwarmAgentic evolution engine
    swarm_engine: Option<SwarmEngine>,
    /// Current population of evolving agents
    population: Population<EvolvableAgent>,
    /// Metrics tracking
    metrics: EvolutionMetrics,
    /// GPU device for accelerated evolution
    device: Arc<cudarc::driver::CudaContext>,
}

impl EvolutionEngineAdapter {
    /// Create a new evolution engine adapter
    pub async fn new(device: Arc<cudarc::driver::CudaContext>) -> Result<Self> {
        // Initialize ADAS engine
        let mut adas_config = AdasConfig::default();
        adas_config.base.population_size = 100;
        adas_config.base.mutation_rate = 0.1;
        adas_config.base.max_generations = 1000;
        adas_config.base.target_fitness = Some(0.95);
        adas_config.architecture_mutation_prob = 0.3;
        adas_config.behavior_mutation_prob = 0.7;
        let adas_engine = AdasEngine::initialize(adas_config)
            .await
            .context("Failed to initialize ADAS engine")?;

        // Initialize DGM engine
        let mut dgm_config = DgmConfig::default();
        dgm_config.base.population_size = 80;
        dgm_config.base.mutation_rate = 0.12;
        dgm_config.discovery_rate = 0.15;
        dgm_config.pattern_retention_threshold = 0.8;
        let dgm_engine = DgmEngine::initialize(dgm_config)
            .await
            .context("Failed to initialize DGM engine")?;

        // Initialize SwarmAgentic engine
        let mut swarm_config = SwarmConfig::default();
        swarm_config.base.population_size = 50;
        swarm_config.base.mutation_rate = 0.08;
        swarm_config.neighborhood_size = 5;
        swarm_config.social_influence = 2.3;
        swarm_config.cognitive_influence = 1.7;
        let swarm_engine = SwarmEngine::initialize(swarm_config)
            .await
            .context("Failed to initialize SwarmAgentic engine")?;

        // Create initial population
        let population = Self::create_initial_population(100).await?;

        Ok(Self {
            adas_engine: Some(adas_engine),
            dgm_engine: Some(dgm_engine),
            swarm_engine: Some(swarm_engine),
            population,
            metrics: EvolutionMetrics::default(),
            device,
        })
    }

    /// Create initial population of evolvable agents
    async fn create_initial_population(size: usize) -> Result<Population<EvolvableAgent>> {
        let mut individuals = Vec::with_capacity(size);

        for i in 0..size {
            let genome = AgentGenome {
                goal: Goal::new(
                    format!("Consensus optimization agent {}", i),
                    GoalPriority::Normal,
                ),
                architecture: ArchitectureGenes {
                    memory_capacity: 1024 + (i * 512),
                    processing_units: 4 + (i % 8) as u32,
                    network_topology: vec![
                        (10 + i % 20) as u32,
                        (20 + i % 30) as u32,
                        (10 + i % 15) as u32,
                    ],
                },
                behavior: BehaviorGenes {
                    exploration_rate: 0.1 + (i as f64 * 0.01) % 0.8,
                    learning_rate: 0.001 + (i as f64 * 0.001) % 0.1,
                    risk_tolerance: 0.1 + (i as f64 * 0.02) % 0.8,
                },
            };

            let agent = EvolvableAgent::from_genome(genome)
                .await
                .context("Failed to create agent from genome")?;
            let individual = Individual::new(agent);
            individuals.push(individual);
        }

        Ok(Population::from_individuals(individuals))
    }

    /// Evolve agents using ADAS algorithm
    pub async fn evolve_with_adas(&mut self) -> Result<EvolutionMetrics> {
        let engine = self
            .adas_engine
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("ADAS engine not initialized"))?;

        // Run evolution step
        self.population = engine
            .evolve_step(self.population.clone())
            .await
            .context("Failed to evolve with ADAS")?;

        // Update metrics
        self.metrics = engine.metrics().clone();

        Ok(self.metrics.clone())
    }

    /// Evolve agents using DGM algorithm  
    pub async fn evolve_with_dgm(&mut self) -> Result<EvolutionMetrics> {
        let engine = self
            .dgm_engine
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("DGM engine not initialized"))?;

        // Run evolution step
        self.population = engine
            .evolve_step(self.population.clone())
            .await
            .context("Failed to evolve with DGM")?;

        // Update metrics
        self.metrics = engine.metrics().clone();

        Ok(self.metrics.clone())
    }

    /// Evolve agents using SwarmAgentic algorithm
    pub async fn evolve_with_swarm_agentic(&mut self) -> Result<EvolutionMetrics> {
        let engine = self
            .swarm_engine
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("SwarmAgentic engine not initialized"))?;

        // Run evolution step
        self.population = engine
            .evolve_step(self.population.clone())
            .await
            .context("Failed to evolve with SwarmAgentic")?;

        // Update metrics
        self.metrics = engine.metrics().clone();

        Ok(self.metrics.clone())
    }

    /// Run hybrid evolution using multiple engines
    pub async fn evolve_hybrid(&mut self) -> Result<Vec<EvolutionMetrics>> {
        let mut results = Vec::new();

        // Split population for different engines
        let pop_size = self.population.size();
        let third = pop_size / 3;

        let adas_pop = Population::from_individuals(self.population.individuals[0..third].to_vec());
        let dgm_pop =
            Population::from_individuals(self.population.individuals[third..2 * third].to_vec());
        let swarm_pop =
            Population::from_individuals(self.population.individuals[2 * third..].to_vec());

        // Evolve with different engines in parallel
        let adas_future = async {
            if let Some(engine) = &mut self.adas_engine {
                engine.evolve_step(adas_pop).await
            } else {
                Err(EvolutionEngineError::InitializationError {
                    message: "ADAS engine not available".to_string(),
                })
            }
        };

        let dgm_future = async {
            if let Some(engine) = &mut self.dgm_engine {
                engine.evolve_step(dgm_pop).await
            } else {
                Err(EvolutionEngineError::InitializationError {
                    message: "DGM engine not available".to_string(),
                })
            }
        };

        let swarm_future = async {
            if let Some(engine) = &mut self.swarm_engine {
                engine.evolve_step(swarm_pop).await
            } else {
                Err(EvolutionEngineError::InitializationError {
                    message: "SwarmAgentic engine not available".to_string(),
                })
            }
        };

        // Wait for all engines to complete
        let (adas_result, dgm_result, swarm_result) =
            tokio::join!(adas_future, dgm_future, swarm_future);

        // Combine results
        let mut combined_individuals = Vec::new();
        if let Ok(adas_pop) = adas_result {
            combined_individuals.extend(adas_pop.individuals);
            if let Some(engine) = &self.adas_engine {
                results.push(engine.metrics().clone());
            }
        }
        if let Ok(dgm_pop) = dgm_result {
            combined_individuals.extend(dgm_pop.individuals);
            if let Some(engine) = &self.dgm_engine {
                results.push(engine.metrics().clone());
            }
        }
        if let Ok(swarm_pop) = swarm_result {
            combined_individuals.extend(swarm_pop.individuals);
            if let Some(engine) = &self.swarm_engine {
                results.push(engine.metrics().clone());
            }
        }

        // Update population with evolved agents
        self.population = Population::from_individuals(combined_individuals);

        Ok(results)
    }

    /// Get best agents from current population
    pub async fn get_elite_agents(&self, count: usize) -> Result<Vec<EvolvableAgent>> {
        let mut agents_with_fitness = Vec::new();

        for individual in &self.population.individuals {
            let fitness = individual
                .entity
                .evaluate_fitness()
                .await
                .context("Failed to evaluate agent fitness")?;
            agents_with_fitness.push((individual.entity.clone(), fitness));
        }

        // Sort by fitness (descending)
        agents_with_fitness
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N agents
        Ok(agents_with_fitness
            .into_iter()
            .take(count)
            .map(|(agent, _)| agent)
            .collect())
    }

    /// Optimize consensus weights using evolution
    pub async fn optimize_consensus_weights(
        &mut self,
        consensus_engine: &ConsensusSynthesisEngine,
    ) -> Result<ConsensusWeights> {
        // Create specialized agents for consensus optimization
        let mut consensus_population = Vec::new();

        for i in 0..50 {
            let genome = AgentGenome {
                goal: Goal::new("Optimize consensus weights".to_string(), GoalPriority::High),
                architecture: ArchitectureGenes {
                    memory_capacity: 2048,
                    processing_units: 8,
                    network_topology: vec![16, 32, 16],
                },
                behavior: BehaviorGenes {
                    exploration_rate: 0.2 + (i as f64 * 0.01),
                    learning_rate: 0.01 + (i as f64 * 0.001),
                    risk_tolerance: 0.3 + (i as f64 * 0.01),
                },
            };

            let agent = EvolvableAgent::from_genome(genome).await?;
            consensus_population.push(agent);
        }

        let consensus_individuals: Vec<Individual<EvolvableAgent>> = consensus_population
            .into_iter()
            .map(|agent| Individual::new(agent))
            .collect();
        let consensus_pop = Population::from_individuals(consensus_individuals);

        // Evolve for consensus optimization
        let mut evolved_pop = consensus_pop;
        for _ in 0..10 {
            if let Some(engine) = &mut self.adas_engine {
                evolved_pop = engine.evolve_step(evolved_pop).await?;
            }
        }

        // Extract optimal weights from best agent
        let elite = self.get_elite_agents(1).await?;
        if let Some(best_agent) = elite.first() {
            Ok(ConsensusWeights::from_agent_genome(&best_agent.genome))
        } else {
            Ok(ConsensusWeights::default())
        }
    }

    /// Get current evolution metrics
    pub fn get_metrics(&self) -> &EvolutionMetrics {
        &self.metrics
    }

    /// Get population statistics
    pub async fn get_population_stats(&self) -> Result<PopulationStats> {
        let mut fitness_values = Vec::new();

        for individual in &self.population.individuals {
            let fitness = individual.entity.evaluate_fitness().await?;
            fitness_values.push(fitness);
        }

        let size = fitness_values.len();
        let mean = fitness_values.iter().sum::<f64>() / size as f64;
        let min = fitness_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = fitness_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate standard deviation
        let variance = fitness_values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / size as f64;
        let std_dev = variance.sqrt();

        Ok(PopulationStats {
            size,
            mean_fitness: mean,
            min_fitness: min,
            max_fitness: max,
            std_deviation: std_dev,
        })
    }
}

/// Consensus weights optimized by evolution
#[derive(Debug, Clone)]
pub struct ConsensusWeights {
    pub node_weights: HashMap<String, f64>,
    pub algorithm_weights: HashMap<String, f64>,
    pub performance_weight: f64,
    pub reliability_weight: f64,
}

impl ConsensusWeights {
    /// Create weights from agent genome
    pub fn from_agent_genome(genome: &AgentGenome) -> Self {
        let mut node_weights = HashMap::new();
        let mut algorithm_weights = HashMap::new();

        // Extract weights from behavioral genes
        node_weights.insert("node_1".to_string(), genome.behavior.exploration_rate);
        node_weights.insert("node_2".to_string(), genome.behavior.learning_rate * 10.0);
        node_weights.insert("node_3".to_string(), genome.behavior.risk_tolerance);

        algorithm_weights.insert("adas".to_string(), genome.behavior.exploration_rate);
        algorithm_weights.insert("dgm".to_string(), genome.behavior.learning_rate * 50.0);
        algorithm_weights.insert("swarm".to_string(), genome.behavior.risk_tolerance);

        Self {
            node_weights,
            algorithm_weights,
            performance_weight: genome.behavior.learning_rate * 100.0,
            reliability_weight: 1.0 - genome.behavior.risk_tolerance,
        }
    }
}

impl Default for ConsensusWeights {
    fn default() -> Self {
        let mut node_weights = HashMap::new();
        node_weights.insert("node_1".to_string(), 0.33);
        node_weights.insert("node_2".to_string(), 0.33);
        node_weights.insert("node_3".to_string(), 0.34);

        let mut algorithm_weights = HashMap::new();
        algorithm_weights.insert("adas".to_string(), 0.4);
        algorithm_weights.insert("dgm".to_string(), 0.3);
        algorithm_weights.insert("swarm".to_string(), 0.3);

        Self {
            node_weights,
            algorithm_weights,
            performance_weight: 0.7,
            reliability_weight: 0.3,
        }
    }
}

/// Population statistics
#[derive(Debug, Clone)]
pub struct PopulationStats {
    pub size: usize,
    pub mean_fitness: f64,
    pub min_fitness: f64,
    pub max_fitness: f64,
    pub std_deviation: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_evolution_adapter_creation() -> Result<(), Box<dyn std::error::Error>> {
        let device = cudarc::driver::CudaContext::new(0)?;
        let adapter = EvolutionEngineAdapter::new(device).await;
        assert!(adapter.is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn test_initial_population_creation() {
        let population = EvolutionEngineAdapter::create_initial_population(10).await;
        assert!(population.is_ok());
        let pop = population?;
        assert_eq!(pop.size(), 10);
    }

    #[tokio::test]
    async fn test_adas_evolution() -> Result<(), Box<dyn std::error::Error>> {
        let device = cudarc::driver::CudaContext::new(0)?;
        let mut adapter = EvolutionEngineAdapter::new(device).await?;

        let result = adapter.evolve_with_adas().await;
        assert!(result.is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn test_dgm_evolution() -> Result<(), Box<dyn std::error::Error>> {
        let device = cudarc::driver::CudaContext::new(0)?;
        let mut adapter = EvolutionEngineAdapter::new(device).await?;

        let result = adapter.evolve_with_dgm().await;
        assert!(result.is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn test_swarm_evolution() -> Result<(), Box<dyn std::error::Error>> {
        let device = cudarc::driver::CudaContext::new(0)?;
        let mut adapter = EvolutionEngineAdapter::new(device).await?;

        let result = adapter.evolve_with_swarm_agentic().await;
        assert!(result.is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn test_elite_agents_selection() -> Result<(), Box<dyn std::error::Error>> {
        let device = cudarc::driver::CudaContext::new(0)?;
        let adapter = EvolutionEngineAdapter::new(device).await?;

        let elite = adapter.get_elite_agents(5).await;
        assert!(elite.is_ok());
        assert_eq!(elite?.len(), 5);
        Ok(())
    }

    #[tokio::test]
    async fn test_population_stats() -> Result<(), Box<dyn std::error::Error>> {
        let device = cudarc::driver::CudaContext::new(0)?;
        let adapter = EvolutionEngineAdapter::new(device).await?;

        let stats = adapter.get_population_stats().await;
        assert!(stats.is_ok());
        let s = stats?;
        assert!(s.size > 0);
        assert!(s.mean_fitness >= 0.0);
        assert!(s.mean_fitness <= 1.0);
    }

    #[test]
    fn test_consensus_weights_default() {
        let weights = ConsensusWeights::default();
        assert_eq!(weights.node_weights.len(), 3);
        assert_eq!(weights.algorithm_weights.len(), 3);
        assert!(weights.performance_weight > 0.0);
        assert!(weights.reliability_weight > 0.0);
    }

    #[test]
    fn test_consensus_weights_from_genome() {
        let genome = AgentGenome {
            goal: Goal::new("Test".to_string(), GoalPriority::Normal),
            architecture: ArchitectureGenes {
                memory_capacity: 1024,
                processing_units: 4,
                network_topology: vec![10, 20, 10],
            },
            behavior: BehaviorGenes {
                exploration_rate: 0.5,
                learning_rate: 0.01,
                risk_tolerance: 0.3,
            },
        };

        let weights = ConsensusWeights::from_agent_genome(&genome);
        assert_eq!(weights.node_weights["node_1"], 0.5);
        assert_eq!(weights.algorithm_weights["adas"], 0.5);
        assert_eq!(weights.performance_weight, 1.0);
        assert_eq!(weights.reliability_weight, 0.7);
    }
}
