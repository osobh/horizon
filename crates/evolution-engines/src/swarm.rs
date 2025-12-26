//! SwarmAgentic evolution engine for population-based optimization
//!
//! This engine implements swarm intelligence algorithms for agent evolution

use crate::{
    error::{EvolutionEngineError, EvolutionEngineResult},
    metrics::{EvolutionMetrics, MetricsCollector},
    population::{Individual, Population},
    swarm_particle::{Particle, ParticleParameters},
    swarm_topology::{SwarmTopology, TopologyManager},
    swarm_velocity::VelocityUpdater,
    traits::{EngineConfig, EvolutionEngine, Evolvable, EvolvableAgent},
};
use async_trait::async_trait;
use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// SwarmAgentic engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Base configuration
    pub base: crate::config::EvolutionEngineConfig,
    /// Swarm topology
    pub topology: SwarmTopology,
    /// Particle parameters
    pub particle_params: ParticleParameters,
    /// Social influence factor
    pub social_influence: f64,
    /// Cognitive influence factor
    pub cognitive_influence: f64,
    /// Inertia weight
    pub inertia_weight: f64,
    /// Neighborhood size
    pub neighborhood_size: usize,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            base: crate::config::EvolutionEngineConfig::default(),
            topology: SwarmTopology::default(),
            particle_params: ParticleParameters::default(),
            social_influence: 2.0,
            cognitive_influence: 2.0,
            inertia_weight: 0.7,
            neighborhood_size: 5,
        }
    }
}

impl EngineConfig for SwarmConfig {
    fn validate(&self) -> EvolutionEngineResult<()> {
        self.base.validate()?;

        if self.social_influence < 0.0 || self.social_influence > 4.0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Social influence must be between 0 and 4".to_string(),
            });
        }

        if self.cognitive_influence < 0.0 || self.cognitive_influence > 4.0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Cognitive influence must be between 0 and 4".to_string(),
            });
        }

        if self.inertia_weight < 0.0 || self.inertia_weight > 1.0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Inertia weight must be between 0 and 1".to_string(),
            });
        }

        Ok(())
    }

    fn engine_name(&self) -> &str {
        "SwarmAgentic"
    }
}

/// SwarmAgentic evolution engine
pub struct SwarmEngine {
    /// Configuration
    config: SwarmConfig,
    /// Metrics collector
    metrics_collector: MetricsCollector,
    /// Random number generator
    rng: Arc<RwLock<StdRng>>,
    /// Global best position
    global_best: Arc<RwLock<Option<EvolvableAgent>>>,
    /// Global best fitness
    global_best_fitness: Arc<RwLock<Option<f64>>>,
    /// Particle history
    particles: Arc<RwLock<Vec<Particle<EvolvableAgent>>>>,
    /// Topology manager
    topology_manager: TopologyManager,
    /// Velocity updater
    velocity_updater: VelocityUpdater,
}

impl SwarmEngine {
    /// Create new swarm engine
    pub fn new(config: SwarmConfig) -> EvolutionEngineResult<Self> {
        config.validate()?;

        let seed = config.base.seed.unwrap_or_else(|| rand::random());
        let rng = Arc::new(RwLock::new(StdRng::seed_from_u64(seed)));

        Ok(Self {
            config,
            metrics_collector: MetricsCollector::new(),
            topology_manager: TopologyManager::new(rng.clone()),
            velocity_updater: VelocityUpdater::new(rng.clone()),
            rng,
            global_best: Arc::new(RwLock::new(None)),
            global_best_fitness: Arc::new(RwLock::new(None)),
            particles: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Initialize particles from population
    fn initialize_particles(
        &self,
        population: &Population<EvolvableAgent>,
    ) -> Vec<Particle<EvolvableAgent>> {
        let mut particles = Vec::new();

        for individual in &population.individuals {
            let velocity_size = 10; // Simplified for now
            let velocity = self
                .velocity_updater
                .initialize_velocities(velocity_size, &self.config.particle_params);
            let fitness = individual.fitness.unwrap_or(0.0);

            particles.push(Particle::new(
                individual.entity.clone(),
                velocity,
                individual.entity.clone(),
                fitness,
            ));
        }

        particles
    }

    /// Get neighbors for a particle
    fn get_neighbors(&self, particle_idx: usize, num_particles: usize) -> Vec<usize> {
        self.topology_manager.get_neighbors(
            &self.config.topology,
            particle_idx,
            num_particles,
            self.config.neighborhood_size,
        )
    }
}

#[async_trait]
impl EvolutionEngine for SwarmEngine {
    type Entity = EvolvableAgent;
    type Config = SwarmConfig;

    async fn initialize(config: Self::Config) -> EvolutionEngineResult<Self> {
        Self::new(config)
    }

    async fn evolve_step(
        &mut self,
        mut population: Population<Self::Entity>,
    ) -> EvolutionEngineResult<Population<Self::Entity>> {
        self.metrics_collector.start_generation();

        // Initialize particles if needed
        if self.particles.read().is_empty() {
            let particles = self.initialize_particles(&population);
            *self.particles.write() = particles;
        }

        // Evaluate fitness
        let mut total_fitness = 0.0;
        let mut best_fitness = 0.0;

        for individual in &mut population.individuals {
            let fitness = individual.entity.evaluate_fitness().await?;
            individual.fitness = Some(fitness);
            total_fitness += fitness;
            best_fitness = f64::max(best_fitness, fitness);
        }

        // Update personal and global bests
        {
            let mut particles = self.particles.write();
            let mut global_best = self.global_best.write();
            let mut global_best_fitness = self.global_best_fitness.write();

            for (i, individual) in population.individuals.iter().enumerate() {
                if let Some(fitness) = individual.fitness {
                    // Update personal best
                    if i < particles.len() {
                        particles[i].update_personal_best(fitness);
                    }

                    // Update global best
                    if global_best_fitness.is_none() || fitness > global_best_fitness.unwrap() {
                        *global_best = Some(individual.entity.clone());
                        *global_best_fitness = Some(fitness);
                    }
                }
            }
        }

        // Update velocities
        {
            let mut particles = self.particles.write();
            let global_best = self.global_best.read();
            self.velocity_updater.update_velocities(
                &mut particles,
                &*global_best,
                self.config.social_influence,
                self.config.cognitive_influence,
                self.config.inertia_weight,
                &self.config.particle_params,
            );
        }

        // Move particles (apply velocity to position)
        let mut new_individuals = Vec::new();
        let particles_to_mutate: Vec<_> = {
            let particles = self.particles.read();
            particles
                .iter()
                .map(|p| (p.position.clone(), p.velocity.clone()))
                .collect()
        };

        for (position, velocity) in particles_to_mutate {
            let mutated = self
                .velocity_updater
                .apply_velocity_to_position(&position, &velocity)
                .await?;
            new_individuals.push(Individual::new(mutated));
        }

        // Update metrics
        let average_fitness = total_fitness / population.individuals.len() as f64;
        self.metrics_collector.end_generation(
            best_fitness,
            average_fitness,
            0.5, // Simplified diversity
            population.individuals.len() as u64,
        );

        let mut new_population = Population::from_individuals(new_individuals);
        new_population.generation = population.generation + 1;

        Ok(new_population)
    }

    async fn generate_initial_population(
        &self,
        size: usize,
    ) -> EvolutionEngineResult<Population<Self::Entity>> {
        let mut individuals = Vec::new();

        for i in 0..size {
            // Generate random genome for swarm particle
            let genome = {
                let mut rng = self.rng.write();
                crate::traits::AgentGenome {
                    goal: stratoswarm_agent_core::Goal::new(
                        "Swarm optimization".to_string(),
                        stratoswarm_agent_core::GoalPriority::Normal,
                    ),
                    architecture: crate::traits::ArchitectureGenes {
                        memory_capacity: rng.gen_range(1024..1024 * 1024),
                        processing_units: rng.gen_range(1..8),
                        network_topology: vec![
                            rng.gen_range(10..100),
                            rng.gen_range(10..100),
                            rng.gen_range(10..100),
                        ],
                    },
                    behavior: crate::traits::BehaviorGenes {
                        exploration_rate: rng.gen_range(0.0..1.0),
                        learning_rate: rng.gen_range(0.001..0.1),
                        risk_tolerance: rng.gen_range(0.0..1.0),
                    },
                }
            };

            let config = stratoswarm_agent_core::AgentConfig {
                name: format!("swarm_particle_{i}"),
                agent_type: "swarm".to_string(),
                max_memory: genome.architecture.memory_capacity,
                max_gpu_memory: genome.architecture.memory_capacity / 4,
                priority: 1,
                metadata: serde_json::Value::Null,
            };

            let agent = stratoswarm_agent_core::Agent::new(config).map_err(|e| {
                EvolutionEngineError::InitializationError {
                    message: format!("Failed to create agent: {e}"),
                }
            })?;

            individuals.push(Individual::new(EvolvableAgent { agent, genome }));
        }

        Ok(Population::from_individuals(individuals))
    }

    async fn should_terminate(&self, metrics: &EvolutionMetrics) -> bool {
        // Check generation limit
        if metrics.generation >= self.config.base.max_generations {
            return true;
        }

        // Check fitness target
        if let Some(target) = self.config.base.target_fitness {
            if metrics.best_fitness >= target {
                return true;
            }
        }

        // Check convergence
        if metrics.convergence_rate > 0.99 && metrics.generation > 20 {
            return true;
        }

        false
    }

    fn metrics(&self) -> &EvolutionMetrics {
        self.metrics_collector.metrics()
    }

    async fn adapt_parameters(&mut self, metrics: &EvolutionMetrics) -> EvolutionEngineResult<()> {
        if self.config.base.adaptive_parameters {
            // Adapt inertia weight
            let progress = metrics.generation as f64 / self.config.base.max_generations as f64;
            self.config.inertia_weight = 0.9 - 0.5 * progress; // Linear decrease

            // Adapt social/cognitive influence based on convergence
            if metrics.convergence_rate > 0.8 {
                // Increase exploration
                self.config.cognitive_influence *= 1.1;
                self.config.social_influence *= 0.9;
            } else if metrics.convergence_rate < 0.2 {
                // Increase exploitation
                self.config.cognitive_influence *= 0.9;
                self.config.social_influence *= 1.1;
            }

            // Clamp values
            self.config.cognitive_influence = self.config.cognitive_influence.clamp(0.5, 3.5);
            self.config.social_influence = self.config.social_influence.clamp(0.5, 3.5);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_config_default() {
        let config = SwarmConfig::default();
        assert_eq!(config.social_influence, 2.0);
        assert_eq!(config.cognitive_influence, 2.0);
        assert_eq!(config.inertia_weight, 0.7);
        assert_eq!(config.neighborhood_size, 5);
    }

    #[test]
    fn test_swarm_config_validation() {
        let mut config = SwarmConfig::default();
        assert!(config.validate().is_ok());

        config.social_influence = 5.0;
        assert!(config.validate().is_err());

        config.social_influence = 2.0;
        config.cognitive_influence = -0.5;
        assert!(config.validate().is_err());

        config.cognitive_influence = 2.0;
        config.inertia_weight = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_swarm_engine_creation() {
        let config = SwarmConfig::default();
        let engine = SwarmEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_initial_population_generation() {
        let config = SwarmConfig::default();
        let engine = SwarmEngine::new(config)?;

        let population = engine.generate_initial_population(20).await?;
        assert_eq!(population.size(), 20);
        assert_eq!(population.generation, 0);
    }

    #[test]
    fn test_swarm_config_serialization() {
        let config = SwarmConfig::default();
        let json = serde_json::to_string(&config)?;
        let deserialized: SwarmConfig = serde_json::from_str(&json)?;
        assert_eq!(config.social_influence, deserialized.social_influence);
        assert_eq!(config.cognitive_influence, deserialized.cognitive_influence);
    }

    #[test]
    fn test_engine_name() {
        let config = SwarmConfig::default();
        assert_eq!(config.engine_name(), "SwarmAgentic");
    }

    #[tokio::test]
    async fn test_evolution_step_basic() {
        let mut config = SwarmConfig::default();
        config.base.population_size = 5;
        config.base.mutation_rate = 0.1;

        let mut engine = SwarmEngine::new(config)?;
        let initial_pop = engine.generate_initial_population(5).await?;

        let evolved_pop = engine.evolve_step(initial_pop).await?;
        assert_eq!(evolved_pop.size(), 5);
        assert_eq!(evolved_pop.generation, 1);
    }

    #[tokio::test]
    async fn test_termination_conditions() {
        let mut config = SwarmConfig::default();
        config.base.max_generations = 10;
        config.base.target_fitness = Some(0.9);

        let engine = SwarmEngine::new(config)?;

        // Test generation limit
        let mut metrics = EvolutionMetrics::default();
        metrics.generation = 15;
        assert!(engine.should_terminate(&metrics).await);

        // Test target fitness
        metrics.generation = 5;
        metrics.best_fitness = 0.95;
        assert!(engine.should_terminate(&metrics).await);

        // Test convergence termination
        metrics.generation = 25;
        metrics.best_fitness = 0.5;
        metrics.convergence_rate = 0.995;
        assert!(engine.should_terminate(&metrics).await);

        // Test no termination
        metrics.generation = 5;
        metrics.best_fitness = 0.5;
        metrics.convergence_rate = 0.5;
        assert!(!engine.should_terminate(&metrics).await);
    }

    #[test]
    fn test_parameter_adaptation() {
        let mut config = SwarmConfig::default();
        config.base.adaptive_parameters = true;
        let mut engine = SwarmEngine::new(config)?;

        // Test with high convergence
        let mut metrics = EvolutionMetrics::default();
        metrics.convergence_rate = 0.9;

        let initial_cognitive = engine.config.cognitive_influence;
        let initial_social = engine.config.social_influence;

        futures::executor::block_on(engine.adapt_parameters(&metrics)).unwrap();

        assert!(engine.config.cognitive_influence > initial_cognitive);
        assert!(engine.config.social_influence < initial_social);
    }

    #[tokio::test]
    async fn test_global_best_update() {
        let config = SwarmConfig::default();
        let mut engine = SwarmEngine::new(config)?;

        // Initially no global best
        assert!(engine.global_best.read().is_none());
        assert!(engine.global_best_fitness.read().is_none());

        let population = engine.generate_initial_population(3).await?;
        let _evolved = engine.evolve_step(population).await?;

        // Should now have global best
        assert!(engine.global_best.read().is_some());
        assert!(engine.global_best_fitness.read().is_some());
    }
}
