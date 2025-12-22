//! Particle implementation for SwarmAgentic optimization
//!
//! This module contains the particle structure and related operations for
//! particle swarm optimization algorithms.

use crate::traits::Evolvable;
use serde::{Deserialize, Serialize};

/// Particle parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleParameters {
    /// Velocity clamping factor
    pub velocity_clamp: f64,
    /// Position bounds
    pub position_bounds: Option<(f64, f64)>,
    /// Velocity initialization range
    pub velocity_init_range: f64,
}

impl Default for ParticleParameters {
    fn default() -> Self {
        Self {
            velocity_clamp: 1.0,
            position_bounds: Some((-10.0, 10.0)),
            velocity_init_range: 0.1,
        }
    }
}

/// Particle in the swarm
#[derive(Debug, Clone)]
pub struct Particle<E: Evolvable> {
    /// Current position (entity)
    pub position: E,
    /// Velocity vector
    pub velocity: Vec<f64>,
    /// Personal best position
    pub personal_best: E,
    /// Personal best fitness
    pub personal_best_fitness: E::Fitness,
}

impl<E: Evolvable> Particle<E> {
    /// Create new particle
    pub fn new(
        position: E,
        velocity: Vec<f64>,
        personal_best: E,
        personal_best_fitness: E::Fitness,
    ) -> Self {
        Self {
            position,
            velocity,
            personal_best,
            personal_best_fitness,
        }
    }

    /// Update personal best if current fitness is better
    pub fn update_personal_best(&mut self, fitness: E::Fitness)
    where
        E::Fitness: PartialOrd + Copy,
    {
        if fitness > self.personal_best_fitness {
            self.personal_best = self.position.clone();
            self.personal_best_fitness = fitness;
        }
    }

    /// Apply velocity constraints
    pub fn clamp_velocity(&mut self, clamp_value: f64) {
        for velocity in &mut self.velocity {
            *velocity = velocity.clamp(-clamp_value, clamp_value);
        }
    }

    /// Calculate velocity magnitude
    pub fn velocity_magnitude(&self) -> f64 {
        self.velocity.iter().map(|v| v.abs()).sum::<f64>() / self.velocity.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Evolvable, EvolvableAgent};

    fn create_test_agent() -> EvolvableAgent {
        let genome = crate::traits::AgentGenome {
            goal: exorust_agent_core::Goal::new(
                "test".to_string(),
                exorust_agent_core::GoalPriority::Normal,
            ),
            architecture: crate::traits::ArchitectureGenes {
                memory_capacity: 1024,
                processing_units: 4,
                network_topology: vec![10, 20, 10],
            },
            behavior: crate::traits::BehaviorGenes {
                exploration_rate: 0.1,
                learning_rate: 0.01,
                risk_tolerance: 0.5,
            },
        };

        let config = exorust_agent_core::AgentConfig {
            name: "test_agent".to_string(),
            agent_type: "test".to_string(),
            max_memory: 1024,
            max_gpu_memory: 256,
            priority: 1,
            metadata: serde_json::Value::Null,
        };

        let agent = exorust_agent_core::Agent::new(config)?;
        EvolvableAgent { agent, genome }
    }

    #[test]
    fn test_particle_creation() {
        let agent = create_test_agent();
        let velocity = vec![0.1, 0.2, 0.3];
        let particle = Particle::new(agent.clone(), velocity.clone(), agent.clone(), 0.5);

        assert_eq!(particle.velocity, velocity);
        assert_eq!(particle.personal_best_fitness, 0.5);
    }

    #[test]
    fn test_particle_personal_best_update() {
        let agent = create_test_agent();
        let velocity = vec![0.1, 0.2, 0.3];
        let mut particle = Particle::new(agent.clone(), velocity, agent.clone(), 0.5);

        // Higher fitness should update personal best
        particle.update_personal_best(0.8);
        assert_eq!(particle.personal_best_fitness, 0.8);

        // Lower fitness should not update personal best
        particle.update_personal_best(0.6);
        assert_eq!(particle.personal_best_fitness, 0.8);
    }

    #[test]
    fn test_velocity_clamping() {
        let agent = create_test_agent();
        let velocity = vec![2.0, -3.0, 1.5, -0.8];
        let mut particle = Particle::new(agent.clone(), velocity, agent.clone(), 0.5);

        particle.clamp_velocity(1.0);

        for &vel in &particle.velocity {
            assert!(vel.abs() <= 1.0);
        }
    }

    #[test]
    fn test_velocity_magnitude() {
        let agent = create_test_agent();
        let velocity = vec![0.5, -0.3, 0.8, -0.2];
        let particle = Particle::new(agent.clone(), velocity, agent.clone(), 0.5);

        let magnitude = particle.velocity_magnitude();
        let expected = (0.5 + 0.3 + 0.8 + 0.2) / 4.0;
        assert_eq!(magnitude, expected);
    }

    #[test]
    fn test_particle_parameters_default() {
        let params = ParticleParameters::default();
        assert_eq!(params.velocity_clamp, 1.0);
        assert_eq!(params.position_bounds, Some((-10.0, 10.0)));
        assert_eq!(params.velocity_init_range, 0.1);
    }

    #[test]
    fn test_particle_parameters_serialization() {
        let params = ParticleParameters::default();
        let json = serde_json::to_string(&params)?;
        let deserialized: ParticleParameters = serde_json::from_str(&json)?;
        assert_eq!(params.velocity_clamp, deserialized.velocity_clamp);
        assert_eq!(params.position_bounds, deserialized.position_bounds);
    }
}
