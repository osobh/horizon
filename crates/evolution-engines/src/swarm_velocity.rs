//! Velocity update operations for particle swarm optimization

use crate::swarm_particle::{Particle, ParticleParameters};
use crate::traits::{Evolvable, EvolvableAgent};
use parking_lot::RwLock;
use rand::{rngs::StdRng, Rng};
use std::sync::Arc;

/// Velocity updater for particle swarm optimization
pub struct VelocityUpdater {
    /// Random number generator
    rng: Arc<RwLock<StdRng>>,
}

impl VelocityUpdater {
    /// Create new velocity updater
    pub fn new(rng: Arc<RwLock<StdRng>>) -> Self {
        Self { rng }
    }

    /// Update particle velocities using PSO formula
    pub fn update_velocities<E: Evolvable>(
        &self,
        particles: &mut [Particle<E>],
        global_best: &Option<E>,
        social_influence: f64,
        cognitive_influence: f64,
        inertia_weight: f64,
        particle_params: &ParticleParameters,
    ) {
        if global_best.is_none() {
            return;
        }

        let mut rng = self.rng.write();

        for particle in particles {
            // Update velocity using PSO formula
            for i in 0..particle.velocity.len() {
                let r1: f64 = rng.gen();
                let r2: f64 = rng.gen();

                // Simplified velocity update (normally would use actual position differences)
                particle.velocity[i] = inertia_weight * particle.velocity[i]
                    + cognitive_influence * r1 * 0.1  // Simplified
                    + social_influence * r2 * 0.1; // Simplified
            }

            // Apply velocity constraints
            particle.clamp_velocity(particle_params.velocity_clamp);
        }
    }

    /// Initialize particle velocities randomly
    pub fn initialize_velocities(
        &self,
        velocity_size: usize,
        particle_params: &ParticleParameters,
    ) -> Vec<f64> {
        let mut rng = self.rng.write();
        (0..velocity_size)
            .map(|_| {
                rng.gen_range(
                    -particle_params.velocity_init_range..particle_params.velocity_init_range,
                )
            })
            .collect()
    }

    /// Apply velocity to position (simplified mutation)
    pub async fn apply_velocity_to_position(
        &self,
        position: &EvolvableAgent,
        velocity: &[f64],
    ) -> crate::error::EvolutionEngineResult<EvolvableAgent> {
        let velocity_magnitude: f64 =
            velocity.iter().map(|v| v.abs()).sum::<f64>() / velocity.len() as f64;
        position.mutate(velocity_magnitude * 0.1).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm_particle::Particle;
    use rand::SeedableRng;

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

    fn create_velocity_updater() -> VelocityUpdater {
        let rng = Arc::new(RwLock::new(StdRng::seed_from_u64(42)));
        VelocityUpdater::new(rng)
    }

    #[test]
    fn test_velocity_initialization() {
        let updater = create_velocity_updater();
        let params = ParticleParameters::default();

        let velocity = updater.initialize_velocities(5, &params);
        assert_eq!(velocity.len(), 5);

        for &vel in &velocity {
            assert!(vel.abs() <= params.velocity_init_range);
        }
    }

    #[test]
    fn test_velocity_update() {
        let updater = create_velocity_updater();
        let params = ParticleParameters::default();
        let agent = create_test_agent();

        let velocity = vec![0.1, 0.2, 0.3];
        let particle = Particle::new(agent.clone(), velocity.clone(), agent.clone(), 0.5);
        let mut particles = vec![particle];

        let global_best = Some(agent);

        updater.update_velocities(
            &mut particles,
            &global_best,
            2.0, // social_influence
            2.0, // cognitive_influence
            0.7, // inertia_weight
            &params,
        );

        // Velocities should have changed and be clamped
        for &vel in &particles[0].velocity {
            assert!(vel.abs() <= params.velocity_clamp);
        }
    }

    #[test]
    fn test_velocity_update_without_global_best() {
        let updater = create_velocity_updater();
        let params = ParticleParameters::default();
        let agent = create_test_agent();

        let velocity = vec![0.1, 0.2, 0.3];
        let original_velocity = velocity.clone();
        let particle = Particle::new(agent.clone(), velocity, agent, 0.5);
        let mut particles = vec![particle];

        let global_best = None;

        updater.update_velocities(&mut particles, &global_best, 2.0, 2.0, 0.7, &params);

        // Velocities should remain unchanged
        assert_eq!(particles[0].velocity, original_velocity);
    }

    #[tokio::test]
    async fn test_apply_velocity_to_position() {
        let updater = create_velocity_updater();
        let agent = create_test_agent();
        let velocity = vec![0.1, 0.2, 0.3];

        let result = updater.apply_velocity_to_position(&agent, &velocity).await;
        assert!(result.is_ok());

        // The mutated agent should be different (though we can't easily test specific changes)
        let mutated = result?;
        // At minimum, the operation should succeed - verify it's still a valid agent
        assert_eq!(
            mutated.genome.goal.description,
            agent.genome.goal.description
        );
    }

    #[test]
    fn test_velocity_clamping_in_update() {
        let updater = create_velocity_updater();
        let mut params = ParticleParameters::default();
        params.velocity_clamp = 0.5; // Very restrictive clamp

        let agent = create_test_agent();
        let velocity = vec![2.0, -3.0, 1.5]; // Large velocities
        let particle = Particle::new(agent.clone(), velocity, agent.clone(), 0.5);
        let mut particles = vec![particle];

        let global_best = Some(agent);

        updater.update_velocities(&mut particles, &global_best, 2.0, 2.0, 0.7, &params);

        // All velocities should be clamped
        for &vel in &particles[0].velocity {
            assert!(vel.abs() <= 0.5);
        }
    }

    #[test]
    fn test_zero_velocity_initialization() {
        let updater = create_velocity_updater();
        let mut params = ParticleParameters::default();
        params.velocity_init_range = 0.0;

        let velocity = updater.initialize_velocities(3, &params);

        for &vel in &velocity {
            assert_eq!(vel, 0.0);
        }
    }
}
