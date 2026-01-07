//! Simple agent scenario implementations

use super::config::SimpleBehavior;
use crate::GpuSwarm;

/// Behavior configuration for simple agents
#[derive(Debug, Clone, Copy)]
pub struct BehaviorWeights {
    /// Weight for cohesion (moving towards center of neighbors)
    pub cohesion: f32,
    /// Weight for separation (avoiding crowding neighbors)
    pub separation: f32,
    /// Weight for alignment (matching velocity of neighbors)
    pub alignment: f32,
}

impl BehaviorWeights {
    /// Create weights for random walk behavior
    pub fn random_walk() -> Self {
        Self {
            cohesion: 0.0,
            separation: 0.05,  // Minimal separation to avoid collisions
            alignment: 0.0,
        }
    }

    /// Create weights for seeking behavior
    pub fn seeking() -> Self {
        Self {
            cohesion: 0.5,     // Move towards targets (simulated by cohesion to center)
            separation: 0.1,
            alignment: 0.0,
        }
    }

    /// Create weights for flocking behavior (Reynolds boids)
    pub fn flocking() -> Self {
        Self {
            cohesion: 0.15,    // Stay close to neighbors
            separation: 0.25,  // Avoid crowding
            alignment: 0.20,   // Match neighbor velocities
        }
    }

    /// Create weights for avoidance behavior
    pub fn avoidance() -> Self {
        Self {
            cohesion: 0.0,
            separation: 0.5,   // Strong separation for obstacle avoidance
            alignment: 0.05,   // Minimal alignment for smooth movement
        }
    }

    /// Create weights for composite behavior (balanced mix)
    pub fn composite() -> Self {
        Self {
            cohesion: 0.1,
            separation: 0.2,
            alignment: 0.1,
        }
    }
}

/// Target position for seeking behavior
#[derive(Debug, Clone, Copy)]
pub struct SeekTarget {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Obstacle for avoidance behavior
#[derive(Debug, Clone, Copy)]
pub struct Obstacle {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub radius: f32,
}

/// Simple agent scenario executor
pub struct SimpleAgentScenario {
    behavior: SimpleBehavior,
    interaction_radius: f32,
    update_frequency: f32,
    weights: BehaviorWeights,
    seek_targets: Vec<SeekTarget>,
    obstacles: Vec<Obstacle>,
}

impl SimpleAgentScenario {
    /// Create a new simple agent scenario
    pub fn new(behavior: SimpleBehavior, interaction_radius: f32, update_frequency: f32) -> Self {
        let weights = match behavior {
            SimpleBehavior::RandomWalk => BehaviorWeights::random_walk(),
            SimpleBehavior::Seeking => BehaviorWeights::seeking(),
            SimpleBehavior::Flocking => BehaviorWeights::flocking(),
            SimpleBehavior::Avoidance => BehaviorWeights::avoidance(),
            SimpleBehavior::Composite => BehaviorWeights::composite(),
        };

        Self {
            behavior,
            interaction_radius,
            update_frequency,
            weights,
            seek_targets: Vec::new(),
            obstacles: Vec::new(),
        }
    }

    /// Get the behavior weights
    pub fn weights(&self) -> BehaviorWeights {
        self.weights
    }

    /// Get the interaction radius
    pub fn interaction_radius(&self) -> f32 {
        self.interaction_radius
    }

    /// Get the update frequency
    pub fn update_frequency(&self) -> f32 {
        self.update_frequency
    }

    /// Add a seek target for seeking behavior
    pub fn add_seek_target(&mut self, x: f32, y: f32, z: f32) {
        self.seek_targets.push(SeekTarget { x, y, z });
    }

    /// Add an obstacle for avoidance behavior
    pub fn add_obstacle(&mut self, x: f32, y: f32, z: f32, radius: f32) {
        self.obstacles.push(Obstacle { x, y, z, radius });
    }

    /// Initialize agents for this scenario
    pub fn initialize_agents(
        &mut self,
        swarm: &mut GpuSwarm,
    ) -> Result<(), Box<dyn std::error::Error>> {
        log::info!(
            "Initializing {:?} behavior with weights: cohesion={:.2}, separation={:.2}, alignment={:.2}",
            self.behavior,
            self.weights.cohesion,
            self.weights.separation,
            self.weights.alignment
        );

        // Set behavior-specific parameters
        match self.behavior {
            SimpleBehavior::RandomWalk => {
                // Random walk uses minimal parameters
                log::debug!("Random walk initialized - agents will move randomly");
            }
            SimpleBehavior::Seeking => {
                // Set up default seek targets if none provided
                if self.seek_targets.is_empty() {
                    // Create targets at the corners and center of the simulation space
                    self.seek_targets.push(SeekTarget { x: 0.0, y: 0.0, z: 0.0 });
                    self.seek_targets.push(SeekTarget { x: 100.0, y: 0.0, z: 0.0 });
                    self.seek_targets.push(SeekTarget { x: 0.0, y: 100.0, z: 0.0 });
                    self.seek_targets.push(SeekTarget { x: 100.0, y: 100.0, z: 0.0 });
                    self.seek_targets.push(SeekTarget { x: 50.0, y: 50.0, z: 0.0 });
                }
                log::debug!("Seeking initialized with {} targets", self.seek_targets.len());
            }
            SimpleBehavior::Flocking => {
                // Flocking uses Reynolds boid parameters (already set in weights)
                log::debug!(
                    "Flocking initialized with Reynolds parameters - \
                    cohesion={:.2}, separation={:.2}, alignment={:.2}",
                    self.weights.cohesion,
                    self.weights.separation,
                    self.weights.alignment
                );
            }
            SimpleBehavior::Avoidance => {
                // Set up default obstacles if none provided
                if self.obstacles.is_empty() {
                    // Create some obstacles in the simulation space
                    self.obstacles.push(Obstacle { x: 25.0, y: 25.0, z: 0.0, radius: 5.0 });
                    self.obstacles.push(Obstacle { x: 75.0, y: 25.0, z: 0.0, radius: 5.0 });
                    self.obstacles.push(Obstacle { x: 25.0, y: 75.0, z: 0.0, radius: 5.0 });
                    self.obstacles.push(Obstacle { x: 75.0, y: 75.0, z: 0.0, radius: 5.0 });
                    self.obstacles.push(Obstacle { x: 50.0, y: 50.0, z: 0.0, radius: 10.0 });
                }
                log::debug!("Avoidance initialized with {} obstacles", self.obstacles.len());
            }
            SimpleBehavior::Composite => {
                // Composite uses a mix of all behaviors
                // Add some targets and obstacles
                if self.seek_targets.is_empty() {
                    self.seek_targets.push(SeekTarget { x: 50.0, y: 50.0, z: 0.0 });
                }
                if self.obstacles.is_empty() {
                    self.obstacles.push(Obstacle { x: 30.0, y: 30.0, z: 0.0, radius: 5.0 });
                    self.obstacles.push(Obstacle { x: 70.0, y: 70.0, z: 0.0, radius: 5.0 });
                }
                log::debug!(
                    "Composite behavior initialized with {} targets and {} obstacles",
                    self.seek_targets.len(),
                    self.obstacles.len()
                );
            }
        }

        // Update swarm configuration with behavior weights
        // The GPU swarm will use these weights during the update step
        swarm.set_behavior_weights(
            self.weights.cohesion,
            self.weights.separation,
            self.weights.alignment,
        )?;

        Ok(())
    }

    /// Update agents based on behavior
    pub fn update(&self, swarm: &mut GpuSwarm) -> Result<(), Box<dyn std::error::Error>> {
        // Behavior-specific updates are handled by GPU kernels
        swarm.step()?;
        Ok(())
    }

    /// Get seek targets
    pub fn seek_targets(&self) -> &[SeekTarget] {
        &self.seek_targets
    }

    /// Get obstacles
    pub fn obstacles(&self) -> &[Obstacle] {
        &self.obstacles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavior_weights() {
        let random = BehaviorWeights::random_walk();
        assert_eq!(random.cohesion, 0.0);
        assert!(random.separation > 0.0);

        let flocking = BehaviorWeights::flocking();
        assert!(flocking.cohesion > 0.0);
        assert!(flocking.separation > 0.0);
        assert!(flocking.alignment > 0.0);
    }

    #[test]
    fn test_scenario_creation() {
        let scenario = SimpleAgentScenario::new(SimpleBehavior::Flocking, 10.0, 60.0);
        let weights = scenario.weights();
        assert!(weights.cohesion > 0.0);
        assert!(weights.separation > 0.0);
        assert!(weights.alignment > 0.0);
    }
}
