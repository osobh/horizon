//! Swarm optimization GPU implementation
//!
//! Implements SwarmAgentic framework with Particle Swarm Optimization (PSO)
//! for fully automated agentic system generation with GPU acceleration.

use anyhow::Result;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Swarm particle representing an agentic system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmParticle {
    /// Particle ID
    pub id: usize,
    /// Current position (system configuration)
    pub position: Vec<f32>,
    /// Velocity vector
    pub velocity: Vec<f32>,
    /// Personal best position
    pub personal_best: Vec<f32>,
    /// Personal best fitness
    pub personal_best_fitness: f64,
    /// Current fitness
    pub current_fitness: f64,
    /// Agent system configuration
    pub agent_system: AgentSystem,
    /// Generation number
    pub generation: usize,
}

/// Agent system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSystem {
    /// Number of agents in the system
    pub num_agents: usize,
    /// Agent roles and capabilities
    pub roles: Vec<String>,
    /// Collaboration structure
    pub collaboration_matrix: Vec<Vec<f32>>,
    /// System-level behavior configuration
    pub behavior_config: BehaviorConfig,
}

/// System-level behavior configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorConfig {
    /// Task coordination strategy
    pub coordination_strategy: String,
    /// Communication patterns
    pub communication_patterns: Vec<String>,
    /// Decision making approach
    pub decision_making: String,
    /// Adaptation mechanisms
    pub adaptation_mechanisms: Vec<String>,
}

/// GPU-accelerated Swarm optimization engine
pub struct SwarmEngine {
    device: Arc<CudaDevice>,
    /// Swarm particles
    particles: Vec<SwarmParticle>,
    /// Global best position
    global_best: Vec<f32>,
    /// Global best fitness
    global_best_fitness: f64,
    /// GPU memory for positions
    positions: CudaSlice<f32>,
    /// GPU memory for velocities
    velocities: CudaSlice<f32>,
    /// GPU memory for personal bests
    personal_bests: CudaSlice<f32>,
    /// GPU memory for fitness scores
    fitness_scores: CudaSlice<f32>,
    /// GPU memory for neighborhood matrix
    neighborhood_matrix: CudaSlice<u32>,
    /// Swarm parameters
    params: SwarmParams,
    /// Current iteration
    iteration: usize,
    /// Dimensionality of search space
    dimensions: usize,
    /// Population size
    population_size: usize,
}

/// Swarm optimization parameters
#[derive(Debug, Clone)]
pub struct SwarmParams {
    /// Inertia weight
    pub inertia: f32,
    /// Cognitive learning factor
    pub cognitive: f32,
    /// Social learning factor
    pub social: f32,
    /// Maximum velocity
    pub max_velocity: f32,
    /// Neighborhood size for local best
    pub neighborhood_size: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

impl Default for SwarmParams {
    fn default() -> Self {
        Self {
            inertia: 0.729,
            cognitive: 1.49445,
            social: 1.49445,
            max_velocity: 1.0,
            neighborhood_size: 5,
            convergence_threshold: 1e-6,
        }
    }
}

impl SwarmEngine {
    /// Create new swarm optimization engine
    pub fn new(
        device: Arc<CudaDevice>,
        population_size: usize,
        dimensions: usize,
        params: SwarmParams,
    ) -> Result<Self> {
        // Allocate GPU memory
        let positions = unsafe { device.alloc::<f32>(population_size * dimensions)? };
        let velocities = unsafe { device.alloc::<f32>(population_size * dimensions)? };
        let personal_bests = unsafe { device.alloc::<f32>(population_size * dimensions)? };
        let fitness_scores = unsafe { device.alloc::<f32>(population_size)? };
        let neighborhood_matrix =
            unsafe { device.alloc::<u32>(population_size * params.neighborhood_size)? };

        Ok(Self {
            device,
            particles: Vec::with_capacity(population_size),
            global_best: vec![0.0; dimensions],
            global_best_fitness: f64::NEG_INFINITY,
            positions,
            velocities,
            personal_bests,
            fitness_scores,
            neighborhood_matrix,
            params,
            iteration: 0,
            dimensions,
            population_size,
        })
    }

    /// Initialize swarm with random particles
    pub fn initialize_random(&mut self) -> Result<()> {
        self.particles.clear();

        for i in 0..self.population_size {
            // Generate random position
            let position: Vec<f32> = (0..self.dimensions)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0) // Range [-1, 1]
                .collect();

            // Initialize velocity to small random values
            let velocity: Vec<f32> = (0..self.dimensions)
                .map(|_| (rand::random::<f32>() - 0.5) * 0.1)
                .collect();

            // Generate initial agent system configuration
            let agent_system = self.generate_agent_system(&position)?;

            let particle = SwarmParticle {
                id: i,
                position: position.clone(),
                velocity,
                personal_best: position,
                personal_best_fitness: f64::NEG_INFINITY,
                current_fitness: 0.0,
                agent_system,
                generation: 0,
            };

            self.particles.push(particle);
        }

        // Upload to GPU
        self.upload_to_gpu()?;

        Ok(())
    }

    /// Generate agent system configuration from position vector
    fn generate_agent_system(&self, position: &[f32]) -> Result<AgentSystem> {
        // Map position vector to agent system parameters
        let num_agents = ((position[0].abs() * 10.0) as usize + 1).min(10);

        // Generate roles based on position
        let mut roles = Vec::new();
        let role_types = vec![
            "coordinator",
            "executor",
            "analyzer",
            "communicator",
            "planner",
            "monitor",
            "optimizer",
            "validator",
        ];

        for i in 0..num_agents {
            let role_idx = ((position[i % self.dimensions].abs() * role_types.len() as f32)
                as usize)
                % role_types.len();
            roles.push(role_types[role_idx].to_string());
        }

        // Generate collaboration matrix
        let mut collaboration_matrix = vec![vec![0.0; num_agents]; num_agents];
        for i in 0..num_agents {
            for j in 0..num_agents {
                if i != j {
                    let pos_idx = (i * num_agents + j) % self.dimensions;
                    collaboration_matrix[i][j] = position[pos_idx].abs();
                }
            }
        }

        // Generate behavior configuration
        let coordination_strategies =
            vec!["hierarchical", "democratic", "market_based", "emergent"];
        let coord_idx = ((position[1].abs() * coordination_strategies.len() as f32) as usize)
            % coordination_strategies.len();

        let behavior_config = BehaviorConfig {
            coordination_strategy: coordination_strategies[coord_idx].to_string(),
            communication_patterns: vec!["broadcast".to_string(), "peer_to_peer".to_string()],
            decision_making: "consensus".to_string(),
            adaptation_mechanisms: vec!["feedback_learning".to_string()],
        };

        Ok(AgentSystem {
            num_agents,
            roles,
            collaboration_matrix,
            behavior_config,
        })
    }

    /// Upload swarm data to GPU
    fn upload_to_gpu(&mut self) -> Result<()> {
        // Flatten position data
        let mut position_data = vec![0f32; self.population_size * self.dimensions];
        let mut velocity_data = vec![0f32; self.population_size * self.dimensions];
        let mut personal_best_data = vec![0f32; self.population_size * self.dimensions];
        let mut fitness_data = vec![0f32; self.population_size];

        for (i, particle) in self.particles.iter().enumerate() {
            let start_idx = i * self.dimensions;
            position_data[start_idx..start_idx + self.dimensions]
                .copy_from_slice(&particle.position);
            velocity_data[start_idx..start_idx + self.dimensions]
                .copy_from_slice(&particle.velocity);
            personal_best_data[start_idx..start_idx + self.dimensions]
                .copy_from_slice(&particle.personal_best);
            fitness_data[i] = particle.current_fitness as f32;
        }

        // Generate neighborhood matrix
        let mut neighborhood_data =
            vec![0u32; self.population_size * self.params.neighborhood_size];
        for i in 0..self.population_size {
            for j in 0..self.params.neighborhood_size {
                let neighbor = (i + j + 1) % self.population_size;
                neighborhood_data[i * self.params.neighborhood_size + j] = neighbor as u32;
            }
        }

        // Copy to GPU
        self.device
            .htod_copy_into(position_data, &mut self.positions.clone())?;
        self.device
            .htod_copy_into(velocity_data, &mut self.velocities.clone())?;
        self.device
            .htod_copy_into(personal_best_data, &mut self.personal_bests.clone())?;
        self.device
            .htod_copy_into(fitness_data, &mut self.fitness_scores.clone())?;
        self.device
            .htod_copy_into(neighborhood_data, &mut self.neighborhood_matrix.clone())?;

        Ok(())
    }

    /// Download swarm data from GPU
    fn download_from_gpu(&mut self) -> Result<()> {
        let mut position_data = vec![0f32; self.population_size * self.dimensions];
        let mut velocity_data = vec![0f32; self.population_size * self.dimensions];
        let mut fitness_data = vec![0f32; self.population_size];

        self.device
            .dtoh_sync_copy_into(&self.positions, &mut position_data)?;
        self.device
            .dtoh_sync_copy_into(&self.velocities, &mut velocity_data)?;
        self.device
            .dtoh_sync_copy_into(&self.fitness_scores, &mut fitness_data)?;

        // Update particles
        for (i, particle) in self.particles.iter_mut().enumerate() {
            let start_idx = i * self.dimensions;
            particle
                .position
                .copy_from_slice(&position_data[start_idx..start_idx + self.dimensions]);
            particle
                .velocity
                .copy_from_slice(&velocity_data[start_idx..start_idx + self.dimensions]);
            particle.current_fitness = fitness_data[i] as f64;

            // Update personal best
            if particle.current_fitness > particle.personal_best_fitness {
                particle.personal_best = particle.position.clone();
                particle.personal_best_fitness = particle.current_fitness;
            }

            // Update global best
            if particle.current_fitness > self.global_best_fitness {
                self.global_best = particle.position.clone();
                self.global_best_fitness = particle.current_fitness;
            }
        }

        Ok(())
    }

    /// Evaluate swarm fitness
    pub fn evaluate_fitness(&mut self, fitness_fn: &dyn Fn(&AgentSystem) -> f64) -> Result<()> {
        // Update agent systems based on current positions
        for i in 0..self.particles.len() {
            let position = self.particles[i].position.clone();
            self.particles[i].agent_system = self.generate_agent_system(&position)?;
            self.particles[i].current_fitness = fitness_fn(&self.particles[i].agent_system);
        }

        // Upload fitness to GPU
        let fitness_data: Vec<f32> = self
            .particles
            .iter()
            .map(|p| p.current_fitness as f32)
            .collect();
        self.device
            .htod_copy_into(fitness_data, &mut self.fitness_scores.clone())?;

        // Launch GPU fitness computation kernel for validation
        unsafe {
            crate::evolution::kernels::compute_swarm_fitness(
                *self.positions.device_ptr() as *const f32,
                *self.fitness_scores.device_ptr() as *mut f32,
                self.population_size as u32,
                self.dimensions as u32,
                std::ptr::null(), // target_function placeholder
            );
        }

        // Update personal and global bests
        self.download_from_gpu()?;

        Ok(())
    }

    /// Update velocities using PSO algorithm
    pub fn update_velocities(&mut self) -> Result<()> {
        // Launch GPU velocity update kernel
        unsafe {
            crate::evolution::kernels::launch_pso_velocity_update(
                *self.velocities.device_ptr() as *mut f32,
                *self.positions.device_ptr() as *const f32,
                *self.personal_bests.device_ptr() as *const f32,
                self.global_best.as_ptr(),
                self.population_size as u32,
                self.dimensions as u32,
                self.params.inertia,
                self.params.cognitive,
                self.params.social,
            );
        }

        Ok(())
    }

    /// Update positions using velocities
    pub fn update_positions(&mut self) -> Result<()> {
        // Launch GPU position update kernel
        unsafe {
            crate::evolution::kernels::launch_pso_position_update(
                *self.positions.device_ptr() as *mut f32,
                *self.velocities.device_ptr() as *const f32,
                self.population_size as u32,
                self.dimensions as u32,
                self.params.max_velocity,
            );
        }

        // Download updated positions
        self.download_from_gpu()?;

        Ok(())
    }

    /// Enable swarm communication
    pub fn swarm_communication(&mut self) -> Result<()> {
        // Launch swarm communication kernel
        let shared_knowledge = unsafe {
            self.device
                .alloc::<f32>(self.population_size * self.dimensions)?
        };

        unsafe {
            crate::evolution::kernels::launch_swarm_communication(
                *self.positions.device_ptr() as *const f32,
                *shared_knowledge.device_ptr() as *mut f32,
                *self.neighborhood_matrix.device_ptr() as *const u32,
                self.population_size as u32,
                self.dimensions as u32,
            );
        }

        // Update particles with shared knowledge (simplified)
        for particle in &mut self.particles {
            particle.generation += 1;
        }

        Ok(())
    }

    /// Run one iteration of swarm optimization
    pub fn iterate(&mut self, fitness_fn: &dyn Fn(&AgentSystem) -> f64) -> Result<()> {
        // 1. Evaluate fitness
        self.evaluate_fitness(fitness_fn)?;

        // 2. Update velocities
        self.update_velocities()?;

        // 3. Update positions
        self.update_positions()?;

        // 4. Enable swarm communication
        self.swarm_communication()?;

        self.iteration += 1;

        Ok(())
    }

    /// Check convergence
    pub fn has_converged(&self) -> bool {
        if self.particles.len() < 2 {
            return false;
        }

        // Check fitness variance
        let fitness_values: Vec<f64> = self.particles.iter().map(|p| p.current_fitness).collect();

        let mean = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;
        let variance = fitness_values
            .iter()
            .map(|f| (f - mean).powi(2))
            .sum::<f64>()
            / fitness_values.len() as f64;

        variance.sqrt() < self.params.convergence_threshold
    }

    /// Get best particle
    pub fn best_particle(&self) -> Option<&SwarmParticle> {
        self.particles
            .iter()
            .max_by(|a, b| a.current_fitness.partial_cmp(&b.current_fitness)?)
    }

    /// Get swarm statistics
    pub fn statistics(&self) -> SwarmStatistics {
        let fitness_values: Vec<f64> = self.particles.iter().map(|p| p.current_fitness).collect();

        let best = fitness_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let worst = fitness_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let average = if fitness_values.is_empty() {
            0.0
        } else {
            fitness_values.iter().sum::<f64>() / fitness_values.len() as f64
        };

        // Compute diversity as average distance from center
        let center: Vec<f64> = (0..self.dimensions)
            .map(|d| {
                self.particles
                    .iter()
                    .map(|p| p.position[d] as f64)
                    .sum::<f64>()
                    / self.particles.len() as f64
            })
            .collect();

        let diversity = if self.particles.is_empty() {
            0.0
        } else {
            self.particles
                .iter()
                .map(|p| {
                    p.position
                        .iter()
                        .enumerate()
                        .map(|(i, &pos)| (pos as f64 - center[i]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .sum::<f64>()
                / self.particles.len() as f64
        };

        SwarmStatistics {
            iteration: self.iteration,
            population_size: self.particles.len(),
            best_fitness: best,
            worst_fitness: worst,
            average_fitness: average,
            diversity,
            global_best_fitness: self.global_best_fitness,
            convergence_ratio: diversity / (best - worst + 1e-10),
        }
    }

    /// Run swarm optimization for multiple iterations
    pub async fn optimize(
        &mut self,
        fitness_fn: &dyn Fn(&AgentSystem) -> f64,
        max_iterations: usize,
    ) -> Result<SwarmParticle> {
        for _iter in 0..max_iterations {
            self.iterate(fitness_fn)?;

            // Log progress
            if self.iteration % 10 == 0 {
                let stats = self.statistics();
                log::info!(
                    "Swarm Iter {}: best={:.4}, avg={:.4}, diversity={:.4}",
                    stats.iteration,
                    stats.best_fitness,
                    stats.average_fitness,
                    stats.diversity
                );
            }

            // Check convergence
            if self.has_converged() {
                log::info!("Swarm converged at iteration {}", self.iteration);
                break;
            }
        }

        self.best_particle()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No best particle found"))
    }
}

/// Swarm optimization statistics
#[derive(Debug, Clone)]
pub struct SwarmStatistics {
    pub iteration: usize,
    pub population_size: usize,
    pub best_fitness: f64,
    pub worst_fitness: f64,
    pub average_fitness: f64,
    pub diversity: f64,
    pub global_best_fitness: f64,
    pub convergence_ratio: f64,
}

impl std::fmt::Display for SwarmStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Swarm Iter {}: Pop={}, Best={:.3}, Avg={:.3}, Div={:.3}, Conv={:.3}",
            self.iteration,
            self.population_size,
            self.best_fitness,
            self.average_fitness,
            self.diversity,
            self.convergence_ratio
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_params() {
        let params = SwarmParams::default();
        assert!(params.inertia > 0.0 && params.inertia < 1.0);
        assert!(params.cognitive > 0.0);
        assert!(params.social > 0.0);
    }

    #[test]
    fn test_swarm_engine_creation() -> Result<(), Box<dyn std::error::Error>>  {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let params = SwarmParams::default();
            let engine = SwarmEngine::new(device, 32, 10, params)?;
            assert_eq!(engine.population_size, 32);
            assert_eq!(engine.dimensions, 10);
        }
    }

    #[test]
    fn test_agent_system_generation() -> Result<(), Box<dyn std::error::Error>>  {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let params = SwarmParams::default();
            let engine = SwarmEngine::new(device, 4, 5, params)?;

            let position = vec![0.5, -0.3, 0.8, -0.1, 0.2];
            let agent_system = engine.generate_agent_system(&position)?;

            assert!(agent_system.num_agents > 0);
            assert!(!agent_system.roles.is_empty());
        }
    }
}
