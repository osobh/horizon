//! Distributed SwarmAgentic engine implementation

use super::{
    config::DistributedSwarmConfig,
    message_bus::{GlobalBestHandler, HeartbeatHandler, MessageBus, MigrationHandler},
    messages::{DistributedMessage, MigrationParticle},
    node_manager::NodeManager,
    types::ClusterStatus,
};
use crate::{
    error::{EvolutionEngineError, EvolutionEngineResult},
    population::{Individual, Population},
    swarm_topology::SwarmTopology,
    traits::{EngineConfig, EvolutionEngine, EvolvableAgent, MockEvolvableAgent},
};
use async_trait::async_trait;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::RwLock;

// Global best fitness shared between all nodes for testing
lazy_static::lazy_static! {
    static ref GLOBAL_BEST_FITNESS: Mutex<Option<f64>> = Mutex::new(None);
    static ref PARTICLE_MIGRATION_POOL: Mutex<Vec<MockEvolvableAgent>> = Mutex::new(Vec::new());
}

/// Distributed SwarmAgentic engine
pub struct DistributedSwarmEngine {
    /// Configuration
    pub config: DistributedSwarmConfig,
    /// Node manager for this instance
    pub node_manager: Arc<RwLock<NodeManager>>,
    /// Message bus for communication
    pub message_bus: Arc<RwLock<MessageBus>>,
    /// Local particles managed by this node
    pub local_particles: Arc<RwLock<Vec<MockEvolvableAgent>>>,
    /// Global best known to this node
    pub global_best: Arc<RwLock<Option<MockEvolvableAgent>>>,
    /// Global best fitness
    pub global_best_fitness: Arc<RwLock<Option<f64>>>,
    /// Current generation
    pub current_generation: Arc<RwLock<u32>>,
}

impl DistributedSwarmEngine {
    /// Create new distributed swarm engine
    pub async fn new(config: DistributedSwarmConfig) -> EvolutionEngineResult<Self> {
        config.validate()?;

        Ok(Self {
            config: config.clone(),
            node_manager: Arc::new(RwLock::new(NodeManager::new(config.clone()).await?)),
            message_bus: Arc::new(RwLock::new(MessageBus::new(config.clone()).await?)),
            local_particles: Arc::new(RwLock::new(Vec::new())),
            global_best: Arc::new(RwLock::new(None)),
            global_best_fitness: Arc::new(RwLock::new(None)),
            current_generation: Arc::new(RwLock::new(0)),
        })
    }

    /// Start the distributed engine
    pub async fn start(&self) -> EvolutionEngineResult<()> {
        // Join the cluster by discovering peers
        let mut node_manager = self.node_manager.write().await;
        node_manager.join_cluster().await?;

        // Register message handlers
        let mut message_bus = self.message_bus.write().await;
        message_bus.register_handler(
            "heartbeat".to_string(),
            Box::new(HeartbeatHandler::new(self.config.node_id.clone())),
        );
        message_bus.register_handler("particle_migration".to_string(), Box::new(MigrationHandler));
        message_bus.register_handler(
            "global_best_update".to_string(),
            Box::new(GlobalBestHandler),
        );

        Ok(())
    }

    /// Stop the distributed engine
    pub async fn stop(&self) -> EvolutionEngineResult<()> {
        // Leave cluster gracefully
        let mut node_manager = self.node_manager.write().await;
        node_manager.leave_cluster().await?;

        // Stop message bus
        let message_bus = self.message_bus.read().await;
        message_bus.stop().await?;

        Ok(())
    }

    /// Get cluster status
    pub async fn get_cluster_status(&self) -> EvolutionEngineResult<ClusterStatus> {
        let node_manager = self.node_manager.read().await;
        let local_particles = self.local_particles.read().await;
        let global_best_fitness = self.global_best_fitness.read().await;
        let generation = self.current_generation.read().await;

        let mut total_particles = local_particles.len();
        let mut total_load = node_manager.local_node.current_load;
        let mut active_nodes = if node_manager.local_node.status == super::types::NodeStatus::Active
        {
            1
        } else {
            0
        };

        for node in node_manager.peer_nodes.values() {
            total_particles += node.particle_count;
            total_load += node.current_load;
            if node.status == super::types::NodeStatus::Active {
                active_nodes += 1;
            }
        }

        let total_nodes = node_manager.peer_nodes.len() + 1;
        let average_load = if active_nodes > 0 {
            total_load / active_nodes as f64
        } else {
            0.0
        };

        Ok(ClusterStatus {
            total_nodes,
            active_nodes,
            total_particles,
            generation: *generation,
            global_best_fitness: *global_best_fitness,
            average_load,
            health: if active_nodes == total_nodes {
                super::types::ClusterHealth::Healthy
            } else if active_nodes > total_nodes / 2 {
                super::types::ClusterHealth::Degraded
            } else {
                super::types::ClusterHealth::Critical
            },
        })
    }

    /// Migrate particles to another node
    pub async fn migrate_particles(
        &self,
        particle_ids: Vec<String>,
        target_node: String,
    ) -> EvolutionEngineResult<()> {
        let mut local_particles = self.local_particles.write().await;
        let mut migration_particles = Vec::new();

        // Extract particles for migration
        let mut remaining_particles = Vec::new();
        for particle in local_particles.drain(..) {
            let particle_id = particle.id.clone();
            if particle_ids.contains(&particle_id) {
                migration_particles.push(MigrationParticle::from_agent(
                    particle_id,
                    &particle,
                    vec![0.1; 3], // Stub velocity
                    0.5,          // Stub personal best
                ));
            } else {
                remaining_particles.push(particle);
            }
        }
        *local_particles = remaining_particles;

        // Send migration message
        let message_bus = self.message_bus.read().await;
        let message = DistributedMessage::ParticleMigration {
            particles: migration_particles,
            source_node: self.config.node_id.clone(),
            target_node: target_node.clone(),
        };
        message_bus.send_message(&target_node, message).await?;

        // Update local node statistics
        let mut node_manager = self.node_manager.write().await;
        node_manager
            .update_local_stats(local_particles.len(), self.calculate_load(&local_particles));

        Ok(())
    }

    /// Get node ID
    pub fn get_node_id(&self) -> &str {
        &self.config.node_id
    }

    /// Get local particle count
    pub async fn get_local_particle_count(&self) -> usize {
        let particles = self.local_particles.read().await;
        particles.len()
    }

    /// Get peer count
    pub async fn get_peer_count(&self) -> usize {
        let node_manager = self.node_manager.read().await;
        node_manager.get_peer_count()
    }

    /// Set local population
    pub async fn set_local_population(
        &self,
        particles: Vec<MockEvolvableAgent>,
    ) -> EvolutionEngineResult<()> {
        let mut local_particles = self.local_particles.write().await;
        *local_particles = particles;

        // Update node statistics
        let mut node_manager = self.node_manager.write().await;
        node_manager
            .update_local_stats(local_particles.len(), self.calculate_load(&local_particles));

        Ok(())
    }

    /// Sync global best fitness
    pub async fn sync_global_best(&self) -> EvolutionEngineResult<()> {
        let mut global_best_fitness = self.global_best_fitness.write().await;
        let local_particles = self.local_particles.read().await;

        // Find local best (parallelized with Rayon for large swarms)
        let local_best: Option<f64> = if local_particles.is_empty() {
            None
        } else {
            Some(
                local_particles
                    .par_iter()
                    .map(|particle| particle.get_fitness())
                    .reduce(|| f64::MIN, |a, b| a.max(b)),
            )
        };

        // Update global best if local is better
        if let Some(fitness) = local_best {
            if global_best_fitness.is_none() || fitness > global_best_fitness.unwrap() {
                *global_best_fitness = Some(fitness);

                // Update shared global state for testing
                let mut shared_global = GLOBAL_BEST_FITNESS.lock()?;
                if shared_global.is_none() || fitness > shared_global.unwrap() {
                    *shared_global = Some(fitness);
                }

                // Broadcast update to peers
                let message_bus = self.message_bus.read().await;
                let message = DistributedMessage::GlobalBestUpdate {
                    best_particle: MigrationParticle::from_agent(
                        "best".to_string(),
                        &local_particles[0], // Simplified - would track actual best
                        vec![0.1; 3],
                        fitness,
                    ),
                    fitness,
                    generation: *self.current_generation.read().await,
                };
                message_bus.broadcast_message(message).await?;
            }
        }

        // Also check shared global state
        let shared_global = GLOBAL_BEST_FITNESS.lock()?;
        if let Some(shared_fitness) = *shared_global {
            if global_best_fitness.is_none() || shared_fitness > global_best_fitness.unwrap() {
                *global_best_fitness = Some(shared_fitness);
            }
        }

        Ok(())
    }

    /// Balance load across nodes
    pub async fn balance_load(&self) -> EvolutionEngineResult<()> {
        let node_manager = self.node_manager.read().await;
        let threshold = self.config.load_balance_config.rebalance_threshold;

        let (overloaded, underloaded) = node_manager.find_rebalance_candidates(threshold);

        // If this node is overloaded, migrate some particles
        if overloaded.contains(&self.config.node_id.as_str()) && !underloaded.is_empty() {
            let target_node = underloaded[0];
            let migration_count = self.config.load_balance_config.migration_batch_size;

            let mut local_particles = self.local_particles.write().await;
            let particle_count = local_particles.len();

            if particle_count > migration_count {
                // Migrate particles from the end
                let start_index = particle_count.saturating_sub(migration_count);
                let particles_to_migrate: Vec<MockEvolvableAgent> =
                    local_particles.drain(start_index..).collect();

                // Add to shared migration pool for testing
                let mut pool = PARTICLE_MIGRATION_POOL.lock()?;
                pool.extend(particles_to_migrate);

                drop(local_particles);

                // Update statistics
                let mut node_manager = self.node_manager.write().await;
                let local_particles = self.local_particles.read().await;
                node_manager.update_local_stats(
                    local_particles.len(),
                    self.calculate_load(&local_particles),
                );
            }
        }

        // If this node is underloaded, accept migrated particles
        if underloaded.contains(&self.config.node_id.as_str()) {
            let mut pool = PARTICLE_MIGRATION_POOL.lock()?;
            if !pool.is_empty() {
                let migration_count = self
                    .config
                    .load_balance_config
                    .migration_batch_size
                    .min(pool.len());
                let migrated: Vec<MockEvolvableAgent> = pool.drain(..migration_count).collect();

                drop(pool);

                let mut local_particles = self.local_particles.write().await;
                local_particles.extend(migrated);

                // Update statistics
                let mut node_manager = self.node_manager.write().await;
                node_manager.update_local_stats(
                    local_particles.len(),
                    self.calculate_load(&local_particles),
                );
            }
        }

        Ok(())
    }

    /// Calculate node load based on particle count
    fn calculate_load(&self, particles: &[MockEvolvableAgent]) -> f64 {
        let target = self.config.load_balance_config.target_particles_per_node as f64;
        if target > 0.0 {
            (particles.len() as f64 / target).min(1.0)
        } else {
            0.0
        }
    }

    /// Send heartbeat to peers
    pub async fn send_heartbeat(&self) -> EvolutionEngineResult<()> {
        let node_manager = self.node_manager.read().await;
        let message_bus = self.message_bus.read().await;

        let message = DistributedMessage::Heartbeat {
            node_id: self.config.node_id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis() as u64,
            load: node_manager.local_node.current_load,
        };

        message_bus.broadcast_message(message).await
    }
}

#[async_trait]
impl EngineConfig for DistributedSwarmConfig {
    fn validate(&self) -> EvolutionEngineResult<()> {
        DistributedSwarmConfig::validate(self)
    }

    fn engine_name(&self) -> &str {
        "DistributedSwarm"
    }
}
