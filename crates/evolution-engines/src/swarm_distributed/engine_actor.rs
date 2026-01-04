//! Actor-based distributed swarm engine for cancel-safe async operations
//!
//! This module provides an actor-based implementation that eliminates sequential
//! lock acquisitions and provides cancel-safe operations.
//!
//! # Cancel Safety
//!
//! The actor model ensures cancel safety by:
//! 1. Actor owns all mutable state exclusively - no Arc<RwLock<...>>
//! 2. No sequential lock acquisitions - single actor processes requests
//! 3. Callers communicate via message passing, not shared memory
//! 4. Graceful shutdown via explicit Shutdown message

use super::{
    config::DistributedSwarmConfig,
    message_bus::{GlobalBestHandler, HeartbeatHandler, MessageBus, MigrationHandler},
    messages::{DistributedMessage, MigrationParticle},
    node_manager::NodeManager,
    types::{ClusterHealth, ClusterStatus, NodeStatus as DistNodeStatus},
};
use crate::{
    error::{EvolutionEngineError, EvolutionEngineResult},
    traits::MockEvolvableAgent,
};
use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

// Global shared state for testing (unchanged from original)
lazy_static::lazy_static! {
    static ref GLOBAL_BEST_FITNESS: Mutex<Option<f64>> = Mutex::new(None);
    static ref PARTICLE_MIGRATION_POOL: Mutex<Vec<MockEvolvableAgent>> = Mutex::new(Vec::new());
}

/// Requests that can be sent to the distributed swarm actor
#[derive(Debug)]
pub enum SwarmEngineRequest {
    /// Start the engine (join cluster, register handlers)
    Start {
        reply: oneshot::Sender<EvolutionEngineResult<()>>,
    },
    /// Stop the engine gracefully
    Stop {
        reply: oneshot::Sender<EvolutionEngineResult<()>>,
    },
    /// Get cluster status
    GetClusterStatus {
        reply: oneshot::Sender<EvolutionEngineResult<ClusterStatus>>,
    },
    /// Migrate particles to another node
    MigrateParticles {
        particle_ids: Vec<String>,
        target_node: String,
        reply: oneshot::Sender<EvolutionEngineResult<()>>,
    },
    /// Get node ID
    GetNodeId { reply: oneshot::Sender<String> },
    /// Get local particle count
    GetLocalParticleCount { reply: oneshot::Sender<usize> },
    /// Get peer count
    GetPeerCount { reply: oneshot::Sender<usize> },
    /// Set local population
    SetLocalPopulation {
        particles: Vec<MockEvolvableAgent>,
        reply: oneshot::Sender<EvolutionEngineResult<()>>,
    },
    /// Sync global best fitness
    SyncGlobalBest {
        reply: oneshot::Sender<EvolutionEngineResult<()>>,
    },
    /// Balance load across nodes
    BalanceLoad {
        reply: oneshot::Sender<EvolutionEngineResult<()>>,
    },
    /// Send heartbeat to peers
    SendHeartbeat {
        reply: oneshot::Sender<EvolutionEngineResult<()>>,
    },
    /// Get global best fitness
    GetGlobalBestFitness { reply: oneshot::Sender<Option<f64>> },
    /// Graceful shutdown
    Shutdown,
}

/// Distributed swarm engine actor that owns all mutable state
///
/// This actor processes requests sequentially, eliminating the need for
/// multiple Arc<RwLock<...>> and providing inherent cancel safety.
pub struct DistributedSwarmActor {
    /// Configuration
    config: DistributedSwarmConfig,
    /// Node manager - owned, not shared
    node_manager: NodeManager,
    /// Message bus - owned, not shared
    message_bus: MessageBus,
    /// Local particles - owned, not shared
    local_particles: Vec<MockEvolvableAgent>,
    /// Global best known to this node - owned, not shared
    global_best: Option<MockEvolvableAgent>,
    /// Global best fitness - owned, not shared
    global_best_fitness: Option<f64>,
    /// Current generation - owned, not shared
    current_generation: u32,
    /// Request receiver
    inbox: mpsc::Receiver<SwarmEngineRequest>,
}

impl DistributedSwarmActor {
    /// Create a new actor with configuration
    pub async fn new(
        config: DistributedSwarmConfig,
        inbox: mpsc::Receiver<SwarmEngineRequest>,
    ) -> EvolutionEngineResult<Self> {
        config.validate()?;

        Ok(Self {
            node_manager: NodeManager::new(config.clone()).await?,
            message_bus: MessageBus::new(config.clone()).await?,
            config,
            local_particles: Vec::new(),
            global_best: None,
            global_best_fitness: None,
            current_generation: 0,
            inbox,
        })
    }

    /// Run the actor's message processing loop
    pub async fn run(mut self) {
        while let Some(request) = self.inbox.recv().await {
            match request {
                SwarmEngineRequest::Start { reply } => {
                    let result = self.start_engine().await;
                    let _ = reply.send(result);
                }
                SwarmEngineRequest::Stop { reply } => {
                    let result = self.stop_engine().await;
                    let _ = reply.send(result);
                }
                SwarmEngineRequest::GetClusterStatus { reply } => {
                    let result = self.get_cluster_status();
                    let _ = reply.send(result);
                }
                SwarmEngineRequest::MigrateParticles {
                    particle_ids,
                    target_node,
                    reply,
                } => {
                    let result = self.migrate_particles(particle_ids, target_node).await;
                    let _ = reply.send(result);
                }
                SwarmEngineRequest::GetNodeId { reply } => {
                    let _ = reply.send(self.config.node_id.clone());
                }
                SwarmEngineRequest::GetLocalParticleCount { reply } => {
                    let _ = reply.send(self.local_particles.len());
                }
                SwarmEngineRequest::GetPeerCount { reply } => {
                    let _ = reply.send(self.node_manager.get_peer_count());
                }
                SwarmEngineRequest::SetLocalPopulation { particles, reply } => {
                    let result = self.set_local_population(particles);
                    let _ = reply.send(result);
                }
                SwarmEngineRequest::SyncGlobalBest { reply } => {
                    let result = self.sync_global_best().await;
                    let _ = reply.send(result);
                }
                SwarmEngineRequest::BalanceLoad { reply } => {
                    let result = self.balance_load().await;
                    let _ = reply.send(result);
                }
                SwarmEngineRequest::SendHeartbeat { reply } => {
                    let result = self.send_heartbeat().await;
                    let _ = reply.send(result);
                }
                SwarmEngineRequest::GetGlobalBestFitness { reply } => {
                    let _ = reply.send(self.global_best_fitness);
                }
                SwarmEngineRequest::Shutdown => {
                    tracing::info!("DistributedSwarmActor shutting down");
                    // Attempt graceful stop
                    let _ = self.stop_engine().await;
                    break;
                }
            }
        }
        tracing::debug!("DistributedSwarmActor terminated gracefully");
    }

    /// Start the distributed engine (internal implementation)
    async fn start_engine(&mut self) -> EvolutionEngineResult<()> {
        // Join the cluster
        self.node_manager.join_cluster().await?;

        // Register message handlers
        self.message_bus.register_handler(
            "heartbeat".to_string(),
            Box::new(HeartbeatHandler::new(self.config.node_id.clone())),
        );
        self.message_bus
            .register_handler("particle_migration".to_string(), Box::new(MigrationHandler));
        self.message_bus.register_handler(
            "global_best_update".to_string(),
            Box::new(GlobalBestHandler),
        );

        Ok(())
    }

    /// Stop the distributed engine (internal implementation)
    async fn stop_engine(&mut self) -> EvolutionEngineResult<()> {
        // Leave cluster gracefully
        self.node_manager.leave_cluster().await?;

        // Stop message bus
        self.message_bus.stop().await?;

        Ok(())
    }

    /// Get cluster status (internal implementation - no locks needed!)
    fn get_cluster_status(&self) -> EvolutionEngineResult<ClusterStatus> {
        let mut total_particles = self.local_particles.len();
        let mut total_load = self.node_manager.local_node.current_load;
        let mut active_nodes = if self.node_manager.local_node.status == DistNodeStatus::Active {
            1
        } else {
            0
        };

        for node in self.node_manager.peer_nodes.values() {
            total_particles += node.particle_count;
            total_load += node.current_load;
            if node.status == DistNodeStatus::Active {
                active_nodes += 1;
            }
        }

        let total_nodes = self.node_manager.peer_nodes.len() + 1;
        let average_load = if active_nodes > 0 {
            total_load / active_nodes as f64
        } else {
            0.0
        };

        Ok(ClusterStatus {
            total_nodes,
            active_nodes,
            total_particles,
            generation: self.current_generation,
            global_best_fitness: self.global_best_fitness,
            average_load,
            health: if active_nodes == total_nodes {
                ClusterHealth::Healthy
            } else if active_nodes > total_nodes / 2 {
                ClusterHealth::Degraded
            } else {
                ClusterHealth::Critical
            },
        })
    }

    /// Migrate particles to another node (internal implementation)
    async fn migrate_particles(
        &mut self,
        particle_ids: Vec<String>,
        target_node: String,
    ) -> EvolutionEngineResult<()> {
        let mut migration_particles = Vec::new();
        let mut remaining_particles = Vec::new();

        // Extract particles for migration
        for particle in self.local_particles.drain(..) {
            let particle_id = particle.id.clone();
            if particle_ids.contains(&particle_id) {
                migration_particles.push(MigrationParticle::from_agent(
                    particle_id,
                    &particle,
                    vec![0.1; 3],
                    0.5,
                ));
            } else {
                remaining_particles.push(particle);
            }
        }
        self.local_particles = remaining_particles;

        // Send migration message
        let message = DistributedMessage::ParticleMigration {
            particles: migration_particles,
            source_node: self.config.node_id.clone(),
            target_node: target_node.clone(),
        };
        self.message_bus.send_message(&target_node, message).await?;

        // Update local node statistics
        self.node_manager.update_local_stats(
            self.local_particles.len(),
            self.calculate_load(&self.local_particles),
        );

        Ok(())
    }

    /// Set local population (internal implementation)
    fn set_local_population(
        &mut self,
        particles: Vec<MockEvolvableAgent>,
    ) -> EvolutionEngineResult<()> {
        self.local_particles = particles;

        // Update node statistics
        let load = self.calculate_load(&self.local_particles);
        self.node_manager
            .update_local_stats(self.local_particles.len(), load);

        Ok(())
    }

    /// Sync global best fitness (internal implementation)
    async fn sync_global_best(&mut self) -> EvolutionEngineResult<()> {
        // Find local best (parallelized with Rayon)
        let local_best: Option<f64> = if self.local_particles.is_empty() {
            None
        } else {
            Some(
                self.local_particles
                    .par_iter()
                    .map(|particle| particle.get_fitness())
                    .reduce(|| f64::MIN, |a, b| a.max(b)),
            )
        };

        // Update global best if local is better
        if let Some(fitness) = local_best {
            if self.global_best_fitness.is_none() || fitness > self.global_best_fitness.unwrap() {
                self.global_best_fitness = Some(fitness);

                // Update shared global state for testing
                if let Ok(mut shared_global) = GLOBAL_BEST_FITNESS.lock() {
                    if shared_global.is_none() || fitness > shared_global.unwrap() {
                        *shared_global = Some(fitness);
                    }
                }

                // Broadcast update to peers
                if !self.local_particles.is_empty() {
                    let message = DistributedMessage::GlobalBestUpdate {
                        best_particle: MigrationParticle::from_agent(
                            "best".to_string(),
                            &self.local_particles[0],
                            vec![0.1; 3],
                            fitness,
                        ),
                        fitness,
                        generation: self.current_generation,
                    };
                    self.message_bus.broadcast_message(message).await?;
                }
            }
        }

        // Also check shared global state
        if let Ok(shared_global) = GLOBAL_BEST_FITNESS.lock() {
            if let Some(shared_fitness) = *shared_global {
                if self.global_best_fitness.is_none()
                    || shared_fitness > self.global_best_fitness.unwrap()
                {
                    self.global_best_fitness = Some(shared_fitness);
                }
            }
        }

        Ok(())
    }

    /// Balance load across nodes (internal implementation)
    async fn balance_load(&mut self) -> EvolutionEngineResult<()> {
        let threshold = self.config.load_balance_config.rebalance_threshold;
        // Clone the results to avoid holding immutable borrow during mutable operations
        let (overloaded, underloaded): (Vec<String>, Vec<String>) = {
            let (o, u) = self.node_manager.find_rebalance_candidates(threshold);
            (
                o.into_iter().map(String::from).collect(),
                u.into_iter().map(String::from).collect(),
            )
        };

        // If this node is overloaded, migrate some particles
        if overloaded.iter().any(|n| n == &self.config.node_id) && !underloaded.is_empty() {
            let migration_count = self.config.load_balance_config.migration_batch_size;
            let particle_count = self.local_particles.len();

            if particle_count > migration_count {
                let start_index = particle_count.saturating_sub(migration_count);
                let particles_to_migrate: Vec<MockEvolvableAgent> =
                    self.local_particles.drain(start_index..).collect();

                // Add to shared migration pool for testing
                if let Ok(mut pool) = PARTICLE_MIGRATION_POOL.lock() {
                    pool.extend(particles_to_migrate);
                }

                // Update statistics
                let load = self.calculate_load(&self.local_particles);
                self.node_manager
                    .update_local_stats(self.local_particles.len(), load);
            }
        }

        // If this node is underloaded, accept migrated particles
        if underloaded.iter().any(|n| n == &self.config.node_id) {
            if let Ok(mut pool) = PARTICLE_MIGRATION_POOL.lock() {
                if !pool.is_empty() {
                    let migration_count = self
                        .config
                        .load_balance_config
                        .migration_batch_size
                        .min(pool.len());
                    let migrated: Vec<MockEvolvableAgent> = pool.drain(..migration_count).collect();
                    self.local_particles.extend(migrated);

                    // Update statistics
                    let load = self.calculate_load(&self.local_particles);
                    self.node_manager
                        .update_local_stats(self.local_particles.len(), load);
                }
            }
        }

        Ok(())
    }

    /// Send heartbeat to peers (internal implementation)
    async fn send_heartbeat(&self) -> EvolutionEngineResult<()> {
        let message = DistributedMessage::Heartbeat {
            node_id: self.config.node_id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| EvolutionEngineError::DistributedError(e.to_string()))?
                .as_millis() as u64,
            load: self.node_manager.local_node.current_load,
        };

        self.message_bus.broadcast_message(message).await
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
}

/// Handle for interacting with the distributed swarm actor
///
/// This handle is cheap to clone and can be used from multiple tasks.
/// All operations are cancel-safe.
#[derive(Clone)]
pub struct DistributedSwarmHandle {
    sender: mpsc::Sender<SwarmEngineRequest>,
}

impl DistributedSwarmHandle {
    /// Create a new handle from a sender
    pub fn new(sender: mpsc::Sender<SwarmEngineRequest>) -> Self {
        Self { sender }
    }

    /// Start the distributed engine
    pub async fn start(&self) -> EvolutionEngineResult<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmEngineRequest::Start { reply: tx })
            .await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor dropped".to_string()))?
    }

    /// Stop the distributed engine
    pub async fn stop(&self) -> EvolutionEngineResult<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmEngineRequest::Stop { reply: tx })
            .await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor dropped".to_string()))?
    }

    /// Get cluster status
    pub async fn get_cluster_status(&self) -> EvolutionEngineResult<ClusterStatus> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmEngineRequest::GetClusterStatus { reply: tx })
            .await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor dropped".to_string()))?
    }

    /// Migrate particles to another node
    pub async fn migrate_particles(
        &self,
        particle_ids: Vec<String>,
        target_node: String,
    ) -> EvolutionEngineResult<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmEngineRequest::MigrateParticles {
                particle_ids,
                target_node,
                reply: tx,
            })
            .await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor dropped".to_string()))?
    }

    /// Get node ID
    pub async fn get_node_id(&self) -> EvolutionEngineResult<String> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmEngineRequest::GetNodeId { reply: tx })
            .await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor dropped".to_string()))
    }

    /// Get local particle count
    pub async fn get_local_particle_count(&self) -> EvolutionEngineResult<usize> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmEngineRequest::GetLocalParticleCount { reply: tx })
            .await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor dropped".to_string()))
    }

    /// Get peer count
    pub async fn get_peer_count(&self) -> EvolutionEngineResult<usize> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmEngineRequest::GetPeerCount { reply: tx })
            .await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor dropped".to_string()))
    }

    /// Set local population
    pub async fn set_local_population(
        &self,
        particles: Vec<MockEvolvableAgent>,
    ) -> EvolutionEngineResult<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmEngineRequest::SetLocalPopulation {
                particles,
                reply: tx,
            })
            .await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor dropped".to_string()))?
    }

    /// Sync global best fitness
    pub async fn sync_global_best(&self) -> EvolutionEngineResult<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmEngineRequest::SyncGlobalBest { reply: tx })
            .await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor dropped".to_string()))?
    }

    /// Balance load across nodes
    pub async fn balance_load(&self) -> EvolutionEngineResult<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmEngineRequest::BalanceLoad { reply: tx })
            .await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor dropped".to_string()))?
    }

    /// Send heartbeat to peers
    pub async fn send_heartbeat(&self) -> EvolutionEngineResult<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmEngineRequest::SendHeartbeat { reply: tx })
            .await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor dropped".to_string()))?
    }

    /// Get global best fitness
    pub async fn get_global_best_fitness(&self) -> EvolutionEngineResult<Option<f64>> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmEngineRequest::GetGlobalBestFitness { reply: tx })
            .await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionEngineError::DistributedError("Actor dropped".to_string()))
    }

    /// Request graceful shutdown
    pub async fn shutdown(&self) -> EvolutionEngineResult<()> {
        self.sender
            .send(SwarmEngineRequest::Shutdown)
            .await
            .map_err(|_| {
                EvolutionEngineError::DistributedError("Actor already stopped".to_string())
            })
    }
}

/// Create an actor and its handle
///
/// # Returns
/// A tuple of (actor, handle). The actor should be spawned as a task,
/// and the handle used to communicate with it.
///
/// # Example
/// ```ignore
/// let (actor, handle) = create_distributed_swarm_actor(config).await?;
///
/// // Spawn the actor
/// tokio::spawn(actor.run());
///
/// // Use the handle
/// handle.start().await?;
/// let status = handle.get_cluster_status().await?;
///
/// // Shutdown
/// handle.shutdown().await?;
/// ```
pub async fn create_distributed_swarm_actor(
    config: DistributedSwarmConfig,
) -> EvolutionEngineResult<(DistributedSwarmActor, DistributedSwarmHandle)> {
    let (tx, rx) = mpsc::channel(64);

    let actor = DistributedSwarmActor::new(config, rx).await?;
    let handle = DistributedSwarmHandle::new(tx);

    Ok((actor, handle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm_distributed::config::LoadBalanceConfig;

    fn create_test_config() -> DistributedSwarmConfig {
        DistributedSwarmConfig::new("test-node-1".to_string())
    }

    #[tokio::test]
    async fn test_actor_creation() {
        let config = create_test_config();
        let result = create_distributed_swarm_actor(config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_actor_get_node_id() {
        let config = create_test_config();
        let (actor, handle) = create_distributed_swarm_actor(config).await.unwrap();

        tokio::spawn(actor.run());

        let node_id = handle.get_node_id().await.unwrap();
        assert_eq!(node_id, "test-node-1");

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_cancel_safety() {
        let config = create_test_config();
        let (actor, handle) = create_distributed_swarm_actor(config).await.unwrap();

        tokio::spawn(actor.run());

        // Start a request but cancel it
        let handle_clone = handle.clone();
        let future = handle_clone.get_local_particle_count();
        drop(future);

        // Give actor time to process
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Actor should still be responsive
        let count = handle.get_local_particle_count().await.unwrap();
        assert_eq!(count, 0);

        handle.shutdown().await.unwrap();
    }
}
