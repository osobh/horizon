//! Actor-based consistency manager for cancel-safe async operations
//!
//! This module provides an actor-based implementation that eliminates shared
//! mutable state and provides cancel-safe operations for distributed consistency.
//!
//! # Cancel Safety
//!
//! The actor model ensures cancel safety by:
//! 1. Actor owns all mutable state exclusively - no Arc<DashMap<...>>
//! 2. Callers communicate via message passing, not shared memory
//! 3. If a caller's future is cancelled, the actor continues processing
//! 4. Graceful shutdown via explicit Shutdown message
//! 5. Background tasks (convergence checks, pruning) are managed internally

use crate::consistency_manager::{
    Conflict, ConflictStatus, ConflictVersion, ConsistencyConfig, ConsistencyLevel,
    ConvergenceState, ResolutionStrategy, ResourceType, VectorClock,
};
use crate::error::{GlobalKnowledgeGraphError, GlobalKnowledgeGraphResult};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::interval;
use uuid::Uuid;

/// Requests that can be sent to the consistency actor
#[derive(Debug)]
pub enum ConsistencyRequest {
    /// Update resource with consistency tracking
    UpdateResource {
        resource_id: String,
        resource_type: ResourceType,
        region: String,
        version: u64,
        reply: oneshot::Sender<GlobalKnowledgeGraphResult<()>>,
    },
    /// Register a conflict manually
    RegisterConflict {
        resource_id: String,
        resource_type: ResourceType,
        versions: Vec<ConflictVersion>,
        reply: oneshot::Sender<GlobalKnowledgeGraphResult<String>>,
    },
    /// Resolve a conflict
    ResolveConflict {
        conflict_id: String,
        winning_region: String,
        reply: oneshot::Sender<GlobalKnowledgeGraphResult<()>>,
    },
    /// Get conflicts by status
    GetConflicts {
        status_filter: Option<ConflictStatus>,
        reply: oneshot::Sender<Vec<Conflict>>,
    },
    /// Set resolution strategy
    SetResolutionStrategy {
        resource_id: String,
        strategy: ResolutionStrategy,
    },
    /// Get vector clock for resource
    GetVectorClock {
        resource_id: String,
        reply: oneshot::Sender<Option<VectorClock>>,
    },
    /// Check convergence status
    CheckConvergenceStatus {
        region: String,
        reply: oneshot::Sender<Option<ConvergenceState>>,
    },
    /// Update convergence state
    UpdateConvergenceState {
        region: String,
        is_converged: bool,
        divergence_ms: u64,
    },
    /// Monitor divergence between regions
    MonitorDivergence {
        region1: String,
        region2: String,
        reply: oneshot::Sender<GlobalKnowledgeGraphResult<u64>>,
    },
    /// Force convergence for a region
    ForceConvergence {
        region: String,
        reply: oneshot::Sender<GlobalKnowledgeGraphResult<()>>,
    },
    /// Get consistency level
    GetConsistencyLevel {
        reply: oneshot::Sender<ConsistencyLevel>,
    },
    /// Graceful shutdown
    Shutdown,
}

/// Consistency manager actor that owns all mutable state
///
/// This actor processes requests sequentially, eliminating the need for
/// Arc<DashMap<...>> and providing inherent cancel safety.
pub struct ConsistencyActor {
    /// Configuration - owned, not shared
    config: ConsistencyConfig,
    /// Vector clocks per resource - owned, not shared
    vector_clocks: HashMap<String, VectorClock>,
    /// Active conflicts - owned, not shared
    conflicts: HashMap<String, Conflict>,
    /// Convergence states per region - owned, not shared
    convergence_states: HashMap<String, ConvergenceState>,
    /// Resolution strategies per resource - owned, not shared
    resolution_strategies: HashMap<String, ResolutionStrategy>,
    /// Request receiver
    inbox: mpsc::Receiver<ConsistencyRequest>,
}

impl ConsistencyActor {
    /// Create a new actor with configuration
    pub fn new(config: ConsistencyConfig, inbox: mpsc::Receiver<ConsistencyRequest>) -> Self {
        Self {
            config,
            vector_clocks: HashMap::new(),
            conflicts: HashMap::new(),
            convergence_states: HashMap::new(),
            resolution_strategies: HashMap::new(),
            inbox,
        }
    }

    /// Run the actor's message processing loop
    ///
    /// This method runs until a Shutdown message is received or the
    /// channel is closed. Each request is processed to completion
    /// before the next is handled.
    ///
    /// Background tasks (convergence checks, pruning) are integrated
    /// into the message loop using tokio::select! safely since we're
    /// not holding any locks across await points.
    pub async fn run(mut self) {
        let mut convergence_interval = interval(self.config.convergence_interval);
        let mut pruning_interval = interval(self.config.pruning_interval);

        loop {
            tokio::select! {
                // Process incoming requests - highest priority
                Some(request) = self.inbox.recv() => {
                    match request {
                        ConsistencyRequest::UpdateResource {
                            resource_id,
                            resource_type,
                            region,
                            version,
                            reply,
                        } => {
                            let result = self.update_resource(&resource_id, resource_type, &region, version);
                            let _ = reply.send(result);
                        }
                        ConsistencyRequest::RegisterConflict {
                            resource_id,
                            resource_type,
                            versions,
                            reply,
                        } => {
                            let result = self.register_conflict(resource_id, resource_type, versions);
                            let _ = reply.send(result);
                        }
                        ConsistencyRequest::ResolveConflict {
                            conflict_id,
                            winning_region,
                            reply,
                        } => {
                            let result = self.resolve_conflict(&conflict_id, &winning_region);
                            let _ = reply.send(result);
                        }
                        ConsistencyRequest::GetConflicts { status_filter, reply } => {
                            let conflicts = self.get_conflicts(status_filter);
                            let _ = reply.send(conflicts);
                        }
                        ConsistencyRequest::SetResolutionStrategy { resource_id, strategy } => {
                            self.resolution_strategies.insert(resource_id, strategy);
                        }
                        ConsistencyRequest::GetVectorClock { resource_id, reply } => {
                            let clock = self.vector_clocks.get(&resource_id).cloned();
                            let _ = reply.send(clock);
                        }
                        ConsistencyRequest::CheckConvergenceStatus { region, reply } => {
                            let state = self.convergence_states.get(&region).cloned();
                            let _ = reply.send(state);
                        }
                        ConsistencyRequest::UpdateConvergenceState {
                            region,
                            is_converged,
                            divergence_ms,
                        } => {
                            if let Some(state) = self.convergence_states.get_mut(&region) {
                                state.last_check = Instant::now();
                                state.is_converged = is_converged;
                                state.divergence_ms = divergence_ms;
                            }
                        }
                        ConsistencyRequest::MonitorDivergence { region1, region2, reply } => {
                            let result = self.monitor_divergence(&region1, &region2);
                            let _ = reply.send(result);
                        }
                        ConsistencyRequest::ForceConvergence { region, reply } => {
                            let result = self.force_convergence(&region);
                            let _ = reply.send(result);
                        }
                        ConsistencyRequest::GetConsistencyLevel { reply } => {
                            let _ = reply.send(self.config.consistency_level.clone());
                        }
                        ConsistencyRequest::Shutdown => {
                            tracing::info!("ConsistencyActor shutting down");
                            break;
                        }
                    }
                }
                // Periodic convergence check - cancel-safe since no locks held
                _ = convergence_interval.tick() => {
                    self.check_convergence();
                }
                // Periodic pruning - cancel-safe since no locks held
                _ = pruning_interval.tick() => {
                    self.prune_vector_clocks();
                }
            }
        }
        // Actor cleanup happens here via Drop
        tracing::debug!("ConsistencyActor terminated gracefully");
    }

    /// Update resource with consistency tracking (internal implementation)
    fn update_resource(
        &mut self,
        resource_id: &str,
        resource_type: ResourceType,
        region: &str,
        version: u64,
    ) -> GlobalKnowledgeGraphResult<()> {
        // Update or create vector clock and get a clone for conflict detection
        let clock_clone = {
            let clock = self
                .vector_clocks
                .entry(resource_id.to_string())
                .or_insert_with(VectorClock::new);
            clock.increment(region);
            clock.clone()
        };

        // Check for conflicts if not eventual consistency
        if self.config.consistency_level != ConsistencyLevel::Eventual {
            self.detect_conflicts(resource_id, resource_type, region, version, clock_clone)?;
        }

        Ok(())
    }

    /// Detect conflicts (internal implementation)
    fn detect_conflicts(
        &mut self,
        resource_id: &str,
        resource_type: ResourceType,
        region: &str,
        version: u64,
        clock: VectorClock,
    ) -> GlobalKnowledgeGraphResult<()> {
        let conflict_key = format!("{}:{:?}", resource_id, resource_type);

        if let Some(conflict) = self.conflicts.get_mut(&conflict_key) {
            // Add new version to existing conflict
            conflict.versions.push(ConflictVersion {
                region: region.to_string(),
                version,
                vector_clock: clock,
                updated_at: chrono::Utc::now(),
            });

            if self.config.enable_auto_resolution {
                self.auto_resolve_conflict(&conflict_key)?;
            }
        }
        // In production, would check against recent updates from other regions

        Ok(())
    }

    /// Auto-resolve conflict (internal implementation)
    fn auto_resolve_conflict(&mut self, conflict_key: &str) -> GlobalKnowledgeGraphResult<()> {
        if let Some(conflict) = self.conflicts.get_mut(conflict_key) {
            let strategy = self
                .resolution_strategies
                .get(&conflict.resource_id)
                .cloned()
                .unwrap_or(ResolutionStrategy::LastWriteWins);

            match strategy {
                ResolutionStrategy::LastWriteWins => {
                    conflict.versions.sort_by_key(|v| v.updated_at);
                    if let Some(winner) = conflict.versions.last() {
                        conflict.status = ConflictStatus::Resolved;
                        tracing::info!(
                            "Resolved conflict {} using LastWriteWins: region {} wins",
                            conflict.id,
                            winner.region
                        );
                    }
                }
                ResolutionStrategy::HigherVersionWins => {
                    if let Some(winner) = conflict.versions.iter().max_by_key(|v| v.version) {
                        conflict.status = ConflictStatus::Resolved;
                        tracing::info!(
                            "Resolved conflict {} using HigherVersionWins: region {} wins",
                            conflict.id,
                            winner.region
                        );
                    }
                }
                _ => {
                    conflict.status = ConflictStatus::Detected;
                }
            }
        }

        Ok(())
    }

    /// Register conflict manually (internal implementation)
    fn register_conflict(
        &mut self,
        resource_id: String,
        resource_type: ResourceType,
        versions: Vec<ConflictVersion>,
    ) -> GlobalKnowledgeGraphResult<String> {
        let conflict_id = Uuid::new_v4().to_string();
        let conflict = Conflict {
            id: conflict_id.clone(),
            resource_id: resource_id.clone(),
            resource_type: resource_type.clone(),
            versions,
            detected_at: Instant::now(),
            status: ConflictStatus::Detected,
        };

        let conflict_key = format!("{}:{:?}", resource_id, resource_type);
        self.conflicts.insert(conflict_key, conflict);

        Ok(conflict_id)
    }

    /// Resolve conflict manually (internal implementation)
    fn resolve_conflict(
        &mut self,
        conflict_id: &str,
        winning_region: &str,
    ) -> GlobalKnowledgeGraphResult<()> {
        for conflict in self.conflicts.values_mut() {
            if conflict.id == conflict_id {
                if conflict.versions.iter().any(|v| v.region == winning_region) {
                    conflict.status = ConflictStatus::Resolved;
                    return Ok(());
                }
            }
        }

        Err(GlobalKnowledgeGraphError::ConsistencyConflict {
            region1: "unknown".to_string(),
            region2: "unknown".to_string(),
            conflict_type: "Conflict not found".to_string(),
        })
    }

    /// Get conflicts by status (internal implementation)
    fn get_conflicts(&self, status_filter: Option<ConflictStatus>) -> Vec<Conflict> {
        self.conflicts
            .values()
            .filter(|conflict| {
                status_filter
                    .as_ref()
                    .map_or(true, |status| &conflict.status == status)
            })
            .cloned()
            .collect()
    }

    /// Monitor divergence between regions (internal implementation)
    fn monitor_divergence(&self, region1: &str, region2: &str) -> GlobalKnowledgeGraphResult<u64> {
        let state1 = self.convergence_states.get(region1);
        let state2 = self.convergence_states.get(region2);

        match (state1, state2) {
            (Some(s1), Some(s2)) => Ok((s1.divergence_ms + s2.divergence_ms) / 2),
            _ => Ok(0),
        }
    }

    /// Force convergence for a region (internal implementation)
    fn force_convergence(&mut self, region: &str) -> GlobalKnowledgeGraphResult<()> {
        if let Some(state) = self.convergence_states.get_mut(region) {
            state.is_converged = true;
            state.divergence_ms = 0;
            state.last_check = Instant::now();
        }
        Ok(())
    }

    /// Check convergence across regions (internal background task)
    fn check_convergence(&mut self) {
        for state in self.convergence_states.values_mut() {
            state.last_check = Instant::now();

            if state.divergence_ms > self.config.max_divergence_ms {
                state.is_converged = false;
                tracing::warn!(
                    "Region {} has diverged beyond threshold: {}ms",
                    state.region,
                    state.divergence_ms
                );
            } else {
                state.is_converged = true;
            }
        }
    }

    /// Prune old vector clocks (internal background task)
    fn prune_vector_clocks(&mut self) {
        // In production, would prune clocks for deleted resources
        tracing::debug!(
            "Pruning vector clocks, current count: {}",
            self.vector_clocks.len()
        );
    }
}

/// Handle for interacting with the consistency actor
///
/// This handle is cheap to clone and can be used from multiple tasks.
/// All operations are cancel-safe - if the caller's future is dropped,
/// the actor continues processing.
#[derive(Clone)]
pub struct ConsistencyHandle {
    sender: mpsc::Sender<ConsistencyRequest>,
}

impl ConsistencyHandle {
    /// Create a new handle from a sender
    pub fn new(sender: mpsc::Sender<ConsistencyRequest>) -> Self {
        Self { sender }
    }

    /// Update resource with consistency tracking
    ///
    /// This operation is cancel-safe. If the caller's future is dropped,
    /// the update continues to completion in the actor.
    pub async fn update_resource(
        &self,
        resource_id: &str,
        resource_type: ResourceType,
        region: &str,
        version: u64,
    ) -> GlobalKnowledgeGraphResult<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ConsistencyRequest::UpdateResource {
                resource_id: resource_id.to_string(),
                resource_type,
                region: region.to_string(),
                version,
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))?
    }

    /// Register a conflict manually
    pub async fn register_conflict(
        &self,
        resource_id: String,
        resource_type: ResourceType,
        versions: Vec<ConflictVersion>,
    ) -> GlobalKnowledgeGraphResult<String> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ConsistencyRequest::RegisterConflict {
                resource_id,
                resource_type,
                versions,
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))?
    }

    /// Resolve a conflict
    pub async fn resolve_conflict(
        &self,
        conflict_id: &str,
        winning_region: &str,
    ) -> GlobalKnowledgeGraphResult<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ConsistencyRequest::ResolveConflict {
                conflict_id: conflict_id.to_string(),
                winning_region: winning_region.to_string(),
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))?
    }

    /// Get conflicts by status
    pub async fn get_conflicts(
        &self,
        status_filter: Option<ConflictStatus>,
    ) -> GlobalKnowledgeGraphResult<Vec<Conflict>> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ConsistencyRequest::GetConflicts {
                status_filter,
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))
    }

    /// Set resolution strategy for a resource
    pub async fn set_resolution_strategy(
        &self,
        resource_id: String,
        strategy: ResolutionStrategy,
    ) -> GlobalKnowledgeGraphResult<()> {
        self.sender
            .send(ConsistencyRequest::SetResolutionStrategy {
                resource_id,
                strategy,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))
    }

    /// Get vector clock for resource
    pub async fn get_vector_clock(
        &self,
        resource_id: &str,
    ) -> GlobalKnowledgeGraphResult<Option<VectorClock>> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ConsistencyRequest::GetVectorClock {
                resource_id: resource_id.to_string(),
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))
    }

    /// Check convergence status for a region
    pub async fn check_convergence_status(
        &self,
        region: &str,
    ) -> GlobalKnowledgeGraphResult<Option<ConvergenceState>> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ConsistencyRequest::CheckConvergenceStatus {
                region: region.to_string(),
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))
    }

    /// Update convergence state for a region
    pub async fn update_convergence_state(
        &self,
        region: String,
        is_converged: bool,
        divergence_ms: u64,
    ) -> GlobalKnowledgeGraphResult<()> {
        self.sender
            .send(ConsistencyRequest::UpdateConvergenceState {
                region,
                is_converged,
                divergence_ms,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))
    }

    /// Monitor divergence between two regions
    pub async fn monitor_divergence(
        &self,
        region1: &str,
        region2: &str,
    ) -> GlobalKnowledgeGraphResult<u64> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ConsistencyRequest::MonitorDivergence {
                region1: region1.to_string(),
                region2: region2.to_string(),
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))?
    }

    /// Force convergence for a region
    pub async fn force_convergence(&self, region: &str) -> GlobalKnowledgeGraphResult<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ConsistencyRequest::ForceConvergence {
                region: region.to_string(),
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))?
    }

    /// Get current consistency level
    pub async fn get_consistency_level(&self) -> GlobalKnowledgeGraphResult<ConsistencyLevel> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(ConsistencyRequest::GetConsistencyLevel { reply: tx })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))
    }

    /// Request graceful shutdown
    pub async fn shutdown(&self) -> GlobalKnowledgeGraphResult<()> {
        self.sender
            .send(ConsistencyRequest::Shutdown)
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor already stopped".to_string()))
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
/// let (actor, handle) = create_consistency_actor(config);
///
/// // Spawn the actor
/// tokio::spawn(actor.run());
///
/// // Use the handle
/// handle.update_resource("node-1", ResourceType::Node, "us-east-1", 1).await?;
///
/// // Shutdown
/// handle.shutdown().await?;
/// ```
pub fn create_consistency_actor(
    config: ConsistencyConfig,
) -> (ConsistencyActor, ConsistencyHandle) {
    let (tx, rx) = mpsc::channel(64);

    let actor = ConsistencyActor::new(config, rx);
    let handle = ConsistencyHandle::new(tx);

    (actor, handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_actor_update_resource() {
        let (actor, handle) = create_consistency_actor(ConsistencyConfig::default());

        // Spawn actor
        let actor_handle = tokio::spawn(actor.run());

        // Update resource
        let result = handle
            .update_resource("node-123", ResourceType::Node, "us-east-1", 1)
            .await;
        assert!(result.is_ok());

        // Check vector clock
        let clock = handle.get_vector_clock("node-123").await.unwrap();
        assert!(clock.is_some());
        assert_eq!(clock.unwrap().get("us-east-1"), 1);

        // Shutdown
        handle.shutdown().await.unwrap();
        actor_handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_register_and_resolve_conflict() {
        let (actor, handle) = create_consistency_actor(ConsistencyConfig::default());
        tokio::spawn(actor.run());

        let versions = vec![
            ConflictVersion {
                region: "us-east-1".to_string(),
                version: 1,
                vector_clock: VectorClock::new(),
                updated_at: chrono::Utc::now(),
            },
            ConflictVersion {
                region: "eu-west-1".to_string(),
                version: 2,
                vector_clock: VectorClock::new(),
                updated_at: chrono::Utc::now(),
            },
        ];

        let conflict_id = handle
            .register_conflict("node-123".to_string(), ResourceType::Node, versions)
            .await
            .unwrap();

        assert!(!conflict_id.is_empty());

        // Resolve conflict
        let result = handle.resolve_conflict(&conflict_id, "us-east-1").await;
        assert!(result.is_ok());

        // Check conflicts
        let resolved = handle
            .get_conflicts(Some(ConflictStatus::Resolved))
            .await
            .unwrap();
        assert_eq!(resolved.len(), 1);

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_cancel_safety() {
        let (actor, handle) = create_consistency_actor(ConsistencyConfig::default());
        tokio::spawn(actor.run());

        // Start an update but cancel it
        let handle_clone = handle.clone();
        let update_future =
            handle_clone.update_resource("node-1", ResourceType::Node, "us-east-1", 1);

        // Drop the future immediately (simulating cancellation)
        drop(update_future);

        // Give actor time to process any pending messages
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Actor should still be running and responsive
        let result = handle
            .update_resource("node-2", ResourceType::Node, "us-east-1", 1)
            .await;
        assert!(result.is_ok());

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_consistency_level() {
        let config = ConsistencyConfig {
            consistency_level: ConsistencyLevel::Strong,
            ..Default::default()
        };

        let (actor, handle) = create_consistency_actor(config);
        tokio::spawn(actor.run());

        let level = handle.get_consistency_level().await.unwrap();
        assert_eq!(level, ConsistencyLevel::Strong);

        handle.shutdown().await.unwrap();
    }
}
