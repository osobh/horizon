//! HPC Channels integration for evolution events.
//!
//! This module bridges evolution events to the hpc-channels message bus,
//! enabling real-time monitoring and coordination of evolution processes.
//!
//! # Channels Used
//!
//! - `hpc.evolution.event` - Evolution lifecycle events
//! - `hpc.evolution.capabilities` - Agent capability updates
//!
//! # Example
//!
//! ```rust,ignore
//! use stratoswarm_evolution::channels::EvolutionChannelBridge;
//!
//! let bridge = EvolutionChannelBridge::new();
//!
//! // Publish generation complete event
//! bridge.publish_generation_complete(42, &stats);
//!
//! // Subscribe to evolution events
//! let mut rx = bridge.subscribe_events();
//! while let Ok(event) = rx.recv().await {
//!     println!("Evolution event: {:?}", event);
//! }
//! ```

use std::sync::Arc;

use tokio::sync::broadcast;

use crate::{EvolutionStats, FitnessScore};

/// Evolution lifecycle events published to hpc-channels.
#[derive(Clone, Debug)]
pub enum EvolutionEvent {
    /// Population initialized.
    PopulationInitialized {
        /// Population size.
        size: usize,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Generation completed.
    GenerationComplete {
        /// Generation number.
        generation: u64,
        /// Best fitness achieved.
        best_fitness: f64,
        /// Average fitness.
        average_fitness: f64,
        /// Diversity index.
        diversity_index: f64,
    },
    /// Mutation applied.
    MutationApplied {
        /// Type of mutation.
        mutation_type: String,
        /// Number of individuals affected.
        affected_count: usize,
    },
    /// Kernel hot-swapped.
    KernelHotSwapped {
        /// Old kernel ID.
        old_kernel_id: String,
        /// New kernel ID.
        new_kernel_id: String,
        /// Performance improvement percentage.
        improvement_percent: f64,
    },
    /// XP reward granted.
    XpRewarded {
        /// Agent ID.
        agent_id: String,
        /// Amount of XP.
        xp_amount: u64,
        /// Reason for reward.
        reason: String,
    },
}

/// Agent capability update events.
#[derive(Clone, Debug)]
pub struct CapabilityUpdate {
    /// Agent ID.
    pub agent_id: String,
    /// Capability name.
    pub capability: String,
    /// New level (0-100).
    pub level: u32,
    /// Change from previous level.
    pub delta: i32,
}

/// Bridge between evolution events and hpc-channels.
pub struct EvolutionChannelBridge {
    /// Broadcast sender for evolution events.
    events_tx: broadcast::Sender<EvolutionEvent>,
    /// Broadcast sender for capability updates.
    capabilities_tx: broadcast::Sender<CapabilityUpdate>,
}

impl EvolutionChannelBridge {
    /// Create a new evolution channel bridge.
    ///
    /// Registers channels with the hpc-channels global registry.
    pub fn new() -> Self {
        let events_tx = hpc_channels::broadcast::<EvolutionEvent>(
            hpc_channels::channels::EVOLUTION_EVENT,
            1024,
        );
        let capabilities_tx = hpc_channels::broadcast::<CapabilityUpdate>(
            hpc_channels::channels::EVOLUTION_CAPABILITIES,
            256,
        );

        Self {
            events_tx,
            capabilities_tx,
        }
    }

    /// Publish a population initialized event.
    pub fn publish_population_initialized(&self, size: usize) {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let _ = self.events_tx.send(EvolutionEvent::PopulationInitialized {
            size,
            timestamp_ms,
        });
    }

    /// Publish a generation complete event.
    pub fn publish_generation_complete(&self, generation: u64, stats: &EvolutionStats) {
        let _ = self.events_tx.send(EvolutionEvent::GenerationComplete {
            generation,
            best_fitness: stats.best_fitness,
            average_fitness: stats.average_fitness,
            diversity_index: stats.diversity_index,
        });
    }

    /// Publish a mutation applied event.
    pub fn publish_mutation_applied(&self, mutation_type: &str, affected_count: usize) {
        let _ = self.events_tx.send(EvolutionEvent::MutationApplied {
            mutation_type: mutation_type.to_string(),
            affected_count,
        });
    }

    /// Publish a kernel hot-swap event.
    pub fn publish_kernel_hot_swapped(
        &self,
        old_kernel_id: &str,
        new_kernel_id: &str,
        improvement_percent: f64,
    ) {
        let _ = self.events_tx.send(EvolutionEvent::KernelHotSwapped {
            old_kernel_id: old_kernel_id.to_string(),
            new_kernel_id: new_kernel_id.to_string(),
            improvement_percent,
        });
    }

    /// Publish an XP reward event.
    pub fn publish_xp_rewarded(&self, agent_id: &str, xp_amount: u64, reason: &str) {
        let _ = self.events_tx.send(EvolutionEvent::XpRewarded {
            agent_id: agent_id.to_string(),
            xp_amount,
            reason: reason.to_string(),
        });
    }

    /// Publish a capability update.
    pub fn publish_capability_update(&self, update: CapabilityUpdate) {
        let _ = self.capabilities_tx.send(update);
    }

    /// Subscribe to evolution events.
    pub fn subscribe_events(&self) -> broadcast::Receiver<EvolutionEvent> {
        self.events_tx.subscribe()
    }

    /// Subscribe to capability updates.
    pub fn subscribe_capabilities(&self) -> broadcast::Receiver<CapabilityUpdate> {
        self.capabilities_tx.subscribe()
    }
}

impl Default for EvolutionChannelBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared channel bridge type.
pub type SharedEvolutionChannelBridge = Arc<EvolutionChannelBridge>;

/// Create a new shared channel bridge.
#[must_use]
pub fn shared_channel_bridge() -> SharedEvolutionChannelBridge {
    Arc::new(EvolutionChannelBridge::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = EvolutionChannelBridge::new();
        assert!(hpc_channels::exists(hpc_channels::channels::EVOLUTION_EVENT));
        assert!(hpc_channels::exists(hpc_channels::channels::EVOLUTION_CAPABILITIES));
        let _ = bridge;
    }

    #[tokio::test]
    async fn test_generation_event_publishing() {
        let bridge = EvolutionChannelBridge::new();
        let mut rx = bridge.subscribe_events();

        let stats = EvolutionStats {
            generation: 42,
            population_size: 100,
            best_fitness: 0.95,
            average_fitness: 0.72,
            mutations_per_second: 1000.0,
            diversity_index: 0.85,
        };

        bridge.publish_generation_complete(42, &stats);

        let event = rx.recv().await.expect("Should receive event");
        match event {
            EvolutionEvent::GenerationComplete {
                generation,
                best_fitness,
                ..
            } => {
                assert_eq!(generation, 42);
                assert!((best_fitness - 0.95).abs() < f64::EPSILON);
            }
            _ => panic!("Expected GenerationComplete event"),
        }
    }

    #[tokio::test]
    async fn test_capability_update_publishing() {
        let bridge = EvolutionChannelBridge::new();
        let mut rx = bridge.subscribe_capabilities();

        let update = CapabilityUpdate {
            agent_id: "agent-42".to_string(),
            capability: "matrix_multiply".to_string(),
            level: 75,
            delta: 5,
        };

        bridge.publish_capability_update(update);

        let received = rx.recv().await.expect("Should receive update");
        assert_eq!(received.agent_id, "agent-42");
        assert_eq!(received.capability, "matrix_multiply");
        assert_eq!(received.level, 75);
    }
}
