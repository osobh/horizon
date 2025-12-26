//! HPC Channels integration for swarmlet agent lifecycle events.
//!
//! This module bridges agent lifecycle events to the hpc-channels message bus,
//! enabling real-time monitoring of agent spawn/terminate across the cluster.
//!
//! # Channels Used
//!
//! - `hpc.agent.spawn` - Agent spawn events
//! - `hpc.agent.terminate` - Agent terminate events
//! - `hpc.agent.message` - Agent message events
//!
//! # Example
//!
//! ```rust,ignore
//! use swarmlet::hpc_bridge::AgentChannelBridge;
//!
//! let bridge = AgentChannelBridge::new();
//!
//! // Publish agent started event
//! bridge.publish_agent_started("node-123");
//!
//! // Subscribe to agent events
//! let mut rx = bridge.subscribe();
//! while let Ok(event) = rx.recv().await {
//!     println!("Agent event: {:?}", event);
//! }
//! ```

use std::sync::Arc;
use tokio::sync::broadcast;

/// Agent lifecycle events published to hpc-channels.
#[derive(Clone, Debug)]
pub enum AgentEvent {
    /// Agent has started and is ready.
    AgentStarted {
        /// Node ID of the agent.
        node_id: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Agent is healthy and operational.
    AgentHealthy {
        /// Node ID of the agent.
        node_id: String,
        /// Health status.
        status: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Agent is shutting down.
    AgentShutdown {
        /// Node ID of the agent.
        node_id: String,
        /// Reason for shutdown.
        reason: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Agent health status changed.
    HealthStatusChanged {
        /// Node ID of the agent.
        node_id: String,
        /// Previous status.
        previous_status: String,
        /// New status.
        new_status: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Workload started on this agent.
    WorkloadStarted {
        /// Node ID of the agent.
        node_id: String,
        /// Workload ID.
        workload_id: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Heartbeat sent to controller.
    HeartbeatSent {
        /// Node ID of the agent.
        node_id: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
}

/// Bridge between agent events and hpc-channels.
pub struct AgentChannelBridge {
    /// Broadcast sender for agent spawn events.
    spawn_tx: broadcast::Sender<AgentEvent>,
    /// Broadcast sender for agent terminate events.
    terminate_tx: broadcast::Sender<AgentEvent>,
}

impl AgentChannelBridge {
    /// Create a new agent channel bridge.
    ///
    /// Registers channels with the hpc-channels global registry.
    pub fn new() -> Self {
        let spawn_tx = hpc_channels::broadcast::<AgentEvent>(
            hpc_channels::channels::AGENT_SPAWN,
            256,
        );
        let terminate_tx = hpc_channels::broadcast::<AgentEvent>(
            hpc_channels::channels::AGENT_TERMINATE,
            256,
        );

        Self { spawn_tx, terminate_tx }
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Publish an agent started event.
    pub fn publish_agent_started(&self, node_id: &str) {
        let _ = self.spawn_tx.send(AgentEvent::AgentStarted {
            node_id: node_id.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish an agent healthy event.
    pub fn publish_agent_healthy(&self, node_id: &str, status: &str) {
        let _ = self.spawn_tx.send(AgentEvent::AgentHealthy {
            node_id: node_id.to_string(),
            status: status.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish an agent shutdown event.
    pub fn publish_agent_shutdown(&self, node_id: &str, reason: &str) {
        let _ = self.terminate_tx.send(AgentEvent::AgentShutdown {
            node_id: node_id.to_string(),
            reason: reason.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish a health status changed event.
    pub fn publish_health_status_changed(&self, node_id: &str, previous_status: &str, new_status: &str) {
        let _ = self.spawn_tx.send(AgentEvent::HealthStatusChanged {
            node_id: node_id.to_string(),
            previous_status: previous_status.to_string(),
            new_status: new_status.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish a workload started event.
    pub fn publish_workload_started(&self, node_id: &str, workload_id: &str) {
        let _ = self.spawn_tx.send(AgentEvent::WorkloadStarted {
            node_id: node_id.to_string(),
            workload_id: workload_id.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish a heartbeat sent event.
    pub fn publish_heartbeat_sent(&self, node_id: &str) {
        let _ = self.spawn_tx.send(AgentEvent::HeartbeatSent {
            node_id: node_id.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Subscribe to agent spawn events.
    pub fn subscribe_spawn(&self) -> broadcast::Receiver<AgentEvent> {
        self.spawn_tx.subscribe()
    }

    /// Subscribe to agent terminate events.
    pub fn subscribe_terminate(&self) -> broadcast::Receiver<AgentEvent> {
        self.terminate_tx.subscribe()
    }
}

impl Default for AgentChannelBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared channel bridge type.
pub type SharedAgentChannelBridge = Arc<AgentChannelBridge>;

/// Create a new shared channel bridge.
#[must_use]
pub fn shared_channel_bridge() -> SharedAgentChannelBridge {
    Arc::new(AgentChannelBridge::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = AgentChannelBridge::new();
        assert!(hpc_channels::exists(hpc_channels::channels::AGENT_SPAWN));
        assert!(hpc_channels::exists(hpc_channels::channels::AGENT_TERMINATE));
        let _ = bridge;
    }

    #[tokio::test]
    async fn test_agent_started_event() {
        let bridge = AgentChannelBridge::new();
        let mut rx = bridge.subscribe_spawn();

        bridge.publish_agent_started("test-node-123");

        let event = rx.recv().await.expect("Should receive event");
        match event {
            AgentEvent::AgentStarted { node_id, .. } => {
                assert_eq!(node_id, "test-node-123");
            }
            _ => panic!("Expected AgentStarted event"),
        }
    }

    #[tokio::test]
    async fn test_agent_shutdown_event() {
        let bridge = AgentChannelBridge::new();
        let mut rx = bridge.subscribe_terminate();

        bridge.publish_agent_shutdown("test-node-456", "graceful shutdown");

        let event = rx.recv().await.expect("Should receive event");
        match event {
            AgentEvent::AgentShutdown { node_id, reason, .. } => {
                assert_eq!(node_id, "test-node-456");
                assert_eq!(reason, "graceful shutdown");
            }
            _ => panic!("Expected AgentShutdown event"),
        }
    }

    #[tokio::test]
    async fn test_health_status_changed_event() {
        let bridge = AgentChannelBridge::new();
        let mut rx = bridge.subscribe_spawn();

        bridge.publish_health_status_changed("test-node-789", "Healthy", "Degraded");

        let event = rx.recv().await.expect("Should receive event");
        match event {
            AgentEvent::HealthStatusChanged {
                node_id,
                previous_status,
                new_status,
                ..
            } => {
                assert_eq!(node_id, "test-node-789");
                assert_eq!(previous_status, "Healthy");
                assert_eq!(new_status, "Degraded");
            }
            _ => panic!("Expected HealthStatusChanged event"),
        }
    }
}
