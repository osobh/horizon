//! HPC Channels integration for runtime container lifecycle events.
//!
//! This module bridges container runtime events to the hpc-channels message bus,
//! enabling real-time monitoring of container lifecycle across the Stratoswarm cluster.
//!
//! # Channels Used
//!
//! - `hpc.runtime.container.start` - Container start events
//! - `hpc.runtime.container.stop` - Container stop events
//!
//! # Example
//!
//! ```rust,ignore
//! use stratoswarm_runtime::hpc_bridge::RuntimeChannelBridge;
//!
//! let bridge = RuntimeChannelBridge::new();
//!
//! // Publish container started event
//! bridge.publish_container_started("container-123", "Running");
//!
//! // Subscribe to container events
//! let mut rx = bridge.subscribe_start_events();
//! while let Ok(event) = rx.recv().await {
//!     println!("Container event: {:?}", event);
//! }
//! ```

use std::sync::Arc;
use tokio::sync::broadcast;
use tracing;

/// Container lifecycle event published when a container starts.
#[derive(Clone, Debug)]
pub struct ContainerStartEvent {
    /// Container ID.
    pub container_id: String,
    /// New state after starting.
    pub state: String,
    /// Timestamp (epoch ms).
    pub timestamp_ms: u64,
}

/// Container lifecycle event published when a container stops.
#[derive(Clone, Debug)]
pub struct ContainerStopEvent {
    /// Container ID.
    pub container_id: String,
    /// Final state after stopping.
    pub state: String,
    /// Timestamp (epoch ms).
    pub timestamp_ms: u64,
}

/// Container lifecycle event for all state changes.
#[derive(Clone, Debug)]
pub enum RuntimeEvent {
    /// Container was created.
    ContainerCreated {
        /// Container ID.
        container_id: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Container was started.
    ContainerStarted {
        /// Container ID.
        container_id: String,
        /// State after starting.
        state: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Container was stopped.
    ContainerStopped {
        /// Container ID.
        container_id: String,
        /// State after stopping.
        state: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Container was removed.
    ContainerRemoved {
        /// Container ID.
        container_id: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
}

/// Bridge between container runtime events and hpc-channels.
pub struct RuntimeChannelBridge {
    /// Broadcast sender for container start events.
    start_tx: broadcast::Sender<ContainerStartEvent>,
    /// Broadcast sender for container stop events.
    stop_tx: broadcast::Sender<ContainerStopEvent>,
}

impl RuntimeChannelBridge {
    /// Create a new runtime channel bridge.
    ///
    /// Registers channels with the hpc-channels global registry.
    pub fn new() -> Self {
        let start_tx = hpc_channels::broadcast::<ContainerStartEvent>(
            hpc_channels::channels::RUNTIME_CONTAINER_START,
            256,
        );
        let stop_tx = hpc_channels::broadcast::<ContainerStopEvent>(
            hpc_channels::channels::RUNTIME_CONTAINER_STOP,
            256,
        );

        Self { start_tx, stop_tx }
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or_else(|e| {
                tracing::error!(error = ?e, "System clock is before UNIX epoch - using 0 timestamp");
                0
            })
    }

    /// Publish a container created event.
    pub fn publish_container_created(&self, container_id: &str) {
        // Use start channel for created events too
        if let Err(e) = self.start_tx.send(ContainerStartEvent {
            container_id: container_id.to_string(),
            state: "Created".to_string(),
            timestamp_ms: Self::now_ms(),
        }) {
            tracing::warn!(
                container_id = container_id,
                error = ?e,
                "Failed to publish container created event - no subscribers"
            );
        }
    }

    /// Publish a container started event.
    pub fn publish_container_started(&self, container_id: &str, state: &str) {
        if let Err(e) = self.start_tx.send(ContainerStartEvent {
            container_id: container_id.to_string(),
            state: state.to_string(),
            timestamp_ms: Self::now_ms(),
        }) {
            tracing::warn!(
                container_id = container_id,
                state = state,
                error = ?e,
                "Failed to publish container started event - no subscribers"
            );
        }
    }

    /// Publish a container stopped event.
    pub fn publish_container_stopped(&self, container_id: &str, state: &str) {
        if let Err(e) = self.stop_tx.send(ContainerStopEvent {
            container_id: container_id.to_string(),
            state: state.to_string(),
            timestamp_ms: Self::now_ms(),
        }) {
            tracing::warn!(
                container_id = container_id,
                state = state,
                error = ?e,
                "Failed to publish container stopped event - no subscribers"
            );
        }
    }

    /// Publish a container removed event.
    pub fn publish_container_removed(&self, container_id: &str) {
        if let Err(e) = self.stop_tx.send(ContainerStopEvent {
            container_id: container_id.to_string(),
            state: "Removed".to_string(),
            timestamp_ms: Self::now_ms(),
        }) {
            tracing::warn!(
                container_id = container_id,
                error = ?e,
                "Failed to publish container removed event - no subscribers"
            );
        }
    }

    /// Subscribe to container start events.
    pub fn subscribe_start_events(&self) -> broadcast::Receiver<ContainerStartEvent> {
        self.start_tx.subscribe()
    }

    /// Subscribe to container stop events.
    pub fn subscribe_stop_events(&self) -> broadcast::Receiver<ContainerStopEvent> {
        self.stop_tx.subscribe()
    }
}

impl Default for RuntimeChannelBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared channel bridge type.
pub type SharedRuntimeChannelBridge = Arc<RuntimeChannelBridge>;

/// Create a new shared channel bridge.
#[must_use]
pub fn shared_channel_bridge() -> SharedRuntimeChannelBridge {
    Arc::new(RuntimeChannelBridge::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = RuntimeChannelBridge::new();
        assert!(hpc_channels::exists(
            hpc_channels::channels::RUNTIME_CONTAINER_START
        ));
        assert!(hpc_channels::exists(
            hpc_channels::channels::RUNTIME_CONTAINER_STOP
        ));
        let _ = bridge;
    }

    #[tokio::test]
    async fn test_container_start_event_publishing() {
        let bridge = RuntimeChannelBridge::new();
        let mut rx = bridge.subscribe_start_events();

        bridge.publish_container_started("test-container-123", "Running");

        let event = rx.recv().await.expect("Should receive event");
        assert_eq!(event.container_id, "test-container-123");
        assert_eq!(event.state, "Running");
        assert!(event.timestamp_ms > 0);
    }

    #[tokio::test]
    async fn test_container_stop_event_publishing() {
        let bridge = RuntimeChannelBridge::new();
        let mut rx = bridge.subscribe_stop_events();

        bridge.publish_container_stopped("test-container-456", "Stopped");

        let event = rx.recv().await.expect("Should receive event");
        assert_eq!(event.container_id, "test-container-456");
        assert_eq!(event.state, "Stopped");
        assert!(event.timestamp_ms > 0);
    }

    #[tokio::test]
    async fn test_container_removed_event() {
        let bridge = RuntimeChannelBridge::new();
        let mut rx = bridge.subscribe_stop_events();

        bridge.publish_container_removed("test-container-789");

        let event = rx.recv().await.expect("Should receive event");
        assert_eq!(event.container_id, "test-container-789");
        assert_eq!(event.state, "Removed");
    }
}
