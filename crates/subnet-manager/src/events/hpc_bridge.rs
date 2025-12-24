//! HPC-Channels bridge for subnet events
//!
//! Connects SubnetEventPublisher to hpc-channels broadcast infrastructure,
//! enabling platform-wide subnet topology synchronization.
//!
//! # Example
//!
//! ```rust,ignore
//! use subnet_manager::events::{SubnetHpcBridge, SubnetEventPublisher};
//! use std::sync::Arc;
//!
//! let bridge = Arc::new(SubnetHpcBridge::new());
//! let publisher = SubnetEventPublisher::with_transport(bridge);
//!
//! // Events now flow through hpc-channels broadcast
//! publisher.subnet_created(&subnet, None).await?;
//! ```

use super::channels::{
    ChannelId, SUBNET_ASSIGNMENTS, SUBNET_LIFECYCLE, SUBNET_POLICIES, SUBNET_ROUTES,
    SUBNET_TOPOLOGY, SUBNET_WIREGUARD,
};
use super::messages::SubnetMessage;
use super::publisher::{EventSubscription, EventTransport, PublishError, SubscribeError};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, error, instrument};

#[cfg(feature = "hpc-channels")]
use hpc_channels::channels as hpc;

/// Channel for migration events (maps to hpc.subnet.migrations)
pub const SUBNET_MIGRATIONS: ChannelId = ChannelId::from_static("subnet.migrations");

/// Bridge connecting subnet events to hpc-channels broadcast infrastructure.
///
/// Creates broadcast senders for each subnet channel and routes messages
/// based on their type.
pub struct SubnetHpcBridge {
    /// Broadcast senders for each channel
    lifecycle_tx: broadcast::Sender<SubnetMessage>,
    topology_tx: broadcast::Sender<SubnetMessage>,
    assignments_tx: broadcast::Sender<SubnetMessage>,
    routes_tx: broadcast::Sender<SubnetMessage>,
    wireguard_tx: broadcast::Sender<SubnetMessage>,
    policies_tx: broadcast::Sender<SubnetMessage>,
    migrations_tx: broadcast::Sender<SubnetMessage>,
}

impl SubnetHpcBridge {
    /// Create a new bridge with hpc-channels registration.
    ///
    /// Registers all subnet channels with hpc-channels' global broadcast registry.
    #[cfg(feature = "hpc-channels")]
    pub fn new() -> Self {
        Self {
            lifecycle_tx: hpc_channels::broadcast::<SubnetMessage>(hpc::SUBNET_LIFECYCLE, 256),
            topology_tx: hpc_channels::broadcast::<SubnetMessage>(hpc::SUBNET_TOPOLOGY, 512),
            assignments_tx: hpc_channels::broadcast::<SubnetMessage>(hpc::SUBNET_ASSIGNMENTS, 1024),
            routes_tx: hpc_channels::broadcast::<SubnetMessage>(hpc::SUBNET_ROUTES, 256),
            wireguard_tx: hpc_channels::broadcast::<SubnetMessage>(hpc::SUBNET_WIREGUARD, 512),
            policies_tx: hpc_channels::broadcast::<SubnetMessage>(hpc::SUBNET_POLICIES, 128),
            migrations_tx: hpc_channels::broadcast::<SubnetMessage>(hpc::SUBNET_MIGRATIONS, 256),
        }
    }

    /// Create a new bridge without hpc-channels (for testing without feature)
    #[cfg(not(feature = "hpc-channels"))]
    pub fn new() -> Self {
        Self {
            lifecycle_tx: broadcast::channel::<SubnetMessage>(256).0,
            topology_tx: broadcast::channel::<SubnetMessage>(512).0,
            assignments_tx: broadcast::channel::<SubnetMessage>(1024).0,
            routes_tx: broadcast::channel::<SubnetMessage>(256).0,
            wireguard_tx: broadcast::channel::<SubnetMessage>(512).0,
            policies_tx: broadcast::channel::<SubnetMessage>(128).0,
            migrations_tx: broadcast::channel::<SubnetMessage>(256).0,
        }
    }

    /// Get the appropriate sender for a message type
    fn get_sender(&self, message: &SubnetMessage) -> &broadcast::Sender<SubnetMessage> {
        match message {
            SubnetMessage::SubnetCreated(_)
            | SubnetMessage::SubnetStatusChanged(_)
            | SubnetMessage::SubnetDeleted(_)
            | SubnetMessage::SubnetUpdated(_) => &self.lifecycle_tx,

            SubnetMessage::TopologySnapshot(_) | SubnetMessage::TopologyDelta(_) => {
                &self.topology_tx
            }

            SubnetMessage::NodeAssigned(_)
            | SubnetMessage::NodeUnassigned(_)
            | SubnetMessage::BulkAssignmentCompleted(_) => &self.assignments_tx,

            SubnetMessage::RouteCreated(_)
            | SubnetMessage::RouteDeleted(_)
            | SubnetMessage::RouteStatusChanged(_) => &self.routes_tx,

            SubnetMessage::PeerConfigUpdated(_)
            | SubnetMessage::InterfaceCreated(_)
            | SubnetMessage::InterfaceDeleted(_)
            | SubnetMessage::KeyRotated(_) => &self.wireguard_tx,

            SubnetMessage::PolicyCreated(_)
            | SubnetMessage::PolicyUpdated(_)
            | SubnetMessage::PolicyDeleted(_) => &self.policies_tx,

            SubnetMessage::MigrationStarted(_)
            | SubnetMessage::MigrationCompleted(_)
            | SubnetMessage::MigrationFailed(_) => &self.migrations_tx,
        }
    }

    /// Get the sender for a specific channel ID
    fn get_sender_for_channel(&self, channel: ChannelId) -> &broadcast::Sender<SubnetMessage> {
        if channel == SUBNET_LIFECYCLE {
            &self.lifecycle_tx
        } else if channel == SUBNET_TOPOLOGY {
            &self.topology_tx
        } else if channel == SUBNET_ASSIGNMENTS {
            &self.assignments_tx
        } else if channel == SUBNET_ROUTES {
            &self.routes_tx
        } else if channel == SUBNET_WIREGUARD {
            &self.wireguard_tx
        } else if channel == SUBNET_POLICIES {
            &self.policies_tx
        } else if channel == SUBNET_MIGRATIONS {
            &self.migrations_tx
        } else {
            &self.lifecycle_tx // fallback
        }
    }

    /// Publish a message to the appropriate channel based on message type.
    ///
    /// Note: For broadcast channels, having no subscribers is a valid state.
    /// Messages will be silently dropped if there are no receivers.
    #[instrument(skip(self, message), fields(event = %message.description()))]
    pub fn publish_sync(&self, message: &SubnetMessage) -> Result<(), PublishError> {
        let tx = self.get_sender(message);
        match tx.send(message.clone()) {
            Ok(receiver_count) => {
                debug!(receivers = receiver_count, "Published event to hpc-channels");
                Ok(())
            }
            Err(_) => {
                // No receivers is valid for broadcast - message is dropped
                debug!("No receivers for event, message dropped");
                Ok(())
            }
        }
    }

    /// Subscribe to a specific channel.
    ///
    /// Returns a broadcast receiver that will receive all messages
    /// published to the channel.
    pub fn subscribe_broadcast(&self, channel: ChannelId) -> broadcast::Receiver<SubnetMessage> {
        self.get_sender_for_channel(channel).subscribe()
    }

    /// Subscribe to all channels (for topology sync).
    ///
    /// Returns receivers for all 7 subnet channels.
    pub fn subscribe_all(&self) -> Vec<broadcast::Receiver<SubnetMessage>> {
        vec![
            self.lifecycle_tx.subscribe(),
            self.topology_tx.subscribe(),
            self.assignments_tx.subscribe(),
            self.routes_tx.subscribe(),
            self.wireguard_tx.subscribe(),
            self.policies_tx.subscribe(),
            self.migrations_tx.subscribe(),
        ]
    }

    /// Get subscriber count for a channel
    pub fn receiver_count(&self, channel: ChannelId) -> usize {
        self.get_sender_for_channel(channel).receiver_count()
    }

    /// Get total subscriber count across all channels
    pub fn total_receiver_count(&self) -> usize {
        self.lifecycle_tx.receiver_count()
            + self.topology_tx.receiver_count()
            + self.assignments_tx.receiver_count()
            + self.routes_tx.receiver_count()
            + self.wireguard_tx.receiver_count()
            + self.policies_tx.receiver_count()
            + self.migrations_tx.receiver_count()
    }
}

impl Default for SubnetHpcBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Implement EventTransport trait for hpc-channels integration.
///
/// This allows SubnetHpcBridge to be used as the transport backend
/// for SubnetEventPublisher.
#[async_trait]
impl EventTransport for SubnetHpcBridge {
    async fn publish(&self, _channel: ChannelId, message: &SubnetMessage) -> Result<(), PublishError> {
        // Route based on message type, not the provided channel
        // This ensures messages go to the correct hpc-channels channel
        self.publish_sync(message)
    }

    async fn subscribe(&self, channel: ChannelId) -> Result<EventSubscription, SubscribeError> {
        // Create a broadcast receiver and spawn a task to forward to mpsc
        let mut broadcast_rx = self.subscribe_broadcast(channel);
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        // Spawn a task to forward broadcast messages to mpsc
        tokio::spawn(async move {
            loop {
                match broadcast_rx.recv().await {
                    Ok(msg) => {
                        if tx.send(msg).await.is_err() {
                            break; // Receiver dropped
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue, // Skip lagged messages
                }
            }
        });

        Ok(EventSubscription { channel, receiver: rx })
    }

    fn is_connected(&self) -> bool {
        // hpc-channels broadcast is always "connected" in-process
        true
    }
}

/// Shared bridge type alias for convenience
pub type SharedSubnetHpcBridge = Arc<SubnetHpcBridge>;

/// Create a shared bridge instance
pub fn create_shared_bridge() -> SharedSubnetHpcBridge {
    Arc::new(SubnetHpcBridge::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::messages::*;
    use crate::models::SubnetPurpose;
    use chrono::Utc;
    use std::net::Ipv4Addr;
    use uuid::Uuid;

    fn create_test_event() -> SubnetMessage {
        SubnetMessage::SubnetCreated(SubnetCreatedEvent {
            subnet_id: Uuid::new_v4(),
            name: "test-subnet".to_string(),
            cidr: "10.100.0.0/20".to_string(),
            purpose: SubnetPurpose::Tenant,
            wg_interface: "wg-test".to_string(),
            wg_listen_port: 51820,
            created_at: Utc::now(),
            created_by: None,
        })
    }

    #[test]
    fn test_bridge_creation() {
        let bridge = SubnetHpcBridge::new();
        assert_eq!(bridge.total_receiver_count(), 0);
    }

    #[test]
    fn test_message_routing() {
        let bridge = SubnetHpcBridge::new();

        // Test lifecycle routing
        let lifecycle_event = SubnetMessage::SubnetCreated(SubnetCreatedEvent {
            subnet_id: Uuid::new_v4(),
            name: "test".to_string(),
            cidr: "10.0.0.0/24".to_string(),
            purpose: SubnetPurpose::Tenant,
            wg_interface: "wg0".to_string(),
            wg_listen_port: 51820,
            created_at: Utc::now(),
            created_by: None,
        });
        let sender = bridge.get_sender(&lifecycle_event);
        assert!(std::ptr::eq(sender, &bridge.lifecycle_tx));

        // Test assignment routing
        let assignment_event = SubnetMessage::NodeAssigned(NodeAssignedEvent {
            node_id: Uuid::new_v4(),
            subnet_id: Uuid::new_v4(),
            assigned_ip: Ipv4Addr::new(10, 0, 0, 1),
            wg_public_key: "key".to_string(),
            policy_id: None,
            assignment_method: "test".to_string(),
            assigned_at: Utc::now(),
        });
        let sender = bridge.get_sender(&assignment_event);
        assert!(std::ptr::eq(sender, &bridge.assignments_tx));

        // Test migration routing
        let migration_event = SubnetMessage::MigrationStarted(MigrationStartedEvent {
            migration_id: Uuid::new_v4(),
            node_id: Uuid::new_v4(),
            source_subnet_id: Uuid::new_v4(),
            target_subnet_id: Uuid::new_v4(),
            source_ip: Ipv4Addr::new(10, 0, 0, 1),
            target_ip: Ipv4Addr::new(10, 0, 1, 1),
            started_at: Utc::now(),
        });
        let sender = bridge.get_sender(&migration_event);
        assert!(std::ptr::eq(sender, &bridge.migrations_tx));
    }

    #[tokio::test]
    async fn test_publish_subscribe() {
        let bridge = SubnetHpcBridge::new();
        let mut rx = bridge.subscribe_broadcast(SUBNET_LIFECYCLE);

        let event = create_test_event();
        bridge.publish_sync(&event).unwrap();

        let received = rx.recv().await.unwrap();
        match received {
            SubnetMessage::SubnetCreated(e) => {
                assert_eq!(e.name, "test-subnet");
            }
            _ => panic!("Expected SubnetCreated event"),
        }
    }

    #[tokio::test]
    async fn test_subscribe_all() {
        let bridge = SubnetHpcBridge::new();
        let receivers = bridge.subscribe_all();
        assert_eq!(receivers.len(), 7);
    }

    #[tokio::test]
    async fn test_event_transport_trait() {
        let bridge = SubnetHpcBridge::new();
        let transport: &dyn EventTransport = &bridge;

        assert!(transport.is_connected());

        // Must have a subscriber before publishing to broadcast channel
        let _rx = bridge.subscribe_broadcast(SUBNET_LIFECYCLE);

        let event = create_test_event();
        transport.publish(SUBNET_LIFECYCLE, &event).await.unwrap();
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let bridge = SubnetHpcBridge::new();

        let subscription = bridge.subscribe(SUBNET_LIFECYCLE).await.unwrap();
        assert_eq!(subscription.channel, SUBNET_LIFECYCLE);

        // Give the spawned task time to start
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Publish and receive
        let event = create_test_event();
        bridge.publish_sync(&event).unwrap();

        let mut receiver = subscription.receiver;
        let received = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            receiver.recv(),
        )
        .await
        .expect("timeout")
        .expect("recv");

        match received {
            SubnetMessage::SubnetCreated(e) => {
                assert_eq!(e.name, "test-subnet");
            }
            _ => panic!("Expected SubnetCreated event"),
        }
    }

    #[test]
    fn test_receiver_count() {
        let bridge = SubnetHpcBridge::new();

        assert_eq!(bridge.receiver_count(SUBNET_LIFECYCLE), 0);

        let _rx1 = bridge.subscribe_broadcast(SUBNET_LIFECYCLE);
        assert_eq!(bridge.receiver_count(SUBNET_LIFECYCLE), 1);

        let _rx2 = bridge.subscribe_broadcast(SUBNET_LIFECYCLE);
        assert_eq!(bridge.receiver_count(SUBNET_LIFECYCLE), 2);
    }

    #[test]
    fn test_shared_bridge() {
        let bridge = create_shared_bridge();
        assert_eq!(bridge.total_receiver_count(), 0);
    }
}
