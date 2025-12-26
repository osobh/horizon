//! Subnet event publisher for hpc-channels integration
//!
//! Publishes subnet events to the appropriate broadcast channels
//! for real-time topology synchronization across the platform.

use super::channels::ChannelId;
use super::messages::*;
use crate::models::{
    AssignmentPolicy, CrossSubnetRoute, RouteDirection, Subnet, SubnetAssignment, SubnetPurpose,
    SubnetStatus,
};
use crate::migration::Migration;
use async_trait::async_trait;
use chrono::Utc;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::{debug, error, info, instrument};
use uuid::Uuid;

/// Trait for event transport backends
#[async_trait]
pub trait EventTransport: Send + Sync {
    /// Publish a message to a channel
    async fn publish(&self, channel: ChannelId, message: &SubnetMessage) -> Result<(), PublishError>;

    /// Subscribe to a channel
    async fn subscribe(&self, channel: ChannelId) -> Result<EventSubscription, SubscribeError>;

    /// Check if the transport is connected
    fn is_connected(&self) -> bool;
}

/// Error type for publish operations
#[derive(Debug, thiserror::Error)]
pub enum PublishError {
    #[error("Transport not connected")]
    NotConnected,

    #[error("Serialization failed: {0}")]
    SerializationFailed(#[from] serde_json::Error),

    #[error("Transport error: {0}")]
    TransportError(String),

    #[error("Channel not found: {0}")]
    ChannelNotFound(String),
}

/// Error type for subscribe operations
#[derive(Debug, thiserror::Error)]
pub enum SubscribeError {
    #[error("Transport not connected")]
    NotConnected,

    #[error("Transport error: {0}")]
    TransportError(String),
}

/// A subscription to receive events
pub struct EventSubscription {
    pub channel: ChannelId,
    pub receiver: tokio::sync::mpsc::Receiver<SubnetMessage>,
}

/// Statistics for the event publisher
#[derive(Debug, Default, Clone)]
pub struct PublisherStats {
    pub messages_published: u64,
    pub messages_failed: u64,
    pub messages_queued: u64,
    pub subscribers: u64,
}

/// Main subnet event publisher
pub struct SubnetEventPublisher {
    transport: Option<Arc<dyn EventTransport>>,
    buffer: RwLock<VecDeque<(ChannelId, SubnetMessage)>>,
    buffer_capacity: usize,
    stats: RwLock<PublisherStats>,
    local_subscribers: RwLock<Vec<tokio::sync::mpsc::Sender<SubnetMessage>>>,
}

impl SubnetEventPublisher {
    /// Create a new publisher without a transport (local-only mode)
    pub fn new() -> Self {
        Self {
            transport: None,
            buffer: RwLock::new(VecDeque::with_capacity(1000)),
            buffer_capacity: 1000,
            stats: RwLock::new(PublisherStats::default()),
            local_subscribers: RwLock::new(Vec::new()),
        }
    }

    /// Create a publisher with a transport backend
    pub fn with_transport(transport: Arc<dyn EventTransport>) -> Self {
        Self {
            transport: Some(transport),
            buffer: RwLock::new(VecDeque::with_capacity(1000)),
            buffer_capacity: 1000,
            stats: RwLock::new(PublisherStats::default()),
            local_subscribers: RwLock::new(Vec::new()),
        }
    }

    /// Create a publisher with hpc-channels backend.
    ///
    /// This enables platform-wide event broadcast via hpc-channels.
    /// Requires the `hpc-channels` feature to be enabled.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let publisher = SubnetEventPublisher::with_hpc_channels();
    /// publisher.subnet_created(&subnet, None).await?;
    /// // Events now flow through hpc-channels broadcast
    /// ```
    #[cfg(feature = "hpc-channels")]
    pub fn with_hpc_channels() -> Self {
        let bridge = Arc::new(super::hpc_bridge::SubnetHpcBridge::new());
        Self::with_transport(bridge)
    }

    /// Set the buffer capacity for offline queuing
    pub fn with_buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self.buffer = RwLock::new(VecDeque::with_capacity(capacity));
        self
    }

    /// Subscribe to all subnet events locally
    pub fn subscribe_local(&self) -> tokio::sync::mpsc::Receiver<SubnetMessage> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        self.local_subscribers.write().push(tx);
        self.stats.write().subscribers += 1;
        rx
    }

    /// Get publisher statistics
    pub fn stats(&self) -> PublisherStats {
        self.stats.read().clone()
    }

    /// Publish a message to the appropriate channel
    #[instrument(skip(self, message), fields(channel = %message.channel()))]
    pub async fn publish(&self, message: SubnetMessage) -> Result<(), PublishError> {
        let channel = message.channel();

        // Notify local subscribers first
        self.notify_local(&message);

        // Publish via transport if available
        if let Some(transport) = &self.transport {
            if transport.is_connected() {
                match transport.publish(channel, &message).await {
                    Ok(()) => {
                        self.stats.write().messages_published += 1;
                        debug!(event = %message.description(), "Published subnet event");
                        Ok(())
                    }
                    Err(e) => {
                        self.stats.write().messages_failed += 1;
                        error!(error = %e, "Failed to publish event");
                        self.buffer_message(channel, message);
                        Err(e)
                    }
                }
            } else {
                self.buffer_message(channel, message);
                Err(PublishError::NotConnected)
            }
        } else {
            // No transport, just count as published (local-only mode)
            self.stats.write().messages_published += 1;
            debug!(event = %message.description(), "Published event locally");
            Ok(())
        }
    }

    /// Buffer a message for later delivery
    fn buffer_message(&self, channel: ChannelId, message: SubnetMessage) {
        let mut buffer = self.buffer.write();
        if buffer.len() >= self.buffer_capacity {
            buffer.pop_front(); // Drop oldest
        }
        buffer.push_back((channel, message));
        self.stats.write().messages_queued = buffer.len() as u64;
    }

    /// Notify local subscribers
    fn notify_local(&self, message: &SubnetMessage) {
        let subscribers = self.local_subscribers.read();
        for tx in subscribers.iter() {
            let _ = tx.try_send(message.clone());
        }
    }

    /// Flush buffered messages (call when transport reconnects)
    pub async fn flush_buffer(&self) -> Result<usize, PublishError> {
        let transport = self
            .transport
            .as_ref()
            .ok_or(PublishError::NotConnected)?;

        if !transport.is_connected() {
            return Err(PublishError::NotConnected);
        }

        let messages: Vec<_> = {
            let mut buffer = self.buffer.write();
            buffer.drain(..).collect()
        };

        let count = messages.len();
        for (channel, message) in messages {
            transport.publish(channel, &message).await?;
            self.stats.write().messages_published += 1;
        }

        self.stats.write().messages_queued = 0;
        info!(count, "Flushed buffered messages");
        Ok(count)
    }

    // ========================================================================
    // Convenience methods for publishing specific events
    // ========================================================================

    /// Publish a subnet created event
    pub async fn subnet_created(
        &self,
        subnet: &Subnet,
        created_by: Option<Uuid>,
    ) -> Result<(), PublishError> {
        self.publish(SubnetMessage::SubnetCreated(SubnetCreatedEvent {
            subnet_id: subnet.id,
            name: subnet.name.clone(),
            cidr: subnet.cidr.to_string(),
            purpose: subnet.purpose,
            wg_interface: subnet.wg_interface.clone(),
            wg_listen_port: subnet.wg_listen_port,
            created_at: subnet.created_at,
            created_by,
        }))
        .await
    }

    /// Publish a subnet status changed event
    pub async fn subnet_status_changed(
        &self,
        subnet_id: Uuid,
        old_status: SubnetStatus,
        new_status: SubnetStatus,
        reason: Option<String>,
    ) -> Result<(), PublishError> {
        self.publish(SubnetMessage::SubnetStatusChanged(SubnetStatusChangedEvent {
            subnet_id,
            old_status,
            new_status,
            reason,
            changed_at: Utc::now(),
        }))
        .await
    }

    /// Publish a subnet deleted event
    pub async fn subnet_deleted(
        &self,
        subnet: &Subnet,
        deleted_by: Option<Uuid>,
    ) -> Result<(), PublishError> {
        self.publish(SubnetMessage::SubnetDeleted(SubnetDeletedEvent {
            subnet_id: subnet.id,
            name: subnet.name.clone(),
            cidr: subnet.cidr.to_string(),
            deleted_at: Utc::now(),
            deleted_by,
        }))
        .await
    }

    /// Publish a node assigned event
    pub async fn node_assigned(&self, assignment: &SubnetAssignment) -> Result<(), PublishError> {
        self.publish(SubnetMessage::NodeAssigned(NodeAssignedEvent {
            node_id: assignment.node_id,
            subnet_id: assignment.subnet_id,
            assigned_ip: assignment.assigned_ip,
            wg_public_key: assignment.wg_public_key.clone(),
            policy_id: assignment.policy_id,
            assignment_method: assignment.assignment_method.clone(),
            assigned_at: assignment.assigned_at,
        }))
        .await
    }

    /// Publish a node unassigned event
    pub async fn node_unassigned(
        &self,
        node_id: Uuid,
        subnet_id: Uuid,
        released_ip: std::net::Ipv4Addr,
        reason: &str,
    ) -> Result<(), PublishError> {
        self.publish(SubnetMessage::NodeUnassigned(NodeUnassignedEvent {
            node_id,
            subnet_id,
            released_ip,
            reason: reason.to_string(),
            unassigned_at: Utc::now(),
        }))
        .await
    }

    /// Publish a route created event
    pub async fn route_created(&self, route: &CrossSubnetRoute) -> Result<(), PublishError> {
        self.publish(SubnetMessage::RouteCreated(RouteCreatedEvent {
            route_id: route.id,
            source_subnet_id: route.source_subnet_id,
            destination_subnet_id: route.destination_subnet_id,
            direction: route.direction,
            created_at: route.created_at,
        }))
        .await
    }

    /// Publish a route deleted event
    pub async fn route_deleted(
        &self,
        route_id: Uuid,
        source_subnet_id: Uuid,
        destination_subnet_id: Uuid,
    ) -> Result<(), PublishError> {
        self.publish(SubnetMessage::RouteDeleted(RouteDeletedEvent {
            route_id,
            source_subnet_id,
            destination_subnet_id,
            deleted_at: Utc::now(),
        }))
        .await
    }

    /// Publish a policy created event
    pub async fn policy_created(&self, policy: &AssignmentPolicy) -> Result<(), PublishError> {
        self.publish(SubnetMessage::PolicyCreated(PolicyCreatedEvent {
            policy_id: policy.id,
            name: policy.name.clone(),
            priority: policy.priority,
            target_subnet_id: policy.target_subnet_id,
            rule_count: policy.rules.len(),
            created_at: policy.created_at,
        }))
        .await
    }

    /// Publish a migration started event
    pub async fn migration_started(&self, migration: &Migration) -> Result<(), PublishError> {
        let target_ip = migration.target_ip.unwrap_or(std::net::Ipv4Addr::new(0, 0, 0, 0));
        self.publish(SubnetMessage::MigrationStarted(MigrationStartedEvent {
            migration_id: migration.id,
            node_id: migration.node_id,
            source_subnet_id: migration.source_subnet_id,
            target_subnet_id: migration.target_subnet_id,
            source_ip: migration.source_ip,
            target_ip,
            started_at: migration.started_at.unwrap_or_else(Utc::now),
        }))
        .await
    }

    /// Publish a migration completed event
    pub async fn migration_completed(
        &self,
        migration: &Migration,
        duration_ms: u64,
    ) -> Result<(), PublishError> {
        let final_ip = migration.target_ip.unwrap_or(std::net::Ipv4Addr::new(0, 0, 0, 0));
        self.publish(SubnetMessage::MigrationCompleted(MigrationCompletedEvent {
            migration_id: migration.id,
            node_id: migration.node_id,
            source_subnet_id: migration.source_subnet_id,
            target_subnet_id: migration.target_subnet_id,
            final_ip,
            duration_ms,
            completed_at: Utc::now(),
        }))
        .await
    }

    /// Publish a migration failed event
    pub async fn migration_failed(
        &self,
        migration: &Migration,
        error: &str,
        rollback_successful: bool,
    ) -> Result<(), PublishError> {
        self.publish(SubnetMessage::MigrationFailed(MigrationFailedEvent {
            migration_id: migration.id,
            node_id: migration.node_id,
            source_subnet_id: migration.source_subnet_id,
            target_subnet_id: migration.target_subnet_id,
            error: error.to_string(),
            rollback_successful,
            failed_at: Utc::now(),
        }))
        .await
    }

    /// Publish a full topology snapshot
    pub async fn topology_snapshot(
        &self,
        subnets: Vec<SubnetInfo>,
        routes: Vec<RouteInfo>,
        version: u64,
    ) -> Result<(), PublishError> {
        self.publish(SubnetMessage::TopologySnapshot(TopologySnapshotEvent {
            snapshot_id: Uuid::new_v4(),
            subnets,
            routes,
            generated_at: Utc::now(),
            version,
        }))
        .await
    }

    // =========================================================================
    // WireGuard Events
    // =========================================================================

    /// Publish a peer config updated event
    pub async fn peer_config_updated(
        &self,
        subnet_id: Uuid,
        node_id: Uuid,
        public_key: &str,
        endpoint: Option<std::net::SocketAddr>,
        allowed_ips: Vec<String>,
    ) -> Result<(), PublishError> {
        use super::messages::PeerConfigUpdatedEvent;
        self.publish(SubnetMessage::PeerConfigUpdated(PeerConfigUpdatedEvent {
            subnet_id,
            node_id,
            public_key: public_key.to_string(),
            endpoint: endpoint.map(|e| e.to_string()),
            allowed_ips,
            updated_at: Utc::now(),
        }))
        .await
    }

    /// Publish an interface created event
    pub async fn interface_created(
        &self,
        subnet_id: Uuid,
        interface_name: &str,
        listen_port: u16,
        public_key: &str,
    ) -> Result<(), PublishError> {
        use super::messages::InterfaceCreatedEvent;
        self.publish(SubnetMessage::InterfaceCreated(InterfaceCreatedEvent {
            subnet_id,
            interface_name: interface_name.to_string(),
            listen_port,
            public_key: public_key.to_string(),
            created_at: Utc::now(),
        }))
        .await
    }

    /// Publish an interface deleted event
    pub async fn interface_deleted(
        &self,
        subnet_id: Uuid,
        interface_name: &str,
    ) -> Result<(), PublishError> {
        use super::messages::InterfaceDeletedEvent;
        self.publish(SubnetMessage::InterfaceDeleted(InterfaceDeletedEvent {
            subnet_id,
            interface_name: interface_name.to_string(),
            deleted_at: Utc::now(),
        }))
        .await
    }

    /// Publish a key rotated event
    pub async fn key_rotated(
        &self,
        subnet_id: Uuid,
        old_public_key: &str,
        new_public_key: &str,
    ) -> Result<(), PublishError> {
        use super::messages::KeyRotatedEvent;
        self.publish(SubnetMessage::KeyRotated(KeyRotatedEvent {
            subnet_id,
            old_public_key: old_public_key.to_string(),
            new_public_key: new_public_key.to_string(),
            rotated_at: Utc::now(),
        }))
        .await
    }
}

impl Default for SubnetEventPublisher {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// In-memory transport for testing
// ============================================================================

/// Simple in-memory transport for testing
pub struct InMemoryTransport {
    connected: std::sync::atomic::AtomicBool,
    messages: RwLock<Vec<(ChannelId, SubnetMessage)>>,
}

impl InMemoryTransport {
    pub fn new() -> Self {
        Self {
            connected: std::sync::atomic::AtomicBool::new(true),
            messages: RwLock::new(Vec::new()),
        }
    }

    pub fn disconnect(&self) {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn connect(&self) {
        self.connected
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn messages(&self) -> Vec<(ChannelId, SubnetMessage)> {
        self.messages.read().clone()
    }

    pub fn clear(&self) {
        self.messages.write().clear();
    }
}

impl Default for InMemoryTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EventTransport for InMemoryTransport {
    async fn publish(&self, channel: ChannelId, message: &SubnetMessage) -> Result<(), PublishError> {
        if !self.is_connected() {
            return Err(PublishError::NotConnected);
        }
        self.messages.write().push((channel, message.clone()));
        Ok(())
    }

    async fn subscribe(&self, channel: ChannelId) -> Result<EventSubscription, SubscribeError> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        Ok(EventSubscription { channel, receiver: rx })
    }

    fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SubnetPurpose;
    use ipnet::Ipv4Net;
    use std::str::FromStr;

    #[tokio::test]
    async fn test_local_only_publisher() {
        let publisher = SubnetEventPublisher::new();

        let mut rx = publisher.subscribe_local();

        let subnet = Subnet::new(
            "test-subnet",
            Ipv4Net::from_str("10.100.0.0/20").unwrap(),
            SubnetPurpose::Tenant,
            51820,
        );

        publisher.subnet_created(&subnet, None).await.unwrap();

        let msg = rx.try_recv().unwrap();
        match msg {
            SubnetMessage::SubnetCreated(e) => {
                assert_eq!(e.name, "test-subnet");
            }
            _ => panic!("Expected SubnetCreated"),
        }

        assert_eq!(publisher.stats().messages_published, 1);
    }

    #[tokio::test]
    async fn test_transport_publisher() {
        let transport = Arc::new(InMemoryTransport::new());
        let publisher = SubnetEventPublisher::with_transport(transport.clone());

        let subnet = Subnet::new(
            "test-subnet",
            Ipv4Net::from_str("10.100.0.0/20").unwrap(),
            SubnetPurpose::Tenant,
            51820,
        );

        publisher.subnet_created(&subnet, None).await.unwrap();

        let messages = transport.messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].0, super::super::channels::SUBNET_LIFECYCLE);
    }

    #[tokio::test]
    async fn test_message_buffering() {
        let transport = Arc::new(InMemoryTransport::new());
        transport.disconnect();

        let publisher = SubnetEventPublisher::with_transport(transport.clone());

        let subnet = Subnet::new(
            "test-subnet",
            Ipv4Net::from_str("10.100.0.0/20").unwrap(),
            SubnetPurpose::Tenant,
            51820,
        );

        // Should fail but buffer
        let result = publisher.subnet_created(&subnet, None).await;
        assert!(result.is_err());
        assert_eq!(publisher.stats().messages_queued, 1);

        // Reconnect and flush
        transport.connect();
        let flushed = publisher.flush_buffer().await.unwrap();
        assert_eq!(flushed, 1);
        assert_eq!(publisher.stats().messages_queued, 0);

        let messages = transport.messages();
        assert_eq!(messages.len(), 1);
    }
}
