//! Message bus for inter-node communication

use super::{
    config::DistributedSwarmConfig, messages::DistributedMessage, network::NetworkTransport,
};
use crate::error::EvolutionEngineResult;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Message handler trait
pub trait MessageHandler: Send + Sync {
    fn handle(&self, message: DistributedMessage) -> EvolutionEngineResult<()>;
}

/// Handles inter-node message passing
pub struct MessageBus {
    /// Node ID
    pub node_id: String,
    /// Network transport layer
    pub transport: Arc<RwLock<NetworkTransport>>,
    /// Message handlers
    pub handlers: HashMap<String, Box<dyn MessageHandler>>,
}

impl MessageBus {
    /// Create new message bus
    pub async fn new(config: DistributedSwarmConfig) -> EvolutionEngineResult<Self> {
        let transport = Arc::new(RwLock::new(
            NetworkTransport::new(
                format!(
                    "{}:{}",
                    config.network_config.listen_addr, config.network_config.port
                ),
                config.network_config.max_connections,
            )
            .await?,
        ));

        // Start the transport
        transport.read().await.start().await?;

        Ok(Self {
            node_id: config.node_id,
            transport,
            handlers: HashMap::new(),
        })
    }

    /// Register a message handler
    pub fn register_handler(&mut self, message_type: String, handler: Box<dyn MessageHandler>) {
        self.handlers.insert(message_type, handler);
    }

    /// Send a message to a specific node
    pub async fn send_message(
        &self,
        target_node: &str,
        message: DistributedMessage,
    ) -> EvolutionEngineResult<()> {
        let serialized = serde_json::to_vec(&message)?;
        let transport = self.transport.read().await;
        transport.send(target_node, serialized).await
    }

    /// Broadcast a message to all connected nodes
    pub async fn broadcast_message(
        &self,
        message: DistributedMessage,
    ) -> EvolutionEngineResult<()> {
        let serialized = serde_json::to_vec(&message)?;
        let transport = self.transport.read().await;
        transport.broadcast(serialized).await
    }

    /// Process incoming message
    pub fn process_message(&self, message: DistributedMessage) -> EvolutionEngineResult<()> {
        let message_type = message.message_type().to_string();

        if let Some(handler) = self.handlers.get(&message_type) {
            handler.handle(message)
        } else {
            // No handler registered, log and continue
            tracing::warn!("No handler registered for message type: {}", message_type);
            Ok(())
        }
    }

    /// Stop the message bus
    pub async fn stop(&self) -> EvolutionEngineResult<()> {
        let transport = self.transport.read().await;
        transport.stop().await
    }

    /// Get transport statistics
    pub async fn get_stats(&self) -> TransportStats {
        let transport = self.transport.read().await;
        TransportStats {
            connected_peers: transport.connections.len(),
            listen_address: transport.listen_addr.clone(),
        }
    }
}

/// Transport statistics
#[derive(Debug)]
pub struct TransportStats {
    pub connected_peers: usize,
    pub listen_address: String,
}

/// Default heartbeat handler
pub struct HeartbeatHandler {
    node_id: String,
}

impl HeartbeatHandler {
    pub fn new(node_id: String) -> Self {
        Self { node_id }
    }
}

impl MessageHandler for HeartbeatHandler {
    fn handle(&self, message: DistributedMessage) -> EvolutionEngineResult<()> {
        if let DistributedMessage::Heartbeat {
            node_id,
            timestamp,
            load,
        } = message
        {
            tracing::debug!(
                "Received heartbeat from {} at {} with load {}",
                node_id,
                timestamp,
                load
            );
        }
        Ok(())
    }
}

/// Particle migration handler
pub struct MigrationHandler;

impl MessageHandler for MigrationHandler {
    fn handle(&self, message: DistributedMessage) -> EvolutionEngineResult<()> {
        if let DistributedMessage::ParticleMigration {
            particles,
            source_node,
            target_node,
        } = message
        {
            tracing::info!(
                "Handling migration of {} particles from {} to {}",
                particles.len(),
                source_node,
                target_node
            );
        }
        Ok(())
    }
}

/// Global best update handler
pub struct GlobalBestHandler;

impl MessageHandler for GlobalBestHandler {
    fn handle(&self, message: DistributedMessage) -> EvolutionEngineResult<()> {
        if let DistributedMessage::GlobalBestUpdate {
            best_particle,
            fitness,
            generation,
        } = message
        {
            tracing::info!(
                "Global best updated: fitness {} at generation {}",
                fitness,
                generation
            );
        }
        Ok(())
    }
}
