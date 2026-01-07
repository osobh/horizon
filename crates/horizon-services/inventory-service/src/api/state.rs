//! Application state for the inventory service.

use hpc_channels::{broadcast, channels, InventoryMessage};
use sqlx::PgPool;
use tokio::sync::broadcast::Sender as BroadcastSender;
use tracing::warn;

/// Application state shared across all handlers.
#[derive(Clone)]
pub struct AppState {
    /// Database connection pool.
    pub pool: PgPool,
    /// Broadcast channel for asset lifecycle events.
    pub asset_events: BroadcastSender<InventoryMessage>,
    /// Broadcast channel for discovery events.
    pub discovery_events: BroadcastSender<InventoryMessage>,
}

impl AppState {
    /// Create a new application state.
    pub fn new(pool: PgPool) -> Self {
        // Create broadcast channels for inventory events
        let asset_events = broadcast::<InventoryMessage>(channels::INVENTORY_ASSETS, 256);
        let discovery_events = broadcast::<InventoryMessage>(channels::INVENTORY_DISCOVERY, 64);

        Self {
            pool,
            asset_events,
            discovery_events,
        }
    }

    /// Publish an asset lifecycle event.
    pub fn publish_asset_event(&self, event: InventoryMessage) {
        if let Err(e) = self.asset_events.send(event) {
            warn!(error = ?e, "No subscribers for asset event");
        }
    }

    /// Publish a discovery event.
    pub fn publish_discovery_event(&self, event: InventoryMessage) {
        if let Err(e) = self.discovery_events.send(event) {
            warn!(error = ?e, "No subscribers for discovery event");
        }
    }
}
