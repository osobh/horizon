use crate::db::VendorRepository;
use crate::error::Result;
use crate::models::*;
use axum::{extract::State, Json};
use hpc_channels::{broadcast, channels, VendorMessage};
use std::sync::Arc;
use tokio::sync::broadcast::Sender as BroadcastSender;

#[derive(Clone)]
pub struct AppState {
    pub repository: VendorRepository,
    /// Channel for vendor lifecycle events.
    pub lifecycle_events: BroadcastSender<VendorMessage>,
    /// Channel for contract events.
    pub contract_events: BroadcastSender<VendorMessage>,
}

impl AppState {
    pub fn new(repository: VendorRepository) -> Self {
        let lifecycle_events = broadcast::<VendorMessage>(channels::VENDOR_LIFECYCLE, 256);
        let contract_events = broadcast::<VendorMessage>(channels::VENDOR_CONTRACTS, 256);

        Self {
            repository,
            lifecycle_events,
            contract_events,
        }
    }

    /// Publish a lifecycle event (non-blocking).
    pub fn publish_lifecycle_event(&self, event: VendorMessage) {
        let _ = self.lifecycle_events.send(event);
    }

    /// Publish a contract event (non-blocking).
    pub fn publish_contract_event(&self, event: VendorMessage) {
        let _ = self.contract_events.send(event);
    }
}

pub async fn health() -> &'static str {
    "OK"
}

pub async fn list_vendors(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Vendor>>> {
    let vendors = state.repository.list_vendors().await?;
    Ok(Json(vendors))
}

pub async fn get_summary(
    State(state): State<Arc<AppState>>,
) -> Result<Json<VendorSummary>> {
    let summary = state.repository.get_summary().await?;
    Ok(Json(summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health() {
        assert_eq!(health().await, "OK");
    }
}
