use crate::db::InitiativeRepository;
use crate::error::Result;
use crate::models::*;
use axum::{extract::State, Json};
use hpc_channels::{broadcast, channels, InitiativeMessage};
use std::sync::Arc;
use tokio::sync::broadcast::Sender as BroadcastSender;
use tracing::warn;

#[derive(Clone)]
pub struct AppState {
    pub repository: InitiativeRepository,
    /// Channel for initiative lifecycle events.
    pub lifecycle_events: BroadcastSender<InitiativeMessage>,
    /// Channel for portfolio analysis events.
    pub portfolio_events: BroadcastSender<InitiativeMessage>,
}

impl AppState {
    pub fn new(repository: InitiativeRepository) -> Self {
        let lifecycle_events = broadcast::<InitiativeMessage>(channels::INITIATIVE_LIFECYCLE, 256);
        let portfolio_events = broadcast::<InitiativeMessage>(channels::INITIATIVE_PORTFOLIO, 64);

        Self {
            repository,
            lifecycle_events,
            portfolio_events,
        }
    }

    /// Publish a lifecycle event (non-blocking).
    pub fn publish_lifecycle_event(&self, event: InitiativeMessage) {
        if let Err(e) = self.lifecycle_events.send(event) {
            warn!(error = ?e, "No subscribers for lifecycle event");
        }
    }

    /// Publish a portfolio event (non-blocking).
    pub fn publish_portfolio_event(&self, event: InitiativeMessage) {
        if let Err(e) = self.portfolio_events.send(event) {
            warn!(error = ?e, "No subscribers for portfolio event");
        }
    }
}

pub async fn health() -> &'static str {
    "OK"
}

pub async fn list_initiatives(State(state): State<Arc<AppState>>) -> Result<Json<Vec<Initiative>>> {
    let initiatives = state.repository.list_initiatives().await?;
    Ok(Json(initiatives))
}

pub async fn get_portfolio(State(state): State<Arc<AppState>>) -> Result<Json<PortfolioSummary>> {
    let summary = state.repository.get_portfolio_summary().await?;
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
