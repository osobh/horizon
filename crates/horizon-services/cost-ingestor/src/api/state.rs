use crate::db::BillingRepository;
use crate::normalize::NormalizedBillingSchema;
use hpc_channels::{broadcast, channels, CostMessage};
use std::sync::Arc;
use tokio::sync::broadcast::Sender as BroadcastSender;

#[derive(Clone)]
pub struct AppState {
    pub repository: Arc<BillingRepository>,
    pub schema: Arc<NormalizedBillingSchema>,
    /// Channel for ingestion events.
    pub ingestion_events: BroadcastSender<CostMessage>,
    /// Channel for cost alerts (failures).
    pub alert_events: BroadcastSender<CostMessage>,
}

impl AppState {
    /// Create a new AppState with broadcast channels.
    pub fn new(repository: BillingRepository, schema: NormalizedBillingSchema) -> Self {
        let ingestion_events = broadcast::<CostMessage>(channels::COST_INGESTION, 256);
        let alert_events = broadcast::<CostMessage>(channels::COST_ALERTS, 64);

        Self {
            repository: Arc::new(repository),
            schema: Arc::new(schema),
            ingestion_events,
            alert_events,
        }
    }

    /// Publish an ingestion event (non-blocking).
    pub fn publish_ingestion_event(&self, event: CostMessage) {
        let _ = self.ingestion_events.send(event);
    }

    /// Publish an alert event (non-blocking).
    pub fn publish_alert_event(&self, event: CostMessage) {
        let _ = self.alert_events.send(event);
    }
}
