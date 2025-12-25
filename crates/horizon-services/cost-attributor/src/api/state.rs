use crate::{config::Config, db::Repository};
use hpc_channels::{broadcast, channels, CostMessage};
use std::sync::Arc;
use tokio::sync::broadcast::Sender as BroadcastSender;

#[derive(Clone)]
pub struct AppState {
    pub repository: Arc<Repository>,
    pub config: Arc<Config>,
    /// Channel for attribution events.
    pub attribution_events: BroadcastSender<CostMessage>,
}

impl AppState {
    pub fn new(repository: Repository, config: Config) -> Self {
        let attribution_events = broadcast::<CostMessage>(channels::COST_ATTRIBUTION, 256);

        Self {
            repository: Arc::new(repository),
            config: Arc::new(config),
            attribution_events,
        }
    }

    /// Publish an attribution event (non-blocking).
    pub fn publish_attribution_event(&self, event: CostMessage) {
        let _ = self.attribution_events.send(event);
    }
}
