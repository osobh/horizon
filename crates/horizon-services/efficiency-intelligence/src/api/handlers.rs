use crate::db::EfficiencyRepository;
use crate::error::Result;
use crate::models::*;
use axum::{extract::State, Json};
use hpc_channels::{broadcast, channels, EfficiencyMessage};
use std::sync::Arc;
use tokio::sync::broadcast::Sender as BroadcastSender;

#[derive(Clone)]
pub struct AppState {
    pub repository: EfficiencyRepository,
    /// Channel for efficiency detection events.
    pub detection_events: BroadcastSender<EfficiencyMessage>,
    /// Channel for efficiency summary events.
    pub summary_events: BroadcastSender<EfficiencyMessage>,
}

impl AppState {
    pub fn new(repository: EfficiencyRepository) -> Self {
        let detection_events = broadcast::<EfficiencyMessage>(channels::EFFICIENCY_DETECTIONS, 256);
        let summary_events = broadcast::<EfficiencyMessage>(channels::EFFICIENCY_SUMMARY, 64);

        Self {
            repository,
            detection_events,
            summary_events,
        }
    }

    /// Publish a detection event (non-blocking).
    pub fn publish_detection_event(&self, event: EfficiencyMessage) {
        let _ = self.detection_events.send(event);
    }

    /// Publish a summary event (non-blocking).
    pub fn publish_summary_event(&self, event: EfficiencyMessage) {
        let _ = self.summary_events.send(event);
    }
}

pub async fn health() -> &'static str {
    "OK"
}

pub async fn list_detections(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<WasteDetection>>> {
    let detections = state.repository.list_detections().await?;
    Ok(Json(detections))
}

pub async fn get_summary(State(state): State<Arc<AppState>>) -> Result<Json<SavingsSummary>> {
    let summary = state.repository.get_savings_summary().await?;
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
