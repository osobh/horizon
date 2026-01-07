//! Channel integration for telemetry events.

use hpc_channels::{broadcast, channels, TelemetryMessage};
use tokio::sync::broadcast::Sender as BroadcastSender;
use tracing::warn;

/// Telemetry event publisher.
#[derive(Clone)]
pub struct TelemetryChannels {
    /// Channel for metrics events.
    pub metrics: BroadcastSender<TelemetryMessage>,
    /// Channel for alerts.
    pub alerts: BroadcastSender<TelemetryMessage>,
    /// Channel for backpressure events.
    pub backpressure: BroadcastSender<TelemetryMessage>,
}

impl TelemetryChannels {
    /// Create new telemetry channels.
    pub fn new() -> Self {
        Self {
            metrics: broadcast::<TelemetryMessage>(channels::TELEMETRY_METRICS, 256),
            alerts: broadcast::<TelemetryMessage>(channels::TELEMETRY_ALERTS, 64),
            backpressure: broadcast::<TelemetryMessage>(channels::TELEMETRY_BACKPRESSURE, 32),
        }
    }

    /// Publish a metrics event.
    pub fn publish_metrics(&self, event: TelemetryMessage) {
        if let Err(e) = self.metrics.send(event) {
            warn!(error = ?e, "No subscribers for metrics event");
        }
    }

    /// Publish an alert event.
    pub fn publish_alert(&self, event: TelemetryMessage) {
        if let Err(e) = self.alerts.send(event) {
            warn!(error = ?e, "No subscribers for alert event");
        }
    }

    /// Publish a backpressure event.
    pub fn publish_backpressure(&self, event: TelemetryMessage) {
        if let Err(e) = self.backpressure.send(event) {
            warn!(error = ?e, "No subscribers for backpressure event");
        }
    }
}

impl Default for TelemetryChannels {
    fn default() -> Self {
        Self::new()
    }
}
