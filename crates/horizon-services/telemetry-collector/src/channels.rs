//! Channel integration for telemetry events.

use hpc_channels::{broadcast, channels, TelemetryMessage};
use tokio::sync::broadcast::Sender as BroadcastSender;

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
        let _ = self.metrics.send(event);
    }

    /// Publish an alert event.
    pub fn publish_alert(&self, event: TelemetryMessage) {
        let _ = self.alerts.send(event);
    }

    /// Publish a backpressure event.
    pub fn publish_backpressure(&self, event: TelemetryMessage) {
        let _ = self.backpressure.send(event);
    }
}

impl Default for TelemetryChannels {
    fn default() -> Self {
        Self::new()
    }
}
