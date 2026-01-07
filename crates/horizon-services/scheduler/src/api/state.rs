use crate::config::PricingConfig;
use crate::scheduler::Scheduler;
use hpc_channels::{broadcast, channels, SchedulerMessage};
use std::sync::Arc;
use tokio::sync::broadcast::Sender as BroadcastSender;
use tracing::warn;

/// Application state shared across all handlers
#[derive(Clone)]
pub struct AppState {
    pub scheduler: Arc<Scheduler>,
    /// Broadcast channel for job events (submit, cancel, complete, etc.)
    pub job_events: BroadcastSender<SchedulerMessage>,
    /// Broadcast channel for tenant events
    pub tenant_events: BroadcastSender<SchedulerMessage>,
    /// Pricing configuration for cost estimates
    pub pricing: PricingConfig,
}

impl AppState {
    pub fn new(scheduler: Arc<Scheduler>) -> Self {
        Self::with_pricing(scheduler, PricingConfig::default())
    }

    pub fn with_pricing(scheduler: Arc<Scheduler>, pricing: PricingConfig) -> Self {
        // Create broadcast channels for scheduler events
        let job_events = broadcast::<SchedulerMessage>(channels::SCHEDULER_JOBS, 256);
        let tenant_events = broadcast::<SchedulerMessage>(channels::SCHEDULER_TENANTS, 64);

        Self {
            scheduler,
            job_events,
            tenant_events,
            pricing,
        }
    }

    /// Publish a job event to subscribers
    pub fn publish_job_event(&self, event: SchedulerMessage) {
        if let Err(e) = self.job_events.send(event) {
            warn!(error = ?e, "No subscribers for job event");
        }
    }

    /// Publish a tenant event to subscribers
    pub fn publish_tenant_event(&self, event: SchedulerMessage) {
        if let Err(e) = self.tenant_events.send(event) {
            warn!(error = ?e, "No subscribers for tenant event");
        }
    }
}
