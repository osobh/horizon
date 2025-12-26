use crate::scheduler::Scheduler;
use hpc_channels::{broadcast, channels, SchedulerMessage};
use std::sync::Arc;
use tokio::sync::broadcast::Sender as BroadcastSender;

/// Application state shared across all handlers
#[derive(Clone)]
pub struct AppState {
    pub scheduler: Arc<Scheduler>,
    /// Broadcast channel for job events (submit, cancel, complete, etc.)
    pub job_events: BroadcastSender<SchedulerMessage>,
    /// Broadcast channel for tenant events
    pub tenant_events: BroadcastSender<SchedulerMessage>,
}

impl AppState {
    pub fn new(scheduler: Arc<Scheduler>) -> Self {
        // Create broadcast channels for scheduler events
        let job_events = broadcast::<SchedulerMessage>(channels::SCHEDULER_JOBS, 256);
        let tenant_events = broadcast::<SchedulerMessage>(channels::SCHEDULER_TENANTS, 64);

        Self {
            scheduler,
            job_events,
            tenant_events,
        }
    }

    /// Publish a job event to subscribers
    pub fn publish_job_event(&self, event: SchedulerMessage) {
        // Ignore errors - no subscribers is fine
        let _ = self.job_events.send(event);
    }

    /// Publish a tenant event to subscribers
    pub fn publish_tenant_event(&self, event: SchedulerMessage) {
        let _ = self.tenant_events.send(event);
    }
}
