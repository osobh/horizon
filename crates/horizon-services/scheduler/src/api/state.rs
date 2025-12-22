use crate::scheduler::Scheduler;
use std::sync::Arc;

/// Application state shared across all handlers
#[derive(Clone)]
pub struct AppState {
    pub scheduler: Arc<Scheduler>,
}

impl AppState {
    pub fn new(scheduler: Arc<Scheduler>) -> Self {
        Self { scheduler }
    }
}
