use axum::{
    routing::{delete, get, post},
    Router,
};
use std::sync::Arc;

use crate::scheduler::Scheduler;

use super::{handlers, state::AppState};

/// Create the main application router with all API routes
pub fn create_router(scheduler: Arc<Scheduler>) -> Router {
    let state = AppState::new(scheduler);

    Router::new()
        // Health check
        .route("/health", get(handlers::health_check))

        // Job endpoints (v1 paths)
        .route("/api/v1/jobs", post(handlers::submit_job))
        .route("/api/v1/jobs", get(handlers::list_jobs))
        .route("/api/v1/jobs/:id", get(handlers::get_job))
        .route("/api/v1/jobs/:id", delete(handlers::cancel_job))

        // User-specific job endpoints (alternate paths for frontend compatibility)
        .route("/api/users/:user_id/jobs", post(handlers::submit_user_job))
        .route("/api/users/:user_id/jobs", get(handlers::list_user_jobs))
        .route("/api/users/:user_id/activity", get(handlers::get_user_activity))

        // Job endpoints (non-versioned paths for frontend compatibility)
        .route("/api/jobs/:job_id", get(handlers::get_job))
        .route("/api/jobs/:job_id/cancel", post(handlers::cancel_job))

        // Checkpoint endpoints
        .route("/api/jobs/:job_id/checkpoint", post(handlers::create_checkpoint))
        .route("/api/jobs/:job_id/checkpoint", get(handlers::get_checkpoint))

        // Resource endpoints
        .route("/api/gpu/availability", get(handlers::get_gpu_availability))
        .route("/api/jobs/estimate", post(handlers::estimate_job_cost))

        // Queue endpoints
        .route("/api/v1/queue", get(handlers::get_queue_status))
        .route("/api/queue/status", get(handlers::get_queue_status))

        .with_state(state)
}
