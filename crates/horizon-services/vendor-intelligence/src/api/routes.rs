use super::handlers::*;
use axum::{routing::get, Router};
use std::sync::Arc;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/v1/vendors", get(list_vendors))
        .route("/api/v1/summary", get(get_summary))
        .with_state(state)
}
