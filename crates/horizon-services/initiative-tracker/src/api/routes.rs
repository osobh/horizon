use super::handlers::*;
use axum::{routing::get, Router};
use std::sync::Arc;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/v1/initiatives", get(list_initiatives))
        .route("/api/v1/portfolio/summary", get(get_portfolio))
        .with_state(state)
}
