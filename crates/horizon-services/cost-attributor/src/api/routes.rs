use axum::{
    routing::{get, post, put},
    Router,
};

use super::{
    handlers::{attributions, health, pricing},
    state::AppState,
};

pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Health endpoints
        .route("/health", get(health::health_check))
        .route("/ready", get(health::readiness_check))
        // Attribution endpoints
        .route("/api/v1/attributions", post(attributions::create_attribution))
        .route("/api/v1/attributions", get(attributions::query_attributions))
        .route("/api/v1/attributions/:id", get(attributions::get_attribution))
        .route("/api/v1/attributions/calculate", post(attributions::calculate_attribution))
        // Rollup endpoints
        .route("/api/v1/attributions/rollup/user/:user_id", get(attributions::rollup_by_user))
        .route("/api/v1/attributions/rollup/team/:team_id", get(attributions::rollup_by_team))
        // Pricing endpoints
        .route("/api/v1/pricing/gpu", post(pricing::create_pricing))
        .route("/api/v1/pricing/gpu", get(pricing::query_pricing))
        .route("/api/v1/pricing/gpu/current/:type", get(pricing::get_pricing_by_type))
        .route("/api/v1/pricing/gpu/:id", put(pricing::update_pricing))
        .with_state(state)
}
