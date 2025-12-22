use axum::{
    routing::{delete, get, post},
    Router,
};
use std::sync::Arc;
use tower_http::trace::TraceLayer;

use crate::api::handlers;
use crate::api::state::AppState;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(handlers::health_check))
        .route("/ready", get(handlers::readiness_check))
        .route("/api/v1/billing/ingest", post(handlers::ingest_billing_data))
        .route("/api/v1/billing/records", post(handlers::create_billing_record))
        .route("/api/v1/billing/records", get(handlers::query_billing_records))
        .route("/api/v1/billing/records/:id", get(handlers::get_billing_record))
        .route("/api/v1/billing/records/:id", delete(handlers::delete_billing_record))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::BillingRepository;
    use crate::normalize::NormalizedBillingSchema;

    #[tokio::test]
    async fn test_router_creation() {
        let pool = sqlx::PgPool::connect_lazy("postgres://localhost/test").unwrap();
        let repository = BillingRepository::new(pool);
        let schema = NormalizedBillingSchema::new();

        let state = Arc::new(AppState { repository, schema });
        let _router = create_router(state);

        // Router created successfully
        assert!(true);
    }
}
