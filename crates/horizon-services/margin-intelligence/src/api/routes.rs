use super::handlers::*;
use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tower_http::trace::TraceLayer;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        // Customer profiles
        .route("/api/v1/customers", get(list_customers))
        .route("/api/v1/customers/:id/profile", get(get_customer_profile))
        .route(
            "/api/v1/customers/:id/profile/refresh",
            post(refresh_customer_profile),
        )
        // Analysis
        .route("/api/v1/analysis/margin", get(get_margin_analysis))
        .route("/api/v1/analysis/by-segment", get(get_segment_analysis))
        .route("/api/v1/analysis/top-contributors", get(get_top_contributors))
        .route("/api/v1/analysis/at-risk", get(get_at_risk_customers))
        // Simulations
        .route("/api/v1/simulations", post(create_simulation))
        .route("/api/v1/simulations/:id", get(get_simulation))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculator::MarginCalculator;
    use crate::config::DatabaseConfig;
    use crate::db::pool::create_pool;
    use crate::db::MarginRepository;
    use crate::profiler::CustomerProfiler;
    use crate::simulator::PricingSimulator;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use rust_decimal_macros::dec;
    use tower::ServiceExt;

    async fn setup_test_state() -> Arc<AppState> {
        let config = DatabaseConfig {
            url: std::env::var("TEST_DATABASE_URL")
                .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/margin_test".to_string()),
            max_connections: 5,
        };

        let pool = create_pool(&config).await.unwrap();
        let repository = MarginRepository::new(pool);
        let profiler = CustomerProfiler::new(repository.clone(), 12);
        let simulator = PricingSimulator::new(repository.clone());

        Arc::new(AppState {
            repository,
            profiler,
            simulator,
            at_risk_threshold: dec!(10.0),
        })
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = setup_test_state().await;
        let app = create_router(state);

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_ready_endpoint() {
        let state = setup_test_state().await;
        let app = create_router(state);

        let response = app
            .oneshot(Request::builder().uri("/ready").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_customers_endpoint() {
        let state = setup_test_state().await;
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/customers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
