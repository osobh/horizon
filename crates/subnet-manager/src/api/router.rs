//! API router configuration
//!
//! Defines all API routes and middleware.

use super::handlers;
use super::state::AppState;
use axum::{
    routing::{delete, get, post, put},
    Router,
};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;


/// Create the API router with all routes
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        // Subnets
        .route("/api/v1/subnets", get(handlers::list_subnets))
        .route("/api/v1/subnets", post(handlers::create_subnet))
        .route("/api/v1/subnets/:id", get(handlers::get_subnet))
        .route("/api/v1/subnets/:id", put(handlers::update_subnet))
        .route("/api/v1/subnets/:id", delete(handlers::delete_subnet))
        .route("/api/v1/subnets/:id/stats", get(handlers::get_subnet_stats))
        .route("/api/v1/subnets/:id/nodes", get(handlers::list_subnet_nodes))
        .route("/api/v1/subnets/:id/nodes", post(handlers::assign_node))
        .route(
            "/api/v1/subnets/:subnet_id/nodes/:node_id",
            delete(handlers::unassign_node),
        )
        // Policies
        .route("/api/v1/policies", get(handlers::list_policies))
        .route("/api/v1/policies/:id", get(handlers::get_policy))
        .route("/api/v1/policies/evaluate", post(handlers::evaluate_policy))
        // Migrations
        .route("/api/v1/migrations", get(handlers::list_migrations))
        .route("/api/v1/migrations", post(handlers::start_migration))
        .route("/api/v1/migrations/:id", get(handlers::get_migration))
        .route(
            "/api/v1/migrations/:id/cancel",
            post(handlers::cancel_migration),
        )
        .route("/api/v1/migrations/stats", get(handlers::get_migration_stats))
        // Routes
        .route("/api/v1/routes", get(handlers::list_routes))
        .route("/api/v1/routes", post(handlers::create_route))
        .route(
            "/api/v1/routes/:source_id/:dest_id",
            delete(handlers::delete_route),
        )
        // Templates
        .route("/api/v1/templates", get(handlers::list_templates))
        // Stats
        .route("/api/v1/stats", get(handlers::get_manager_stats))
        // Health check
        .route("/health", get(health_check))
        .route("/ready", get(readiness_check))
        // Add state
        .with_state(state)
        // Add middleware
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
}

/// Health check endpoint
async fn health_check() -> &'static str {
    "OK"
}

/// Readiness check endpoint
async fn readiness_check() -> &'static str {
    "READY"
}

/// API server configuration
#[derive(Debug, Clone)]
pub struct ApiServerConfig {
    /// Host to bind to
    pub host: String,
    /// Port to bind to
    pub port: u16,
}

impl Default for ApiServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
        }
    }
}

impl ApiServerConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        Self {
            host: std::env::var("API_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: std::env::var("API_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
        }
    }

    /// Get the bind address
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_check() {
        let state = Arc::new(AppState::new());
        let app = create_router(state);

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_readiness_check() {
        let state = Arc::new(AppState::new());
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_subnets_empty() {
        let state = Arc::new(AppState::new());
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/subnets")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_policies_empty() {
        let state = Arc::new(AppState::new());
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/policies")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_templates() {
        let state = Arc::new(AppState::new());
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/templates")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_stats() {
        let state = Arc::new(AppState::new());
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_subnet_not_found() {
        let state = Arc::new(AppState::new());
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/subnets/00000000-0000-0000-0000-000000000000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_api_config_default() {
        let config = ApiServerConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert_eq!(config.bind_addr(), "0.0.0.0:8080");
    }
}
