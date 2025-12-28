//! HTTP API router for cluster coordinator.
//!
//! This module defines the API routes for the cluster coordinator service,
//! including the install script endpoint for curl-based node bootstrapping.

use axum::{routing::get, Router};
use std::sync::Arc;
use tower_http::trace::TraceLayer;

use super::{handlers, state::AppState};

/// Create the API router with all endpoints.
///
/// # Routes
///
/// ## Health & Readiness
/// - `GET /health` - Health check
/// - `GET /ready` - Readiness check
///
/// ## Install & Join
/// - `GET /api/v1/install` - Generate install script (requires `token` query param)
/// - `GET /api/v1/join/validate` - Validate a join token
///
/// # Example
///
/// ```rust,ignore
/// use stratoswarm_cluster_mesh::api::{create_router, AppState, AppStateConfig};
/// use std::sync::Arc;
///
/// let config = AppStateConfig {
///     cluster_host: "cluster.example.com".to_string(),
///     cluster_port: 7946,
///     ..Default::default()
/// };
///
/// let state = Arc::new(AppState::new(config));
/// let app = create_router(state);
///
/// // Run with axum::serve()
/// ```
pub fn create_router(state: Arc<AppState>) -> Router {
    // API v1 routes
    let api_v1 = Router::new()
        .route("/install", get(handlers::install_handler))
        .route("/join/validate", get(handlers::validate_token_handler))
        .with_state(state.clone());

    // Build the complete router
    Router::new()
        .route("/health", get(handlers::health_handler))
        .route("/ready", get(handlers::ready_handler).with_state(state))
        .nest("/api/v1", api_v1)
        .layer(TraceLayer::new_for_http())
}

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind to (e.g., "0.0.0.0:7946")
    pub bind_addr: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:7946".to_string(),
        }
    }
}

/// Start the HTTP server.
///
/// # Arguments
///
/// * `state` - Shared application state
/// * `config` - Server configuration
///
/// # Example
///
/// ```rust,ignore
/// use stratoswarm_cluster_mesh::api::{start_server, AppState, AppStateConfig, ServerConfig};
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() {
///     let state = Arc::new(AppState::new(AppStateConfig::default()));
///     let config = ServerConfig { bind_addr: "0.0.0.0:8080".to_string() };
///
///     start_server(state, config).await.unwrap();
/// }
/// ```
pub async fn start_server(
    state: Arc<AppState>,
    config: ServerConfig,
) -> Result<(), std::io::Error> {
    let app = create_router(state);

    tracing::info!("Starting cluster coordinator API on {}", config.bind_addr);

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    fn create_test_state() -> Arc<AppState> {
        use super::super::state::AppStateConfig;

        Arc::new(AppState::new(AppStateConfig {
            cluster_host: "test.local".to_string(),
            cluster_port: 7946,
            ..Default::default()
        }))
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = create_test_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_ready_endpoint() {
        let state = create_test_state();
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
    async fn test_install_endpoint_requires_token() {
        let state = create_test_state();
        let app = create_router(state);

        // Request without token should fail
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/install")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Should get a 400 Bad Request because token is required
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_install_endpoint_with_valid_token() {
        let state = create_test_state();

        // Generate a valid token
        let (token, _) = state.generate_token(chrono::Duration::hours(1), 0, vec![]);

        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/v1/install?token={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // Check content type
        let content_type = response.headers().get("content-type").unwrap();
        assert!(content_type.to_str().unwrap().contains("shellscript"));
    }

    #[tokio::test]
    async fn test_validate_token_endpoint() {
        let state = create_test_state();

        // Generate a valid token
        let (token, _) = state.generate_token(chrono::Duration::hours(1), 0, vec![]);

        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/v1/join/validate?token={}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
