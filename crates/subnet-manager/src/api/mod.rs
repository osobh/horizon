//! REST API for subnet management
//!
//! Provides HTTP endpoints for managing subnets, policies, migrations, and routes.
//!
//! # Endpoints
//!
//! ## Subnets
//! - `GET /api/v1/subnets` - List all subnets
//! - `POST /api/v1/subnets` - Create a new subnet
//! - `GET /api/v1/subnets/:id` - Get subnet details
//! - `PUT /api/v1/subnets/:id` - Update a subnet
//! - `DELETE /api/v1/subnets/:id` - Delete a subnet
//! - `GET /api/v1/subnets/:id/stats` - Get subnet statistics
//! - `GET /api/v1/subnets/:id/nodes` - List nodes in subnet
//! - `POST /api/v1/subnets/:id/nodes` - Assign a node to subnet
//! - `DELETE /api/v1/subnets/:id/nodes/:node_id` - Remove node from subnet
//!
//! ## Policies
//! - `GET /api/v1/policies` - List all policies
//! - `GET /api/v1/policies/:id` - Get policy details
//! - `POST /api/v1/policies/evaluate` - Evaluate policy for node (dry run)
//!
//! ## Migrations
//! - `GET /api/v1/migrations` - List active migrations
//! - `POST /api/v1/migrations` - Start a new migration
//! - `GET /api/v1/migrations/:id` - Get migration progress
//! - `POST /api/v1/migrations/:id/cancel` - Cancel a migration
//! - `GET /api/v1/migrations/stats` - Get migration statistics
//!
//! ## Routes
//! - `GET /api/v1/routes` - List cross-subnet routes
//! - `POST /api/v1/routes` - Create a route
//! - `DELETE /api/v1/routes/:source_id/:dest_id` - Delete a route
//!
//! ## Templates
//! - `GET /api/v1/templates` - List subnet templates
//!
//! ## Stats & Health
//! - `GET /api/v1/stats` - Get manager statistics
//! - `GET /health` - Health check
//! - `GET /ready` - Readiness check

pub mod dto;
pub mod handlers;
pub mod router;
pub mod state;

pub use dto::*;
pub use router::{create_router, ApiServerConfig};
pub use state::AppState;

use std::sync::Arc;

/// Start the API server
///
/// # Example
///
/// ```ignore
/// use subnet_manager::api::{start_server, AppState, ApiServerConfig};
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() {
///     let state = Arc::new(AppState::new());
///     let config = ApiServerConfig::default();
///     start_server(state, config).await.unwrap();
/// }
/// ```
pub async fn start_server(
    state: Arc<AppState>,
    config: ApiServerConfig,
) -> Result<(), std::io::Error> {
    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind(&config.bind_addr()).await?;

    tracing::info!("Starting API server on {}", config.bind_addr());

    axum::serve(listener, app).await?;

    Ok(())
}
