//! HTTP API for cluster coordinator operations.
//!
//! This module provides the HTTP API endpoints for the cluster coordinator,
//! including the install script endpoint that enables curl-based node bootstrapping.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use stratoswarm_cluster_mesh::api::{
//!     create_router, start_server, AppState, AppStateConfig, ServerConfig
//! };
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create application state
//!     let config = AppStateConfig {
//!         cluster_host: "cluster.example.com".to_string(),
//!         cluster_port: 7946,
//!         ..Default::default()
//!     };
//!     let state = Arc::new(AppState::new(config));
//!
//!     // Generate a join token
//!     let (token, _info) = state.generate_token(
//!         chrono::Duration::hours(24),
//!         10,  // max 10 uses
//!         vec!["gpu_workloads".to_string()],
//!     );
//!     println!("Install URL: https://cluster.example.com/api/v1/install?token={}", token);
//!
//!     // Start the server
//!     let server_config = ServerConfig {
//!         bind_addr: "0.0.0.0:7946".to_string(),
//!     };
//!     start_server(state, server_config).await.unwrap();
//! }
//! ```
//!
//! # API Endpoints
//!
//! ## Health & Readiness
//!
//! - `GET /health` - Returns health status
//! - `GET /ready` - Returns readiness status
//!
//! ## Install & Join
//!
//! - `GET /api/v1/install?token=TOKEN` - Returns a shell script for node bootstrap
//! - `GET /api/v1/join/validate?token=TOKEN` - Validates a join token
//!
//! # Install Script Usage
//!
//! The install endpoint generates a dynamic shell script that can be executed via curl:
//!
//! ```bash
//! # One-liner installation
//! curl -sSL "https://cluster.example.com/api/v1/install?token=TOKEN" | bash
//!
//! # Download and review first
//! curl -sSL "https://cluster.example.com/api/v1/install?token=TOKEN" -o install.sh
//! chmod +x install.sh
//! ./install.sh
//! ```
//!
//! The script auto-detects the environment:
//! - If Docker is available, uses container deployment
//! - Otherwise, downloads native binary and installs systemd/launchd service

pub mod handlers;
pub mod router;
pub mod state;

// Re-export main types for convenience
pub use router::{create_router, start_server, ServerConfig};
pub use state::{AppState, AppStateConfig, TokenError, TokenInfo, TokenStats};
