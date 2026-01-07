//! Embedded HTTP server for bootstrap script serving
//!
//! Provides an axum-based HTTP server that serves bootstrap scripts
//! and receives callbacks from bootstrapped nodes.

use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

use super::scripts::ScriptGenerator;
use crate::core::inventory::{Architecture, NodeMode, OsType};

/// Bootstrap callback data sent by nodes after bootstrap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapCallback {
    /// Node identifier
    pub node_id: String,
    /// Bootstrap status
    pub status: String,
    /// Agent version installed
    pub agent_version: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// Hardware info JSON
    pub hardware: Option<serde_json::Value>,
}

/// Server state shared across handlers
#[derive(Debug)]
pub struct ServerState {
    /// Token to node ID mapping
    tokens: RwLock<HashMap<String, TokenInfo>>,
    /// Callback sender
    callback_tx: mpsc::Sender<BootstrapCallback>,
    /// Script generator
    script_gen: ScriptGenerator,
}

/// Token information
#[derive(Debug, Clone)]
struct TokenInfo {
    node_id: String,
    os: OsType,
    arch: Architecture,
    mode: NodeMode,
}

/// Bootstrap server
pub struct BootstrapServer {
    state: Arc<ServerState>,
    callback_rx: mpsc::Receiver<BootstrapCallback>,
    addr: SocketAddr,
}

impl BootstrapServer {
    /// Create a new bootstrap server
    pub fn new() -> Result<Self> {
        let (callback_tx, callback_rx) = mpsc::channel(100);

        let state = Arc::new(ServerState {
            tokens: RwLock::new(HashMap::new()),
            callback_tx,
            script_gen: ScriptGenerator::new(),
        });

        // Use a random available port
        let addr: SocketAddr = "0.0.0.0:0".parse()?;

        Ok(Self {
            state,
            callback_rx,
            addr,
        })
    }

    /// Register a node for bootstrap and return a token
    pub async fn register_node(
        &self,
        node_id: String,
        os: OsType,
        arch: Architecture,
        mode: NodeMode,
    ) -> String {
        let token = uuid::Uuid::new_v4().to_string();

        let info = TokenInfo {
            node_id,
            os,
            arch,
            mode,
        };

        self.state.tokens.write().await.insert(token.clone(), info);
        token
    }

    /// Start the server and return the bound address
    pub async fn start(self) -> Result<(BootstrapServerHandle, mpsc::Receiver<BootstrapCallback>)> {
        let app = Router::new()
            .route("/bootstrap", get(handle_bootstrap))
            .route("/callback", post(handle_callback))
            .route("/health", get(handle_health))
            .with_state(self.state.clone());

        let listener = tokio::net::TcpListener::bind(&self.addr).await?;
        let local_addr = listener.local_addr()?;

        let state = self.state.clone();
        let server_task = tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });

        Ok((
            BootstrapServerHandle {
                addr: local_addr,
                state,
                task: server_task,
            },
            self.callback_rx,
        ))
    }
}

impl Default for BootstrapServer {
    fn default() -> Self {
        Self::new().expect("Failed to create bootstrap server")
    }
}

/// Handle to a running bootstrap server
pub struct BootstrapServerHandle {
    /// Server address
    pub addr: SocketAddr,
    /// Server state
    state: Arc<ServerState>,
    /// Server task handle
    task: tokio::task::JoinHandle<()>,
}

impl BootstrapServerHandle {
    /// Get the server URL
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Get bootstrap URL for a token
    pub fn bootstrap_url(&self, token: &str) -> String {
        format!("{}/bootstrap?token={}", self.url(), token)
    }

    /// Register a node for bootstrap
    pub async fn register_node(
        &self,
        node_id: String,
        os: OsType,
        arch: Architecture,
        mode: NodeMode,
    ) -> String {
        let token = uuid::Uuid::new_v4().to_string();

        let info = TokenInfo {
            node_id,
            os,
            arch,
            mode,
        };

        self.state.tokens.write().await.insert(token.clone(), info);
        token
    }

    /// Stop the server
    pub async fn stop(self) {
        self.task.abort();
    }
}

/// Query parameters for bootstrap endpoint
#[derive(Debug, Deserialize)]
struct BootstrapQuery {
    token: String,
}

/// Handle bootstrap script request
async fn handle_bootstrap(
    State(state): State<Arc<ServerState>>,
    Query(query): Query<BootstrapQuery>,
) -> impl IntoResponse {
    let tokens = state.tokens.read().await;

    match tokens.get(&query.token) {
        Some(info) => {
            let script = state.script_gen.generate(&info.os, &info.arch, &info.mode);
            (
                StatusCode::OK,
                [("Content-Type", "text/plain; charset=utf-8")],
                script,
            )
        }
        None => (
            StatusCode::UNAUTHORIZED,
            [("Content-Type", "text/plain; charset=utf-8")],
            "Invalid or expired token".to_string(),
        ),
    }
}

/// Handle callback from bootstrapped node
async fn handle_callback(
    State(state): State<Arc<ServerState>>,
    Json(callback): Json<BootstrapCallback>,
) -> impl IntoResponse {
    // Send callback to receiver
    if state.callback_tx.send(callback).await.is_ok() {
        (StatusCode::OK, "Callback received")
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, "Failed to process callback")
    }
}

/// Health check endpoint
async fn handle_health() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_callback_serialize() {
        let callback = BootstrapCallback {
            node_id: "test-node".to_string(),
            status: "success".to_string(),
            agent_version: Some("0.1.0".to_string()),
            error: None,
            hardware: None,
        };

        let json = serde_json::to_string(&callback).unwrap();
        assert!(json.contains("test-node"));
        assert!(json.contains("success"));
    }

    #[tokio::test]
    async fn test_server_creation() {
        let server = BootstrapServer::new().unwrap();
        let (handle, _rx) = server.start().await.unwrap();

        assert!(handle.addr.port() > 0);
        assert!(handle.url().starts_with("http://"));

        handle.stop().await;
    }

    #[tokio::test]
    async fn test_token_registration() {
        let server = BootstrapServer::new().unwrap();
        let (handle, _rx) = server.start().await.unwrap();

        let token = handle
            .register_node(
                "node-1".to_string(),
                OsType::Linux,
                Architecture::Amd64,
                NodeMode::Docker,
            )
            .await;

        assert!(!token.is_empty());
        assert!(handle.bootstrap_url(&token).contains(&token));

        handle.stop().await;
    }
}
