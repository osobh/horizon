//! HTTP API handlers for cluster mesh operations.

use crate::install_script::{generate_install_script, InstallScriptConfig};
use axum::{
    extract::{Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::net::Ipv4Addr;
use std::sync::Arc;
use uuid::Uuid;

use super::state::AppState;

/// Query parameters for the install endpoint.
#[derive(Debug, Deserialize)]
pub struct InstallQuery {
    /// Join token (required)
    pub token: String,
    /// Optional node name override
    pub node_name: Option<String>,
}

/// Response for health check endpoint.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub timestamp: String,
}

/// Response for token validation.
#[derive(Debug, Serialize)]
pub struct TokenValidationResponse {
    pub valid: bool,
    pub reason: Option<String>,
    pub expires_at: Option<String>,
}

/// Error response structure.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        let status = match self.code.as_str() {
            "INVALID_TOKEN" => StatusCode::UNAUTHORIZED,
            "TOKEN_EXPIRED" => StatusCode::UNAUTHORIZED,
            "NOT_FOUND" => StatusCode::NOT_FOUND,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (status, Json(self)).into_response()
    }
}

// ============== WireGuard Join Protocol ==============

/// Request body for node join with WireGuard support.
///
/// This is the new join protocol that includes WireGuard key exchange
/// for secure mesh networking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinRequest {
    /// Join token for authentication
    pub token: String,
    /// Node's WireGuard public key (base64 encoded)
    pub wg_public_key: String,
    /// Node's hostname
    pub hostname: String,
    /// Node's hardware profile (optional, will be profiled if not provided)
    #[serde(default)]
    pub hardware_profile: Option<HardwareProfileInfo>,
    /// Node's preferred WireGuard listen port (optional)
    pub wg_listen_port: Option<u16>,
    /// Node's public endpoint for WireGuard (if known)
    pub public_endpoint: Option<String>,
}

/// Hardware profile information from the joining node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfileInfo {
    pub cpu_model: String,
    pub cpu_cores: u32,
    pub memory_gb: f32,
    pub storage_gb: f32,
    pub gpu_count: u32,
}

/// Response to a successful join request with WireGuard configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinResponse {
    /// Assigned node ID
    pub node_id: Uuid,
    /// Assigned subnet information
    pub subnet: SubnetAssignmentInfo,
    /// WireGuard configuration for the node
    pub wireguard_config: WireGuardJoinConfig,
    /// Current peers in the subnet
    pub peers: Vec<PeerInfo>,
    /// Cluster coordinator endpoint for heartbeats
    pub coordinator_endpoint: String,
    /// Token for subsequent API calls
    pub api_token: String,
}

/// Subnet assignment information returned to joining node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetAssignmentInfo {
    /// Subnet ID
    pub subnet_id: Uuid,
    /// Subnet name
    pub subnet_name: String,
    /// Subnet CIDR (e.g., "10.100.0.0/24")
    pub cidr: String,
    /// Assigned IP address for this node
    pub assigned_ip: Ipv4Addr,
}

/// WireGuard configuration for the joining node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireGuardJoinConfig {
    /// Interface name to use (e.g., "wg-swarm0")
    pub interface_name: String,
    /// Listen port for WireGuard
    pub listen_port: u16,
    /// Node's assigned address with CIDR (e.g., "10.100.0.5/24")
    pub address: String,
    /// MTU to use
    pub mtu: u16,
    /// Subnet gateway's public key (for routing to other nodes)
    pub gateway_public_key: Option<String>,
}

/// Information about an existing peer in the subnet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Peer's node ID
    pub node_id: Uuid,
    /// Peer's hostname
    pub hostname: String,
    /// Peer's WireGuard public key
    pub public_key: String,
    /// Peer's allowed IPs (their assigned IP)
    pub allowed_ips: Vec<String>,
    /// Peer's endpoint (if known)
    pub endpoint: Option<String>,
    /// Persistent keepalive interval
    pub persistent_keepalive: u16,
}

/// Health check handler.
///
/// Returns the current health status of the cluster coordinator.
pub async fn health_handler() -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: Utc::now().to_rfc3339(),
    })
}

/// Readiness check handler.
///
/// Returns whether the service is ready to accept requests.
pub async fn ready_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.is_ready() {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not ready")
    }
}

/// Install script handler.
///
/// Generates and returns a dynamic install script for node bootstrapping.
///
/// # Query Parameters
///
/// - `token`: Required. The join token for authentication.
/// - `node_name`: Optional. Override the default node name.
///
/// # Returns
///
/// A shell script that can be piped to bash for node installation:
/// ```bash
/// curl -sSL "https://cluster.example.com/api/v1/install?token=TOKEN" | bash
/// ```
pub async fn install_handler(
    Query(params): Query<InstallQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, ErrorResponse> {
    // Validate the token
    let token_info = state.validate_token(&params.token).await.map_err(|e| {
        tracing::warn!("Token validation failed: {}", e);
        ErrorResponse {
            error: e.to_string(),
            code: "INVALID_TOKEN".to_string(),
        }
    })?;

    // Check if token is expired
    if token_info.is_expired() {
        return Err(ErrorResponse {
            error: "Token has expired".to_string(),
            code: "TOKEN_EXPIRED".to_string(),
        });
    }

    // Build the install script configuration
    let config =
        InstallScriptConfig::new(state.cluster_host.clone(), state.cluster_port, params.token)
            .with_expiry(token_info.expires_at)
            .with_version(state.swarmlet_version.clone())
            .with_docker_image(state.docker_image.clone())
            .with_releases_url(state.releases_base_url.clone())
            .with_checksums(state.binary_checksums.clone());

    // Generate the install script
    let script = generate_install_script(&config);

    tracing::info!(
        "Generated install script for token (expires: {})",
        token_info.expires_at
    );

    // Return the script with appropriate content type
    Ok((
        [(header::CONTENT_TYPE, "text/x-shellscript; charset=utf-8")],
        script,
    ))
}

/// Token validation handler.
///
/// Validates a join token without generating an install script.
pub async fn validate_token_handler(
    Query(params): Query<InstallQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, ErrorResponse> {
    let token_info = state
        .validate_token(&params.token)
        .await
        .map_err(|e| ErrorResponse {
            error: e.to_string(),
            code: "INVALID_TOKEN".to_string(),
        })?;

    Ok(Json(TokenValidationResponse {
        valid: !token_info.is_expired(),
        reason: if token_info.is_expired() {
            Some("Token has expired".to_string())
        } else {
            None
        },
        expires_at: Some(token_info.expires_at.to_rfc3339()),
    }))
}

/// Node join request handler with WireGuard integration.
///
/// This is the primary endpoint for nodes to join the cluster with
/// WireGuard mesh networking support.
///
/// # Request Body
///
/// ```json
/// {
///   "token": "join-token-here",
///   "wg_public_key": "base64-encoded-wireguard-public-key",
///   "hostname": "node-hostname",
///   "wg_listen_port": 51820,
///   "public_endpoint": "1.2.3.4:51820"
/// }
/// ```
///
/// # Response
///
/// Returns the node's subnet assignment and WireGuard configuration,
/// including all current peers in the subnet.
pub async fn join_request_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<JoinRequest>,
) -> Result<impl IntoResponse, ErrorResponse> {
    // 1. Validate the join token
    let token_info = state.validate_token(&request.token).await.map_err(|e| {
        tracing::warn!("Join request failed - invalid token: {}", e);
        ErrorResponse {
            error: e.to_string(),
            code: "INVALID_TOKEN".to_string(),
        }
    })?;

    if token_info.is_expired() {
        return Err(ErrorResponse {
            error: "Token has expired".to_string(),
            code: "TOKEN_EXPIRED".to_string(),
        });
    }

    // 2. Validate WireGuard public key format (basic check)
    if request.wg_public_key.len() < 32 {
        return Err(ErrorResponse {
            error: "Invalid WireGuard public key format".to_string(),
            code: "INVALID_WG_KEY".to_string(),
        });
    }

    // 3. Generate node ID
    let node_id = Uuid::new_v4();

    // 4. Assign subnet and IP
    // In a full implementation, this would call the subnet-manager
    // For now, we use a simple allocation from a default subnet
    let subnet_assignment = state
        .assign_node_to_subnet(
            node_id,
            &request.hostname,
            &request.wg_public_key,
            request.public_endpoint.as_deref(),
        )
        .await
        .map_err(|e| {
            tracing::error!("Failed to assign subnet: {}", e);
            ErrorResponse {
                error: format!("Failed to assign subnet: {}", e),
                code: "SUBNET_ASSIGNMENT_FAILED".to_string(),
            }
        })?;

    // 5. Get existing peers in the subnet
    let peers = state
        .get_subnet_peers(subnet_assignment.subnet_id, node_id)
        .await;

    // 6. Build WireGuard configuration for the node
    let wireguard_config = WireGuardJoinConfig {
        interface_name: format!(
            "wg-{}",
            subnet_assignment
                .subnet_name
                .replace(' ', "-")
                .to_lowercase()
        ),
        listen_port: request.wg_listen_port.unwrap_or(51820),
        address: format!("{}/24", subnet_assignment.assigned_ip),
        mtu: 1420,
        gateway_public_key: state
            .get_subnet_gateway_key(subnet_assignment.subnet_id)
            .await,
    };

    // 7. Generate API token for subsequent calls
    let (api_token, _) = state.generate_token(
        chrono::Duration::days(365),
        0, // unlimited uses
        vec!["node".to_string()],
    );

    // 8. Notify existing peers about the new node (async, don't block response)
    let new_peer_info = PeerInfo {
        node_id,
        hostname: request.hostname.clone(),
        public_key: request.wg_public_key.clone(),
        allowed_ips: vec![format!("{}/32", subnet_assignment.assigned_ip)],
        endpoint: request.public_endpoint.clone(),
        persistent_keepalive: 25,
    };

    // Clone for async task
    let state_clone = state.clone();
    let subnet_id = subnet_assignment.subnet_id;
    tokio::spawn(async move {
        state_clone
            .notify_peers_of_new_node(subnet_id, new_peer_info)
            .await;
    });

    tracing::info!(
        "Node {} ({}) joined subnet {} with IP {}",
        node_id,
        request.hostname,
        subnet_assignment.subnet_name,
        subnet_assignment.assigned_ip
    );

    // 9. Return join response
    Ok(Json(JoinResponse {
        node_id,
        subnet: subnet_assignment,
        wireguard_config,
        peers,
        coordinator_endpoint: format!("{}:{}", state.cluster_host, state.cluster_port),
        api_token,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_handler() {
        let response = health_handler().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
