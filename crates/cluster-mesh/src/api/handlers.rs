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
use std::sync::Arc;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_handler() {
        let response = health_handler().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
