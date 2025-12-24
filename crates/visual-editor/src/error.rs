use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VisualEditorError {
    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("WebSocket error: {0}")]
    WebSocketError(String),

    #[error("GraphQL error: {0}")]
    GraphQLError(String),

    #[error("Topology not found: {0}")]
    TopologyNotFound(String),

    #[error("Invalid topology format: {0}")]
    InvalidTopology(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Agent not found: {0}")]
    AgentNotFound(String),

    #[error("Agent XP error: {0}")]
    AgentXPError(String),

    #[error("Evolution not allowed: {0}")]
    EvolutionNotAllowed(String),

    #[error("Internal server error")]
    InternalError,

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error(transparent)]
    SerdeError(#[from] serde_json::Error),

    #[error(transparent)]
    AgentError(#[from] stratoswarm_agent_core::AgentError),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, VisualEditorError>;

impl IntoResponse for VisualEditorError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            VisualEditorError::ValidationError(msg) => (StatusCode::BAD_REQUEST, msg),
            VisualEditorError::TopologyNotFound(id) => {
                (StatusCode::NOT_FOUND, format!("Topology not found: {}", id))
            }
            VisualEditorError::InvalidTopology(msg) => (StatusCode::BAD_REQUEST, msg),
            VisualEditorError::WebSocketError(msg) => (StatusCode::BAD_REQUEST, msg),
            VisualEditorError::GraphQLError(msg) => (StatusCode::BAD_REQUEST, msg),
            VisualEditorError::ConnectionError(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg),
            VisualEditorError::AgentNotFound(id) => {
                (StatusCode::NOT_FOUND, format!("Agent not found: {}", id))
            }
            VisualEditorError::AgentXPError(msg) => (StatusCode::BAD_REQUEST, msg),
            VisualEditorError::EvolutionNotAllowed(msg) => (StatusCode::CONFLICT, msg),
            VisualEditorError::InternalError => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error".to_string(),
            ),
            VisualEditorError::IoError(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "IO error occurred".to_string(),
            ),
            VisualEditorError::SerdeError(_) => {
                (StatusCode::BAD_REQUEST, "Invalid JSON format".to_string())
            }
            VisualEditorError::AgentError(e) => {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Agent error: {}", e))
            }
            VisualEditorError::Other(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "An error occurred".to_string(),
            ),
        };

        let body = Json(json!({
            "error": error_message,
            "status": status.as_u16(),
        }));

        (status, body).into_response()
    }
}
