use axum::{http::StatusCode, Json};
use serde_json::{json, Value};

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Service is healthy")
    )
)]
pub async fn health_check() -> (StatusCode, Json<Value>) {
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "service": "capacity-modeler",
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check() {
        let (status, body) = health_check().await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body.0["status"], "healthy");
        assert_eq!(body.0["service"], "capacity-modeler");
    }
}
