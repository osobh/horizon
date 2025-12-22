use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::api::state::AppState;

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
}

pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        service: "cost-attributor".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

pub async fn readiness_check(
    State(_state): State<AppState>,
) -> Result<Json<HealthResponse>, StatusCode> {
    // In a real implementation, we would check database connectivity here
    Ok(Json(HealthResponse {
        status: "ready".to_string(),
        service: "cost-attributor".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check() {
        let response = health_check().await;
        assert_eq!(response.0.status, "healthy");
        assert_eq!(response.0.service, "cost-attributor");
    }
}
