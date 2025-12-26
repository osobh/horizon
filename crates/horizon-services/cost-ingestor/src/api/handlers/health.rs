use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
        service: "cost-ingestor".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

pub async fn readiness_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let _ = state.repository.count().await;

    Json(HealthResponse {
        status: "ready".to_string(),
        service: "cost-ingestor".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check() {
        let response = health_check().await;
        assert_eq!(response.status, "healthy");
        assert_eq!(response.service, "cost-ingestor");
    }
}
