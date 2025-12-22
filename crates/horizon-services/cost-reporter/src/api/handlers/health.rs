use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::api::AppState;

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
}

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        service: "cost-reporter".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReadyResponse {
    pub ready: bool,
    pub database: bool,
    pub views: bool,
}

pub async fn ready(State(state): State<Arc<AppState>>) -> Result<Json<ReadyResponse>, StatusCode> {
    // Check if materialized views exist
    let views_exist = state
        .view_manager
        .views_exist()
        .await
        .unwrap_or(false);

    let ready = views_exist;

    Ok(Json(ReadyResponse {
        ready,
        database: true, // If we got here, database is accessible
        views: views_exist,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health() {
        let response = health().await;
        assert_eq!(response.status, "healthy");
        assert_eq!(response.service, "cost-reporter");
    }
}
