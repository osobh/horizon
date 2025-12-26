use axum::{extract::State, Json};

use crate::api::models::HealthResponse;
use crate::api::state::AppState;
use crate::error::Result;

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse),
        (status = 500, description = "Service is unhealthy")
    )
)]
#[tracing::instrument(skip(state))]
pub async fn health_check(State(state): State<AppState>) -> Result<Json<HealthResponse>> {
    let db_status = match sqlx::query("SELECT 1").execute(&state.pool).await {
        Ok(_) => "connected",
        Err(_) => "disconnected",
    };

    Ok(Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        database: db_status.to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "healthy".to_string(),
            version: "0.1.0".to_string(),
            database: "connected".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("healthy"));
        assert!(json.contains("connected"));
    }
}
