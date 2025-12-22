use axum::{extract::State, Json};
use sqlx::PgPool;

use crate::api::models::HealthResponse;
use crate::error::Result;

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse),
        (status = 500, description = "Service is unhealthy")
    )
)]
#[tracing::instrument(skip(pool))]
pub async fn health_check(State(pool): State<PgPool>) -> Result<Json<HealthResponse>> {
    let db_status = match sqlx::query("SELECT 1").execute(&pool).await {
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
