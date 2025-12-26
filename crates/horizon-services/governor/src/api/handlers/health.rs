use crate::api::dto::HealthResponse;
use axum::Json;

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Health check successful", body = HealthResponse)
    )
)]
pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}
