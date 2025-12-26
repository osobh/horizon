use axum::{extract::State, Json};

use crate::api::{dto::QueueStatusResponse, state::AppState};

/// Get current queue status and statistics
#[utoipa::path(
    get,
    path = "/api/v1/queue",
    responses(
        (status = 200, description = "Queue status", body = QueueStatusResponse)
    ),
    tag = "queue"
)]
pub async fn get_queue_status(
    State(state): State<AppState>,
) -> Result<Json<QueueStatusResponse>, crate::HpcError> {
    let stats = state.scheduler.get_queue_stats().await?;

    Ok(Json(QueueStatusResponse {
        total: stats.total,
        urgent_priority: stats.urgent_priority,
        high_priority: stats.high_priority,
        normal_priority: stats.normal_priority,
        low_priority: stats.low_priority,
    }))
}
