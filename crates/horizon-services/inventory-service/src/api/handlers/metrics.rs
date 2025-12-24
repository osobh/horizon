use axum::{
    extract::{Path, State},
    Json,
};
use uuid::Uuid;

use crate::api::state::AppState;
use crate::error::Result;
use crate::models::AssetMetrics;
use crate::repository::MetricsRepository;

#[utoipa::path(
    get,
    path = "/api/v1/assets/{id}/metrics",
    params(
        ("id" = Uuid, Path, description = "Asset ID")
    ),
    responses(
        (status = 200, description = "Asset metrics", body = AssetMetrics),
        (status = 404, description = "Metrics not found"),
        (status = 500, description = "Internal server error")
    )
)]
#[tracing::instrument(skip(state))]
pub async fn get_asset_metrics(
    Path(asset_id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<AssetMetrics>> {
    let repo = MetricsRepository::new(state.pool.clone());
    let metrics = repo.get_by_asset(asset_id).await?;

    Ok(Json(metrics))
}
