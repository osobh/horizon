use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::Deserialize;
use sqlx::PgPool;
use uuid::Uuid;

use crate::api::models::{ListHistoryResponse, Pagination};
use crate::error::Result;
use crate::repository::HistoryRepository;

#[derive(Debug, Deserialize)]
pub struct HistoryQuery {
    pub page: Option<i64>,
    pub page_size: Option<i64>,
}

impl HistoryQuery {
    pub fn page(&self) -> i64 {
        self.page.unwrap_or(1).max(1)
    }

    pub fn page_size(&self) -> i64 {
        self.page_size.unwrap_or(50).clamp(1, 1000)
    }
}

#[utoipa::path(
    get,
    path = "/api/v1/assets/{id}/history",
    params(
        ("id" = Uuid, Path, description = "Asset ID"),
        ("page" = Option<i64>, Query, description = "Page number"),
        ("page_size" = Option<i64>, Query, description = "Page size")
    ),
    responses(
        (status = 200, description = "Asset history", body = ListHistoryResponse),
        (status = 404, description = "Asset not found"),
        (status = 500, description = "Internal server error")
    )
)]
#[tracing::instrument(skip(pool))]
pub async fn list_asset_history(
    Path(asset_id): Path<Uuid>,
    Query(query): Query<HistoryQuery>,
    State(pool): State<PgPool>,
) -> Result<Json<ListHistoryResponse>> {
    let repo = HistoryRepository::new(pool);

    let history = repo
        .list_by_asset(asset_id, query.page(), query.page_size())
        .await?;
    let total = repo.count_by_asset(asset_id).await?;

    let pagination = Pagination::new(query.page(), query.page_size(), total);

    Ok(Json(ListHistoryResponse {
        data: history,
        pagination,
    }))
}
