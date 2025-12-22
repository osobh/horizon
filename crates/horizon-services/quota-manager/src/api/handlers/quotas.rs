use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    api::routes::AppState,
    models::*,
    error::HpcError,
};

#[derive(Deserialize)]
pub struct ListQuotasQuery {
    entity_type: Option<String>,
}

pub async fn create_quota(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateQuotaRequest>,
) -> Result<(StatusCode, Json<Quota>), HpcError> {
    let quota = state.quota_service.create_quota(req).await?;
    Ok((StatusCode::CREATED, Json(quota)))
}

pub async fn get_quota(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Quota>, HpcError> {
    let quota = state.quota_service.get_quota(id).await?;
    Ok(Json(quota))
}

pub async fn list_quotas(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListQuotasQuery>,
) -> Result<Json<Vec<Quota>>, HpcError> {
    let entity_type = if let Some(et_str) = query.entity_type {
        Some(EntityType::from_str(&et_str)?)
    } else {
        None
    };

    let quotas = state.quota_service.list_quotas(entity_type).await?;
    Ok(Json(quotas))
}

pub async fn update_quota(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateQuotaRequest>,
) -> Result<Json<Quota>, HpcError> {
    let quota = state.quota_service.update_quota(id, req).await?;
    Ok(Json(quota))
}

pub async fn delete_quota(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, HpcError> {
    state.quota_service.delete_quota(id).await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn get_usage_stats(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<QuotaUsageStats>, HpcError> {
    let stats = state.quota_service.get_usage_stats(id).await?;
    Ok(Json(stats))
}

#[derive(Deserialize)]
pub struct UsageHistoryQuery {
    limit: Option<i64>,
}

pub async fn get_usage_history(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Query(query): Query<UsageHistoryQuery>,
) -> Result<Json<Vec<UsageHistory>>, HpcError> {
    let history = state.quota_service.get_usage_history(id, query.limit).await?;
    Ok(Json(history))
}
