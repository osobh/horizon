use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use hpc_channels::QuotaMessage;
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
    let resource_type = req.resource_type.as_str().to_string();
    let limit = req.limit_value;

    let quota = state.quota_service.create_quota(req).await?;

    // Publish quota created event
    state.publish_quota_event(QuotaMessage::QuotaCreated {
        quota_id: quota.id.to_string(),
        resource_type,
        limit: limit.try_into().unwrap_or(0.0),
    });

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
    // Get the current quota to capture previous limit
    let previous_quota = state.quota_service.get_quota(id).await?;
    let previous_limit: f64 = previous_quota.limit_value.try_into().unwrap_or(0.0);

    let quota = state.quota_service.update_quota(id, req.clone()).await?;

    // Publish quota updated event if limit changed
    if let Some(new_limit) = req.limit_value {
        let new_limit_f64: f64 = new_limit.try_into().unwrap_or(0.0);
        if (new_limit_f64 - previous_limit).abs() > f64::EPSILON {
            state.publish_quota_event(QuotaMessage::QuotaUpdated {
                quota_id: quota.id.to_string(),
                limit: new_limit_f64,
                previous_limit,
            });
        }
    }

    Ok(Json(quota))
}

pub async fn delete_quota(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, HpcError> {
    state.quota_service.delete_quota(id).await?;

    // Publish quota deleted event
    state.publish_quota_event(QuotaMessage::QuotaDeleted {
        quota_id: id.to_string(),
    });

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
