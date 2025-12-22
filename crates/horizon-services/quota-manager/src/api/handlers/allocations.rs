use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    api::routes::AppState,
    error::HpcError,
    models::*,
};

#[derive(Debug, Deserialize)]
pub struct AllocateRequest {
    pub entity_type: String,
    pub entity_id: String,
    pub job_id: Uuid,
    pub resource_type: String,
    pub requested_value: rust_decimal::Decimal,
    pub metadata: Option<serde_json::Value>,
}

pub async fn check_allocation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AllocationCheckRequest>,
) -> Result<Json<AllocationCheckResponse>, HpcError> {
    let entity_type = EntityType::from_str(&req.entity_type)?;
    let resource_type = ResourceType::from_str(req.resource_type.as_str())?;

    let response = state
        .allocation_service
        .check_allocation(entity_type, &req.entity_id, resource_type, req.requested_value)
        .await?;

    Ok(Json(response))
}

pub async fn create_allocation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AllocateRequest>,
) -> Result<(StatusCode, Json<Allocation>), HpcError> {
    let entity_type = EntityType::from_str(&req.entity_type)?;
    let resource_type = ResourceType::from_str(&req.resource_type)?;

    let allocation = state
        .allocation_service
        .allocate(
            entity_type,
            &req.entity_id,
            req.job_id,
            resource_type,
            req.requested_value,
            req.metadata,
        )
        .await?;

    Ok((StatusCode::CREATED, Json(allocation)))
}

pub async fn get_allocation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Allocation>, HpcError> {
    let allocation = state.allocation_service.get_allocation(id).await?;
    Ok(Json(allocation))
}

pub async fn release_allocation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Allocation>, HpcError> {
    let allocation = state.allocation_service.release(id).await?;
    Ok(Json(allocation))
}

pub async fn list_allocations(
    State(state): State<Arc<AppState>>,
    Path(quota_id): Path<Uuid>,
) -> Result<Json<Vec<Allocation>>, HpcError> {
    let allocations = state.allocation_service.list_active_allocations(quota_id).await?;
    Ok(Json(allocations))
}
