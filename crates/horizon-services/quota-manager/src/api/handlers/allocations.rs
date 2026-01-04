use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use hpc_channels::QuotaMessage;
use serde::Deserialize;
use std::sync::Arc;
use uuid::Uuid;

use crate::{api::routes::AppState, error::HpcError, models::*};

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
        .check_allocation(
            entity_type,
            &req.entity_id,
            resource_type,
            req.requested_value,
        )
        .await?;

    Ok(Json(response))
}

pub async fn create_allocation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AllocateRequest>,
) -> Result<(StatusCode, Json<Allocation>), HpcError> {
    let entity_type = EntityType::from_str(&req.entity_type)?;
    let resource_type = ResourceType::from_str(&req.resource_type)?;
    let requested_amount: f64 = req.requested_value.try_into().unwrap_or(0.0);

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

    // Publish allocation granted event
    state.publish_allocation_event(QuotaMessage::AllocationGranted {
        allocation_id: allocation.id.to_string(),
        quota_id: allocation.quota_id.to_string(),
        amount: requested_amount,
    });

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

    // Publish allocation released event
    let amount: f64 = allocation.allocated_value.try_into().unwrap_or(0.0);
    state.publish_allocation_event(QuotaMessage::AllocationReleased {
        allocation_id: allocation.id.to_string(),
        quota_id: allocation.quota_id.to_string(),
        amount,
    });

    Ok(Json(allocation))
}

pub async fn list_allocations(
    State(state): State<Arc<AppState>>,
    Path(quota_id): Path<Uuid>,
) -> Result<Json<Vec<Allocation>>, HpcError> {
    let allocations = state
        .allocation_service
        .list_active_allocations(quota_id)
        .await?;
    Ok(Json(allocations))
}
