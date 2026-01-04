use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use hpc_channels::InventoryMessage;
use uuid::Uuid;
use validator::Validate;

use crate::api::models::{
    CreateAssetRequest, DiscoverAssetsRequest, DiscoverAssetsResponse, ListAssetsQuery,
    ListAssetsResponse, Pagination, UpdateAssetRequest,
};
use crate::api::state::AppState;
use crate::error::{HpcError, InventoryErrorExt, Result};
use crate::models::Asset;
use crate::repository::AssetRepository;

#[utoipa::path(
    get,
    path = "/api/v1/assets",
    responses(
        (status = 200, description = "List of assets", body = ListAssetsResponse),
        (status = 400, description = "Bad request"),
        (status = 500, description = "Internal server error")
    )
)]
#[tracing::instrument(skip(state))]
pub async fn list_assets(
    Query(query): Query<ListAssetsQuery>,
    State(state): State<AppState>,
) -> Result<Json<ListAssetsResponse>> {
    let repo = AssetRepository::new(state.pool.clone());

    let assets = repo.list(&query).await?;
    let total = repo.count(&query).await?;

    let pagination = Pagination::new(query.page(), query.page_size(), total);

    Ok(Json(ListAssetsResponse {
        data: assets,
        pagination,
    }))
}

#[utoipa::path(
    post,
    path = "/api/v1/assets",
    request_body = CreateAssetRequest,
    responses(
        (status = 201, description = "Asset created", body = Asset),
        (status = 400, description = "Bad request"),
        (status = 409, description = "Conflict - asset already exists"),
        (status = 500, description = "Internal server error")
    )
)]
#[tracing::instrument(skip(state))]
pub async fn create_asset(
    State(state): State<AppState>,
    Json(req): Json<CreateAssetRequest>,
) -> Result<(StatusCode, Json<Asset>)> {
    req.validate()
        .map_err(|e| HpcError::validation(e.to_string()))?;

    let asset_type = format!("{:?}", req.asset_type);
    let provider = format!("{:?}", req.provider);
    let hostname = req.hostname.clone();

    let repo = AssetRepository::new(state.pool.clone());
    let asset = repo.create(req, "api".to_string()).await?;

    // Publish asset created event via hpc-channels
    state.publish_asset_event(InventoryMessage::AssetCreated {
        asset_id: asset.id.to_string(),
        asset_type,
        provider,
        hostname,
    });

    Ok((StatusCode::CREATED, Json(asset)))
}

#[utoipa::path(
    get,
    path = "/api/v1/assets/{id}",
    params(
        ("id" = Uuid, Path, description = "Asset ID")
    ),
    responses(
        (status = 200, description = "Asset found", body = Asset),
        (status = 404, description = "Asset not found"),
        (status = 500, description = "Internal server error")
    )
)]
#[tracing::instrument(skip(state))]
pub async fn get_asset(Path(id): Path<Uuid>, State(state): State<AppState>) -> Result<Json<Asset>> {
    let repo = AssetRepository::new(state.pool.clone());
    let asset = repo.get_by_id(id).await?;

    Ok(Json(asset))
}

#[utoipa::path(
    put,
    path = "/api/v1/assets/{id}",
    params(
        ("id" = Uuid, Path, description = "Asset ID")
    ),
    request_body = UpdateAssetRequest,
    responses(
        (status = 200, description = "Asset updated", body = Asset),
        (status = 400, description = "Bad request"),
        (status = 404, description = "Asset not found"),
        (status = 500, description = "Internal server error")
    )
)]
#[tracing::instrument(skip(state))]
pub async fn update_asset(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(req): Json<UpdateAssetRequest>,
) -> Result<Json<Asset>> {
    req.validate()
        .map_err(|e| HpcError::validation(e.to_string()))?;

    let status = req.status.as_ref().map(|s| format!("{:?}", s));

    let repo = AssetRepository::new(state.pool.clone());
    let asset = repo.update(id, req).await?;

    // Publish asset updated event via hpc-channels
    if let Some(status) = status {
        state.publish_asset_event(InventoryMessage::AssetUpdated {
            asset_id: asset.id.to_string(),
            status,
        });
    }

    Ok(Json(asset))
}

#[utoipa::path(
    delete,
    path = "/api/v1/assets/{id}",
    params(
        ("id" = Uuid, Path, description = "Asset ID")
    ),
    responses(
        (status = 204, description = "Asset decommissioned"),
        (status = 404, description = "Asset not found"),
        (status = 500, description = "Internal server error")
    )
)]
#[tracing::instrument(skip(state))]
pub async fn decommission_asset(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<StatusCode> {
    let repo = AssetRepository::new(state.pool.clone());
    repo.decommission(id).await?;

    // Publish asset decommissioned event via hpc-channels
    state.publish_asset_event(InventoryMessage::AssetDecommissioned {
        asset_id: id.to_string(),
    });

    Ok(StatusCode::NO_CONTENT)
}

#[utoipa::path(
    post,
    path = "/api/v1/assets/discover",
    request_body = DiscoverAssetsRequest,
    responses(
        (status = 200, description = "Assets discovered", body = DiscoverAssetsResponse),
        (status = 400, description = "Bad request"),
        (status = 500, description = "Internal server error")
    )
)]
#[tracing::instrument(skip(state))]
pub async fn discover_assets(
    State(state): State<AppState>,
    Json(req): Json<DiscoverAssetsRequest>,
) -> Result<Json<DiscoverAssetsResponse>> {
    let repo = AssetRepository::new(state.pool.clone());

    let node_metadata = req.node.metadata.unwrap_or_else(|| serde_json::json!({}));
    let (node_id, node_created) = repo
        .upsert_node(
            req.node.hostname,
            req.node.provider,
            req.node.provider_id,
            req.node.location,
            node_metadata,
            "node-agent".to_string(),
        )
        .await?;

    let mut gpu_ids = Vec::new();
    let mut created = if node_created { 1 } else { 0 };
    let mut updated = if node_created { 0 } else { 1 };

    for gpu in req.gpus {
        let (gpu_id, is_created) = repo
            .upsert_gpu(
                node_id,
                gpu.gpu_uuid,
                gpu.metadata,
                "node-agent".to_string(),
            )
            .await?;

        gpu_ids.push(gpu_id);
        if is_created {
            created += 1;
        } else {
            updated += 1;
        }
    }

    let response = DiscoverAssetsResponse {
        node_id,
        gpu_ids: gpu_ids.clone(),
        created,
        updated,
    };

    // Publish discovery event via hpc-channels
    state.publish_discovery_event(InventoryMessage::AssetsDiscovered {
        node_id: node_id.to_string(),
        created: created as u32,
        updated: updated as u32,
        gpu_ids: gpu_ids.into_iter().map(|id| id.to_string()).collect(),
    });

    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AssetStatus, AssetType, ProviderType};

    #[test]
    fn test_create_asset_request_validation() {
        let req = CreateAssetRequest {
            asset_type: AssetType::Gpu,
            provider: ProviderType::Baremetal,
            provider_id: None,
            parent_id: None,
            hostname: Some("gpu-node-01".to_string()),
            status: Some(AssetStatus::Available),
            location: Some("us-west-1a".to_string()),
            metadata: Some(serde_json::json!({"gpu_model": "H100"})),
        };

        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_update_asset_request_validation() {
        let req = UpdateAssetRequest {
            status: Some(AssetStatus::Maintenance),
            location: Some("us-west-2a".to_string()),
            metadata: Some(serde_json::json!({"reason": "scheduled maintenance"})),
        };

        assert!(req.validate().is_ok());
    }
}
