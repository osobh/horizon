use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use sqlx::PgPool;
use uuid::Uuid;
use validator::Validate;

use crate::api::models::{
    CreateAssetRequest, DiscoverAssetsRequest, DiscoverAssetsResponse, ListAssetsQuery,
    ListAssetsResponse, Pagination, UpdateAssetRequest,
};
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
#[tracing::instrument(skip(pool))]
pub async fn list_assets(
    Query(query): Query<ListAssetsQuery>,
    State(pool): State<PgPool>,
) -> Result<Json<ListAssetsResponse>> {
    let repo = AssetRepository::new(pool);

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
#[tracing::instrument(skip(pool))]
pub async fn create_asset(
    State(pool): State<PgPool>,
    Json(req): Json<CreateAssetRequest>,
) -> Result<(StatusCode, Json<Asset>)> {
    req.validate()
        .map_err(|e| HpcError::validation(e.to_string()))?;

    let repo = AssetRepository::new(pool);
    let asset = repo.create(req, "api".to_string()).await?;

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
#[tracing::instrument(skip(pool))]
pub async fn get_asset(
    Path(id): Path<Uuid>,
    State(pool): State<PgPool>,
) -> Result<Json<Asset>> {
    let repo = AssetRepository::new(pool);
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
#[tracing::instrument(skip(pool))]
pub async fn update_asset(
    Path(id): Path<Uuid>,
    State(pool): State<PgPool>,
    Json(req): Json<UpdateAssetRequest>,
) -> Result<Json<Asset>> {
    req.validate()
        .map_err(|e| HpcError::validation(e.to_string()))?;

    let repo = AssetRepository::new(pool);
    let asset = repo.update(id, req).await?;

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
#[tracing::instrument(skip(pool))]
pub async fn decommission_asset(
    Path(id): Path<Uuid>,
    State(pool): State<PgPool>,
) -> Result<StatusCode> {
    let repo = AssetRepository::new(pool);
    repo.decommission(id).await?;

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
#[tracing::instrument(skip(pool))]
pub async fn discover_assets(
    State(pool): State<PgPool>,
    Json(req): Json<DiscoverAssetsRequest>,
) -> Result<Json<DiscoverAssetsResponse>> {
    let repo = AssetRepository::new(pool);

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
            .upsert_gpu(node_id, gpu.gpu_uuid, gpu.metadata, "node-agent".to_string())
            .await?;

        gpu_ids.push(gpu_id);
        if is_created {
            created += 1;
        } else {
            updated += 1;
        }
    }

    Ok(Json(DiscoverAssetsResponse {
        node_id,
        gpu_ids,
        created,
        updated,
    }))
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
