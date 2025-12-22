use sqlx::{PgPool, QueryBuilder, Postgres};
use uuid::Uuid;

use crate::api::models::{CreateAssetRequest, ListAssetsQuery, UpdateAssetRequest};
use crate::error::{HpcError, InventoryErrorExt, Result};
use crate::models::{Asset, AssetStatus, AssetType, ProviderType};

pub struct AssetRepository {
    pool: PgPool,
}

impl AssetRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    #[tracing::instrument(skip(self))]
    pub async fn create(&self, req: CreateAssetRequest, created_by: String) -> Result<Asset> {
        let id = Uuid::new_v4();
        let status = req.status.unwrap_or(AssetStatus::Provisioning);
        let metadata = req.metadata.unwrap_or_else(|| serde_json::json!({}));

        let asset = sqlx::query_as!(
            Asset,
            r#"
            INSERT INTO assets (
                id, asset_type, provider, provider_id, parent_id,
                hostname, status, location, metadata, created_by
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING
                id,
                asset_type AS "asset_type: AssetType",
                provider AS "provider: ProviderType",
                provider_id,
                parent_id,
                hostname,
                status AS "status: AssetStatus",
                location,
                metadata AS "metadata: serde_json::Value",
                created_at,
                updated_at,
                decommissioned_at,
                created_by
            "#,
            id,
            req.asset_type as AssetType,
            req.provider as ProviderType,
            req.provider_id,
            req.parent_id,
            req.hostname,
            status as AssetStatus,
            req.location,
            metadata,
            created_by
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| {
            if let sqlx::Error::Database(db_err) = &e {
                if db_err.is_unique_violation() {
                    return HpcError::conflict("Asset with this provider_id already exists");
                }
            }
            e.into()
        })?;

        Ok(asset)
    }

    #[tracing::instrument(skip(self))]
    pub async fn get_by_id(&self, id: Uuid) -> Result<Asset> {
        let asset = sqlx::query_as!(
            Asset,
            r#"
            SELECT
                id,
                asset_type AS "asset_type: AssetType",
                provider AS "provider: ProviderType",
                provider_id,
                parent_id,
                hostname,
                status AS "status: AssetStatus",
                location,
                metadata AS "metadata: serde_json::Value",
                created_at,
                updated_at,
                decommissioned_at,
                created_by
            FROM assets
            WHERE id = $1
            "#,
            id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| HpcError::asset_not_found(id.to_string()))?;

        Ok(asset)
    }

    #[tracing::instrument(skip(self))]
    pub async fn get_by_hostname(&self, hostname: &str) -> Result<Asset> {
        let asset = sqlx::query_as!(
            Asset,
            r#"
            SELECT
                id,
                asset_type AS "asset_type: AssetType",
                provider AS "provider: ProviderType",
                provider_id,
                parent_id,
                hostname,
                status AS "status: AssetStatus",
                location,
                metadata AS "metadata: serde_json::Value",
                created_at,
                updated_at,
                decommissioned_at,
                created_by
            FROM assets
            WHERE hostname = $1
            "#,
            hostname
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| HpcError::not_found("asset", format!("Asset with hostname {} not found", hostname)))?;

        Ok(asset)
    }

    #[tracing::instrument(skip(self))]
    pub async fn list(&self, query: &ListAssetsQuery) -> Result<Vec<Asset>> {
        let mut qb: QueryBuilder<Postgres> = QueryBuilder::new(
            r#"
            SELECT
                id,
                asset_type,
                provider,
                provider_id,
                parent_id,
                hostname,
                status,
                location,
                metadata,
                created_at,
                updated_at,
                decommissioned_at,
                created_by
            FROM assets
            WHERE 1=1
            "#,
        );

        if let Some(asset_type) = query.asset_type {
            qb.push(" AND asset_type = ");
            qb.push_bind(asset_type);
        }

        if let Some(status) = query.status {
            qb.push(" AND status = ");
            qb.push_bind(status);
        }

        if let Some(provider) = query.provider {
            qb.push(" AND provider = ");
            qb.push_bind(provider);
        }

        if let Some(ref location) = query.location {
            qb.push(" AND location LIKE ");
            qb.push_bind(format!("{}%", location));
        }

        let sort_field = query.sort_field();
        let sort_order = query.sort_order();

        qb.push(format!(" ORDER BY {} {}", sort_field, sort_order));

        qb.push(" LIMIT ");
        qb.push_bind(query.page_size());
        qb.push(" OFFSET ");
        qb.push_bind(query.offset());

        let assets = qb
            .build_query_as::<Asset>()
            .fetch_all(&self.pool)
            .await?;

        Ok(assets)
    }

    #[tracing::instrument(skip(self))]
    pub async fn count(&self, query: &ListAssetsQuery) -> Result<i64> {
        let mut qb: QueryBuilder<Postgres> = QueryBuilder::new("SELECT COUNT(*) FROM assets WHERE 1=1");

        if let Some(asset_type) = query.asset_type {
            qb.push(" AND asset_type = ");
            qb.push_bind(asset_type);
        }

        if let Some(status) = query.status {
            qb.push(" AND status = ");
            qb.push_bind(status);
        }

        if let Some(provider) = query.provider {
            qb.push(" AND provider = ");
            qb.push_bind(provider);
        }

        if let Some(ref location) = query.location {
            qb.push(" AND location LIKE ");
            qb.push_bind(format!("{}%", location));
        }

        let count: (i64,) = qb
            .build_query_as()
            .fetch_one(&self.pool)
            .await?;

        Ok(count.0)
    }

    #[tracing::instrument(skip(self))]
    pub async fn update(&self, id: Uuid, req: UpdateAssetRequest) -> Result<Asset> {
        let asset = self.get_by_id(id).await?;

        let status = req.status.unwrap_or(asset.status);
        let location = req.location.or(asset.location);
        let metadata = req.metadata.unwrap_or(asset.metadata);

        let updated = sqlx::query_as!(
            Asset,
            r#"
            UPDATE assets
            SET
                status = $2,
                location = $3,
                metadata = $4
            WHERE id = $1
            RETURNING
                id,
                asset_type AS "asset_type: AssetType",
                provider AS "provider: ProviderType",
                provider_id,
                parent_id,
                hostname,
                status AS "status: AssetStatus",
                location,
                metadata AS "metadata: serde_json::Value",
                created_at,
                updated_at,
                decommissioned_at,
                created_by
            "#,
            id,
            status as AssetStatus,
            location,
            metadata,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(updated)
    }

    #[tracing::instrument(skip(self))]
    pub async fn decommission(&self, id: Uuid) -> Result<Asset> {
        let asset = sqlx::query_as!(
            Asset,
            r#"
            UPDATE assets
            SET
                status = 'decommissioned',
                decommissioned_at = NOW()
            WHERE id = $1
            RETURNING
                id,
                asset_type AS "asset_type: AssetType",
                provider AS "provider: ProviderType",
                provider_id,
                parent_id,
                hostname,
                status AS "status: AssetStatus",
                location,
                metadata AS "metadata: serde_json::Value",
                created_at,
                updated_at,
                decommissioned_at,
                created_by
            "#,
            id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| HpcError::asset_not_found(id.to_string()))?;

        Ok(asset)
    }

    #[tracing::instrument(skip(self))]
    pub async fn upsert_node(
        &self,
        hostname: String,
        provider: ProviderType,
        provider_id: Option<String>,
        location: Option<String>,
        metadata: serde_json::Value,
        created_by: String,
    ) -> Result<(Uuid, bool)> {
        let id = Uuid::new_v4();

        let result = sqlx::query!(
            r#"
            INSERT INTO assets (
                id, asset_type, provider, provider_id, hostname,
                status, location, metadata, created_by
            )
            VALUES ($1, 'node', $2, $3, $4, 'available', $5, $6, $7)
            ON CONFLICT (provider, provider_id)
            DO UPDATE SET
                hostname = EXCLUDED.hostname,
                location = EXCLUDED.location,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING id, (xmax = 0) AS created
            "#,
            id,
            provider as ProviderType,
            provider_id,
            hostname,
            location,
            metadata,
            created_by
        )
        .fetch_one(&self.pool)
        .await?;

        Ok((result.id, result.created.unwrap_or(true)))
    }

    #[tracing::instrument(skip(self))]
    pub async fn upsert_gpu(
        &self,
        parent_id: Uuid,
        gpu_uuid: String,
        metadata: serde_json::Value,
        created_by: String,
    ) -> Result<(Uuid, bool)> {
        let id = Uuid::new_v4();

        let result = sqlx::query!(
            r#"
            INSERT INTO assets (
                id, asset_type, provider, provider_id, parent_id,
                status, metadata, created_by
            )
            VALUES ($1, 'gpu', 'baremetal', $2, $3, 'available', $4, $5)
            ON CONFLICT (provider, provider_id)
            DO UPDATE SET
                parent_id = EXCLUDED.parent_id,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING id, (xmax = 0) AS created
            "#,
            id,
            gpu_uuid,
            parent_id,
            metadata,
            created_by
        )
        .fetch_one(&self.pool)
        .await?;

        Ok((result.id, result.created.unwrap_or(true)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_asset_repository_creation() {
        let pool = PgPool::connect_lazy("postgres://localhost/test").unwrap();
        let repo = AssetRepository::new(pool);
        assert!(std::mem::size_of_val(&repo) > 0);
    }
}
