use rust_decimal::Decimal;
use sqlx::PgPool;
use uuid::Uuid;

use crate::{
    error::{HpcError, QuotaErrorExt, Result},
    models::*,
};

#[derive(Clone)]
pub struct QuotaRepository {
    pool: PgPool,
}

impl QuotaRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a test repository (for unit tests only).
    /// This creates an unconnected pool that should not be used for actual queries.
    #[cfg(test)]
    pub fn new_test() -> Self {
        use sqlx::postgres::PgPoolOptions;
        // Create a pool that will fail on actual connection - only for tests
        // that don't actually need database access
        let pool = PgPoolOptions::new()
            .max_connections(0)
            .connect_lazy("postgres://test:test@localhost:5432/test")
            .expect("Failed to create test pool");
        Self { pool }
    }

    // Quota CRUD operations
    pub async fn create_quota(&self, req: CreateQuotaRequest) -> Result<Quota> {
        let quota = sqlx::query_as::<_, Quota>(
            r#"
            INSERT INTO quotas (
                entity_type, entity_id, parent_id, resource_type,
                limit_value, soft_limit, burst_limit, overcommit_ratio
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
            "#,
        )
        .bind(req.entity_type)
        .bind(&req.entity_id)
        .bind(req.parent_id)
        .bind(req.resource_type)
        .bind(req.limit_value)
        .bind(req.soft_limit)
        .bind(req.burst_limit)
        .bind(req.overcommit_ratio.unwrap_or(Decimal::ONE))
        .fetch_one(&self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::Database(db_err) if db_err.is_unique_violation() => {
                HpcError::quota_already_exists(format!(
                    "{} for entity {} resource {}",
                    "quota",
                    req.entity_id,
                    req.resource_type.as_str()
                ))
            }
            _ => HpcError::from(e),
        })?;

        quota.validate()?;
        Ok(quota)
    }

    pub async fn get_quota(&self, id: Uuid) -> Result<Quota> {
        sqlx::query_as::<_, Quota>("SELECT * FROM quotas WHERE id = $1")
            .bind(id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| match e {
                sqlx::Error::RowNotFound => HpcError::quota_not_found(id),
                _ => HpcError::from(e),
            })
    }

    pub async fn get_quota_by_entity(
        &self,
        entity_type: EntityType,
        entity_id: &str,
        resource_type: ResourceType,
    ) -> Result<Quota> {
        sqlx::query_as::<_, Quota>(
            "SELECT * FROM quotas WHERE entity_type = $1 AND entity_id = $2 AND resource_type = $3",
        )
        .bind(entity_type)
        .bind(entity_id)
        .bind(resource_type)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => HpcError::quota_not_found(format!(
                "{} {} {}",
                entity_type.as_str(),
                entity_id,
                resource_type.as_str()
            )),
            _ => HpcError::from(e),
        })
    }

    pub async fn list_quotas(&self, entity_type: Option<EntityType>) -> Result<Vec<Quota>> {
        let quotas = if let Some(et) = entity_type {
            sqlx::query_as::<_, Quota>(
                "SELECT * FROM quotas WHERE entity_type = $1 ORDER BY created_at DESC",
            )
            .bind(et)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query_as::<_, Quota>("SELECT * FROM quotas ORDER BY created_at DESC")
                .fetch_all(&self.pool)
                .await?
        };

        Ok(quotas)
    }

    pub async fn update_quota(&self, id: Uuid, req: UpdateQuotaRequest) -> Result<Quota> {
        let mut query = String::from("UPDATE quotas SET updated_at = NOW()");
        let params: Vec<(&str, String)> = vec![];

        if req.limit_value.is_some() {
            query.push_str(", limit_value = $");
            query.push_str(&(params.len() + 2).to_string());
        }
        if req.soft_limit.is_some() {
            query.push_str(", soft_limit = $");
            query.push_str(&(params.len() + 2).to_string());
        }
        if req.burst_limit.is_some() {
            query.push_str(", burst_limit = $");
            query.push_str(&(params.len() + 2).to_string());
        }
        if req.overcommit_ratio.is_some() {
            query.push_str(", overcommit_ratio = $");
            query.push_str(&(params.len() + 2).to_string());
        }

        query.push_str(" WHERE id = $1 RETURNING *");

        let mut sql_query = sqlx::query_as::<_, Quota>(&query).bind(id);

        if let Some(limit) = req.limit_value {
            sql_query = sql_query.bind(limit);
        }
        if let Some(soft) = req.soft_limit {
            sql_query = sql_query.bind(soft);
        }
        if let Some(burst) = req.burst_limit {
            sql_query = sql_query.bind(burst);
        }
        if let Some(overcommit) = req.overcommit_ratio {
            sql_query = sql_query.bind(overcommit);
        }

        let quota = sql_query.fetch_one(&self.pool).await.map_err(|e| match e {
            sqlx::Error::RowNotFound => HpcError::quota_not_found(id),
            _ => HpcError::from(e),
        })?;

        quota.validate()?;
        Ok(quota)
    }

    pub async fn delete_quota(&self, id: Uuid) -> Result<()> {
        let result = sqlx::query("DELETE FROM quotas WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;

        if result.rows_affected() == 0 {
            return Err(HpcError::quota_not_found(id));
        }

        Ok(())
    }

    // Allocation operations with optimistic locking
    pub async fn create_allocation(&self, req: CreateAllocationRequest) -> Result<Allocation> {
        let allocation = sqlx::query_as::<_, Allocation>(
            r#"
            INSERT INTO allocations (quota_id, job_id, resource_type, allocated_value, metadata)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            "#,
        )
        .bind(req.quota_id)
        .bind(req.job_id)
        .bind(req.resource_type)
        .bind(req.allocated_value)
        .bind(req.metadata)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::Database(db_err) if db_err.is_unique_violation() => {
                HpcError::already_exists(
                    "allocation",
                    format!("job {} resource {}", req.job_id, req.resource_type.as_str()),
                )
            }
            _ => HpcError::from(e),
        })?;

        Ok(allocation)
    }

    pub async fn get_allocation(&self, id: Uuid) -> Result<Allocation> {
        sqlx::query_as::<_, Allocation>("SELECT * FROM allocations WHERE id = $1")
            .bind(id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| match e {
                sqlx::Error::RowNotFound => HpcError::allocation_not_found(id),
                _ => HpcError::from(e),
            })
    }

    pub async fn release_allocation(&self, id: Uuid) -> Result<Allocation> {
        // Use optimistic locking: increment version when updating
        let allocation = sqlx::query_as::<_, Allocation>(
            "UPDATE allocations SET released_at = NOW(), version = version + 1 WHERE id = $1 AND released_at IS NULL RETURNING *",
        )
        .bind(id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => {
                HpcError::allocation_not_found(format!("active {}", id))
            }
            _ => HpcError::from(e),
        })?;

        Ok(allocation)
    }

    pub async fn list_active_allocations(&self, quota_id: Uuid) -> Result<Vec<Allocation>> {
        let allocations = sqlx::query_as::<_, Allocation>(
            "SELECT * FROM allocations WHERE quota_id = $1 AND released_at IS NULL ORDER BY allocated_at DESC",
        )
        .bind(quota_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(allocations)
    }

    pub async fn get_current_usage(&self, quota_id: Uuid) -> Result<Decimal> {
        let usage: Option<Decimal> = sqlx::query_scalar(
            "SELECT COALESCE(SUM(allocated_value), 0) FROM allocations WHERE quota_id = $1 AND released_at IS NULL",
        )
        .bind(quota_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(usage.unwrap_or(Decimal::ZERO))
    }

    // Usage history
    #[allow(clippy::too_many_arguments)]
    pub async fn record_usage(
        &self,
        quota_id: Uuid,
        entity_type: EntityType,
        entity_id: &str,
        resource_type: ResourceType,
        allocated_value: Decimal,
        operation: OperationType,
        job_id: Option<Uuid>,
        metadata: Option<serde_json::Value>,
    ) -> Result<UsageHistory> {
        let history = sqlx::query_as::<_, UsageHistory>(
            r#"
            INSERT INTO usage_history (
                quota_id, entity_type, entity_id, resource_type,
                allocated_value, operation, job_id, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
            "#,
        )
        .bind(quota_id)
        .bind(entity_type.as_str())
        .bind(entity_id)
        .bind(resource_type)
        .bind(allocated_value)
        .bind(operation)
        .bind(job_id)
        .bind(metadata)
        .fetch_one(&self.pool)
        .await?;

        Ok(history)
    }

    pub async fn get_usage_history(
        &self,
        quota_id: Uuid,
        limit: Option<i64>,
    ) -> Result<Vec<UsageHistory>> {
        let histories = if let Some(lim) = limit {
            sqlx::query_as::<_, UsageHistory>(
                "SELECT * FROM usage_history WHERE quota_id = $1 ORDER BY timestamp DESC LIMIT $2",
            )
            .bind(quota_id)
            .bind(lim)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query_as::<_, UsageHistory>(
                "SELECT * FROM usage_history WHERE quota_id = $1 ORDER BY timestamp DESC",
            )
            .bind(quota_id)
            .fetch_all(&self.pool)
            .await?
        };

        Ok(histories)
    }
}
