use sqlx::PgPool;
use uuid::Uuid;

use crate::error::Result;
use crate::models::AssetHistory;

pub struct HistoryRepository {
    pool: PgPool,
}

impl HistoryRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    #[tracing::instrument(skip(self))]
    pub async fn list_by_asset(
        &self,
        asset_id: Uuid,
        page: i64,
        page_size: i64,
    ) -> Result<Vec<AssetHistory>> {
        let page = page.max(1);
        let page_size = page_size.clamp(1, 1000);
        let offset = (page - 1) * page_size;

        let history = sqlx::query_as!(
            AssetHistory,
            r#"
            SELECT
                id,
                asset_id,
                operation AS "operation: _",
                previous_status AS "previous_status: _",
                previous_metadata AS "previous_metadata: serde_json::Value",
                new_status AS "new_status: _",
                new_metadata AS "new_metadata: serde_json::Value",
                metadata_delta AS "metadata_delta: serde_json::Value",
                changed_at,
                changed_by,
                reason
            FROM asset_history
            WHERE asset_id = $1
            ORDER BY changed_at DESC
            LIMIT $2 OFFSET $3
            "#,
            asset_id,
            page_size,
            offset
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(history)
    }

    #[tracing::instrument(skip(self))]
    pub async fn count_by_asset(&self, asset_id: Uuid) -> Result<i64> {
        let count = sqlx::query_scalar!(
            r#"
            SELECT COUNT(*) as "count!"
            FROM asset_history
            WHERE asset_id = $1
            "#,
            asset_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_history_repository_creation() {
        let pool = PgPool::connect_lazy("postgres://localhost/test").unwrap();
        let repo = HistoryRepository::new(pool);
        assert!(std::mem::size_of_val(&repo) > 0);
    }
}
