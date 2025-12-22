use crate::error::Result;
use sqlx::PgPool;

pub struct ViewManager {
    pool: PgPool,
}

impl ViewManager {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Refresh materialized views (concurrent mode)
    pub async fn refresh_views(&self) -> Result<()> {
        sqlx::query("SELECT refresh_cost_summaries()")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Check if views exist
    pub async fn views_exist(&self) -> Result<bool> {
        let result = sqlx::query_scalar::<_, i64>(
            r#"
            SELECT COUNT(*)
            FROM pg_matviews
            WHERE matviewname IN ('daily_cost_summary', 'monthly_cost_summary')
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(result == 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires database
    async fn test_view_manager_creation() {
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost/horizon_cost_test".to_string());

        let pool = crate::db::create_pool(&database_url).await.unwrap();
        let manager = ViewManager::new(pool);

        // Just test construction
        let _ = manager;
    }
}
