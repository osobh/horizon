use crate::error::Result;
use crate::models::*;
use sqlx::{PgPool, Row};

#[derive(Clone)]
pub struct InitiativeRepository {
    pool: PgPool,
}

impl InitiativeRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn list_initiatives(&self) -> Result<Vec<Initiative>> {
        let initiatives = sqlx::query_as::<_, Initiative>(
            "SELECT * FROM initiatives ORDER BY created_at DESC LIMIT 100",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(initiatives)
    }

    pub async fn get_portfolio_summary(&self) -> Result<PortfolioSummary> {
        let row = sqlx::query(
            r#"
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress,
                COUNT(*) FILTER (WHERE status = 'completed') as completed
            FROM initiatives
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(PortfolioSummary {
            total_initiatives: row.try_get::<Option<i64>, _>("total")?.unwrap_or(0),
            in_progress: row.try_get::<Option<i64>, _>("in_progress")?.unwrap_or(0),
            completed: row.try_get::<Option<i64>, _>("completed")?.unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_repository() {
        let pool = sqlx::PgPool::connect_lazy("postgres://localhost/test").unwrap();
        let _repo = InitiativeRepository::new(pool);
    }
}
