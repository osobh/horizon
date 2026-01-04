use crate::error::Result;
use crate::models::*;
use chrono::NaiveDate;
use sqlx::{PgPool, Row};

#[derive(Clone)]
pub struct ExecutiveRepository {
    pool: PgPool,
}

impl ExecutiveRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn list_reports(&self) -> Result<Vec<ExecutiveReport>> {
        let reports = sqlx::query_as::<_, ExecutiveReport>(
            "SELECT * FROM executive_reports ORDER BY report_period DESC LIMIT 100",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(reports)
    }

    pub async fn get_summary(&self) -> Result<ReportSummary> {
        let row = sqlx::query(
            "SELECT COUNT(*) as count, MAX(report_period) as latest FROM executive_reports",
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(ReportSummary {
            total_reports: row.try_get::<Option<i64>, _>("count")?.unwrap_or(0),
            latest_period: row.try_get::<Option<NaiveDate>, _>("latest")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_repository() {
        let pool = sqlx::PgPool::connect_lazy("postgres://localhost/test").unwrap();
        let _repo = ExecutiveRepository::new(pool);
    }
}
