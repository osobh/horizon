use crate::error::Result;
use crate::models::*;
use rust_decimal::Decimal;
use sqlx::{PgPool, Row};

#[derive(Clone)]
pub struct EfficiencyRepository {
    pool: PgPool,
}

impl EfficiencyRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn list_detections(&self) -> Result<Vec<WasteDetection>> {
        let detections = sqlx::query_as::<_, WasteDetection>(
            "SELECT * FROM waste_detections ORDER BY detected_at DESC LIMIT 100",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(detections)
    }

    pub async fn get_savings_summary(&self) -> Result<SavingsSummary> {
        let row = sqlx::query(
            "SELECT COUNT(*) as count, COALESCE(SUM(cost_impact_monthly), 0) as total FROM waste_detections"
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(SavingsSummary {
            total_detections: row.try_get::<Option<i64>, _>("count")?.unwrap_or(0),
            total_savings: row
                .try_get::<Option<Decimal>, _>("total")?
                .unwrap_or(Decimal::ZERO),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_repository_creation() {
        let pool = sqlx::PgPool::connect_lazy("postgres://localhost/test").unwrap();
        let _repo = EfficiencyRepository::new(pool);
    }
}
