use crate::error::Result;
use crate::models::*;
use rust_decimal::Decimal;
use sqlx::{PgPool, Row};

#[derive(Clone)]
pub struct VendorRepository {
    pool: PgPool,
}

impl VendorRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn list_vendors(&self) -> Result<Vec<Vendor>> {
        let vendors = sqlx::query_as::<_, Vendor>(
            r#"SELECT id, name, type as "type_", website, primary_contact, email, performance_score, created_at, updated_at FROM vendors ORDER BY name"#
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(vendors)
    }

    pub async fn get_summary(&self) -> Result<VendorSummary> {
        let row = sqlx::query(
            r#"
            SELECT
                (SELECT COUNT(*) FROM vendors) as vendors,
                (SELECT COUNT(*) FROM contracts WHERE status = 'active') as contracts,
                COALESCE((SELECT SUM(total_value) FROM contracts), 0) as total
            "#
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(VendorSummary {
            total_vendors: row.try_get::<Option<i64>, _>("vendors")?.unwrap_or(0),
            active_contracts: row.try_get::<Option<i64>, _>("contracts")?.unwrap_or(0),
            total_value: row.try_get::<Option<Decimal>, _>("total")?.unwrap_or(Decimal::ZERO),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_repository() {
        let pool = sqlx::PgPool::connect_lazy("postgres://localhost/test").unwrap();
        let _repo = VendorRepository::new(pool);
    }
}
