use crate::error::Result;
use crate::models::{BillingRecord, BillingRecordQuery, CreateBillingRecord, Provider};
use chrono::Utc;
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Clone)]
pub struct BillingRepository {
    pool: PgPool,
}

impl BillingRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create(&self, record: &CreateBillingRecord) -> Result<BillingRecord> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        let provider_str = record.provider.to_string();

        let row = sqlx::query_as::<_, BillingRecord>(
            r#"
            INSERT INTO raw_billing_records
                (id, provider, account_id, service, resource_id, usage_start, usage_end, amount, currency, raw_data, ingested_at)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(&provider_str)
        .bind(&record.account_id)
        .bind(&record.service)
        .bind(&record.resource_id)
        .bind(record.usage_start)
        .bind(record.usage_end)
        .bind(record.amount)
        .bind(&record.currency)
        .bind(&record.raw_data)
        .bind(now)
        .fetch_one(&self.pool)
        .await?;

        Ok(row)
    }

    pub async fn create_batch(&self, records: &[CreateBillingRecord]) -> Result<Vec<BillingRecord>> {
        let mut results = Vec::new();

        for record in records {
            let created = self.create(record).await?;
            results.push(created);
        }

        Ok(results)
    }

    pub async fn find_by_id(&self, id: Uuid) -> Result<Option<BillingRecord>> {
        let row = sqlx::query_as::<_, BillingRecord>(
            r#"
            SELECT * FROM raw_billing_records
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row)
    }

    pub async fn query(&self, query: &BillingRecordQuery) -> Result<Vec<BillingRecord>> {
        let mut sql = String::from("SELECT * FROM raw_billing_records WHERE 1=1");
        let mut bind_count = 0;

        if query.provider.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND provider = ${}", bind_count));
        }

        if query.account_id.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND account_id = ${}", bind_count));
        }

        if query.service.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND service = ${}", bind_count));
        }

        if query.resource_id.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND resource_id = ${}", bind_count));
        }

        if query.start_date.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND usage_start >= ${}", bind_count));
        }

        if query.end_date.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND usage_end <= ${}", bind_count));
        }

        sql.push_str(" ORDER BY usage_start DESC");

        if let Some(_limit) = query.limit {
            bind_count += 1;
            sql.push_str(&format!(" LIMIT ${}", bind_count));
        }

        if let Some(_offset) = query.offset {
            bind_count += 1;
            sql.push_str(&format!(" OFFSET ${}", bind_count));
        }

        let mut q = sqlx::query_as::<_, BillingRecord>(&sql);

        if let Some(provider) = &query.provider {
            q = q.bind(provider.to_string());
        }
        if let Some(account_id) = &query.account_id {
            q = q.bind(account_id);
        }
        if let Some(service) = &query.service {
            q = q.bind(service);
        }
        if let Some(resource_id) = &query.resource_id {
            q = q.bind(resource_id);
        }
        if let Some(start_date) = query.start_date {
            q = q.bind(start_date);
        }
        if let Some(end_date) = query.end_date {
            q = q.bind(end_date);
        }
        if let Some(limit) = query.limit {
            q = q.bind(limit);
        }
        if let Some(offset) = query.offset {
            q = q.bind(offset);
        }

        let rows = q.fetch_all(&self.pool).await?;
        Ok(rows)
    }

    pub async fn delete_by_id(&self, id: Uuid) -> Result<bool> {
        let result = sqlx::query(
            r#"
            DELETE FROM raw_billing_records
            WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    pub async fn count(&self) -> Result<i64> {
        let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM raw_billing_records")
            .fetch_one(&self.pool)
            .await?;

        Ok(row.0)
    }

    pub async fn count_by_provider(&self, provider: Provider) -> Result<i64> {
        let row: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM raw_billing_records WHERE provider = $1"
        )
        .bind(provider.to_string())
        .fetch_one(&self.pool)
        .await?;

        Ok(row.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn create_test_record() -> CreateBillingRecord {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);

        CreateBillingRecord::new(Provider::Aws, now, later, dec!(100.50))
            .with_account_id("test-account".to_string())
            .with_service("EC2".to_string())
    }

    #[tokio::test]
    #[ignore]
    async fn test_repository_create() {
        let database_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost/horizon_cost_test".to_string());

        let pool = PgPool::connect(&database_url).await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();

        let repo = BillingRepository::new(pool.clone());
        let record = create_test_record();

        let created = repo.create(&record).await.unwrap();
        assert_eq!(created.provider, "aws");
        assert_eq!(created.amount, dec!(100.50));
        assert_eq!(created.account_id, Some("test-account".to_string()));

        sqlx::query("DELETE FROM raw_billing_records")
            .execute(&pool)
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn test_repository_find_by_id() {
        let database_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost/horizon_cost_test".to_string());

        let pool = PgPool::connect(&database_url).await.unwrap();
        let repo = BillingRepository::new(pool.clone());
        let record = create_test_record();

        let created = repo.create(&record).await.unwrap();
        let found = repo.find_by_id(created.id).await.unwrap();

        assert!(found.is_some());
        assert_eq!(found.unwrap().id, created.id);

        sqlx::query("DELETE FROM raw_billing_records")
            .execute(&pool)
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn test_repository_query_by_provider() {
        let database_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost/horizon_cost_test".to_string());

        let pool = PgPool::connect(&database_url).await.unwrap();
        let repo = BillingRepository::new(pool.clone());

        let aws_record = create_test_record();
        repo.create(&aws_record).await.unwrap();

        let gcp_record = CreateBillingRecord::new(
            Provider::Gcp,
            Utc::now(),
            Utc::now() + chrono::Duration::hours(1),
            dec!(50.0),
        );
        repo.create(&gcp_record).await.unwrap();

        let query = BillingRecordQuery {
            provider: Some(Provider::Aws),
            ..Default::default()
        };

        let results = repo.query(&query).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.provider == "aws"));

        sqlx::query("DELETE FROM raw_billing_records")
            .execute(&pool)
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn test_repository_delete() {
        let database_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost/horizon_cost_test".to_string());

        let pool = PgPool::connect(&database_url).await.unwrap();
        let repo = BillingRepository::new(pool.clone());
        let record = create_test_record();

        let created = repo.create(&record).await.unwrap();
        let deleted = repo.delete_by_id(created.id).await.unwrap();
        assert!(deleted);

        let found = repo.find_by_id(created.id).await.unwrap();
        assert!(found.is_none());
    }

    #[tokio::test]
    #[ignore]
    async fn test_repository_count() {
        let database_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost/horizon_cost_test".to_string());

        let pool = PgPool::connect(&database_url).await.unwrap();
        sqlx::query("DELETE FROM raw_billing_records")
            .execute(&pool)
            .await
            .unwrap();

        let repo = BillingRepository::new(pool.clone());

        let initial_count = repo.count().await.unwrap();
        repo.create(&create_test_record()).await.unwrap();
        let new_count = repo.count().await.unwrap();

        assert_eq!(new_count, initial_count + 1);

        sqlx::query("DELETE FROM raw_billing_records")
            .execute(&pool)
            .await
            .unwrap();
    }
}
