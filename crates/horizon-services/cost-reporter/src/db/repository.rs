use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::{HpcError, Result, ReporterErrorExt};
use crate::models::summary::{CostAttribution, DailyCostSummary, MonthlyCostSummary, TopSpender};

#[derive(Clone)]
pub struct Repository {
    pool: PgPool,
}

impl Repository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get cost attributions for a time range
    pub async fn get_attributions(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        team_id: Option<&str>,
        user_id: Option<&str>,
        customer_id: Option<&str>,
    ) -> Result<Vec<CostAttribution>> {
        let mut query = String::from(
            "SELECT * FROM cost_attributions WHERE period_start >= $1 AND period_end <= $2",
        );
        let mut param_num = 3;

        if team_id.is_some() {
            query.push_str(&format!(" AND team_id = ${}", param_num));
            param_num += 1;
        }
        if user_id.is_some() {
            query.push_str(&format!(" AND user_id = ${}", param_num));
            param_num += 1;
        }
        if customer_id.is_some() {
            query.push_str(&format!(" AND customer_id = ${}", param_num));
        }

        query.push_str(" ORDER BY period_start DESC LIMIT 1000");

        let mut q = sqlx::query_as::<_, CostAttribution>(&query)
            .bind(start)
            .bind(end);

        if let Some(tid) = team_id {
            q = q.bind(tid);
        }
        if let Some(uid) = user_id {
            q = q.bind(uid);
        }
        if let Some(cid) = customer_id {
            q = q.bind(cid);
        }

        let attributions = q.fetch_all(&self.pool).await?;
        Ok(attributions)
    }

    /// Get daily cost summaries
    pub async fn get_daily_summaries(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        team_id: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<Vec<DailyCostSummary>> {
        let mut query = String::from(
            "SELECT * FROM daily_cost_summary WHERE day >= $1 AND day <= $2",
        );
        let mut param_num = 3;

        if team_id.is_some() {
            query.push_str(&format!(" AND team_id = ${}", param_num));
            param_num += 1;
        }
        if user_id.is_some() {
            query.push_str(&format!(" AND user_id = ${}", param_num));
        }

        query.push_str(" ORDER BY day ASC");

        let mut q = sqlx::query_as::<_, DailyCostSummary>(&query)
            .bind(start)
            .bind(end);

        if let Some(tid) = team_id {
            q = q.bind(tid);
        }
        if let Some(uid) = user_id {
            q = q.bind(uid);
        }

        let summaries = q.fetch_all(&self.pool).await?;
        Ok(summaries)
    }

    /// Get monthly cost summaries
    pub async fn get_monthly_summaries(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        team_id: Option<&str>,
        user_id: Option<&str>,
        customer_id: Option<&str>,
    ) -> Result<Vec<MonthlyCostSummary>> {
        let mut query = String::from(
            "SELECT * FROM monthly_cost_summary WHERE month >= $1 AND month <= $2",
        );
        let mut param_num = 3;

        if team_id.is_some() {
            query.push_str(&format!(" AND team_id = ${}", param_num));
            param_num += 1;
        }
        if user_id.is_some() {
            query.push_str(&format!(" AND user_id = ${}", param_num));
            param_num += 1;
        }
        if customer_id.is_some() {
            query.push_str(&format!(" AND customer_id = ${}", param_num));
        }

        query.push_str(" ORDER BY month ASC");

        let mut q = sqlx::query_as::<_, MonthlyCostSummary>(&query)
            .bind(start)
            .bind(end);

        if let Some(tid) = team_id {
            q = q.bind(tid);
        }
        if let Some(uid) = user_id {
            q = q.bind(uid);
        }
        if let Some(cid) = customer_id {
            q = q.bind(cid);
        }

        let summaries = q.fetch_all(&self.pool).await?;
        Ok(summaries)
    }

    /// Get top spenders (users or teams)
    pub async fn get_top_spenders(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        entity_type: &str, // "user" or "team"
        limit: i64,
    ) -> Result<Vec<TopSpender>> {
        let group_by_field = match entity_type {
            "user" => "user_id",
            "team" => "team_id",
            _ => return Err(HpcError::invalid_query("Invalid entity type")),
        };

        let query = format!(
            r#"
            SELECT
                {} as entity_id,
                $3 as entity_type,
                SUM(total_cost) as total_cost,
                COUNT(DISTINCT job_id) as job_count
            FROM cost_attributions
            WHERE period_start >= $1 AND period_end <= $2
                AND {} IS NOT NULL
            GROUP BY {}
            ORDER BY total_cost DESC
            LIMIT $4
            "#,
            group_by_field, group_by_field, group_by_field
        );

        let spenders = sqlx::query_as::<_, TopSpender>(&query)
            .bind(start)
            .bind(end)
            .bind(entity_type)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;

        Ok(spenders)
    }

    /// Get attribution by ID
    pub async fn get_attribution_by_id(&self, id: Uuid) -> Result<CostAttribution> {
        let attribution =
            sqlx::query_as::<_, CostAttribution>("SELECT * FROM cost_attributions WHERE id = $1")
                .bind(id)
                .fetch_optional(&self.pool)
                .await?
                .ok_or_else(|| HpcError::report_not_found(format!("Attribution {}", id)))?;

        Ok(attribution)
    }
}

// Manual FromRow implementation for TopSpender
impl sqlx::FromRow<'_, sqlx::postgres::PgRow> for TopSpender {
    fn from_row(row: &sqlx::postgres::PgRow) -> sqlx::Result<Self> {
        use sqlx::Row;

        Ok(TopSpender {
            entity_id: row.try_get("entity_id")?,
            entity_type: row.try_get("entity_type")?,
            total_cost: row.try_get("total_cost")?,
            job_count: row.try_get("job_count")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires database
    async fn test_repository_creation() {
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost/horizon_cost_test".to_string());

        let pool = crate::db::create_pool(&database_url).await.unwrap();
        let repo = Repository::new(pool);

        // Just test construction
        let _ = repo;
    }
}
