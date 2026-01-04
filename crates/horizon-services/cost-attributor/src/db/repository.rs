use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

use crate::models::{
    CostAttribution, CostAttributionQuery, CostRollup, CreateCostAttribution, CreateGpuPricing,
    GpuPricing, GpuPricingQuery, PricingModel, UpdateGpuPricing,
};

pub struct Repository {
    pool: PgPool,
}

impl Repository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    // Cost Attribution Operations

    pub async fn create_attribution(
        &self,
        attribution: &CreateCostAttribution,
    ) -> crate::error::Result<CostAttribution> {
        attribution.validate()?;

        let rec = sqlx::query_as::<_, CostAttribution>(
            r#"
            INSERT INTO cost_attributions (
                job_id, user_id, team_id, customer_id,
                gpu_cost, cpu_cost, network_cost, storage_cost, total_cost,
                period_start, period_end
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING *
            "#,
        )
        .bind(attribution.job_id)
        .bind(&attribution.user_id)
        .bind(&attribution.team_id)
        .bind(&attribution.customer_id)
        .bind(attribution.gpu_cost)
        .bind(attribution.cpu_cost)
        .bind(attribution.network_cost)
        .bind(attribution.storage_cost)
        .bind(attribution.total_cost)
        .bind(attribution.period_start)
        .bind(attribution.period_end)
        .fetch_one(&self.pool)
        .await?;

        Ok(rec)
    }

    pub async fn get_attribution(&self, id: Uuid) -> crate::error::Result<CostAttribution> {
        use crate::error::AttributorErrorExt;

        let rec =
            sqlx::query_as::<_, CostAttribution>("SELECT * FROM cost_attributions WHERE id = $1")
                .bind(id)
                .fetch_optional(&self.pool)
                .await?
                .ok_or_else(|| crate::error::HpcError::attribution_not_found(id.to_string()))?;

        Ok(rec)
    }

    pub async fn query_attributions(
        &self,
        query: &CostAttributionQuery,
    ) -> crate::error::Result<Vec<CostAttribution>> {
        let mut sql = String::from("SELECT * FROM cost_attributions WHERE 1=1");
        let mut bind_count = 0;

        if query.job_id.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND job_id = ${}", bind_count));
        }
        if query.user_id.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND user_id = ${}", bind_count));
        }
        if query.team_id.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND team_id = ${}", bind_count));
        }
        if query.customer_id.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND customer_id = ${}", bind_count));
        }
        if query.start_date.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND period_start >= ${}", bind_count));
        }
        if query.end_date.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND period_end <= ${}", bind_count));
        }

        sql.push_str(" ORDER BY created_at DESC");

        if query.limit.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" LIMIT ${}", bind_count));
        }
        if query.offset.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" OFFSET ${}", bind_count));
        }

        let mut q = sqlx::query_as::<_, CostAttribution>(&sql);

        if let Some(job_id) = query.job_id {
            q = q.bind(job_id);
        }
        if let Some(ref user_id) = query.user_id {
            q = q.bind(user_id);
        }
        if let Some(ref team_id) = query.team_id {
            q = q.bind(team_id);
        }
        if let Some(ref customer_id) = query.customer_id {
            q = q.bind(customer_id);
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

        let recs = q.fetch_all(&self.pool).await?;
        Ok(recs)
    }

    // Rollup Queries

    pub async fn rollup_by_user(
        &self,
        user_id: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> crate::error::Result<CostRollup> {
        #[derive(sqlx::FromRow)]
        struct RollupRow {
            entity_id: String,
            total_gpu_cost: Option<rust_decimal::Decimal>,
            total_cpu_cost: Option<rust_decimal::Decimal>,
            total_network_cost: Option<rust_decimal::Decimal>,
            total_storage_cost: Option<rust_decimal::Decimal>,
            total_cost: Option<rust_decimal::Decimal>,
            job_count: Option<i64>,
        }

        let rec = sqlx::query_as::<_, RollupRow>(
            r#"
            SELECT
                user_id as entity_id,
                SUM(gpu_cost) as total_gpu_cost,
                SUM(cpu_cost) as total_cpu_cost,
                SUM(network_cost) as total_network_cost,
                SUM(storage_cost) as total_storage_cost,
                SUM(total_cost) as total_cost,
                COUNT(*) as job_count
            FROM cost_attributions
            WHERE user_id = $1
              AND period_start >= $2
              AND period_end <= $3
            GROUP BY user_id
            "#,
        )
        .bind(user_id)
        .bind(start_date)
        .bind(end_date)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| {
            use crate::error::AttributorErrorExt;
            crate::error::HpcError::attribution_not_found(format!(
                "No attributions found for user {}",
                user_id
            ))
        })?;

        Ok(CostRollup {
            entity_id: rec.entity_id,
            entity_type: "user".to_string(),
            total_gpu_cost: rec.total_gpu_cost.unwrap_or_default(),
            total_cpu_cost: rec.total_cpu_cost.unwrap_or_default(),
            total_network_cost: rec.total_network_cost.unwrap_or_default(),
            total_storage_cost: rec.total_storage_cost.unwrap_or_default(),
            total_cost: rec.total_cost.unwrap_or_default(),
            job_count: rec.job_count.unwrap_or(0),
            period_start: start_date,
            period_end: end_date,
        })
    }

    pub async fn rollup_by_team(
        &self,
        team_id: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> crate::error::Result<CostRollup> {
        #[derive(sqlx::FromRow)]
        struct RollupRow {
            entity_id: Option<String>,
            total_gpu_cost: Option<rust_decimal::Decimal>,
            total_cpu_cost: Option<rust_decimal::Decimal>,
            total_network_cost: Option<rust_decimal::Decimal>,
            total_storage_cost: Option<rust_decimal::Decimal>,
            total_cost: Option<rust_decimal::Decimal>,
            job_count: Option<i64>,
        }

        let rec = sqlx::query_as::<_, RollupRow>(
            r#"
            SELECT
                team_id as entity_id,
                SUM(gpu_cost) as total_gpu_cost,
                SUM(cpu_cost) as total_cpu_cost,
                SUM(network_cost) as total_network_cost,
                SUM(storage_cost) as total_storage_cost,
                SUM(total_cost) as total_cost,
                COUNT(*) as job_count
            FROM cost_attributions
            WHERE team_id = $1
              AND period_start >= $2
              AND period_end <= $3
            GROUP BY team_id
            "#,
        )
        .bind(team_id)
        .bind(start_date)
        .bind(end_date)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| {
            use crate::error::AttributorErrorExt;
            crate::error::HpcError::attribution_not_found(format!(
                "No attributions found for team {}",
                team_id
            ))
        })?;

        Ok(CostRollup {
            entity_id: rec.entity_id.unwrap_or_default(),
            entity_type: "team".to_string(),
            total_gpu_cost: rec.total_gpu_cost.unwrap_or_default(),
            total_cpu_cost: rec.total_cpu_cost.unwrap_or_default(),
            total_network_cost: rec.total_network_cost.unwrap_or_default(),
            total_storage_cost: rec.total_storage_cost.unwrap_or_default(),
            total_cost: rec.total_cost.unwrap_or_default(),
            job_count: rec.job_count.unwrap_or(0),
            period_start: start_date,
            period_end: end_date,
        })
    }

    // GPU Pricing Operations

    pub async fn create_pricing(
        &self,
        pricing: &CreateGpuPricing,
    ) -> crate::error::Result<GpuPricing> {
        pricing.validate()?;

        let rec = sqlx::query_as::<_, GpuPricing>(
            r#"
            INSERT INTO gpu_pricing (
                gpu_type, region, pricing_model, hourly_rate, effective_start, effective_end
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
            "#,
        )
        .bind(&pricing.gpu_type)
        .bind(&pricing.region)
        .bind(pricing.pricing_model.to_string())
        .bind(pricing.hourly_rate)
        .bind(pricing.effective_start)
        .bind(pricing.effective_end)
        .fetch_one(&self.pool)
        .await?;

        Ok(rec)
    }

    pub async fn get_pricing(&self, id: Uuid) -> crate::error::Result<GpuPricing> {
        use crate::error::AttributorErrorExt;

        let rec = sqlx::query_as::<_, GpuPricing>("SELECT * FROM gpu_pricing WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await?
            .ok_or_else(|| crate::error::HpcError::pricing_not_found(id.to_string()))?;

        Ok(rec)
    }

    pub async fn get_current_pricing(
        &self,
        gpu_type: &str,
        pricing_model: PricingModel,
        at_time: DateTime<Utc>,
    ) -> crate::error::Result<GpuPricing> {
        let rec = sqlx::query_as::<_, GpuPricing>(
            r#"
            SELECT * FROM gpu_pricing
            WHERE gpu_type = $1
              AND pricing_model = $2
              AND effective_start <= $3
              AND (effective_end IS NULL OR effective_end > $3)
            ORDER BY effective_start DESC
            LIMIT 1
            "#,
        )
        .bind(gpu_type)
        .bind(pricing_model.to_string())
        .bind(at_time)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| {
            use crate::error::AttributorErrorExt;
            crate::error::HpcError::pricing_not_found(format!("{} ({})", gpu_type, pricing_model))
        })?;

        Ok(rec)
    }

    pub async fn query_pricing(
        &self,
        query: &GpuPricingQuery,
    ) -> crate::error::Result<Vec<GpuPricing>> {
        let mut sql = String::from("SELECT * FROM gpu_pricing WHERE 1=1");
        let mut bind_count = 0;

        if query.gpu_type.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND gpu_type = ${}", bind_count));
        }
        if query.region.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND region = ${}", bind_count));
        }
        if query.pricing_model.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND pricing_model = ${}", bind_count));
        }
        if query.effective_at.is_some() {
            bind_count += 1;
            sql.push_str(&format!(" AND effective_start <= ${}", bind_count));
            bind_count += 1;
            sql.push_str(&format!(
                " AND (effective_end IS NULL OR effective_end > ${})",
                bind_count - 1
            ));
        }

        sql.push_str(" ORDER BY effective_start DESC");

        let mut q = sqlx::query_as::<_, GpuPricing>(&sql);

        if let Some(ref gpu_type) = query.gpu_type {
            q = q.bind(gpu_type);
        }
        if let Some(ref region) = query.region {
            q = q.bind(region);
        }
        if let Some(pricing_model) = query.pricing_model {
            q = q.bind(pricing_model.to_string());
        }
        if let Some(effective_at) = query.effective_at {
            q = q.bind(effective_at);
        }

        let recs = q.fetch_all(&self.pool).await?;
        Ok(recs)
    }

    pub async fn update_pricing(
        &self,
        id: Uuid,
        update: &UpdateGpuPricing,
    ) -> crate::error::Result<GpuPricing> {
        let mut sql = String::from("UPDATE gpu_pricing SET ");
        let mut updates = Vec::new();
        let mut bind_count = 0;

        if update.hourly_rate.is_some() {
            bind_count += 1;
            updates.push(format!("hourly_rate = ${}", bind_count));
        }
        if update.effective_end.is_some() {
            bind_count += 1;
            updates.push(format!("effective_end = ${}", bind_count));
        }

        if updates.is_empty() {
            return self.get_pricing(id).await;
        }

        sql.push_str(&updates.join(", "));
        bind_count += 1;
        sql.push_str(&format!(" WHERE id = ${} RETURNING *", bind_count));

        let mut q = sqlx::query_as::<_, GpuPricing>(&sql);

        if let Some(hourly_rate) = update.hourly_rate {
            q = q.bind(hourly_rate);
        }
        if let Some(effective_end) = update.effective_end {
            q = q.bind(effective_end);
        }
        q = q.bind(id);

        let rec = q.fetch_one(&self.pool).await?;
        Ok(rec)
    }
}

#[cfg(test)]
mod tests {
    // Note: These tests require a running PostgreSQL database
    // They are marked as #[ignore] and should be run with: cargo test -- --ignored

    #[test]
    fn test_repository_compiles() {
        // This test just ensures the module compiles correctly
        assert!(true);
    }
}
