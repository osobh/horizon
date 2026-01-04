use crate::error::{HpcError, MarginErrorExt, Result};
use crate::models::*;
use chrono::Utc;
use rust_decimal::Decimal;
use sqlx::{PgPool, Row};
use uuid::Uuid;

#[derive(Clone)]
pub struct MarginRepository {
    pool: PgPool,
}

impl MarginRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    // Customer Profile operations

    pub async fn create_profile(&self, request: &CreateProfileRequest) -> Result<CustomerProfile> {
        let segment_str = request.segment.to_string();

        let profile = sqlx::query_as::<_, CustomerProfile>(
            r#"
            INSERT INTO customer_profiles (customer_id, segment, total_revenue, total_cost)
            VALUES ($1, $2, 0, 0)
            RETURNING *
            "#,
        )
        .bind(&request.customer_id)
        .bind(&segment_str)
        .fetch_one(&self.pool)
        .await?;

        Ok(profile)
    }

    pub async fn get_profile(&self, customer_id: &str) -> Result<CustomerProfile> {
        let profile = sqlx::query_as::<_, CustomerProfile>(
            "SELECT * FROM customer_profiles WHERE customer_id = $1",
        )
        .bind(customer_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| HpcError::profile_not_found(customer_id))?;

        Ok(profile)
    }

    pub async fn update_profile_metrics(
        &self,
        customer_id: &str,
        revenue: Decimal,
        cost: Decimal,
    ) -> Result<CustomerProfile> {
        // Calculate margins
        let gross_margin = if !revenue.is_zero() {
            Some((revenue - cost) / revenue * Decimal::from(100))
        } else {
            None
        };

        let contribution_margin = revenue - cost;

        let profile = sqlx::query_as::<_, CustomerProfile>(
            r#"
            UPDATE customer_profiles
            SET total_revenue = $2,
                total_cost = $3,
                gross_margin = $4,
                contribution_margin = $5,
                last_usage_at = $6,
                updated_at = $6
            WHERE customer_id = $1
            RETURNING *
            "#,
        )
        .bind(customer_id)
        .bind(revenue)
        .bind(cost)
        .bind(gross_margin)
        .bind(contribution_margin)
        .bind(Utc::now())
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| HpcError::profile_not_found(customer_id))?;

        Ok(profile)
    }

    pub async fn update_lifetime_value(
        &self,
        customer_id: &str,
        ltv: Decimal,
    ) -> Result<CustomerProfile> {
        let profile = sqlx::query_as::<_, CustomerProfile>(
            r#"
            UPDATE customer_profiles
            SET lifetime_value = $2,
                updated_at = $3
            WHERE customer_id = $1
            RETURNING *
            "#,
        )
        .bind(customer_id)
        .bind(ltv)
        .bind(Utc::now())
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| HpcError::profile_not_found(customer_id))?;

        Ok(profile)
    }

    pub async fn list_profiles(&self, limit: i64, offset: i64) -> Result<Vec<CustomerProfile>> {
        let profiles = sqlx::query_as::<_, CustomerProfile>(
            "SELECT * FROM customer_profiles ORDER BY updated_at DESC LIMIT $1 OFFSET $2",
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        Ok(profiles)
    }

    pub async fn get_profiles_by_segment(&self, segment: &str) -> Result<Vec<CustomerProfile>> {
        let profiles = sqlx::query_as::<_, CustomerProfile>(
            "SELECT * FROM customer_profiles WHERE segment = $1 ORDER BY gross_margin DESC",
        )
        .bind(segment)
        .fetch_all(&self.pool)
        .await?;

        Ok(profiles)
    }

    pub async fn get_top_contributors(&self, limit: i64) -> Result<Vec<CustomerProfile>> {
        let profiles = sqlx::query_as::<_, CustomerProfile>(
            r#"
            SELECT * FROM customer_profiles
            WHERE contribution_margin IS NOT NULL
            ORDER BY contribution_margin DESC
            LIMIT $1
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(profiles)
    }

    pub async fn get_at_risk_customers(&self, threshold: Decimal) -> Result<Vec<CustomerProfile>> {
        let profiles = sqlx::query_as::<_, CustomerProfile>(
            r#"
            SELECT * FROM customer_profiles
            WHERE gross_margin < $1 OR gross_margin IS NULL
            ORDER BY gross_margin ASC NULLS FIRST
            "#,
        )
        .bind(threshold)
        .fetch_all(&self.pool)
        .await?;

        Ok(profiles)
    }

    // Margin analysis operations

    pub async fn get_margin_analysis(&self) -> Result<MarginAnalysis> {
        let row = sqlx::query(
            r#"
            SELECT
                COUNT(*) as total_customers,
                AVG(gross_margin) as avg_gross_margin,
                AVG(contribution_margin) as avg_contribution_margin,
                COUNT(*) FILTER (WHERE total_revenue > total_cost) as profitable_count,
                COUNT(*) FILTER (WHERE gross_margin < 10.0 OR gross_margin IS NULL) as at_risk_count,
                COALESCE(SUM(total_revenue), 0) as total_revenue,
                COALESCE(SUM(total_cost), 0) as total_cost
            FROM customer_profiles
            "#
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(MarginAnalysis {
            total_customers: row
                .try_get::<Option<i64>, _>("total_customers")?
                .unwrap_or(0),
            avg_gross_margin: row.try_get::<Option<Decimal>, _>("avg_gross_margin")?,
            avg_contribution_margin: row
                .try_get::<Option<Decimal>, _>("avg_contribution_margin")?,
            profitable_count: row
                .try_get::<Option<i64>, _>("profitable_count")?
                .unwrap_or(0),
            at_risk_count: row.try_get::<Option<i64>, _>("at_risk_count")?.unwrap_or(0),
            total_revenue: row
                .try_get::<Option<Decimal>, _>("total_revenue")?
                .unwrap_or(Decimal::ZERO),
            total_cost: row
                .try_get::<Option<Decimal>, _>("total_cost")?
                .unwrap_or(Decimal::ZERO),
        })
    }

    pub async fn get_segment_analysis(&self) -> Result<Vec<SegmentAnalysis>> {
        let rows = sqlx::query(
            r#"
            SELECT
                segment,
                COUNT(*) as customer_count,
                AVG(gross_margin) as avg_margin,
                COALESCE(SUM(total_revenue), 0) as total_revenue,
                COALESCE(SUM(total_cost), 0) as total_cost
            FROM customer_profiles
            GROUP BY segment
            ORDER BY segment
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|r| SegmentAnalysis {
                segment: r.try_get("segment").unwrap(),
                customer_count: r
                    .try_get::<Option<i64>, _>("customer_count")
                    .unwrap()
                    .unwrap_or(0),
                avg_margin: r.try_get::<Option<Decimal>, _>("avg_margin").unwrap(),
                total_revenue: r
                    .try_get::<Option<Decimal>, _>("total_revenue")
                    .unwrap()
                    .unwrap_or(Decimal::ZERO),
                total_cost: r
                    .try_get::<Option<Decimal>, _>("total_cost")
                    .unwrap()
                    .unwrap_or(Decimal::ZERO),
            })
            .collect())
    }

    // Pricing simulation operations

    pub async fn create_simulation(
        &self,
        request: &CreateSimulationRequest,
    ) -> Result<PricingSimulation> {
        // Calculate margin impact
        let price_change =
            (request.simulated_price - request.current_price) / request.current_price;
        let margin_impact = price_change * Decimal::from(100);

        let simulation = sqlx::query_as::<_, PricingSimulation>(
            r#"
            INSERT INTO pricing_simulations (
                customer_id, scenario_name, current_price, simulated_price,
                estimated_margin_impact, elasticity_factor
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
            "#,
        )
        .bind(&request.customer_id)
        .bind(&request.scenario_name)
        .bind(request.current_price)
        .bind(request.simulated_price)
        .bind(margin_impact)
        .bind(request.elasticity_factor)
        .fetch_one(&self.pool)
        .await?;

        Ok(simulation)
    }

    pub async fn get_simulation(&self, id: Uuid) -> Result<PricingSimulation> {
        let simulation = sqlx::query_as::<_, PricingSimulation>(
            "SELECT * FROM pricing_simulations WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| HpcError::simulation_not_found(id.to_string()))?;

        Ok(simulation)
    }

    pub async fn list_simulations_for_customer(
        &self,
        customer_id: &str,
    ) -> Result<Vec<PricingSimulation>> {
        let simulations = sqlx::query_as::<_, PricingSimulation>(
            "SELECT * FROM pricing_simulations WHERE customer_id = $1 ORDER BY created_at DESC",
        )
        .bind(customer_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(simulations)
    }

    // Recommendation operations

    pub async fn create_recommendation(
        &self,
        customer_id: &str,
        rec_type: &str,
        current_margin: Decimal,
        projected_margin: Decimal,
        impact_amount: Decimal,
        confidence: Decimal,
    ) -> Result<MarginRecommendation> {
        let rec = sqlx::query_as::<_, MarginRecommendation>(
            r#"
            INSERT INTO margin_recommendations (
                customer_id, recommendation_type, current_margin,
                projected_margin, impact_amount, confidence
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
            "#,
        )
        .bind(customer_id)
        .bind(rec_type)
        .bind(current_margin)
        .bind(projected_margin)
        .bind(impact_amount)
        .bind(confidence)
        .fetch_one(&self.pool)
        .await?;

        Ok(rec)
    }

    pub async fn get_recommendations_for_customer(
        &self,
        customer_id: &str,
    ) -> Result<Vec<MarginRecommendation>> {
        let recs = sqlx::query_as::<_, MarginRecommendation>(
            r#"
            SELECT * FROM margin_recommendations
            WHERE customer_id = $1
            ORDER BY created_at DESC
            "#,
        )
        .bind(customer_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(recs)
    }

    pub async fn update_recommendation_status(
        &self,
        id: Uuid,
        status: &str,
    ) -> Result<MarginRecommendation> {
        let rec = sqlx::query_as::<_, MarginRecommendation>(
            "UPDATE margin_recommendations SET status = $2 WHERE id = $1 RETURNING *",
        )
        .bind(id)
        .bind(status)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| HpcError::simulation_not_found(id.to_string()))?;

        Ok(rec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    async fn setup_test_db() -> PgPool {
        let database_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/margin_test".to_string());

        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        sqlx::query(
            "TRUNCATE customer_profiles, pricing_simulations, margin_recommendations CASCADE",
        )
        .execute(&pool)
        .await
        .expect("Failed to truncate tables");

        pool
    }

    #[tokio::test]
    async fn test_create_and_get_profile() {
        let pool = setup_test_db().await;
        let repo = MarginRepository::new(pool);

        let request = CreateProfileRequest {
            customer_id: "test-customer-001".to_string(),
            segment: CustomerSegment::Enterprise,
        };

        let created = repo.create_profile(&request).await.unwrap();
        assert_eq!(created.customer_id, "test-customer-001");
        assert_eq!(created.segment, "enterprise");

        let retrieved = repo.get_profile("test-customer-001").await.unwrap();
        assert_eq!(retrieved.customer_id, created.customer_id);
    }

    #[tokio::test]
    async fn test_update_profile_metrics() {
        let pool = setup_test_db().await;
        let repo = MarginRepository::new(pool);

        let request = CreateProfileRequest {
            customer_id: "test-customer-002".to_string(),
            segment: CustomerSegment::Growth,
        };

        repo.create_profile(&request).await.unwrap();

        let updated = repo
            .update_profile_metrics("test-customer-002", dec!(10000), dec!(7000))
            .await
            .unwrap();

        assert_eq!(updated.total_revenue, dec!(10000));
        assert_eq!(updated.total_cost, dec!(7000));
        assert_eq!(updated.gross_margin.unwrap(), dec!(30.00));
        assert_eq!(updated.contribution_margin.unwrap(), dec!(3000));
    }

    #[tokio::test]
    async fn test_get_margin_analysis() {
        let pool = setup_test_db().await;
        let repo = MarginRepository::new(pool);

        // Create test profiles
        for i in 1..=3 {
            let request = CreateProfileRequest {
                customer_id: format!("customer-{:03}", i),
                segment: CustomerSegment::Enterprise,
            };
            repo.create_profile(&request).await.unwrap();

            repo.update_profile_metrics(
                &format!("customer-{:03}", i),
                dec!(10000) * Decimal::from(i),
                dec!(6000) * Decimal::from(i),
            )
            .await
            .unwrap();
        }

        let analysis = repo.get_margin_analysis().await.unwrap();
        assert_eq!(analysis.total_customers, 3);
        assert!(analysis.avg_gross_margin.unwrap() > Decimal::ZERO);
        assert_eq!(analysis.profitable_count, 3);
    }

    #[tokio::test]
    async fn test_create_and_get_simulation() {
        let pool = setup_test_db().await;
        let repo = MarginRepository::new(pool);

        let request = CreateSimulationRequest {
            customer_id: "customer-sim".to_string(),
            scenario_name: "10% increase".to_string(),
            current_price: dec!(100),
            simulated_price: dec!(110),
            elasticity_factor: dec!(-0.3),
        };

        let created = repo.create_simulation(&request).await.unwrap();
        assert_eq!(created.customer_id, "customer-sim");
        assert_eq!(created.current_price.unwrap(), dec!(100));
        assert_eq!(created.simulated_price.unwrap(), dec!(110));

        let retrieved = repo.get_simulation(created.id).await.unwrap();
        assert_eq!(retrieved.id, created.id);
    }

    #[tokio::test]
    async fn test_create_recommendation() {
        let pool = setup_test_db().await;
        let repo = MarginRepository::new(pool);

        let rec = repo
            .create_recommendation(
                "customer-rec",
                "price_increase",
                dec!(20.00),
                dec!(25.00),
                dec!(500),
                dec!(0.85),
            )
            .await
            .unwrap();

        assert_eq!(rec.customer_id, "customer-rec");
        assert_eq!(rec.recommendation_type, "price_increase");
        assert_eq!(rec.current_margin.unwrap(), dec!(20.00));
    }

    #[tokio::test]
    async fn test_get_top_contributors() {
        let pool = setup_test_db().await;
        let repo = MarginRepository::new(pool);

        // Create profiles with different margins
        for i in 1..=5 {
            let request = CreateProfileRequest {
                customer_id: format!("top-{:03}", i),
                segment: CustomerSegment::Enterprise,
            };
            repo.create_profile(&request).await.unwrap();

            repo.update_profile_metrics(
                &format!("top-{:03}", i),
                dec!(10000) * Decimal::from(i),
                dec!(5000) * Decimal::from(i),
            )
            .await
            .unwrap();
        }

        let top = repo.get_top_contributors(3).await.unwrap();
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].customer_id, "top-005"); // Highest contributor
    }

    #[tokio::test]
    async fn test_get_at_risk_customers() {
        let pool = setup_test_db().await;
        let repo = MarginRepository::new(pool);

        // Create at-risk profile
        let request = CreateProfileRequest {
            customer_id: "at-risk-001".to_string(),
            segment: CustomerSegment::Startup,
        };
        repo.create_profile(&request).await.unwrap();
        repo.update_profile_metrics("at-risk-001", dec!(1000), dec!(950))
            .await
            .unwrap();

        // Create healthy profile
        let request = CreateProfileRequest {
            customer_id: "healthy-001".to_string(),
            segment: CustomerSegment::Enterprise,
        };
        repo.create_profile(&request).await.unwrap();
        repo.update_profile_metrics("healthy-001", dec!(10000), dec!(5000))
            .await
            .unwrap();

        let at_risk = repo.get_at_risk_customers(dec!(10.0)).await.unwrap();
        assert_eq!(at_risk.len(), 1);
        assert_eq!(at_risk[0].customer_id, "at-risk-001");
    }
}
