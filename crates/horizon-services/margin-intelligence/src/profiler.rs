use crate::db::repository::MarginRepository;
use crate::error::{HpcError, Result, MarginErrorExt};
use crate::models::*;
use rust_decimal::Decimal;

#[derive(Clone)]
pub struct CustomerProfiler {
    repository: MarginRepository,
    ltv_months: i32,
}

impl CustomerProfiler {
    pub fn new(repository: MarginRepository, ltv_months: i32) -> Self {
        Self {
            repository,
            ltv_months,
        }
    }

    pub async fn refresh_profile(
        &self,
        customer_id: &str,
        revenue: Decimal,
        cost: Decimal,
    ) -> Result<CustomerProfile> {
        self.repository
            .update_profile_metrics(customer_id, revenue, cost)
            .await
    }

    pub async fn calculate_lifetime_value(
        &self,
        customer_id: &str,
    ) -> Result<Decimal> {
        let profile = self.repository.get_profile(customer_id).await?;

        // Simple LTV: monthly contribution margin * ltv_months
        let monthly_margin = if let Some(cm) = profile.contribution_margin {
            cm / Decimal::from(12) // Assume annual data
        } else {
            return Err(HpcError::invalid_calculation(
                "No contribution margin data",
            ));
        };

        let ltv = monthly_margin * Decimal::from(self.ltv_months);
        self.repository
            .update_lifetime_value(customer_id, ltv)
            .await?;

        Ok(ltv)
    }

    pub async fn get_segment_profiles(
        &self,
        segment: CustomerSegment,
    ) -> Result<Vec<CustomerProfile>> {
        self.repository
            .get_profiles_by_segment(&segment.to_string())
            .await
    }

    pub async fn identify_at_risk(
        &self,
        threshold: Decimal,
    ) -> Result<Vec<CustomerProfile>> {
        self.repository.get_at_risk_customers(threshold).await
    }

    pub async fn identify_top_contributors(&self, limit: i64) -> Result<Vec<TopContributor>> {
        let profiles = self.repository.get_top_contributors(limit).await?;

        Ok(profiles
            .into_iter()
            .map(|p| TopContributor {
                customer_id: p.customer_id,
                segment: p.segment,
                contribution_margin: p.contribution_margin.unwrap_or(Decimal::ZERO),
                gross_margin: p.gross_margin,
                revenue: p.total_revenue,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::pool::create_pool;
    use crate::config::DatabaseConfig;
    use rust_decimal_macros::dec;
    use sqlx::PgPool;

    async fn setup_test_repo() -> MarginRepository {
        let config = DatabaseConfig {
            url: std::env::var("TEST_DATABASE_URL")
                .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/margin_test".to_string()),
            max_connections: 5,
        };

        let pool = create_pool(&config).await.unwrap();
        sqlx::query("TRUNCATE customer_profiles, pricing_simulations, margin_recommendations CASCADE")
            .execute(&pool)
            .await
            .ok();

        MarginRepository::new(pool)
    }

    #[tokio::test]
    async fn test_calculate_lifetime_value() {
        let repo = setup_test_repo().await;
        let profiler = CustomerProfiler::new(repo.clone(), 12);

        // Create profile
        let request = CreateProfileRequest {
            customer_id: "ltv-test".to_string(),
            segment: CustomerSegment::Enterprise,
        };
        repo.create_profile(&request).await.unwrap();
        repo.update_profile_metrics("ltv-test", dec!(12000), dec!(6000))
            .await
            .unwrap();

        // Calculate LTV
        let ltv = profiler.calculate_lifetime_value("ltv-test").await.unwrap();

        // LTV = (12000-6000)/12 * 12 = 6000
        assert_eq!(ltv, dec!(6000));
    }

    #[tokio::test]
    async fn test_identify_at_risk() {
        let repo = setup_test_repo().await;
        let profiler = CustomerProfiler::new(repo.clone(), 12);

        // Create at-risk profile
        let request = CreateProfileRequest {
            customer_id: "risk-test".to_string(),
            segment: CustomerSegment::Startup,
        };
        repo.create_profile(&request).await.unwrap();
        repo.update_profile_metrics("risk-test", dec!(1000), dec!(950))
            .await
            .unwrap();

        let at_risk = profiler.identify_at_risk(dec!(10.0)).await.unwrap();
        assert!(at_risk.len() > 0);
    }
}
