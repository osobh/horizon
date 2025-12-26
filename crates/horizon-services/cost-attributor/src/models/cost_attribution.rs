use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct CostAttribution {
    pub id: Uuid,
    pub job_id: Option<Uuid>,
    pub user_id: String,
    pub team_id: Option<String>,
    pub customer_id: Option<String>,
    pub gpu_cost: Decimal,
    pub cpu_cost: Decimal,
    pub network_cost: Decimal,
    pub storage_cost: Decimal,
    pub total_cost: Decimal,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCostAttribution {
    pub job_id: Option<Uuid>,
    pub user_id: String,
    pub team_id: Option<String>,
    pub customer_id: Option<String>,
    pub gpu_cost: Decimal,
    pub cpu_cost: Decimal,
    pub network_cost: Decimal,
    pub storage_cost: Decimal,
    pub total_cost: Decimal,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
}

impl CreateCostAttribution {
    pub fn new(
        user_id: String,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Self {
        Self {
            job_id: None,
            user_id,
            team_id: None,
            customer_id: None,
            gpu_cost: Decimal::ZERO,
            cpu_cost: Decimal::ZERO,
            network_cost: Decimal::ZERO,
            storage_cost: Decimal::ZERO,
            total_cost: Decimal::ZERO,
            period_start,
            period_end,
        }
    }

    pub fn with_job_id(mut self, job_id: Uuid) -> Self {
        self.job_id = Some(job_id);
        self
    }

    pub fn with_team_id(mut self, team_id: String) -> Self {
        self.team_id = Some(team_id);
        self
    }

    pub fn with_customer_id(mut self, customer_id: String) -> Self {
        self.customer_id = Some(customer_id);
        self
    }

    pub fn with_gpu_cost(mut self, gpu_cost: Decimal) -> Self {
        self.gpu_cost = gpu_cost;
        self.recalculate_total();
        self
    }

    pub fn with_cpu_cost(mut self, cpu_cost: Decimal) -> Self {
        self.cpu_cost = cpu_cost;
        self.recalculate_total();
        self
    }

    pub fn with_network_cost(mut self, network_cost: Decimal) -> Self {
        self.network_cost = network_cost;
        self.recalculate_total();
        self
    }

    pub fn with_storage_cost(mut self, storage_cost: Decimal) -> Self {
        self.storage_cost = storage_cost;
        self.recalculate_total();
        self
    }

    fn recalculate_total(&mut self) {
        self.total_cost = self.gpu_cost + self.cpu_cost + self.network_cost + self.storage_cost;
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::AttributorErrorExt;

        if self.user_id.is_empty() {
            return Err(crate::error::HpcError::invalid_attribution_data(
                "User ID cannot be empty",
            ));
        }

        if self.gpu_cost < Decimal::ZERO {
            return Err(crate::error::HpcError::invalid_attribution_data(
                "GPU cost cannot be negative",
            ));
        }

        if self.cpu_cost < Decimal::ZERO {
            return Err(crate::error::HpcError::invalid_attribution_data(
                "CPU cost cannot be negative",
            ));
        }

        if self.network_cost < Decimal::ZERO {
            return Err(crate::error::HpcError::invalid_attribution_data(
                "Network cost cannot be negative",
            ));
        }

        if self.storage_cost < Decimal::ZERO {
            return Err(crate::error::HpcError::invalid_attribution_data(
                "Storage cost cannot be negative",
            ));
        }

        if self.total_cost < Decimal::ZERO {
            return Err(crate::error::HpcError::invalid_attribution_data(
                "Total cost cannot be negative",
            ));
        }

        if self.period_end <= self.period_start {
            return Err(crate::error::HpcError::invalid_time_range(
                self.period_start.to_rfc3339(),
                self.period_end.to_rfc3339(),
            ));
        }

        // Verify total is sum of components
        let calculated_total = self.gpu_cost + self.cpu_cost + self.network_cost + self.storage_cost;
        if self.total_cost != calculated_total {
            return Err(crate::error::HpcError::invalid_attribution_data(
                format!(
                    "Total cost {} does not match sum of components {}",
                    self.total_cost, calculated_total
                ),
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAttributionQuery {
    pub job_id: Option<Uuid>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub customer_id: Option<String>,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

impl Default for CostAttributionQuery {
    fn default() -> Self {
        Self {
            job_id: None,
            user_id: None,
            team_id: None,
            customer_id: None,
            start_date: None,
            end_date: None,
            limit: Some(100),
            offset: Some(0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRollup {
    pub entity_id: String,
    pub entity_type: String, // user, team, customer
    pub total_gpu_cost: Decimal,
    pub total_cpu_cost: Decimal,
    pub total_network_cost: Decimal,
    pub total_storage_cost: Decimal,
    pub total_cost: Decimal,
    pub job_count: i64,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_create_cost_attribution_builder() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);

        let attribution = CreateCostAttribution::new("user123".to_string(), now, later)
            .with_job_id(Uuid::new_v4())
            .with_team_id("team456".to_string())
            .with_gpu_cost(dec!(100.50))
            .with_network_cost(dec!(5.25));

        assert_eq!(attribution.user_id, "user123");
        assert!(attribution.job_id.is_some());
        assert_eq!(attribution.gpu_cost, dec!(100.50));
        assert_eq!(attribution.network_cost, dec!(5.25));
        assert_eq!(attribution.total_cost, dec!(105.75));
    }

    #[test]
    fn test_recalculate_total() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);

        let attribution = CreateCostAttribution::new("user123".to_string(), now, later)
            .with_gpu_cost(dec!(100.00))
            .with_cpu_cost(dec!(20.00))
            .with_network_cost(dec!(10.00))
            .with_storage_cost(dec!(5.00));

        assert_eq!(attribution.total_cost, dec!(135.00));
    }

    #[test]
    fn test_validate_success() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);

        let attribution = CreateCostAttribution::new("user123".to_string(), now, later)
            .with_gpu_cost(dec!(100.00));

        assert!(attribution.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_user_id() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);

        let attribution = CreateCostAttribution::new(String::new(), now, later);
        let result = attribution.validate();

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("User ID cannot be empty"));
    }

    #[test]
    fn test_validate_negative_gpu_cost() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);

        let mut attribution = CreateCostAttribution::new("user123".to_string(), now, later);
        attribution.gpu_cost = dec!(-10.00);
        attribution.total_cost = dec!(-10.00);

        let result = attribution.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("GPU cost cannot be negative"));
    }

    #[test]
    fn test_validate_invalid_time_range() {
        let now = Utc::now();
        let earlier = now - chrono::Duration::hours(1);

        let attribution = CreateCostAttribution::new("user123".to_string(), now, earlier);
        let result = attribution.validate();

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid time range"));
    }

    #[test]
    fn test_validate_total_mismatch() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);

        let mut attribution = CreateCostAttribution::new("user123".to_string(), now, later);
        attribution.gpu_cost = dec!(100.00);
        attribution.total_cost = dec!(50.00); // Intentionally wrong

        let result = attribution.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Total cost"));
    }

    #[test]
    fn test_cost_attribution_query_default() {
        let query = CostAttributionQuery::default();
        assert_eq!(query.limit, Some(100));
        assert_eq!(query.offset, Some(0));
        assert!(query.user_id.is_none());
    }

    #[test]
    fn test_serialization() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);

        let attribution = CreateCostAttribution::new("user123".to_string(), now, later)
            .with_gpu_cost(dec!(100.50));

        let json = serde_json::to_string(&attribution).unwrap();
        assert!(json.contains("user123"));
        // rust_decimal serializes as "100.5" not "100.50"
        assert!(json.contains("100.5"));
    }
}
