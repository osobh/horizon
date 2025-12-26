use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

/// Cost attribution model (matches cost-attributor schema)
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

/// Daily cost summary from materialized view
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct DailyCostSummary {
    pub day: DateTime<Utc>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    pub total_cost: Decimal,
    pub gpu_cost: Decimal,
    pub cpu_cost: Decimal,
    pub network_cost: Decimal,
    pub storage_cost: Decimal,
    pub job_count: i64,
}

/// Monthly cost summary from materialized view
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MonthlyCostSummary {
    pub month: DateTime<Utc>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    pub customer_id: Option<String>,
    pub total_cost: Decimal,
    pub gpu_cost: Decimal,
    pub cpu_cost: Decimal,
    pub network_cost: Decimal,
    pub storage_cost: Decimal,
    pub job_count: i64,
}

/// Cost breakdown by resource type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub gpu_cost: Decimal,
    pub cpu_cost: Decimal,
    pub network_cost: Decimal,
    pub storage_cost: Decimal,
    pub total_cost: Decimal,
}

impl CostBreakdown {
    pub fn new() -> Self {
        Self {
            gpu_cost: Decimal::ZERO,
            cpu_cost: Decimal::ZERO,
            network_cost: Decimal::ZERO,
            storage_cost: Decimal::ZERO,
            total_cost: Decimal::ZERO,
        }
    }

    pub fn from_summaries<T>(summaries: &[T]) -> Self
    where
        T: HasCostBreakdown,
    {
        let mut breakdown = Self::new();
        for summary in summaries {
            breakdown.gpu_cost += summary.gpu_cost();
            breakdown.cpu_cost += summary.cpu_cost();
            breakdown.network_cost += summary.network_cost();
            breakdown.storage_cost += summary.storage_cost();
        }
        breakdown.total_cost = breakdown.gpu_cost
            + breakdown.cpu_cost
            + breakdown.network_cost
            + breakdown.storage_cost;
        breakdown
    }
}

impl Default for CostBreakdown {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that have cost breakdown
pub trait HasCostBreakdown {
    fn gpu_cost(&self) -> Decimal;
    fn cpu_cost(&self) -> Decimal;
    fn network_cost(&self) -> Decimal;
    fn storage_cost(&self) -> Decimal;
    fn total_cost(&self) -> Decimal;
}

impl HasCostBreakdown for DailyCostSummary {
    fn gpu_cost(&self) -> Decimal {
        self.gpu_cost
    }
    fn cpu_cost(&self) -> Decimal {
        self.cpu_cost
    }
    fn network_cost(&self) -> Decimal {
        self.network_cost
    }
    fn storage_cost(&self) -> Decimal {
        self.storage_cost
    }
    fn total_cost(&self) -> Decimal {
        self.total_cost
    }
}

impl HasCostBreakdown for MonthlyCostSummary {
    fn gpu_cost(&self) -> Decimal {
        self.gpu_cost
    }
    fn cpu_cost(&self) -> Decimal {
        self.cpu_cost
    }
    fn network_cost(&self) -> Decimal {
        self.network_cost
    }
    fn storage_cost(&self) -> Decimal {
        self.storage_cost
    }
    fn total_cost(&self) -> Decimal {
        self.total_cost
    }
}

impl HasCostBreakdown for CostAttribution {
    fn gpu_cost(&self) -> Decimal {
        self.gpu_cost
    }
    fn cpu_cost(&self) -> Decimal {
        self.cpu_cost
    }
    fn network_cost(&self) -> Decimal {
        self.network_cost
    }
    fn storage_cost(&self) -> Decimal {
        self.storage_cost
    }
    fn total_cost(&self) -> Decimal {
        self.total_cost
    }
}

/// Top spender entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopSpender {
    pub entity_id: String,
    pub entity_type: String, // user, team, customer
    pub total_cost: Decimal,
    pub job_count: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_cost_breakdown_new() {
        let breakdown = CostBreakdown::new();
        assert_eq!(breakdown.gpu_cost, Decimal::ZERO);
        assert_eq!(breakdown.total_cost, Decimal::ZERO);
    }

    #[test]
    fn test_cost_breakdown_from_daily_summaries() {
        let summaries = vec![
            DailyCostSummary {
                day: Utc::now(),
                team_id: Some("team1".to_string()),
                user_id: Some("user1".to_string()),
                total_cost: dec!(100.00),
                gpu_cost: dec!(80.00),
                cpu_cost: dec!(10.00),
                network_cost: dec!(5.00),
                storage_cost: dec!(5.00),
                job_count: 10,
            },
            DailyCostSummary {
                day: Utc::now(),
                team_id: Some("team1".to_string()),
                user_id: Some("user2".to_string()),
                total_cost: dec!(50.00),
                gpu_cost: dec!(40.00),
                cpu_cost: dec!(5.00),
                network_cost: dec!(3.00),
                storage_cost: dec!(2.00),
                job_count: 5,
            },
        ];

        let breakdown = CostBreakdown::from_summaries(&summaries);
        assert_eq!(breakdown.gpu_cost, dec!(120.00));
        assert_eq!(breakdown.cpu_cost, dec!(15.00));
        assert_eq!(breakdown.network_cost, dec!(8.00));
        assert_eq!(breakdown.storage_cost, dec!(7.00));
        assert_eq!(breakdown.total_cost, dec!(150.00));
    }
}
