use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

use crate::db::Repository;
use crate::error::Result;
use crate::models::report::{ChargebackLineItem, ChargebackReport};
use crate::models::summary::CostBreakdown;

pub struct ChargebackGenerator {
    repository: Repository,
}

impl ChargebackGenerator {
    pub fn new(repository: Repository) -> Self {
        Self { repository }
    }

    /// Generate customer chargeback report
    pub async fn generate_customer_report(
        &self,
        customer_id: &str,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Result<ChargebackReport> {
        // Get monthly summaries for this customer
        let summaries = self
            .repository
            .get_monthly_summaries(period_start, period_end, None, None, Some(customer_id))
            .await?;

        // Calculate breakdown
        let breakdown = CostBreakdown::from_summaries(&summaries);

        // Create line items
        let mut line_items = Vec::new();

        if breakdown.gpu_cost > Decimal::ZERO {
            line_items.push(ChargebackLineItem::new(
                "GPU Compute".to_string(),
                "gpu".to_string(),
                breakdown.gpu_cost,
                "USD".to_string(),
                Decimal::ONE,
            ));
        }

        if breakdown.cpu_cost > Decimal::ZERO {
            line_items.push(ChargebackLineItem::new(
                "CPU Compute".to_string(),
                "cpu".to_string(),
                breakdown.cpu_cost,
                "USD".to_string(),
                Decimal::ONE,
            ));
        }

        if breakdown.network_cost > Decimal::ZERO {
            line_items.push(ChargebackLineItem::new(
                "Network Egress".to_string(),
                "network".to_string(),
                breakdown.network_cost,
                "USD".to_string(),
                Decimal::ONE,
            ));
        }

        if breakdown.storage_cost > Decimal::ZERO {
            line_items.push(ChargebackLineItem::new(
                "Storage".to_string(),
                "storage".to_string(),
                breakdown.storage_cost,
                "USD".to_string(),
                Decimal::ONE,
            ));
        }

        Ok(ChargebackReport::new(
            customer_id.to_string(),
            period_start,
            period_end,
        )
        .with_breakdown(breakdown)
        .with_line_items(line_items))
    }

    /// Generate detailed line items from attributions
    pub async fn generate_detailed_report(
        &self,
        customer_id: &str,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Result<ChargebackReport> {
        // Get all attributions for this customer
        let attributions = self
            .repository
            .get_attributions(period_start, period_end, None, None, Some(customer_id))
            .await?;

        let mut report = ChargebackReport::new(
            customer_id.to_string(),
            period_start,
            period_end,
        );

        // Create breakdown
        let breakdown = CostBreakdown::from_summaries(&attributions);
        report = report.with_breakdown(breakdown);

        // Create detailed line items per job
        for attr in attributions {
            if attr.gpu_cost > Decimal::ZERO {
                let desc = format!(
                    "GPU - Job {} - User {}",
                    attr.job_id.map(|id| id.to_string()).unwrap_or_else(|| "N/A".to_string()),
                    attr.user_id
                );
                report.add_line_item(ChargebackLineItem::new(
                    desc,
                    "gpu".to_string(),
                    attr.gpu_cost,
                    "USD".to_string(),
                    Decimal::ONE,
                ));
            }

            if attr.cpu_cost > Decimal::ZERO {
                let desc = format!(
                    "CPU - Job {} - User {}",
                    attr.job_id.map(|id| id.to_string()).unwrap_or_else(|| "N/A".to_string()),
                    attr.user_id
                );
                report.add_line_item(ChargebackLineItem::new(
                    desc,
                    "cpu".to_string(),
                    attr.cpu_cost,
                    "USD".to_string(),
                    Decimal::ONE,
                ));
            }
        }

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires database
    async fn test_chargeback_generator_creation() {
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost/horizon_cost_test".to_string());

        let pool = crate::db::create_pool(&database_url).await.unwrap();
        let repo = Repository::new(pool);
        let generator = ChargebackGenerator::new(repo);

        // Just test construction
        let _ = generator;
    }
}
