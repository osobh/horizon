//! Cost tracking for ephemeral quotas.
//!
//! Provides detailed cost attribution and tracking for ephemeral resource usage,
//! enabling accurate sponsor billing and usage analytics.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use utoipa::ToSchema;
use uuid::Uuid;

use super::ResourceType;

/// A cost record for ephemeral resource usage.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize, ToSchema)]
pub struct EphemeralCostRecord {
    /// Unique identifier for this cost record
    pub id: Uuid,
    /// The ephemeral quota this cost is associated with
    pub ephemeral_quota_id: Uuid,
    /// The sponsor being billed
    pub sponsor_id: String,
    /// The beneficiary who used the resource
    pub beneficiary_id: String,
    /// Organization/tenant context
    pub tenant_id: Uuid,
    /// Type of resource
    pub resource_type: ResourceType,
    /// Amount of resource used
    pub amount: Decimal,
    /// Unit cost rate applied
    pub unit_rate: Decimal,
    /// Total cost for this usage
    pub total_cost: Decimal,
    /// Any discounts applied
    pub discount: Decimal,
    /// Final cost after discounts
    pub final_cost: Decimal,
    /// Optional job ID that incurred this cost
    pub job_id: Option<Uuid>,
    /// Billing period start
    pub period_start: DateTime<Utc>,
    /// Billing period end
    pub period_end: DateTime<Utc>,
    /// Status of this cost record
    pub status: CostRecordStatus,
    /// When this record was created
    pub created_at: DateTime<Utc>,
    /// When this record was finalized
    pub finalized_at: Option<DateTime<Utc>>,
}

/// Status of a cost record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type, ToSchema)]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum CostRecordStatus {
    /// Cost is pending finalization
    Pending,
    /// Cost has been finalized
    Finalized,
    /// Cost has been invoiced
    Invoiced,
    /// Cost has been paid
    Paid,
    /// Cost was disputed
    Disputed,
    /// Cost was waived/credited
    Waived,
}

impl CostRecordStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            CostRecordStatus::Pending => "pending",
            CostRecordStatus::Finalized => "finalized",
            CostRecordStatus::Invoiced => "invoiced",
            CostRecordStatus::Paid => "paid",
            CostRecordStatus::Disputed => "disputed",
            CostRecordStatus::Waived => "waived",
        }
    }

    pub fn is_billable(&self) -> bool {
        matches!(
            self,
            CostRecordStatus::Pending | CostRecordStatus::Finalized | CostRecordStatus::Invoiced
        )
    }
}

impl EphemeralCostRecord {
    /// Create a new cost record.
    pub fn new(
        ephemeral_quota_id: Uuid,
        sponsor_id: impl Into<String>,
        beneficiary_id: impl Into<String>,
        tenant_id: Uuid,
        resource_type: ResourceType,
        amount: Decimal,
        unit_rate: Decimal,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Self {
        let total_cost = amount * unit_rate;
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            ephemeral_quota_id,
            sponsor_id: sponsor_id.into(),
            beneficiary_id: beneficiary_id.into(),
            tenant_id,
            resource_type,
            amount,
            unit_rate,
            total_cost,
            discount: Decimal::ZERO,
            final_cost: total_cost,
            job_id: None,
            period_start,
            period_end,
            status: CostRecordStatus::Pending,
            created_at: now,
            finalized_at: None,
        }
    }

    /// Associate this cost with a job.
    pub fn with_job(mut self, job_id: Uuid) -> Self {
        self.job_id = Some(job_id);
        self
    }

    /// Apply a discount to this cost.
    pub fn apply_discount(&mut self, discount_amount: Decimal) {
        self.discount = discount_amount;
        self.final_cost = (self.total_cost - discount_amount).max(Decimal::ZERO);
    }

    /// Apply a percentage discount.
    pub fn apply_discount_percent(&mut self, percent: Decimal) {
        let discount = self.total_cost * (percent / Decimal::from(100));
        self.apply_discount(discount);
    }

    /// Finalize this cost record.
    pub fn finalize(&mut self) {
        self.status = CostRecordStatus::Finalized;
        self.finalized_at = Some(Utc::now());
    }

    /// Mark as invoiced.
    pub fn mark_invoiced(&mut self) {
        self.status = CostRecordStatus::Invoiced;
    }

    /// Mark as paid.
    pub fn mark_paid(&mut self) {
        self.status = CostRecordStatus::Paid;
    }

    /// Mark as disputed.
    pub fn dispute(&mut self) {
        self.status = CostRecordStatus::Disputed;
    }

    /// Waive this cost.
    pub fn waive(&mut self) {
        self.status = CostRecordStatus::Waived;
        self.final_cost = Decimal::ZERO;
    }
}

/// Aggregated cost summary for a sponsor.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SponsorCostSummary {
    pub sponsor_id: String,
    pub tenant_id: Uuid,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_records: i32,
    pub total_cost: Decimal,
    pub total_discount: Decimal,
    pub final_cost: Decimal,
    pub by_resource_type: Vec<ResourceCostBreakdown>,
    pub by_beneficiary: Vec<BeneficiaryCostBreakdown>,
}

/// Cost breakdown by resource type.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResourceCostBreakdown {
    pub resource_type: ResourceType,
    pub total_amount: Decimal,
    pub total_cost: Decimal,
    pub average_rate: Decimal,
}

/// Cost breakdown by beneficiary.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct BeneficiaryCostBreakdown {
    pub beneficiary_id: String,
    pub total_cost: Decimal,
    pub record_count: i32,
}

/// Builder for creating cost summaries.
pub struct CostSummaryBuilder {
    sponsor_id: String,
    tenant_id: Uuid,
    period_start: DateTime<Utc>,
    period_end: DateTime<Utc>,
    records: Vec<EphemeralCostRecord>,
}

impl CostSummaryBuilder {
    pub fn new(
        sponsor_id: impl Into<String>,
        tenant_id: Uuid,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Self {
        Self {
            sponsor_id: sponsor_id.into(),
            tenant_id,
            period_start,
            period_end,
            records: Vec::new(),
        }
    }

    pub fn add_record(mut self, record: EphemeralCostRecord) -> Self {
        self.records.push(record);
        self
    }

    pub fn add_records(mut self, records: Vec<EphemeralCostRecord>) -> Self {
        self.records.extend(records);
        self
    }

    pub fn build(self) -> SponsorCostSummary {
        let total_records = self.records.len() as i32;
        let total_cost: Decimal = self.records.iter().map(|r| r.total_cost).sum();
        let total_discount: Decimal = self.records.iter().map(|r| r.discount).sum();
        let final_cost: Decimal = self.records.iter().map(|r| r.final_cost).sum();

        // Group by resource type
        let mut by_resource: std::collections::HashMap<String, (Decimal, Decimal)> =
            std::collections::HashMap::new();
        for record in &self.records {
            let key = record.resource_type.as_str().to_string();
            let entry = by_resource
                .entry(key)
                .or_insert((Decimal::ZERO, Decimal::ZERO));
            entry.0 += record.amount;
            entry.1 += record.final_cost;
        }

        let by_resource_type: Vec<ResourceCostBreakdown> = by_resource
            .into_iter()
            .map(|(type_str, (amount, cost))| {
                let resource_type =
                    ResourceType::from_str(&type_str).unwrap_or(ResourceType::GpuHours);
                let average_rate = if amount > Decimal::ZERO {
                    cost / amount
                } else {
                    Decimal::ZERO
                };
                ResourceCostBreakdown {
                    resource_type,
                    total_amount: amount,
                    total_cost: cost,
                    average_rate,
                }
            })
            .collect();

        // Group by beneficiary
        let mut by_beneficiary_map: std::collections::HashMap<String, (Decimal, i32)> =
            std::collections::HashMap::new();
        for record in &self.records {
            let entry = by_beneficiary_map
                .entry(record.beneficiary_id.clone())
                .or_insert((Decimal::ZERO, 0));
            entry.0 += record.final_cost;
            entry.1 += 1;
        }

        let by_beneficiary: Vec<BeneficiaryCostBreakdown> = by_beneficiary_map
            .into_iter()
            .map(|(beneficiary_id, (cost, count))| BeneficiaryCostBreakdown {
                beneficiary_id,
                total_cost: cost,
                record_count: count,
            })
            .collect();

        SponsorCostSummary {
            sponsor_id: self.sponsor_id,
            tenant_id: self.tenant_id,
            period_start: self.period_start,
            period_end: self.period_end,
            total_records,
            total_cost,
            total_discount,
            final_cost,
            by_resource_type,
            by_beneficiary,
        }
    }
}

/// Cost rate configuration for ephemeral quotas.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct EphemeralCostRates {
    pub tenant_id: Uuid,
    pub rates: Vec<ResourceCostRate>,
    pub effective_from: DateTime<Utc>,
    pub effective_until: Option<DateTime<Utc>>,
}

/// Cost rate for a specific resource type.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResourceCostRate {
    pub resource_type: ResourceType,
    pub base_rate: Decimal,
    pub burst_rate: Decimal,
    pub discount_tier: Option<DiscountTier>,
}

/// Volume-based discount tier.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DiscountTier {
    pub tier_name: String,
    pub min_usage: Decimal,
    pub discount_percent: Decimal,
}

impl EphemeralCostRates {
    /// Create default cost rates.
    pub fn default_rates(tenant_id: Uuid) -> Self {
        Self {
            tenant_id,
            rates: vec![
                ResourceCostRate {
                    resource_type: ResourceType::GpuHours,
                    base_rate: Decimal::from(2),  // $2/GPU-hour
                    burst_rate: Decimal::from(3), // $3/GPU-hour for burst
                    discount_tier: Some(DiscountTier {
                        tier_name: "Volume".to_string(),
                        min_usage: Decimal::from(100),
                        discount_percent: Decimal::from(10),
                    }),
                },
                ResourceCostRate {
                    resource_type: ResourceType::CpuHours,
                    base_rate: Decimal::new(50, 2), // $0.50/CPU-hour
                    burst_rate: Decimal::new(75, 2),
                    discount_tier: None,
                },
                ResourceCostRate {
                    resource_type: ResourceType::StorageGbHours,
                    base_rate: Decimal::new(1, 2), // $0.01/GB-hour
                    burst_rate: Decimal::new(2, 2),
                    discount_tier: None,
                },
            ],
            effective_from: Utc::now(),
            effective_until: None,
        }
    }

    /// Get the rate for a specific resource type.
    pub fn get_rate(&self, resource_type: ResourceType) -> Option<&ResourceCostRate> {
        self.rates.iter().find(|r| r.resource_type == resource_type)
    }

    /// Calculate cost for a given amount.
    pub fn calculate_cost(
        &self,
        resource_type: ResourceType,
        amount: Decimal,
        is_burst: bool,
    ) -> Option<Decimal> {
        self.get_rate(resource_type).map(|rate| {
            let unit_rate = if is_burst {
                rate.burst_rate
            } else {
                rate.base_rate
            };
            amount * unit_rate
        })
    }

    /// Check if still effective.
    pub fn is_effective(&self) -> bool {
        let now = Utc::now();
        if now < self.effective_from {
            return false;
        }
        if let Some(until) = self.effective_until {
            if now >= until {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn test_record() -> EphemeralCostRecord {
        EphemeralCostRecord::new(
            Uuid::new_v4(),
            "sponsor1",
            "user1",
            Uuid::new_v4(),
            ResourceType::GpuHours,
            dec!(10),
            dec!(2),
            Utc::now(),
            Utc::now() + chrono::Duration::hours(1),
        )
    }

    #[test]
    fn test_cost_record_status_billable() {
        assert!(CostRecordStatus::Pending.is_billable());
        assert!(CostRecordStatus::Finalized.is_billable());
        assert!(CostRecordStatus::Invoiced.is_billable());
        assert!(!CostRecordStatus::Paid.is_billable());
        assert!(!CostRecordStatus::Disputed.is_billable());
        assert!(!CostRecordStatus::Waived.is_billable());
    }

    #[test]
    fn test_cost_record_new() {
        let record = test_record();

        assert_eq!(record.sponsor_id, "sponsor1");
        assert_eq!(record.beneficiary_id, "user1");
        assert_eq!(record.amount, dec!(10));
        assert_eq!(record.unit_rate, dec!(2));
        assert_eq!(record.total_cost, dec!(20));
        assert_eq!(record.final_cost, dec!(20));
        assert_eq!(record.status, CostRecordStatus::Pending);
    }

    #[test]
    fn test_cost_record_apply_discount() {
        let mut record = test_record();

        record.apply_discount(dec!(5));

        assert_eq!(record.discount, dec!(5));
        assert_eq!(record.final_cost, dec!(15));
    }

    #[test]
    fn test_cost_record_apply_discount_percent() {
        let mut record = test_record();

        record.apply_discount_percent(dec!(25));

        assert_eq!(record.discount, dec!(5));
        assert_eq!(record.final_cost, dec!(15));
    }

    #[test]
    fn test_cost_record_lifecycle() {
        let mut record = test_record();

        assert_eq!(record.status, CostRecordStatus::Pending);

        record.finalize();
        assert_eq!(record.status, CostRecordStatus::Finalized);
        assert!(record.finalized_at.is_some());

        record.mark_invoiced();
        assert_eq!(record.status, CostRecordStatus::Invoiced);

        record.mark_paid();
        assert_eq!(record.status, CostRecordStatus::Paid);
    }

    #[test]
    fn test_cost_record_dispute() {
        let mut record = test_record();
        record.finalize();

        record.dispute();

        assert_eq!(record.status, CostRecordStatus::Disputed);
    }

    #[test]
    fn test_cost_record_waive() {
        let mut record = test_record();

        record.waive();

        assert_eq!(record.status, CostRecordStatus::Waived);
        assert_eq!(record.final_cost, dec!(0));
    }

    #[test]
    fn test_cost_summary_builder_empty() {
        let summary = CostSummaryBuilder::new(
            "sponsor1",
            Uuid::new_v4(),
            Utc::now(),
            Utc::now() + chrono::Duration::days(30),
        )
        .build();

        assert_eq!(summary.total_records, 0);
        assert_eq!(summary.total_cost, dec!(0));
        assert_eq!(summary.final_cost, dec!(0));
    }

    #[test]
    fn test_cost_summary_builder_with_records() {
        let tenant_id = Uuid::new_v4();
        let quota_id = Uuid::new_v4();
        let now = Utc::now();

        let record1 = EphemeralCostRecord::new(
            quota_id,
            "sponsor1",
            "user1",
            tenant_id,
            ResourceType::GpuHours,
            dec!(10),
            dec!(2),
            now,
            now + chrono::Duration::hours(1),
        );

        let record2 = EphemeralCostRecord::new(
            quota_id,
            "sponsor1",
            "user2",
            tenant_id,
            ResourceType::GpuHours,
            dec!(5),
            dec!(2),
            now,
            now + chrono::Duration::hours(1),
        );

        let summary =
            CostSummaryBuilder::new("sponsor1", tenant_id, now, now + chrono::Duration::days(30))
                .add_record(record1)
                .add_record(record2)
                .build();

        assert_eq!(summary.total_records, 2);
        assert_eq!(summary.total_cost, dec!(30));
        assert_eq!(summary.final_cost, dec!(30));
        assert_eq!(summary.by_beneficiary.len(), 2);
    }

    #[test]
    fn test_cost_summary_with_discounts() {
        let tenant_id = Uuid::new_v4();
        let quota_id = Uuid::new_v4();
        let now = Utc::now();

        let mut record1 = EphemeralCostRecord::new(
            quota_id,
            "sponsor1",
            "user1",
            tenant_id,
            ResourceType::GpuHours,
            dec!(10),
            dec!(2),
            now,
            now + chrono::Duration::hours(1),
        );
        record1.apply_discount(dec!(5));

        let summary =
            CostSummaryBuilder::new("sponsor1", tenant_id, now, now + chrono::Duration::days(30))
                .add_record(record1)
                .build();

        assert_eq!(summary.total_cost, dec!(20));
        assert_eq!(summary.total_discount, dec!(5));
        assert_eq!(summary.final_cost, dec!(15));
    }

    #[test]
    fn test_ephemeral_cost_rates_default() {
        let rates = EphemeralCostRates::default_rates(Uuid::new_v4());

        assert!(!rates.rates.is_empty());
        assert!(rates.is_effective());
    }

    #[test]
    fn test_ephemeral_cost_rates_get_rate() {
        let rates = EphemeralCostRates::default_rates(Uuid::new_v4());

        let gpu_rate = rates.get_rate(ResourceType::GpuHours);
        assert!(gpu_rate.is_some());
        assert_eq!(gpu_rate.unwrap().base_rate, dec!(2));

        let cpu_rate = rates.get_rate(ResourceType::CpuHours);
        assert!(cpu_rate.is_some());
        assert_eq!(cpu_rate.unwrap().base_rate, dec!(0.50));
    }

    #[test]
    fn test_ephemeral_cost_rates_calculate_cost() {
        let rates = EphemeralCostRates::default_rates(Uuid::new_v4());

        let cost = rates.calculate_cost(ResourceType::GpuHours, dec!(10), false);
        assert_eq!(cost, Some(dec!(20)));

        let burst_cost = rates.calculate_cost(ResourceType::GpuHours, dec!(10), true);
        assert_eq!(burst_cost, Some(dec!(30)));
    }

    #[test]
    fn test_ephemeral_cost_rates_not_effective() {
        let mut rates = EphemeralCostRates::default_rates(Uuid::new_v4());
        rates.effective_until = Some(Utc::now() - chrono::Duration::hours(1));

        assert!(!rates.is_effective());
    }

    #[test]
    fn test_ephemeral_cost_rates_not_started() {
        let mut rates = EphemeralCostRates::default_rates(Uuid::new_v4());
        rates.effective_from = Utc::now() + chrono::Duration::hours(1);

        assert!(!rates.is_effective());
    }

    #[test]
    fn test_cost_record_serialization() {
        let record = test_record();

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: EphemeralCostRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record.sponsor_id, deserialized.sponsor_id);
        assert_eq!(record.amount, deserialized.amount);
        assert_eq!(record.total_cost, deserialized.total_cost);
    }
}
