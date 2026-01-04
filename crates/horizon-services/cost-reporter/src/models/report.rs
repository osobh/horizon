#[cfg(test)]
use chrono::Datelike;
use chrono::{DateTime, Timelike, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::models::summary::{CostBreakdown, TopSpender};
use crate::models::trend::TrendAnalysis;

/// Period type for reports
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Period {
    CurrentMonth,
    Last3Months,
    YearToDate,
    Custom,
}

impl Period {
    pub fn to_date_range(&self, now: DateTime<Utc>) -> (DateTime<Utc>, DateTime<Utc>) {
        use chrono::{Datelike, Duration};

        match self {
            Period::CurrentMonth => {
                let start = now
                    .with_day(1)
                    .unwrap()
                    .with_hour(0)
                    .unwrap()
                    .with_minute(0)
                    .unwrap()
                    .with_second(0)
                    .unwrap()
                    .with_nanosecond(0)
                    .unwrap();
                (start, now)
            }
            Period::Last3Months => {
                let start = (now - Duration::days(90))
                    .with_day(1)
                    .unwrap()
                    .with_hour(0)
                    .unwrap()
                    .with_minute(0)
                    .unwrap()
                    .with_second(0)
                    .unwrap()
                    .with_nanosecond(0)
                    .unwrap();
                (start, now)
            }
            Period::YearToDate => {
                let start = now
                    .with_month(1)
                    .unwrap()
                    .with_day(1)
                    .unwrap()
                    .with_hour(0)
                    .unwrap()
                    .with_minute(0)
                    .unwrap()
                    .with_second(0)
                    .unwrap()
                    .with_nanosecond(0)
                    .unwrap();
                (start, now)
            }
            Period::Custom => (now, now), // Will be overridden by custom dates
        }
    }
}

/// Team showback report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShowbackReport {
    pub entity_id: String,
    pub entity_type: String, // team, user
    pub period: Period,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub breakdown: CostBreakdown,
    pub trends: Option<TrendAnalysis>,
    pub top_users: Vec<TopSpender>,
    pub job_count: i64,
    pub generated_at: DateTime<Utc>,
}

impl ShowbackReport {
    pub fn new(
        entity_id: String,
        entity_type: String,
        period: Period,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Self {
        Self {
            entity_id,
            entity_type,
            period,
            period_start,
            period_end,
            breakdown: CostBreakdown::new(),
            trends: None,
            top_users: Vec::new(),
            job_count: 0,
            generated_at: Utc::now(),
        }
    }

    pub fn with_breakdown(mut self, breakdown: CostBreakdown) -> Self {
        self.breakdown = breakdown;
        self
    }

    pub fn with_trends(mut self, trends: TrendAnalysis) -> Self {
        self.trends = Some(trends);
        self
    }

    pub fn with_top_users(mut self, top_users: Vec<TopSpender>) -> Self {
        self.top_users = top_users;
        self
    }

    pub fn with_job_count(mut self, job_count: i64) -> Self {
        self.job_count = job_count;
        self
    }
}

/// Customer chargeback report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChargebackReport {
    pub customer_id: String,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub breakdown: CostBreakdown,
    pub line_items: Vec<ChargebackLineItem>,
    pub total_amount: Decimal,
    pub currency: String,
    pub generated_at: DateTime<Utc>,
}

impl ChargebackReport {
    pub fn new(
        customer_id: String,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Self {
        Self {
            customer_id,
            period_start,
            period_end,
            breakdown: CostBreakdown::new(),
            line_items: Vec::new(),
            total_amount: Decimal::ZERO,
            currency: "USD".to_string(),
            generated_at: Utc::now(),
        }
    }

    pub fn with_breakdown(mut self, breakdown: CostBreakdown) -> Self {
        self.total_amount = breakdown.total_cost;
        self.breakdown = breakdown;
        self
    }

    pub fn with_line_items(mut self, line_items: Vec<ChargebackLineItem>) -> Self {
        self.line_items = line_items;
        self
    }

    pub fn add_line_item(&mut self, item: ChargebackLineItem) {
        self.total_amount += item.amount;
        self.line_items.push(item);
    }
}

/// Individual line item for chargeback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChargebackLineItem {
    pub description: String,
    pub resource_type: String, // gpu, cpu, network, storage
    pub quantity: Decimal,
    pub unit: String, // hours, GB, TB
    pub unit_price: Decimal,
    pub amount: Decimal,
}

impl ChargebackLineItem {
    pub fn new(
        description: String,
        resource_type: String,
        quantity: Decimal,
        unit: String,
        unit_price: Decimal,
    ) -> Self {
        let amount = quantity * unit_price;
        Self {
            description,
            resource_type,
            quantity,
            unit,
            unit_price,
            amount,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_period_current_month() {
        let now = chrono::Utc::now();
        let (start, end) = Period::CurrentMonth.to_date_range(now);

        assert_eq!(start.day(), 1);
        assert_eq!(start.hour(), 0);
        assert_eq!(start.minute(), 0);
        assert!(end >= start);
    }

    #[test]
    fn test_period_year_to_date() {
        let now = chrono::Utc::now();
        let (start, end) = Period::YearToDate.to_date_range(now);

        assert_eq!(start.month(), 1);
        assert_eq!(start.day(), 1);
        assert!(end >= start);
    }

    #[test]
    fn test_showback_report_builder() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(24);

        let report = ShowbackReport::new(
            "team123".to_string(),
            "team".to_string(),
            Period::CurrentMonth,
            now,
            later,
        )
        .with_job_count(42);

        assert_eq!(report.entity_id, "team123");
        assert_eq!(report.entity_type, "team");
        assert_eq!(report.job_count, 42);
        assert_eq!(report.period, Period::CurrentMonth);
    }

    #[test]
    fn test_chargeback_report_builder() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(24);

        let breakdown = CostBreakdown {
            gpu_cost: dec!(100.00),
            cpu_cost: dec!(20.00),
            network_cost: dec!(10.00),
            storage_cost: dec!(5.00),
            total_cost: dec!(135.00),
        };

        let report =
            ChargebackReport::new("customer123".to_string(), now, later).with_breakdown(breakdown);

        assert_eq!(report.customer_id, "customer123");
        assert_eq!(report.total_amount, dec!(135.00));
        assert_eq!(report.breakdown.gpu_cost, dec!(100.00));
    }

    #[test]
    fn test_chargeback_line_item() {
        let item = ChargebackLineItem::new(
            "GPU Usage".to_string(),
            "gpu".to_string(),
            dec!(10.5),
            "hours".to_string(),
            dec!(5.00),
        );

        assert_eq!(item.amount, dec!(52.50));
        assert_eq!(item.quantity, dec!(10.5));
        assert_eq!(item.unit_price, dec!(5.00));
    }

    #[test]
    fn test_chargeback_add_line_item() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(24);

        let mut report = ChargebackReport::new("customer123".to_string(), now, later);

        let item1 = ChargebackLineItem::new(
            "GPU Usage".to_string(),
            "gpu".to_string(),
            dec!(10.0),
            "hours".to_string(),
            dec!(5.00),
        );

        let item2 = ChargebackLineItem::new(
            "Storage".to_string(),
            "storage".to_string(),
            dec!(100.0),
            "GB".to_string(),
            dec!(0.10),
        );

        report.add_line_item(item1);
        report.add_line_item(item2);

        assert_eq!(report.line_items.len(), 2);
        assert_eq!(report.total_amount, dec!(60.00));
    }
}
