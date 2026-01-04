use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

/// Customer segment classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "VARCHAR")]
#[sqlx(rename_all = "lowercase")]
pub enum CustomerSegment {
    #[serde(rename = "enterprise")]
    Enterprise,
    #[serde(rename = "growth")]
    Growth,
    #[serde(rename = "startup")]
    Startup,
}

impl std::fmt::Display for CustomerSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Enterprise => write!(f, "enterprise"),
            Self::Growth => write!(f, "growth"),
            Self::Startup => write!(f, "startup"),
        }
    }
}

/// Customer profitability profile
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct CustomerProfile {
    pub id: Uuid,
    pub customer_id: String,
    pub segment: String,
    pub total_revenue: Decimal,
    pub total_cost: Decimal,
    pub gross_margin: Option<Decimal>,
    pub contribution_margin: Option<Decimal>,
    pub lifetime_value: Option<Decimal>,
    pub first_usage_at: Option<DateTime<Utc>>,
    pub last_usage_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl CustomerProfile {
    /// Calculate gross margin percentage: (revenue - cost) / revenue * 100
    pub fn calculate_gross_margin(&self) -> Option<Decimal> {
        if self.total_revenue.is_zero() {
            return None;
        }
        let margin =
            (self.total_revenue - self.total_cost) / self.total_revenue * Decimal::from(100);
        Some(margin)
    }

    /// Calculate contribution margin: revenue - cost
    pub fn calculate_contribution_margin(&self) -> Decimal {
        self.total_revenue - self.total_cost
    }

    /// Check if customer is profitable
    pub fn is_profitable(&self) -> bool {
        self.total_revenue > self.total_cost
    }

    /// Check if customer is at risk (low margin)
    pub fn is_at_risk(&self, threshold: Decimal) -> bool {
        if let Some(margin) = self.calculate_gross_margin() {
            margin < threshold
        } else {
            true // Zero revenue = at risk
        }
    }
}

/// Pricing simulation scenario
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct PricingSimulation {
    pub id: Uuid,
    pub customer_id: String,
    pub scenario_name: String,
    pub current_price: Option<Decimal>,
    pub simulated_price: Option<Decimal>,
    pub estimated_margin_impact: Option<Decimal>,
    pub elasticity_factor: Option<Decimal>,
    pub created_at: DateTime<Utc>,
}

impl PricingSimulation {
    /// Calculate price change percentage
    pub fn price_change_percent(&self) -> Option<Decimal> {
        let current = self.current_price?;
        let simulated = self.simulated_price?;

        if current.is_zero() {
            return None;
        }

        Some((simulated - current) / current * Decimal::from(100))
    }

    /// Estimate revenue impact using price elasticity
    pub fn estimate_revenue_impact(&self, baseline_revenue: Decimal) -> Option<Decimal> {
        let price_change = self.price_change_percent()?;
        let elasticity = self.elasticity_factor?;

        // Revenue change = baseline * (1 + price_change% * (1 + elasticity))
        let impact_factor = price_change / Decimal::from(100) * (Decimal::ONE + elasticity);
        Some(baseline_revenue * impact_factor)
    }
}

/// Recommendation type for margin optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "VARCHAR")]
#[sqlx(rename_all = "lowercase")]
pub enum RecommendationType {
    #[serde(rename = "price_increase")]
    PriceIncrease,
    #[serde(rename = "cost_reduction")]
    CostReduction,
    #[serde(rename = "upsell")]
    Upsell,
}

impl std::fmt::Display for RecommendationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PriceIncrease => write!(f, "price_increase"),
            Self::CostReduction => write!(f, "cost_reduction"),
            Self::Upsell => write!(f, "upsell"),
        }
    }
}

/// Recommendation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "VARCHAR")]
#[sqlx(rename_all = "lowercase")]
pub enum RecommendationStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "approved")]
    Approved,
    #[serde(rename = "implemented")]
    Implemented,
    #[serde(rename = "rejected")]
    Rejected,
}

impl std::fmt::Display for RecommendationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Approved => write!(f, "approved"),
            Self::Implemented => write!(f, "implemented"),
            Self::Rejected => write!(f, "rejected"),
        }
    }
}

/// Margin optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MarginRecommendation {
    pub id: Uuid,
    pub customer_id: String,
    pub recommendation_type: String,
    pub current_margin: Option<Decimal>,
    pub projected_margin: Option<Decimal>,
    pub impact_amount: Option<Decimal>,
    pub confidence: Option<Decimal>,
    pub status: String,
    pub created_at: DateTime<Utc>,
}

impl MarginRecommendation {
    /// Calculate margin improvement
    pub fn margin_improvement(&self) -> Option<Decimal> {
        let current = self.current_margin?;
        let projected = self.projected_margin?;
        Some(projected - current)
    }

    /// Check if recommendation is high confidence
    pub fn is_high_confidence(&self) -> bool {
        if let Some(conf) = self.confidence {
            conf >= Decimal::from_str_exact("0.75").unwrap()
        } else {
            false
        }
    }
}

/// Request to create a customer profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateProfileRequest {
    pub customer_id: String,
    pub segment: CustomerSegment,
}

/// Request to refresh customer profile calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshProfileRequest {
    pub include_ltv: bool,
}

/// Request to create a pricing simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSimulationRequest {
    pub customer_id: String,
    pub scenario_name: String,
    pub current_price: Decimal,
    pub simulated_price: Decimal,
    pub elasticity_factor: Decimal,
}

/// Margin analysis summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginAnalysis {
    pub total_customers: i64,
    pub avg_gross_margin: Option<Decimal>,
    pub avg_contribution_margin: Option<Decimal>,
    pub profitable_count: i64,
    pub at_risk_count: i64,
    pub total_revenue: Decimal,
    pub total_cost: Decimal,
}

/// Segment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentAnalysis {
    pub segment: String,
    pub customer_count: i64,
    pub avg_margin: Option<Decimal>,
    pub total_revenue: Decimal,
    pub total_cost: Decimal,
}

/// Top contributor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopContributor {
    pub customer_id: String,
    pub segment: String,
    pub contribution_margin: Decimal,
    pub gross_margin: Option<Decimal>,
    pub revenue: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_customer_segment_display() {
        assert_eq!(CustomerSegment::Enterprise.to_string(), "enterprise");
        assert_eq!(CustomerSegment::Growth.to_string(), "growth");
        assert_eq!(CustomerSegment::Startup.to_string(), "startup");
    }

    #[test]
    fn test_calculate_gross_margin() {
        let profile = CustomerProfile {
            id: Uuid::new_v4(),
            customer_id: "cust-001".to_string(),
            segment: "enterprise".to_string(),
            total_revenue: dec!(10000),
            total_cost: dec!(7000),
            gross_margin: None,
            contribution_margin: None,
            lifetime_value: None,
            first_usage_at: None,
            last_usage_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let margin = profile.calculate_gross_margin().unwrap();
        assert_eq!(margin, dec!(30.00)); // (10000-7000)/10000 * 100 = 30%
    }

    #[test]
    fn test_calculate_gross_margin_zero_revenue() {
        let profile = CustomerProfile {
            id: Uuid::new_v4(),
            customer_id: "cust-002".to_string(),
            segment: "startup".to_string(),
            total_revenue: dec!(0),
            total_cost: dec!(100),
            gross_margin: None,
            contribution_margin: None,
            lifetime_value: None,
            first_usage_at: None,
            last_usage_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        assert_eq!(profile.calculate_gross_margin(), None);
    }

    #[test]
    fn test_calculate_contribution_margin() {
        let profile = CustomerProfile {
            id: Uuid::new_v4(),
            customer_id: "cust-003".to_string(),
            segment: "growth".to_string(),
            total_revenue: dec!(5000),
            total_cost: dec!(3000),
            gross_margin: None,
            contribution_margin: None,
            lifetime_value: None,
            first_usage_at: None,
            last_usage_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        assert_eq!(profile.calculate_contribution_margin(), dec!(2000));
    }

    #[test]
    fn test_is_profitable() {
        let profitable = CustomerProfile {
            id: Uuid::new_v4(),
            customer_id: "cust-004".to_string(),
            segment: "enterprise".to_string(),
            total_revenue: dec!(10000),
            total_cost: dec!(7000),
            gross_margin: None,
            contribution_margin: None,
            lifetime_value: None,
            first_usage_at: None,
            last_usage_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let unprofitable = CustomerProfile {
            id: Uuid::new_v4(),
            customer_id: "cust-005".to_string(),
            segment: "startup".to_string(),
            total_revenue: dec!(1000),
            total_cost: dec!(2000),
            gross_margin: None,
            contribution_margin: None,
            lifetime_value: None,
            first_usage_at: None,
            last_usage_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        assert!(profitable.is_profitable());
        assert!(!unprofitable.is_profitable());
    }

    #[test]
    fn test_is_at_risk() {
        let at_risk = CustomerProfile {
            id: Uuid::new_v4(),
            customer_id: "cust-006".to_string(),
            segment: "startup".to_string(),
            total_revenue: dec!(1000),
            total_cost: dec!(950),
            gross_margin: None,
            contribution_margin: None,
            lifetime_value: None,
            first_usage_at: None,
            last_usage_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let healthy = CustomerProfile {
            id: Uuid::new_v4(),
            customer_id: "cust-007".to_string(),
            segment: "enterprise".to_string(),
            total_revenue: dec!(10000),
            total_cost: dec!(5000),
            gross_margin: None,
            contribution_margin: None,
            lifetime_value: None,
            first_usage_at: None,
            last_usage_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        assert!(at_risk.is_at_risk(dec!(10.0))); // 5% margin < 10% threshold
        assert!(!healthy.is_at_risk(dec!(10.0))); // 50% margin > 10% threshold
    }

    #[test]
    fn test_pricing_simulation_price_change() {
        let sim = PricingSimulation {
            id: Uuid::new_v4(),
            customer_id: "cust-008".to_string(),
            scenario_name: "10% price increase".to_string(),
            current_price: Some(dec!(100)),
            simulated_price: Some(dec!(110)),
            estimated_margin_impact: None,
            elasticity_factor: Some(dec!(-0.5)),
            created_at: Utc::now(),
        };

        assert_eq!(sim.price_change_percent().unwrap(), dec!(10.00));
    }

    #[test]
    fn test_pricing_simulation_revenue_impact() {
        let sim = PricingSimulation {
            id: Uuid::new_v4(),
            customer_id: "cust-009".to_string(),
            scenario_name: "5% increase".to_string(),
            current_price: Some(dec!(100)),
            simulated_price: Some(dec!(105)),
            estimated_margin_impact: None,
            elasticity_factor: Some(dec!(-0.3)),
            created_at: Utc::now(),
        };

        let baseline_revenue = dec!(10000);
        let impact = sim.estimate_revenue_impact(baseline_revenue).unwrap();

        // 5% price change with -0.3 elasticity = 5% * 0.7 = 3.5% revenue change
        // 10000 * 0.035 = 350
        assert_eq!(impact, dec!(350));
    }

    #[test]
    fn test_recommendation_margin_improvement() {
        let rec = MarginRecommendation {
            id: Uuid::new_v4(),
            customer_id: "cust-010".to_string(),
            recommendation_type: "price_increase".to_string(),
            current_margin: Some(dec!(20.00)),
            projected_margin: Some(dec!(25.00)),
            impact_amount: Some(dec!(500)),
            confidence: Some(dec!(0.85)),
            status: "pending".to_string(),
            created_at: Utc::now(),
        };

        assert_eq!(rec.margin_improvement().unwrap(), dec!(5.00));
        assert!(rec.is_high_confidence());
    }

    #[test]
    fn test_recommendation_low_confidence() {
        let rec = MarginRecommendation {
            id: Uuid::new_v4(),
            customer_id: "cust-011".to_string(),
            recommendation_type: "cost_reduction".to_string(),
            current_margin: Some(dec!(15.00)),
            projected_margin: Some(dec!(18.00)),
            impact_amount: Some(dec!(300)),
            confidence: Some(dec!(0.60)),
            status: "pending".to_string(),
            created_at: Utc::now(),
        };

        assert!(!rec.is_high_confidence());
    }
}
