use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "text")]
pub enum PricingModel {
    #[serde(rename = "on_demand")]
    OnDemand,
    #[serde(rename = "spot")]
    Spot,
    #[serde(rename = "reserved")]
    Reserved,
}

impl std::fmt::Display for PricingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PricingModel::OnDemand => write!(f, "on_demand"),
            PricingModel::Spot => write!(f, "spot"),
            PricingModel::Reserved => write!(f, "reserved"),
        }
    }
}

impl std::str::FromStr for PricingModel {
    type Err = crate::error::HpcError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use crate::error::AttributorErrorExt;

        match s.to_lowercase().as_str() {
            "on_demand" => Ok(PricingModel::OnDemand),
            "spot" => Ok(PricingModel::Spot),
            "reserved" => Ok(PricingModel::Reserved),
            _ => Err(crate::error::HpcError::invalid_pricing_model(s.to_string())),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct GpuPricing {
    pub id: Uuid,
    pub gpu_type: String,
    pub region: Option<String>,
    pub pricing_model: String,
    pub hourly_rate: Decimal,
    pub effective_start: DateTime<Utc>,
    pub effective_end: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateGpuPricing {
    pub gpu_type: String,
    pub region: Option<String>,
    pub pricing_model: PricingModel,
    pub hourly_rate: Decimal,
    pub effective_start: DateTime<Utc>,
    pub effective_end: Option<DateTime<Utc>>,
}

impl CreateGpuPricing {
    pub fn new(
        gpu_type: String,
        pricing_model: PricingModel,
        hourly_rate: Decimal,
        effective_start: DateTime<Utc>,
    ) -> Self {
        Self {
            gpu_type,
            region: None,
            pricing_model,
            hourly_rate,
            effective_start,
            effective_end: None,
        }
    }

    pub fn with_region(mut self, region: String) -> Self {
        self.region = Some(region);
        self
    }

    pub fn with_effective_end(mut self, effective_end: DateTime<Utc>) -> Self {
        self.effective_end = Some(effective_end);
        self
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::AttributorErrorExt;

        if self.gpu_type.is_empty() {
            return Err(crate::error::HpcError::invalid_pricing_data(
                "GPU type cannot be empty",
            ));
        }

        if self.hourly_rate <= Decimal::ZERO {
            return Err(crate::error::HpcError::invalid_pricing_data(
                "Hourly rate must be positive",
            ));
        }

        if let Some(end) = self.effective_end {
            if end <= self.effective_start {
                return Err(crate::error::HpcError::invalid_time_range(
                    self.effective_start.to_rfc3339(),
                    end.to_rfc3339(),
                ));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateGpuPricing {
    pub hourly_rate: Option<Decimal>,
    pub effective_end: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuPricingQuery {
    pub gpu_type: Option<String>,
    pub region: Option<String>,
    pub pricing_model: Option<PricingModel>,
    pub effective_at: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use std::str::FromStr;

    #[test]
    fn test_pricing_model_from_str() {
        assert_eq!(
            PricingModel::from_str("on_demand").unwrap(),
            PricingModel::OnDemand
        );
        assert_eq!(
            PricingModel::from_str("ON_DEMAND").unwrap(),
            PricingModel::OnDemand
        );
        assert_eq!(PricingModel::from_str("spot").unwrap(), PricingModel::Spot);
        assert_eq!(
            PricingModel::from_str("reserved").unwrap(),
            PricingModel::Reserved
        );
        assert!(PricingModel::from_str("invalid").is_err());
    }

    #[test]
    fn test_pricing_model_display() {
        assert_eq!(PricingModel::OnDemand.to_string(), "on_demand");
        assert_eq!(PricingModel::Spot.to_string(), "spot");
        assert_eq!(PricingModel::Reserved.to_string(), "reserved");
    }

    #[test]
    fn test_create_gpu_pricing_builder() {
        let now = Utc::now();
        let pricing =
            CreateGpuPricing::new("A100".to_string(), PricingModel::OnDemand, dec!(3.50), now)
                .with_region("us-east-1".to_string());

        assert_eq!(pricing.gpu_type, "A100");
        assert_eq!(pricing.pricing_model, PricingModel::OnDemand);
        assert_eq!(pricing.hourly_rate, dec!(3.50));
        assert_eq!(pricing.region, Some("us-east-1".to_string()));
    }

    #[test]
    fn test_validate_success() {
        let now = Utc::now();
        let pricing =
            CreateGpuPricing::new("A100".to_string(), PricingModel::OnDemand, dec!(3.50), now);

        assert!(pricing.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_gpu_type() {
        let now = Utc::now();
        let pricing = CreateGpuPricing::new(String::new(), PricingModel::OnDemand, dec!(3.50), now);

        let result = pricing.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("GPU type cannot be empty"));
    }

    #[test]
    fn test_validate_zero_rate() {
        let now = Utc::now();
        let pricing = CreateGpuPricing::new(
            "A100".to_string(),
            PricingModel::OnDemand,
            Decimal::ZERO,
            now,
        );

        let result = pricing.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Hourly rate must be positive"));
    }

    #[test]
    fn test_validate_negative_rate() {
        let now = Utc::now();
        let pricing =
            CreateGpuPricing::new("A100".to_string(), PricingModel::OnDemand, dec!(-1.00), now);

        let result = pricing.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_invalid_time_range() {
        let now = Utc::now();
        let earlier = now - chrono::Duration::hours(1);

        let pricing =
            CreateGpuPricing::new("A100".to_string(), PricingModel::OnDemand, dec!(3.50), now)
                .with_effective_end(earlier);

        let result = pricing.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid time range"));
    }

    #[test]
    fn test_gpu_pricing_query_default() {
        let query = GpuPricingQuery::default();
        assert!(query.gpu_type.is_none());
        assert!(query.region.is_none());
        assert!(query.pricing_model.is_none());
    }

    #[test]
    fn test_pricing_model_serialization() {
        let model = PricingModel::OnDemand;
        let json = serde_json::to_string(&model).unwrap();
        assert_eq!(json, "\"on_demand\"");

        let deserialized: PricingModel = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, PricingModel::OnDemand);
    }

    #[test]
    fn test_update_gpu_pricing() {
        let update = UpdateGpuPricing {
            hourly_rate: Some(dec!(4.00)),
            effective_end: Some(Utc::now()),
        };

        assert!(update.hourly_rate.is_some());
        assert!(update.effective_end.is_some());
    }
}
