use crate::pricing::PricingData;
use hpc_provider::{ProviderError, ProviderResult, SpotPrices};

pub struct SpotPriceManager {
    pricing: PricingData,
}

impl SpotPriceManager {
    pub fn new() -> Self {
        Self {
            pricing: PricingData::new(),
        }
    }

    pub fn get_spot_prices(&self, instance_type: &str, region: &str) -> ProviderResult<SpotPrices> {
        if instance_type.is_empty() {
            return Err(ProviderError::InvalidRequest(
                "instance_type cannot be empty".to_string(),
            ));
        }

        let stats = self
            .pricing
            .get_spot_price_stats(instance_type)
            .ok_or_else(|| {
                ProviderError::InvalidRequest(format!(
                    "instance type {} not supported",
                    instance_type
                ))
            })?;

        Ok(SpotPrices {
            instance_type: instance_type.to_string(),
            region: region.to_string(),
            current_price: stats.current,
            average_price: stats.average,
            min_price: stats.min,
            max_price: stats.max,
        })
    }
}

impl Default for SpotPriceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_spot_price_manager_creation() {
        let manager = SpotPriceManager::new();
        assert!(manager.pricing.is_supported("p4d.24xlarge"));
    }

    #[test]
    fn test_get_spot_prices_p4d() {
        let manager = SpotPriceManager::new();
        let prices = manager
            .get_spot_prices("p4d.24xlarge", "us-east-1")
            .unwrap();

        assert_eq!(prices.instance_type, "p4d.24xlarge");
        assert_eq!(prices.region, "us-east-1");
        assert!(prices.current_price > dec!(0));
        assert!(prices.min_price <= prices.current_price);
        assert!(prices.max_price >= prices.current_price);
    }

    #[test]
    fn test_get_spot_prices_invalid_instance() {
        let manager = SpotPriceManager::new();
        let result = manager.get_spot_prices("invalid.type", "us-east-1");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_spot_prices_empty_instance_type() {
        let manager = SpotPriceManager::new();
        let result = manager.get_spot_prices("", "us-east-1");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ProviderError::InvalidRequest(_)
        ));
    }
}
