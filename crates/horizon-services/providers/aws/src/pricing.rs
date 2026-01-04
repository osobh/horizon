use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

/// AWS EC2 GPU instance pricing data
/// Based on public AWS pricing as of 2024
#[allow(dead_code)]
pub struct PricingData {
    on_demand: HashMap<String, Decimal>,
    spot_base: HashMap<String, Decimal>,
}

impl PricingData {
    pub fn new() -> Self {
        let mut on_demand = HashMap::new();
        let mut spot_base = HashMap::new();

        // P4d instances (A100 GPUs)
        on_demand.insert("p4d.24xlarge".to_string(), dec!(32.77)); // 8x A100
        spot_base.insert("p4d.24xlarge".to_string(), dec!(9.83)); // ~30% of on-demand

        // P5 instances (H100 GPUs)
        on_demand.insert("p5.48xlarge".to_string(), dec!(98.32)); // 8x H100
        spot_base.insert("p5.48xlarge".to_string(), dec!(29.50)); // ~30% of on-demand

        // P3 instances (V100 GPUs)
        on_demand.insert("p3.2xlarge".to_string(), dec!(3.06)); // 1x V100
        on_demand.insert("p3.8xlarge".to_string(), dec!(12.24)); // 4x V100
        on_demand.insert("p3.16xlarge".to_string(), dec!(24.48)); // 8x V100
        spot_base.insert("p3.2xlarge".to_string(), dec!(0.92));
        spot_base.insert("p3.8xlarge".to_string(), dec!(3.67));
        spot_base.insert("p3.16xlarge".to_string(), dec!(7.34));

        // G5 instances (A10G GPUs)
        on_demand.insert("g5.xlarge".to_string(), dec!(1.006)); // 1x A10G
        on_demand.insert("g5.12xlarge".to_string(), dec!(5.672)); // 4x A10G
        on_demand.insert("g5.48xlarge".to_string(), dec!(16.288)); // 8x A10G
        spot_base.insert("g5.xlarge".to_string(), dec!(0.302));
        spot_base.insert("g5.12xlarge".to_string(), dec!(1.702));
        spot_base.insert("g5.48xlarge".to_string(), dec!(4.886));

        Self {
            on_demand,
            spot_base,
        }
    }

    pub fn get_on_demand_price(&self, instance_type: &str) -> Option<Decimal> {
        self.on_demand.get(instance_type).copied()
    }

    pub fn get_spot_base_price(&self, instance_type: &str) -> Option<Decimal> {
        self.spot_base.get(instance_type).copied()
    }

    /// Calculate spot price with simulated variability
    pub fn get_spot_price_with_variance(
        &self,
        instance_type: &str,
        variance_factor: f64,
    ) -> Option<Decimal> {
        self.spot_base.get(instance_type).map(|base| {
            let variance = *base * Decimal::try_from(variance_factor).unwrap_or(dec!(1.0));
            *base + variance
        })
    }

    /// Get price statistics for spot instances
    pub fn get_spot_price_stats(&self, instance_type: &str) -> Option<SpotPriceStats> {
        self.spot_base.get(instance_type).map(|base| {
            SpotPriceStats {
                current: *base * dec!(1.1), // Slightly above base
                average: *base * dec!(1.15),
                min: *base * dec!(0.8),
                max: *base * dec!(1.5),
            }
        })
    }

    pub fn is_supported(&self, instance_type: &str) -> bool {
        self.on_demand.contains_key(instance_type)
    }
}

impl Default for PricingData {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SpotPriceStats {
    pub current: Decimal,
    pub average: Decimal,
    pub min: Decimal,
    pub max: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pricing_data_creation() {
        let pricing = PricingData::new();
        assert!(pricing.get_on_demand_price("p4d.24xlarge").is_some());
        assert!(pricing.get_spot_base_price("p4d.24xlarge").is_some());
    }

    #[test]
    fn test_p4d_pricing() {
        let pricing = PricingData::new();
        let on_demand = pricing.get_on_demand_price("p4d.24xlarge").unwrap();
        let spot = pricing.get_spot_base_price("p4d.24xlarge").unwrap();

        assert_eq!(on_demand, dec!(32.77));
        assert_eq!(spot, dec!(9.83));
        assert!(spot < on_demand);
    }

    #[test]
    fn test_p5_pricing() {
        let pricing = PricingData::new();
        let on_demand = pricing.get_on_demand_price("p5.48xlarge").unwrap();
        let spot = pricing.get_spot_base_price("p5.48xlarge").unwrap();

        assert_eq!(on_demand, dec!(98.32));
        assert!(spot < on_demand);
    }

    #[test]
    fn test_spot_price_stats() {
        let pricing = PricingData::new();
        let stats = pricing.get_spot_price_stats("p4d.24xlarge").unwrap();

        assert!(stats.current > dec!(0));
        assert!(stats.average >= stats.min);
        assert!(stats.average <= stats.max);
        assert!(stats.current >= stats.min);
        assert!(stats.current <= stats.max);
    }

    #[test]
    fn test_is_supported() {
        let pricing = PricingData::new();
        assert!(pricing.is_supported("p4d.24xlarge"));
        assert!(pricing.is_supported("p5.48xlarge"));
        assert!(pricing.is_supported("p3.2xlarge"));
        assert!(!pricing.is_supported("t2.micro"));
    }

    #[test]
    fn test_all_gpu_instances_have_pricing() {
        let pricing = PricingData::new();
        let instances = vec![
            "p4d.24xlarge",
            "p5.48xlarge",
            "p3.2xlarge",
            "p3.8xlarge",
            "p3.16xlarge",
            "g5.xlarge",
            "g5.12xlarge",
            "g5.48xlarge",
        ];

        for instance in instances {
            assert!(
                pricing.get_on_demand_price(instance).is_some(),
                "Missing on-demand pricing for {}",
                instance
            );
            assert!(
                pricing.get_spot_base_price(instance).is_some(),
                "Missing spot pricing for {}",
                instance
            );
        }
    }
}
