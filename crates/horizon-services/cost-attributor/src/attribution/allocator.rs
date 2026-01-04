use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use uuid::Uuid;

use crate::models::CreateCostAttribution;

/// Job data for cost attribution
#[derive(Debug, Clone)]
pub struct JobData {
    pub job_id: Uuid,
    pub user_id: String,
    pub team_id: Option<String>,
    pub customer_id: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub gpu_count: usize,
    pub gpu_type: String,
}

/// Resource usage data for a job
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub network_ingress_gb: Decimal,
    pub network_egress_gb: Decimal,
    pub storage_gb: Decimal,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            network_ingress_gb: Decimal::ZERO,
            network_egress_gb: Decimal::ZERO,
            storage_gb: Decimal::ZERO,
        }
    }
}

/// Pricing rates for cost calculation
#[derive(Debug, Clone)]
pub struct PricingRates {
    pub gpu_hourly_rate: Decimal,
    pub cpu_hourly_rate: Decimal,
    pub network_rate_per_gb: Decimal,
    pub storage_rate_per_gb_hour: Decimal,
}

/// Cost allocator engine
pub struct CostAllocator {
    default_rates: PricingRates,
}

impl CostAllocator {
    pub fn new(default_rates: PricingRates) -> Self {
        Self { default_rates }
    }

    /// Allocate costs for a single job
    pub fn allocate_job_cost(
        &self,
        job: &JobData,
        usage: &ResourceUsage,
        pricing_rates: Option<&PricingRates>,
    ) -> crate::error::Result<CreateCostAttribution> {
        let rates = pricing_rates.unwrap_or(&self.default_rates);

        // Calculate GPU cost
        let gpu_hours = crate::attribution::gpu_hours::calculate_gpu_hours(
            job.start_time,
            job.end_time,
            job.gpu_count,
        )?;
        let gpu_cost =
            crate::attribution::gpu_hours::calculate_gpu_cost(gpu_hours, rates.gpu_hourly_rate)?;

        // Calculate CPU cost (if applicable)
        let cpu_cost = Decimal::ZERO; // TODO: implement if needed

        // Calculate network cost
        let network_cost = crate::attribution::network::calculate_network_cost(
            usage.network_ingress_gb,
            usage.network_egress_gb,
            rates.network_rate_per_gb,
        )?;

        // Calculate storage cost
        let storage_cost = crate::attribution::storage::calculate_storage_cost(
            usage.storage_gb,
            job.start_time,
            job.end_time,
            rates.storage_rate_per_gb_hour,
        )?;

        let mut attribution =
            CreateCostAttribution::new(job.user_id.clone(), job.start_time, job.end_time)
                .with_job_id(job.job_id)
                .with_gpu_cost(gpu_cost)
                .with_cpu_cost(cpu_cost)
                .with_network_cost(network_cost)
                .with_storage_cost(storage_cost);

        if let Some(team_id) = &job.team_id {
            attribution = attribution.with_team_id(team_id.clone());
        }

        if let Some(customer_id) = &job.customer_id {
            attribution = attribution.with_customer_id(customer_id.clone());
        }

        attribution.validate()?;

        Ok(attribution)
    }

    /// Validate attribution accuracy against actual billing
    pub fn validate_attribution_accuracy(
        &self,
        attributed_total: Decimal,
        actual_billing: Decimal,
        threshold_percent: Decimal,
    ) -> crate::error::Result<Decimal> {
        use crate::error::AttributorErrorExt;

        if actual_billing == Decimal::ZERO {
            return Err(crate::error::HpcError::calculation_error(
                "Actual billing cannot be zero for accuracy calculation",
            ));
        }

        let variance =
            ((attributed_total - actual_billing).abs() / actual_billing) * Decimal::from(100);

        if variance > threshold_percent {
            return Err(crate::error::HpcError::accuracy_threshold_exceeded(
                variance,
            ));
        }

        Ok(variance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use rust_decimal_macros::dec;

    fn create_test_job() -> JobData {
        let now = Utc::now();
        JobData {
            job_id: Uuid::new_v4(),
            user_id: "user123".to_string(),
            team_id: Some("team456".to_string()),
            customer_id: None,
            start_time: now,
            end_time: now + Duration::hours(1),
            gpu_count: 4,
            gpu_type: "A100".to_string(),
        }
    }

    fn create_test_rates() -> PricingRates {
        PricingRates {
            gpu_hourly_rate: dec!(3.50),
            cpu_hourly_rate: dec!(0.10),
            network_rate_per_gb: dec!(0.09),
            storage_rate_per_gb_hour: dec!(0.00001368),
        }
    }

    #[test]
    fn test_allocate_job_cost_gpu_only() {
        let allocator = CostAllocator::new(create_test_rates());
        let job = create_test_job();
        let usage = ResourceUsage::default();

        let result = allocator.allocate_job_cost(&job, &usage, None).unwrap();

        assert_eq!(result.user_id, "user123");
        assert_eq!(result.gpu_cost, dec!(14.00)); // 4 GPUs * 1 hour * $3.50
        assert_eq!(result.network_cost, Decimal::ZERO);
        assert_eq!(result.storage_cost, Decimal::ZERO);
        assert_eq!(result.total_cost, dec!(14.00));
    }

    #[test]
    fn test_allocate_job_cost_with_network() {
        let allocator = CostAllocator::new(create_test_rates());
        let job = create_test_job();
        let usage = ResourceUsage {
            network_ingress_gb: dec!(10.0),
            network_egress_gb: dec!(50.0),
            storage_gb: Decimal::ZERO,
        };

        let result = allocator.allocate_job_cost(&job, &usage, None).unwrap();

        assert_eq!(result.gpu_cost, dec!(14.00));
        assert_eq!(result.network_cost, dec!(5.40)); // 60 GB * $0.09
        assert_eq!(result.total_cost, dec!(19.40));
    }

    #[test]
    fn test_allocate_job_cost_with_storage() {
        let allocator = CostAllocator::new(create_test_rates());
        let job = create_test_job();
        let usage = ResourceUsage {
            network_ingress_gb: Decimal::ZERO,
            network_egress_gb: Decimal::ZERO,
            storage_gb: dec!(100.0),
        };

        let result = allocator.allocate_job_cost(&job, &usage, None).unwrap();

        assert_eq!(result.gpu_cost, dec!(14.00));
        assert_eq!(result.storage_cost, dec!(0.001368)); // 100 GB * 1 hour * $0.00001368
                                                         // Total should be GPU + storage
        assert!(result.total_cost > dec!(14.00));
        assert!(result.total_cost < dec!(14.01));
    }

    #[test]
    fn test_allocate_job_cost_all_resources() {
        let allocator = CostAllocator::new(create_test_rates());
        let job = create_test_job();
        let usage = ResourceUsage {
            network_ingress_gb: dec!(10.0),
            network_egress_gb: dec!(50.0),
            storage_gb: dec!(100.0),
        };

        let result = allocator.allocate_job_cost(&job, &usage, None).unwrap();

        assert_eq!(result.gpu_cost, dec!(14.00));
        assert_eq!(result.network_cost, dec!(5.40));
        assert!(result.storage_cost > Decimal::ZERO);
        // Total should be sum of all components
        let expected_total =
            result.gpu_cost + result.cpu_cost + result.network_cost + result.storage_cost;
        assert_eq!(result.total_cost, expected_total);
    }

    #[test]
    fn test_allocate_job_cost_custom_rates() {
        let allocator = CostAllocator::new(create_test_rates());
        let job = create_test_job();
        let usage = ResourceUsage::default();

        let custom_rates = PricingRates {
            gpu_hourly_rate: dec!(5.00),
            cpu_hourly_rate: dec!(0.20),
            network_rate_per_gb: dec!(0.12),
            storage_rate_per_gb_hour: dec!(0.00002),
        };

        let result = allocator
            .allocate_job_cost(&job, &usage, Some(&custom_rates))
            .unwrap();

        assert_eq!(result.gpu_cost, dec!(20.00)); // 4 GPUs * 1 hour * $5.00
    }

    #[test]
    fn test_allocate_job_cost_with_team_and_customer() {
        let allocator = CostAllocator::new(create_test_rates());
        let mut job = create_test_job();
        job.customer_id = Some("customer789".to_string());
        let usage = ResourceUsage::default();

        let result = allocator.allocate_job_cost(&job, &usage, None).unwrap();

        assert_eq!(result.team_id, Some("team456".to_string()));
        assert_eq!(result.customer_id, Some("customer789".to_string()));
    }

    #[test]
    fn test_validate_attribution_accuracy_within_threshold() {
        let allocator = CostAllocator::new(create_test_rates());
        let attributed = dec!(100.00);
        let actual = dec!(102.00);
        let threshold = dec!(5.0); // 5%

        let variance = allocator
            .validate_attribution_accuracy(attributed, actual, threshold)
            .unwrap();

        // Variance should be ~1.96%
        assert!(variance < dec!(2.0));
    }

    #[test]
    fn test_validate_attribution_accuracy_exceeds_threshold() {
        let allocator = CostAllocator::new(create_test_rates());
        let attributed = dec!(100.00);
        let actual = dec!(110.00);
        let threshold = dec!(5.0); // 5%

        let result = allocator.validate_attribution_accuracy(attributed, actual, threshold);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Accuracy threshold exceeded"));
    }

    #[test]
    fn test_validate_attribution_accuracy_exact_match() {
        let allocator = CostAllocator::new(create_test_rates());
        let attributed = dec!(100.00);
        let actual = dec!(100.00);
        let threshold = dec!(5.0);

        let variance = allocator
            .validate_attribution_accuracy(attributed, actual, threshold)
            .unwrap();

        assert_eq!(variance, Decimal::ZERO);
    }

    #[test]
    fn test_validate_attribution_accuracy_zero_actual() {
        let allocator = CostAllocator::new(create_test_rates());
        let attributed = dec!(100.00);
        let actual = Decimal::ZERO;
        let threshold = dec!(5.0);

        let result = allocator.validate_attribution_accuracy(attributed, actual, threshold);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Actual billing cannot be zero"));
    }

    #[test]
    fn test_resource_usage_default() {
        let usage = ResourceUsage::default();
        assert_eq!(usage.network_ingress_gb, Decimal::ZERO);
        assert_eq!(usage.network_egress_gb, Decimal::ZERO);
        assert_eq!(usage.storage_gb, Decimal::ZERO);
    }

    #[test]
    fn test_long_running_job_allocation() {
        let allocator = CostAllocator::new(create_test_rates());
        let now = Utc::now();
        let mut job = create_test_job();
        job.start_time = now;
        job.end_time = now + Duration::hours(24); // 1 day
        job.gpu_count = 8;

        let usage = ResourceUsage::default();

        let result = allocator.allocate_job_cost(&job, &usage, None).unwrap();

        assert_eq!(result.gpu_cost, dec!(672.00)); // 8 GPUs * 24 hours * $3.50
    }

    #[test]
    fn test_fractional_hour_allocation() {
        let allocator = CostAllocator::new(create_test_rates());
        let now = Utc::now();
        let mut job = create_test_job();
        job.start_time = now;
        job.end_time = now + Duration::minutes(30);
        job.gpu_count = 2;

        let usage = ResourceUsage::default();

        let result = allocator.allocate_job_cost(&job, &usage, None).unwrap();

        assert_eq!(result.gpu_cost, dec!(3.50)); // 2 GPUs * 0.5 hours * $3.50
    }
}
