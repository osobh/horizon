use chrono::{Duration, Utc};
use cost_attributor::{
    attribution::{
        calculate_gpu_cost, calculate_gpu_hours, calculate_network_cost, calculate_storage_cost,
        CostAllocator, JobData, PricingRates, ResourceUsage, StorageTier,
    },
    models::{CreateCostAttribution, CreateGpuPricing, PricingModel},
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use uuid::Uuid;

// Unit tests for attribution logic

#[test]
fn test_gpu_hours_calculation_precision() {
    let start = Utc::now();
    let end = start + Duration::seconds(3661); // 1 hour and 1 second

    let result = calculate_gpu_hours(start, end, 1).unwrap();
    // Should be approximately 1.01694... hours (3661/3600)
    assert!(result > dec!(1.016) && result < dec!(1.017));
}

#[test]
fn test_gpu_cost_with_high_precision() {
    let gpu_hours = dec!(123.456789);
    let rate = dec!(3.141592);

    let result = calculate_gpu_cost(gpu_hours, rate).unwrap();
    // 123.456789 * 3.141592 = approximately 387.85...
    assert!(result > dec!(387.85) && result < dec!(387.86));
}

#[test]
fn test_network_cost_large_transfer() {
    let ingress = dec!(1000.0);
    let egress = dec!(5000.0);
    let rate = dec!(0.09);

    let result = calculate_network_cost(ingress, egress, rate).unwrap();
    assert_eq!(result, dec!(540.00));
}

#[test]
fn test_storage_cost_one_month() {
    let storage = dec!(500.0);
    let start = Utc::now();
    let end = start + Duration::hours(730); // ~30 days
    let rate = dec!(0.00001368);

    let result = calculate_storage_cost(storage, start, end, rate).unwrap();
    // 500 GB * 730 hours * 0.00001368 ≈ $4.993
    assert!(result > dec!(4.99) && result < dec!(5.01));
}

#[test]
fn test_tiered_storage_cost_comparison() {
    let storage = dec!(100.0);
    let start = Utc::now();
    let end = start + Duration::hours(730);

    let standard = cost_attributor::attribution::calculate_tiered_storage_cost(
        storage,
        start,
        end,
        StorageTier::Standard,
    )
    .unwrap();

    let archive = cost_attributor::attribution::calculate_tiered_storage_cost(
        storage,
        start,
        end,
        StorageTier::Archive,
    )
    .unwrap();

    // Archive should be cheaper than standard
    assert!(archive < standard);
}

#[test]
fn test_full_job_allocation() {
    let rates = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(rates.clone());

    let now = Utc::now();
    let job = JobData {
        job_id: Uuid::new_v4(),
        user_id: "user123".to_string(),
        team_id: Some("team456".to_string()),
        customer_id: Some("customer789".to_string()),
        start_time: now,
        end_time: now + Duration::hours(2),
        gpu_count: 8,
        gpu_type: "A100".to_string(),
    };

    let usage = ResourceUsage {
        network_ingress_gb: dec!(10.0),
        network_egress_gb: dec!(100.0),
        storage_gb: dec!(500.0),
    };

    let result = allocator
        .allocate_job_cost(&job, &usage, Some(&rates))
        .unwrap();

    // GPU: 8 GPUs * 2 hours * $3.50 = $56.00
    assert_eq!(result.gpu_cost, dec!(56.00));

    // Network: 110 GB * $0.09 = $9.90
    assert_eq!(result.network_cost, dec!(9.90));

    // Storage: 500 GB * 2 hours * $0.00001368 ≈ $0.01368
    assert!(result.storage_cost > dec!(0.013) && result.storage_cost < dec!(0.014));

    // Total should match sum
    let expected_total =
        result.gpu_cost + result.cpu_cost + result.network_cost + result.storage_cost;
    assert_eq!(result.total_cost, expected_total);

    // Metadata
    assert_eq!(result.user_id, "user123");
    assert_eq!(result.team_id, Some("team456".to_string()));
    assert_eq!(result.customer_id, Some("customer789".to_string()));
}

#[test]
fn test_accuracy_validation_within_threshold() {
    let rates = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(rates);

    let attributed = dec!(1000.00);
    let actual = dec!(1020.00);
    let threshold = dec!(5.0); // 5%

    let variance = allocator
        .validate_attribution_accuracy(attributed, actual, threshold)
        .unwrap();

    // Variance should be approximately 1.96%
    assert!(variance > dec!(1.96) && variance < dec!(1.97));
}

#[test]
fn test_accuracy_validation_exceeds_threshold() {
    let rates = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(rates);

    let attributed = dec!(1000.00);
    let actual = dec!(1100.00);
    let threshold = dec!(5.0); // 5%

    let result = allocator.validate_attribution_accuracy(attributed, actual, threshold);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Accuracy threshold exceeded"));
}

#[test]
fn test_cost_attribution_validation() {
    let now = Utc::now();
    let attribution =
        CreateCostAttribution::new("user123".to_string(), now, now + Duration::hours(1))
            .with_gpu_cost(dec!(100.00))
            .with_network_cost(dec!(10.00));

    assert!(attribution.validate().is_ok());
    assert_eq!(attribution.total_cost, dec!(110.00));
}

#[test]
fn test_gpu_pricing_validation() {
    let now = Utc::now();
    let pricing =
        CreateGpuPricing::new("A100".to_string(), PricingModel::OnDemand, dec!(3.50), now);

    assert!(pricing.validate().is_ok());
}

#[test]
fn test_multiple_gpu_types_allocation() {
    let rates_a100 = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let rates_v100 = PricingRates {
        gpu_hourly_rate: dec!(2.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator_a100 = CostAllocator::new(rates_a100.clone());
    let allocator_v100 = CostAllocator::new(rates_v100.clone());

    let now = Utc::now();
    let job = JobData {
        job_id: Uuid::new_v4(),
        user_id: "user123".to_string(),
        team_id: None,
        customer_id: None,
        start_time: now,
        end_time: now + Duration::hours(1),
        gpu_count: 4,
        gpu_type: "A100".to_string(),
    };

    let usage = ResourceUsage::default();

    let a100_cost = allocator_a100
        .allocate_job_cost(&job, &usage, Some(&rates_a100))
        .unwrap();
    let v100_cost = allocator_v100
        .allocate_job_cost(&job, &usage, Some(&rates_v100))
        .unwrap();

    // A100 should cost more than V100 for same job
    assert!(a100_cost.gpu_cost > v100_cost.gpu_cost);
}

#[test]
fn test_zero_duration_job_fails() {
    let now = Utc::now();
    let result = calculate_gpu_hours(now, now, 1);

    assert!(result.is_err());
}

#[test]
fn test_very_large_gpu_count() {
    let start = Utc::now();
    let end = start + Duration::hours(1);

    let result = calculate_gpu_hours(start, end, 1000).unwrap();
    assert_eq!(result, dec!(1000.0));
}

#[test]
fn test_cost_attribution_without_team_or_customer() {
    let now = Utc::now();
    let attribution =
        CreateCostAttribution::new("user123".to_string(), now, now + Duration::hours(1))
            .with_gpu_cost(dec!(100.00));

    assert!(attribution.validate().is_ok());
    assert!(attribution.team_id.is_none());
    assert!(attribution.customer_id.is_none());
}
