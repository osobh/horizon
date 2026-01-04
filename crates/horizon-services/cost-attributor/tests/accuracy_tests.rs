use chrono::{Duration, Utc};
use cost_attributor::attribution::{CostAllocator, JobData, PricingRates, ResourceUsage};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use uuid::Uuid;

// Accuracy validation tests

#[test]
fn test_95_percent_accuracy_threshold() {
    let rates = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(rates);

    // Test at exactly 5% variance (95% accuracy)
    let attributed = dec!(100.00);
    let actual = dec!(105.00);
    let threshold = dec!(5.0);

    let result = allocator.validate_attribution_accuracy(attributed, actual, threshold);
    assert!(result.is_ok());
}

#[test]
fn test_accuracy_under_attribution() {
    let rates = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(rates);

    // Under-attributed by 3%
    let attributed = dec!(100.00);
    let actual = dec!(103.09);
    let threshold = dec!(5.0);

    let variance = allocator
        .validate_attribution_accuracy(attributed, actual, threshold)
        .unwrap();

    assert!(variance < dec!(5.0));
    // Variance should be approximately 3%
    assert!(variance > dec!(2.9) && variance < dec!(3.1));
}

#[test]
fn test_accuracy_over_attribution() {
    let rates = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(rates);

    // Over-attributed by 3%
    let attributed = dec!(103.09);
    let actual = dec!(100.00);
    let threshold = dec!(5.0);

    let variance = allocator
        .validate_attribution_accuracy(attributed, actual, threshold)
        .unwrap();

    assert!(variance < dec!(5.0));
    // Variance should be approximately 3%
    assert!(variance > dec!(2.9) && variance < dec!(3.1));
}

#[test]
fn test_accuracy_real_world_scenario() {
    let rates = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(rates.clone());

    // Simulate a real job
    let now = Utc::now();
    let job = JobData {
        job_id: Uuid::new_v4(),
        user_id: "user123".to_string(),
        team_id: Some("team456".to_string()),
        customer_id: None,
        start_time: now,
        end_time: now + Duration::hours(10),
        gpu_count: 8,
        gpu_type: "A100".to_string(),
    };

    let usage = ResourceUsage {
        network_ingress_gb: dec!(50.0),
        network_egress_gb: dec!(200.0),
        storage_gb: dec!(1000.0),
    };

    let attribution = allocator
        .allocate_job_cost(&job, &usage, Some(&rates))
        .unwrap();

    // Simulate actual billing (with slight variance)
    // Expected: GPU: 8*10*3.50 = 280, Network: 250*0.09 = 22.5, Storage: ~0.1368
    // Total: ~302.64
    let expected_total = dec!(302.64);
    let actual_billing = dec!(305.00); // Slight variance from actual provider

    let variance = allocator
        .validate_attribution_accuracy(attribution.total_cost, actual_billing, dec!(5.0))
        .unwrap();

    assert!(variance < dec!(5.0));
}

#[test]
fn test_accuracy_batch_jobs() {
    let rates = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(rates.clone());

    let now = Utc::now();
    let mut total_attributed = Decimal::ZERO;

    // Simulate 10 jobs
    for i in 0..10 {
        let job = JobData {
            job_id: Uuid::new_v4(),
            user_id: format!("user{}", i),
            team_id: Some("team456".to_string()),
            customer_id: None,
            start_time: now,
            end_time: now + Duration::hours(1),
            gpu_count: 4,
            gpu_type: "A100".to_string(),
        };

        let usage = ResourceUsage {
            network_ingress_gb: dec!(10.0),
            network_egress_gb: dec!(50.0),
            storage_gb: dec!(100.0),
        };

        let attribution = allocator
            .allocate_job_cost(&job, &usage, Some(&rates))
            .unwrap();
        total_attributed += attribution.total_cost;
    }

    // Simulate actual billing for all jobs
    let actual_billing = dec!(215.00); // Provider's total

    let variance = allocator
        .validate_attribution_accuracy(total_attributed, actual_billing, dec!(10.0))
        .unwrap();

    assert!(variance < dec!(10.0));
}

#[test]
fn test_accuracy_spot_pricing() {
    let spot_rates = PricingRates {
        gpu_hourly_rate: dec!(1.75), // 50% discount
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(spot_rates.clone());

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

    let attribution = allocator
        .allocate_job_cost(&job, &usage, Some(&spot_rates))
        .unwrap();

    // 4 GPUs * 1 hour * $1.75 = $7.00
    assert_eq!(attribution.gpu_cost, dec!(7.00));

    // Validate against actual spot billing
    let actual_billing = dec!(7.10); // Slight variance
    let variance = allocator
        .validate_attribution_accuracy(attribution.total_cost, actual_billing, dec!(5.0))
        .unwrap();

    assert!(variance < dec!(5.0));
}

#[test]
fn test_accuracy_long_running_job() {
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
        team_id: None,
        customer_id: None,
        start_time: now,
        end_time: now + Duration::hours(168), // 1 week
        gpu_count: 8,
        gpu_type: "A100".to_string(),
    };

    let usage = ResourceUsage {
        network_ingress_gb: dec!(1000.0),
        network_egress_gb: dec!(5000.0),
        storage_gb: dec!(10000.0),
    };

    let attribution = allocator
        .allocate_job_cost(&job, &usage, Some(&rates))
        .unwrap();

    // GPU: 8 * 168 * 3.50 = 4704
    assert_eq!(attribution.gpu_cost, dec!(4704.00));

    // Simulate actual billing
    let actual_billing = dec!(5300.00); // Includes some other minor charges

    let variance = allocator
        .validate_attribution_accuracy(attribution.total_cost, actual_billing, dec!(15.0))
        .unwrap();

    assert!(variance < dec!(15.0));
}

#[test]
fn test_accuracy_fails_over_threshold() {
    let rates = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(rates);

    // 10% variance - should fail 5% threshold
    let attributed = dec!(100.00);
    let actual = dec!(110.00);
    let threshold = dec!(5.0);

    let result = allocator.validate_attribution_accuracy(attributed, actual, threshold);

    assert!(result.is_err());
}

#[test]
fn test_accuracy_perfect_match() {
    let rates = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(rates);

    let attributed = dec!(100.00);
    let actual = dec!(100.00);
    let threshold = dec!(5.0);

    let variance = allocator
        .validate_attribution_accuracy(attributed, actual, threshold)
        .unwrap();

    assert_eq!(variance, Decimal::ZERO);
}

#[test]
fn test_accuracy_with_different_thresholds() {
    let rates = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(rates);

    let attributed = dec!(100.00);
    let actual = dec!(107.00);

    // Should pass 10% threshold
    let result_10 = allocator.validate_attribution_accuracy(attributed, actual, dec!(10.0));
    assert!(result_10.is_ok());

    // Should fail 5% threshold
    let result_5 = allocator.validate_attribution_accuracy(attributed, actual, dec!(5.0));
    assert!(result_5.is_err());

    // Should pass 8% threshold
    let result_8 = allocator.validate_attribution_accuracy(attributed, actual, dec!(8.0));
    assert!(result_8.is_ok());
}
