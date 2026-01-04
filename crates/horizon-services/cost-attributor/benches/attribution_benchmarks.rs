use chrono::{Duration, Utc};
use cost_attributor::attribution::{
    calculate_gpu_cost, calculate_gpu_hours, calculate_network_cost, calculate_storage_cost,
    CostAllocator, JobData, PricingRates, ResourceUsage,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_decimal_macros::dec;
use uuid::Uuid;

fn bench_gpu_hours_calculation(c: &mut Criterion) {
    let start = Utc::now();
    let end = start + Duration::hours(1);

    c.bench_function("calculate_gpu_hours", |b| {
        b.iter(|| calculate_gpu_hours(black_box(start), black_box(end), black_box(4)))
    });
}

fn bench_gpu_cost_calculation(c: &mut Criterion) {
    c.bench_function("calculate_gpu_cost", |b| {
        b.iter(|| calculate_gpu_cost(black_box(dec!(10.5)), black_box(dec!(3.50))))
    });
}

fn bench_network_cost_calculation(c: &mut Criterion) {
    c.bench_function("calculate_network_cost", |b| {
        b.iter(|| {
            calculate_network_cost(
                black_box(dec!(10.0)),
                black_box(dec!(50.0)),
                black_box(dec!(0.09)),
            )
        })
    });
}

fn bench_storage_cost_calculation(c: &mut Criterion) {
    let start = Utc::now();
    let end = start + Duration::hours(730);

    c.bench_function("calculate_storage_cost", |b| {
        b.iter(|| {
            calculate_storage_cost(
                black_box(dec!(100.0)),
                black_box(start),
                black_box(end),
                black_box(dec!(0.00001368)),
            )
        })
    });
}

fn bench_full_job_allocation(c: &mut Criterion) {
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
        customer_id: None,
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

    c.bench_function("allocate_job_cost", |b| {
        b.iter(|| {
            allocator.allocate_job_cost(black_box(&job), black_box(&usage), black_box(Some(&rates)))
        })
    });
}

fn bench_batch_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_allocation");

    for batch_size in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &size| {
                let rates = PricingRates {
                    gpu_hourly_rate: dec!(3.50),
                    cpu_hourly_rate: dec!(0.10),
                    network_rate_per_gb: dec!(0.09),
                    storage_rate_per_gb_hour: dec!(0.00001368),
                };

                let allocator = CostAllocator::new(rates.clone());

                b.iter(|| {
                    let now = Utc::now();
                    for i in 0..size {
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

                        let _ = allocator.allocate_job_cost(&job, &usage, Some(&rates));
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_accuracy_validation(c: &mut Criterion) {
    let rates = PricingRates {
        gpu_hourly_rate: dec!(3.50),
        cpu_hourly_rate: dec!(0.10),
        network_rate_per_gb: dec!(0.09),
        storage_rate_per_gb_hour: dec!(0.00001368),
    };

    let allocator = CostAllocator::new(rates);

    c.bench_function("validate_attribution_accuracy", |b| {
        b.iter(|| {
            allocator.validate_attribution_accuracy(
                black_box(dec!(1000.00)),
                black_box(dec!(1020.00)),
                black_box(dec!(5.0)),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_gpu_hours_calculation,
    bench_gpu_cost_calculation,
    bench_network_cost_calculation,
    bench_storage_cost_calculation,
    bench_full_job_allocation,
    bench_batch_allocation,
    bench_accuracy_validation
);

criterion_main!(benches);
