use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use horizon_quota_manager::models::*;
use rust_decimal_macros::dec;
use uuid::Uuid;

fn bench_quota_validation(c: &mut Criterion) {
    let quota = Quota {
        id: Uuid::new_v4(),
        entity_type: EntityType::User,
        entity_id: "test-user".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(100.0),
        soft_limit: Some(dec!(80.0)),
        burst_limit: Some(dec!(120.0)),
        overcommit_ratio: dec!(1.5),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };

    c.bench_function("quota_validation", |b| {
        b.iter(|| black_box(&quota).validate().unwrap())
    });
}

fn bench_quota_effective_limit(c: &mut Criterion) {
    let quota = Quota {
        id: Uuid::new_v4(),
        entity_type: EntityType::User,
        entity_id: "test-user".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(100.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: dec!(1.5),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };

    c.bench_function("effective_limit_calculation", |b| {
        b.iter(|| black_box(&quota).effective_limit())
    });
}

fn bench_quota_availability_check(c: &mut Criterion) {
    let quota = Quota {
        id: Uuid::new_v4(),
        entity_type: EntityType::User,
        entity_id: "test-user".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(1000.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: dec!(1.2),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };

    c.bench_function("available_quota_calculation", |b| {
        b.iter(|| black_box(&quota).available_quota(dec!(600.0)))
    });
}

fn bench_hierarchy_validation(c: &mut Criterion) {
    let parent = QuotaHierarchy::new(Quota {
        id: Uuid::new_v4(),
        entity_type: EntityType::Organization,
        entity_id: "org".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(1000.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: dec!(1.0),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    });

    let child = QuotaHierarchy::new(Quota {
        id: Uuid::new_v4(),
        entity_type: EntityType::Team,
        entity_id: "team".to_string(),
        parent_id: Some(parent.quota.id),
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(500.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: dec!(1.0),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    })
    .with_parent(parent);

    c.bench_function("hierarchy_validation", |b| {
        b.iter(|| black_box(&child).validate_hierarchy().unwrap())
    });
}

fn bench_usage_stats_calculation(c: &mut Criterion) {
    let quota = Quota {
        id: Uuid::new_v4(),
        entity_type: EntityType::User,
        entity_id: "user".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(1000.0),
        soft_limit: Some(dec!(800.0)),
        burst_limit: None,
        overcommit_ratio: dec!(1.5),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };

    let mut group = c.benchmark_group("usage_stats");
    for usage in [dec!(100.0), dec!(500.0), dec!(900.0), dec!(1400.0)].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(usage), usage, |b, &u| {
            b.iter(|| QuotaUsageStats::from_quota(black_box(&quota), u))
        });
    }
    group.finish();
}

fn bench_allocation_checks(c: &mut Criterion) {
    let allocation = Allocation {
        id: Uuid::new_v4(),
        quota_id: Uuid::new_v4(),
        job_id: Uuid::new_v4(),
        resource_type: ResourceType::GpuHours,
        allocated_value: dec!(50.0),
        allocated_at: chrono::Utc::now(),
        released_at: None,
        version: 0,
        metadata: None,
    };

    c.bench_function("allocation_is_active", |b| {
        b.iter(|| black_box(&allocation).is_active())
    });
}

criterion_group!(
    benches,
    bench_quota_validation,
    bench_quota_effective_limit,
    bench_quota_availability_check,
    bench_hierarchy_validation,
    bench_usage_stats_calculation,
    bench_allocation_checks
);
criterion_main!(benches);
