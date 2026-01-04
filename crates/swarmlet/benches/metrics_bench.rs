//! Benchmarks for build metrics collection
//!
//! Run with: cargo bench --bench metrics_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use uuid::Uuid;

use chrono::{Duration, Utc};
use swarmlet::build_job::BuildResourceUsage;
use swarmlet::build_metrics::{
    BuildMetricsCollector, BuildOutcome, BuildRecord, CommandType, ProfileType,
};

/// Create a test build record
fn create_build_record(outcome: BuildOutcome, duration_secs: f64) -> BuildRecord {
    BuildRecord {
        job_id: Uuid::new_v4(),
        command: CommandType::Build,
        profile: ProfileType::Debug,
        status: outcome,
        started_at: Utc::now() - Duration::seconds(duration_secs as i64),
        completed_at: Utc::now(),
        duration_seconds: duration_secs,
        resource_usage: BuildResourceUsage {
            cpu_seconds: duration_secs * 0.8,
            peak_memory_mb: 512.0,
            disk_read_bytes: 50 * 1024 * 1024,
            disk_write_bytes: 100 * 1024 * 1024,
            compile_time_seconds: duration_secs,
            crates_compiled: 25,
            cache_hits: 10,
            cache_misses: 15,
        },
        cache_enabled: true,
        source_cache_hit: false,
        toolchain: "stable".to_string(),
    }
}

/// Create a test build record with specific command type
fn create_build_record_with_command(
    command: CommandType,
    outcome: BuildOutcome,
    duration_secs: f64,
) -> BuildRecord {
    let mut record = create_build_record(outcome, duration_secs);
    record.command = command;
    record
}

fn bench_record_build(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("metrics_record");
    group.throughput(Throughput::Elements(1));

    group.bench_function("record_single", |b| {
        b.to_async(&rt).iter(|| async {
            let collector = BuildMetricsCollector::new(1000);
            let record = create_build_record(BuildOutcome::Success, 30.0);
            black_box(collector.record_build(record).await);
        });
    });

    group.bench_function("record_100", |b| {
        b.to_async(&rt).iter(|| async {
            let collector = BuildMetricsCollector::new(1000);
            for i in 0..100 {
                let outcome = if i % 5 == 0 {
                    BuildOutcome::Failed
                } else {
                    BuildOutcome::Success
                };
                let record = create_build_record(outcome, 30.0 + i as f64);
                collector.record_build(record).await;
            }
        });
    });

    group.finish();
}

fn bench_get_stats(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("metrics_stats");

    for size in [10, 100, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("get_stats", size), &size, |b, &size| {
            b.to_async(&rt).iter_batched(
                || {
                    let rt = tokio::runtime::Handle::current();
                    rt.block_on(async {
                        let collector = BuildMetricsCollector::new(size + 100);
                        for i in 0..size {
                            let outcome = if i % 5 == 0 {
                                BuildOutcome::Failed
                            } else {
                                BuildOutcome::Success
                            };
                            let command = match i % 4 {
                                0 => CommandType::Build,
                                1 => CommandType::Test,
                                2 => CommandType::Check,
                                _ => CommandType::Clippy,
                            };
                            let record =
                                create_build_record_with_command(command, outcome, 30.0 + i as f64);
                            collector.record_build(record).await;
                        }
                        collector
                    })
                },
                |collector| async move {
                    black_box(collector.get_stats().await);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_get_snapshot(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("metrics_snapshot");

    for size in [10, 100, 500] {
        group.bench_with_input(BenchmarkId::new("get_snapshot", size), &size, |b, &size| {
            b.to_async(&rt).iter_batched(
                || {
                    let rt = tokio::runtime::Handle::current();
                    rt.block_on(async {
                        let collector = BuildMetricsCollector::new(size + 100);
                        for i in 0..size {
                            let outcome = if i % 5 == 0 {
                                BuildOutcome::Failed
                            } else {
                                BuildOutcome::Success
                            };
                            collector
                                .record_build(create_build_record(outcome, 30.0 + i as f64))
                                .await;
                        }
                        collector
                    })
                },
                |collector| async move {
                    black_box(collector.get_snapshot().await);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_get_summary(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("metrics_summary");

    group.bench_function("summary_last_hour", |b| {
        b.to_async(&rt).iter_batched(
            || {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let collector = BuildMetricsCollector::new(1000);
                    for i in 0..100 {
                        let outcome = if i % 5 == 0 {
                            BuildOutcome::Failed
                        } else {
                            BuildOutcome::Success
                        };
                        collector
                            .record_build(create_build_record(outcome, 30.0 + i as f64))
                            .await;
                    }
                    collector
                })
            },
            |collector| async move {
                black_box(collector.get_summary(Duration::hours(1)).await);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("summary_last_24h", |b| {
        b.to_async(&rt).iter_batched(
            || {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let collector = BuildMetricsCollector::new(1000);
                    for i in 0..100 {
                        let outcome = if i % 5 == 0 {
                            BuildOutcome::Failed
                        } else {
                            BuildOutcome::Success
                        };
                        collector
                            .record_build(create_build_record(outcome, 30.0 + i as f64))
                            .await;
                    }
                    collector
                })
            },
            |collector| async move {
                black_box(collector.get_summary(Duration::hours(24)).await);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_concurrent_recording(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("metrics_concurrent");
    group.sample_size(50);

    group.bench_function("concurrent_record", |b| {
        b.to_async(&rt).iter(|| async {
            use std::sync::Arc;

            let collector = Arc::new(BuildMetricsCollector::new(10000));

            let mut handles = vec![];

            // Spawn 8 concurrent recording tasks
            for _ in 0..8 {
                let collector = collector.clone();
                handles.push(tokio::spawn(async move {
                    for i in 0..50 {
                        let outcome = if i % 5 == 0 {
                            BuildOutcome::Failed
                        } else {
                            BuildOutcome::Success
                        };
                        collector
                            .record_build(create_build_record(outcome, 30.0))
                            .await;
                    }
                }));
            }

            for handle in handles {
                handle.await.unwrap();
            }
        });
    });

    group.bench_function("concurrent_record_and_read", |b| {
        b.to_async(&rt).iter(|| async {
            use std::sync::Arc;

            let collector = Arc::new(BuildMetricsCollector::new(10000));

            let mut handles = vec![];

            // Spawn 4 writers
            for _ in 0..4 {
                let collector = collector.clone();
                handles.push(tokio::spawn(async move {
                    for i in 0..50 {
                        let outcome = if i % 5 == 0 {
                            BuildOutcome::Failed
                        } else {
                            BuildOutcome::Success
                        };
                        collector
                            .record_build(create_build_record(outcome, 30.0))
                            .await;
                    }
                }));
            }

            // Spawn 4 readers
            for _ in 0..4 {
                let collector = collector.clone();
                handles.push(tokio::spawn(async move {
                    for _ in 0..20 {
                        let _ = collector.get_stats().await;
                        tokio::time::sleep(std::time::Duration::from_micros(100)).await;
                    }
                }));
            }

            for handle in handles {
                handle.await.unwrap();
            }
        });
    });

    group.finish();
}

fn bench_max_records_eviction(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("metrics_eviction");

    group.bench_function("eviction_at_limit", |b| {
        b.to_async(&rt).iter_batched(
            || {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let collector = BuildMetricsCollector::new(100);
                    // Fill to limit
                    for _ in 0..100 {
                        collector
                            .record_build(create_build_record(BuildOutcome::Success, 30.0))
                            .await;
                    }
                    collector
                })
            },
            |collector| async move {
                // This should trigger eviction
                collector
                    .record_build(create_build_record(BuildOutcome::Success, 30.0))
                    .await;
                black_box(());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_record_build,
    bench_get_stats,
    bench_get_snapshot,
    bench_get_summary,
    bench_concurrent_recording,
    bench_max_records_eviction,
);

criterion_main!(benches);
