//! Benchmarks for build queue operations
//!
//! Run with: cargo bench --bench build_queue_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::PathBuf;

use swarmlet::build_job::{BuildJob, BuildSource, CargoCommand};
use swarmlet::build_queue::{BuildPriority, BuildQueue, QueueConfig, QueuedBuild};

/// Create a test build job
fn create_test_job() -> BuildJob {
    BuildJob::new(
        CargoCommand::Build,
        BuildSource::Local {
            path: PathBuf::from("/tmp/test-project"),
        },
    )
}

/// Create a queued build with specific priority
fn create_queued_build(priority: BuildPriority) -> QueuedBuild {
    QueuedBuild::new(create_test_job()).with_priority(priority)
}

/// Create a queued build for a specific user
fn create_queued_build_for_user(user_id: &str, priority: BuildPriority) -> QueuedBuild {
    QueuedBuild::new(create_test_job())
        .with_priority(priority)
        .with_user(user_id.to_string())
}

fn bench_enqueue(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("queue_enqueue");
    group.throughput(Throughput::Elements(1));

    group.bench_function("enqueue_single", |b| {
        b.to_async(&rt).iter(|| async {
            let queue = BuildQueue::default_new();
            let build = create_queued_build(BuildPriority::Normal);
            black_box(queue.enqueue(build).await.unwrap());
        });
    });

    group.bench_function("enqueue_100", |b| {
        b.to_async(&rt).iter(|| async {
            let queue = BuildQueue::new(QueueConfig {
                max_queue_size: 1000,
                max_per_user: 1000,
                ..Default::default()
            });
            for _ in 0..100 {
                let build = create_queued_build(BuildPriority::Normal);
                queue.enqueue(build).await.unwrap();
            }
        });
    });

    group.finish();
}

fn bench_dequeue(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("queue_dequeue");
    group.throughput(Throughput::Elements(1));

    for size in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("dequeue_from", size), &size, |b, &size| {
            b.to_async(&rt).iter_batched(
                || {
                    let rt = tokio::runtime::Handle::current();
                    rt.block_on(async {
                        let queue = BuildQueue::new(QueueConfig {
                            max_queue_size: size + 100,
                            max_per_user: size + 100,
                            ..Default::default()
                        });
                        for _ in 0..size {
                            queue
                                .enqueue(create_queued_build(BuildPriority::Normal))
                                .await
                                .unwrap();
                        }
                        queue
                    })
                },
                |queue| async move {
                    black_box(queue.dequeue().await);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_priority_ordering(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("queue_priority");

    group.bench_function("mixed_priorities_100", |b| {
        b.to_async(&rt).iter(|| async {
            let queue = BuildQueue::new(QueueConfig {
                max_queue_size: 1000,
                max_per_user: 1000,
                ..Default::default()
            });

            // Enqueue with mixed priorities
            for i in 0..100 {
                let priority = match i % 5 {
                    0 => BuildPriority::Critical,
                    1 => BuildPriority::High,
                    2 => BuildPriority::Normal,
                    3 => BuildPriority::Low,
                    _ => BuildPriority::Batch,
                };
                queue.enqueue(create_queued_build(priority)).await.unwrap();
            }

            // Dequeue all (should be in priority order)
            for _ in 0..100 {
                black_box(queue.dequeue().await);
            }
        });
    });

    group.finish();
}

fn bench_fair_scheduling(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("queue_fair_scheduling");

    group.bench_function("multi_user_enqueue", |b| {
        b.to_async(&rt).iter(|| async {
            let queue = BuildQueue::new(QueueConfig {
                max_queue_size: 1000,
                max_per_user: 50,
                ..Default::default()
            });

            // 10 users each enqueueing 10 jobs
            for user in 0..10 {
                let user_id = format!("user-{}", user);
                for _ in 0..10 {
                    queue
                        .enqueue(create_queued_build_for_user(
                            &user_id,
                            BuildPriority::Normal,
                        ))
                        .await
                        .unwrap();
                }
            }
        });
    });

    group.bench_function("user_limit_check", |b| {
        b.to_async(&rt).iter_batched(
            || {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let queue = BuildQueue::new(QueueConfig {
                        max_queue_size: 100,
                        max_per_user: 5,
                        ..Default::default()
                    });
                    // Fill user's quota
                    for _ in 0..5 {
                        queue
                            .enqueue(create_queued_build_for_user("user1", BuildPriority::Normal))
                            .await
                            .unwrap();
                    }
                    queue
                })
            },
            |queue| async move {
                // This should fail due to user limit
                let _ = black_box(
                    queue
                        .enqueue(create_queued_build_for_user("user1", BuildPriority::Normal))
                        .await,
                );
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_queue_status(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("queue_status");

    for size in [10, 100, 500] {
        group.bench_with_input(BenchmarkId::new("status", size), &size, |b, &size| {
            b.to_async(&rt).iter_batched(
                || {
                    let rt = tokio::runtime::Handle::current();
                    rt.block_on(async {
                        let queue = BuildQueue::new(QueueConfig {
                            max_queue_size: size + 100,
                            max_per_user: size + 100,
                            ..Default::default()
                        });
                        for i in 0..size {
                            let user = format!("user-{}", i % 10);
                            let priority = match i % 5 {
                                0 => BuildPriority::Critical,
                                1 => BuildPriority::High,
                                2 => BuildPriority::Normal,
                                3 => BuildPriority::Low,
                                _ => BuildPriority::Batch,
                            };
                            queue
                                .enqueue(create_queued_build_for_user(&user, priority))
                                .await
                                .unwrap();
                        }
                        queue
                    })
                },
                |queue| async move {
                    black_box(queue.status().await);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("queue_concurrent");
    group.sample_size(50); // Reduce sample size for concurrent tests

    group.bench_function("concurrent_enqueue_dequeue", |b| {
        b.to_async(&rt).iter(|| async {
            use std::sync::Arc;

            let queue = Arc::new(BuildQueue::new(QueueConfig {
                max_queue_size: 10000,
                max_per_user: 1000,
                ..Default::default()
            }));

            let mut handles = vec![];

            // Spawn 4 producers
            for i in 0..4 {
                let queue = queue.clone();
                let user = format!("producer-{}", i);
                handles.push(tokio::spawn(async move {
                    for _ in 0..50 {
                        let _ = queue
                            .enqueue(create_queued_build_for_user(&user, BuildPriority::Normal))
                            .await;
                    }
                }));
            }

            // Spawn 2 consumers
            for _ in 0..2 {
                let queue = queue.clone();
                handles.push(tokio::spawn(async move {
                    for _ in 0..50 {
                        let _ = queue.dequeue().await;
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

criterion_group!(
    benches,
    bench_enqueue,
    bench_dequeue,
    bench_priority_ordering,
    bench_fair_scheduling,
    bench_queue_status,
    bench_concurrent_operations,
);

criterion_main!(benches);
