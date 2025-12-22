use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scheduler::{
    config::Config,
    models::{Job, Priority},
    queue::PriorityQueue,
    scheduler::Scheduler,
};
use sqlx::PgPool;
use std::sync::Arc;

/// Benchmark priority queue operations
fn bench_priority_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("priority_queue");

    // Benchmark enqueue
    group.bench_function("enqueue_single", |b| {
        b.iter(|| {
            let mut queue = PriorityQueue::new();
            let job = Job::builder()
                .user_id("user1")
                .gpu_count(4)
                .priority(Priority::High)
                .build()
                .unwrap();
            queue.enqueue(black_box(job));
        });
    });

    // Benchmark dequeue
    group.bench_function("dequeue_from_queue_of_100", |b| {
        b.iter_batched(
            || {
                let mut queue = PriorityQueue::new();
                for i in 0..100 {
                    let job = Job::builder()
                        .user_id(&format!("user{}", i))
                        .gpu_count(i % 8 + 1)
                        .priority(match i % 3 {
                            0 => Priority::High,
                            1 => Priority::Normal,
                            _ => Priority::Low,
                        })
                        .build()
                        .unwrap();
                    queue.enqueue(job);
                }
                queue
            },
            |mut queue| black_box(queue.dequeue()),
            criterion::BatchSize::SmallInput,
        );
    });

    // Benchmark throughput for bulk operations
    for size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(size));
        group.bench_with_input(BenchmarkId::new("enqueue_bulk", size), &size, |b, &size| {
            b.iter(|| {
                let mut queue = PriorityQueue::new();
                for i in 0..size {
                    let job = Job::builder()
                        .user_id(&format!("user{}", i))
                        .gpu_count((i % 8) as usize + 1)
                        .priority(match i % 3 {
                            0 => Priority::High,
                            1 => Priority::Normal,
                            _ => Priority::Low,
                        })
                        .build()
                        .unwrap();
                    queue.enqueue(black_box(job));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark job builder
fn bench_job_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("job_creation");

    group.bench_function("minimal_job", |b| {
        b.iter(|| {
            Job::builder()
                .user_id("user1")
                .gpu_count(4)
                .build()
                .unwrap()
        });
    });

    group.bench_function("full_job", |b| {
        b.iter(|| {
            Job::builder()
                .user_id("user1")
                .job_name("training-job")
                .gpu_count(8)
                .gpu_type("H100")
                .cpu_cores(64)
                .memory_gb(512)
                .priority(Priority::High)
                .command("python train.py")
                .working_dir("/workspace")
                .build()
                .unwrap()
        });
    });

    group.finish();
}

/// Benchmark scheduler operations (async)
fn bench_scheduler_operations(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("scheduler_operations");
    group.sample_size(10); // Reduce sample size for async benchmarks

    // Setup: Create test database and scheduler
    let (pool, scheduler) = runtime.block_on(async {
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/scheduler_test".to_string());

        let pool = PgPool::connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        // Clean up
        sqlx::query("TRUNCATE jobs, user_usage, job_events CASCADE")
            .execute(&pool)
            .await
            .expect("Failed to clean up");

        let config = Config::from_env().expect("Failed to load config");
        let scheduler = Scheduler::new(config, pool.clone())
            .await
            .expect("Failed to create scheduler");

        (pool, Arc::new(scheduler))
    });

    // Benchmark submit_job
    group.bench_function("submit_job", |b| {
        b.iter(|| {
            let scheduler = scheduler.clone();
            runtime.block_on(async move {
                let job = Job::builder()
                    .user_id("bench_user")
                    .gpu_count(2)
                    .build()
                    .unwrap();

                scheduler.submit_job(job).await.unwrap()
            })
        });
    });

    // Benchmark get_job
    let job_id = runtime.block_on(async {
        let job = Job::builder()
            .user_id("bench_user")
            .gpu_count(2)
            .build()
            .unwrap();
        let created = scheduler.submit_job(job).await.unwrap();
        created.id
    });

    group.bench_function("get_job", |b| {
        b.iter(|| {
            let scheduler = scheduler.clone();
            runtime.block_on(async move {
                scheduler.get_job(job_id).await.unwrap()
            })
        });
    });

    // Benchmark get_queue_stats
    group.bench_function("get_queue_stats", |b| {
        b.iter(|| {
            let scheduler = scheduler.clone();
            runtime.block_on(async move {
                scheduler.get_queue_stats().await.unwrap()
            })
        });
    });

    // Cleanup
    runtime.block_on(async {
        sqlx::query("TRUNCATE jobs, user_usage, job_events CASCADE")
            .execute(&pool)
            .await
            .expect("Failed to clean up");
    });

    group.finish();
}

/// Benchmark concurrent job submissions
fn bench_concurrent_submissions(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("concurrent_operations");
    group.sample_size(10);

    let (pool, scheduler) = runtime.block_on(async {
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/scheduler_test".to_string());

        let pool = PgPool::connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        sqlx::query("TRUNCATE jobs, user_usage, job_events CASCADE")
            .execute(&pool)
            .await
            .expect("Failed to clean up");

        let config = Config::from_env().expect("Failed to load config");
        let scheduler = Scheduler::new(config, pool.clone())
            .await
            .expect("Failed to create scheduler");

        (pool, Arc::new(scheduler))
    });

    for concurrency in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("submit_jobs_concurrent", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    let scheduler = scheduler.clone();
                    runtime.block_on(async move {
                        let mut handles = vec![];
                        for i in 0..concurrency {
                            let scheduler = scheduler.clone();
                            let handle = tokio::spawn(async move {
                                let job = Job::builder()
                                    .user_id(&format!("user{}", i))
                                    .gpu_count(i % 4 + 1)
                                    .build()
                                    .unwrap();
                                scheduler.submit_job(job).await
                            });
                            handles.push(handle);
                        }

                        for handle in handles {
                            handle.await.unwrap().unwrap();
                        }
                    })
                });
            },
        );

        // Clean up after each benchmark
        runtime.block_on(async {
            sqlx::query("TRUNCATE jobs CASCADE")
                .execute(&pool)
                .await
                .unwrap();
        });
    }

    runtime.block_on(async {
        sqlx::query("TRUNCATE jobs, user_usage, job_events CASCADE")
            .execute(&pool)
            .await
            .expect("Failed to clean up");
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_priority_queue,
    bench_job_creation,
    bench_scheduler_operations,
    bench_concurrent_submissions
);
criterion_main!(benches);
