//! Comprehensive benchmarks for GPU communication patterns.
//!
//! This benchmark suite compares the channel-based approach with the old
//! Mutex+VecDeque pattern to demonstrate performance improvements.

#![allow(missing_docs)]

use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;
use tokio::sync::{broadcast, oneshot};

// Import types from stratoswarm-core
use stratoswarm_core::channels::GpuCommand;

/// Old pattern: Mutex-protected VecDeque for command queue
#[derive(Debug, Clone)]
struct LegacyCommandQueue {
    queue: Arc<Mutex<VecDeque<GpuCommand>>>,
}

impl LegacyCommandQueue {
    fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    fn push(&self, command: GpuCommand) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(command);
    }

    fn pop(&self) -> Option<GpuCommand> {
        let mut queue = self.queue.lock().unwrap();
        queue.pop_front()
    }

    fn len(&self) -> usize {
        let queue = self.queue.lock().unwrap();
        queue.len()
    }
}

/// Create a small GPU command for testing
fn create_small_command() -> GpuCommand {
    GpuCommand::Synchronize { stream_id: None }
}

/// Create a medium-sized GPU command
fn create_medium_command() -> GpuCommand {
    GpuCommand::LaunchKernel {
        kernel_id: "test_kernel".to_string(),
        grid_dim: (32, 1, 1),
        block_dim: (256, 1, 1),
        params: Bytes::from(vec![0u8; 128]),
    }
}

/// Create a large GPU command with significant data transfer
fn create_large_command() -> GpuCommand {
    GpuCommand::TransferToDevice {
        buffer_id: "large_buffer".to_string(),
        data: Bytes::from(vec![0u8; 1024 * 1024]), // 1MB
        offset: 0,
    }
}

/// Benchmark channel send latency
fn channel_send_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("channel_send_latency");

    // Benchmark small command
    group.bench_function("small_command_bounded", |b| {
        let (tx, _rx) = broadcast::channel(100);

        b.iter(|| {
            let command = create_small_command();
            let result = tx.send(black_box(command));
            black_box(result)
        });
    });

    // Benchmark medium command
    group.bench_function("medium_command_bounded", |b| {
        let (tx, _rx) = broadcast::channel(100);

        b.iter(|| {
            let command = create_medium_command();
            let result = tx.send(black_box(command));
            black_box(result)
        });
    });

    // Benchmark large command with zero-copy Bytes
    group.bench_function("large_command_bounded", |b| {
        let (tx, _rx) = broadcast::channel(100);

        b.iter(|| {
            let command = create_large_command();
            let result = tx.send(black_box(command));
            black_box(result)
        });
    });

    // Compare with unbounded channel
    group.bench_function("small_command_unbounded", |b| {
        let (tx, _rx) = broadcast::channel(1000);

        b.iter(|| {
            let command = create_small_command();
            let result = tx.send(black_box(command));
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark Mutex+VecDeque latency (old pattern)
fn mutex_vecdeque_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutex_vecdeque_latency");

    // Benchmark push operation
    group.bench_function("push_small_command", |b| {
        let queue = LegacyCommandQueue::new();

        b.iter(|| {
            let command = create_small_command();
            queue.push(black_box(command));
        });
    });

    group.bench_function("push_medium_command", |b| {
        let queue = LegacyCommandQueue::new();

        b.iter(|| {
            let command = create_medium_command();
            queue.push(black_box(command));
        });
    });

    group.bench_function("push_large_command", |b| {
        let queue = LegacyCommandQueue::new();

        b.iter(|| {
            let command = create_large_command();
            queue.push(black_box(command));
        });
    });

    // Benchmark pop operation
    group.bench_function("pop_command", |b| {
        let queue = LegacyCommandQueue::new();
        // Pre-populate queue
        for _ in 0..100 {
            queue.push(create_small_command());
        }

        b.iter(|| {
            let result = queue.pop();
            black_box(result);
            // Replenish to maintain queue size
            queue.push(create_small_command());
        });
    });

    // Benchmark push-pop cycle
    group.bench_function("push_pop_cycle", |b| {
        let queue = LegacyCommandQueue::new();

        b.iter(|| {
            let command = create_small_command();
            queue.push(black_box(command));
            let result = queue.pop();
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark round-trip request/response latency
fn round_trip_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("round_trip_latency");

    // Channel-based round trip with oneshot response
    group.bench_function("channel_oneshot", |b| {
        let rt = Runtime::new().unwrap();

        b.iter(|| {
            rt.block_on(async {
                let (tx, mut rx) = broadcast::channel(100);
                let (response_tx, response_rx) = oneshot::channel();

                // Simulate sending command
                let command = create_small_command();
                let _ = tx.send(command);

                // Simulate receiving and processing
                if let Ok(cmd) = rx.recv().await {
                    black_box(cmd);
                    // Send response
                    let _ = response_tx.send(());
                }

                // Wait for response
                let _ = response_rx.await;
            });
        });
    });

    // Mutex+VecDeque round trip
    group.bench_function("mutex_vecdeque", |b| {
        let rt = Runtime::new().unwrap();
        let queue = LegacyCommandQueue::new();

        b.iter(|| {
            rt.block_on(async {
                // Send command
                let command = create_small_command();
                queue.push(command);

                // Receive and process
                if let Some(cmd) = queue.pop() {
                    black_box(cmd);
                }
            });
        });
    });

    group.finish();
}

/// Benchmark sustained throughput
fn throughput_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // Channel throughput at different batch sizes
    for batch_size in [10, 100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("channel", batch_size),
            batch_size,
            |b, &size| {
                let rt = Runtime::new().unwrap();

                b.iter(|| {
                    rt.block_on(async {
                        let (tx, mut rx) = broadcast::channel(size.max(100));

                        // Spawn consumer task
                        let consumer = tokio::spawn(async move {
                            let mut count = 0;
                            while count < size {
                                if rx.recv().await.is_ok() {
                                    count += 1;
                                }
                            }
                        });

                        // Send commands
                        for _ in 0..size {
                            let command = create_small_command();
                            let _ = tx.send(command);
                        }

                        // Wait for consumer to finish
                        let _ = consumer.await;
                    });
                });
            },
        );

        // Mutex+VecDeque throughput
        group.bench_with_input(
            BenchmarkId::new("mutex_vecdeque", batch_size),
            batch_size,
            |b, &size| {
                let rt = Runtime::new().unwrap();

                b.iter(|| {
                    rt.block_on(async {
                        let queue = LegacyCommandQueue::new();

                        // Producer
                        for _ in 0..size {
                            let command = create_small_command();
                            queue.push(command);
                        }

                        // Consumer
                        for _ in 0..size {
                            while queue.pop().is_none() {
                                tokio::task::yield_now().await;
                            }
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark with backpressure simulation
fn backpressure_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("backpressure");

    // Channel with bounded capacity (simulates backpressure)
    group.bench_function("channel_bounded_100", |b| {
        let rt = Runtime::new().unwrap();

        b.iter(|| {
            rt.block_on(async {
                let (tx, mut rx) = broadcast::channel(100);
                let commands_to_send = 1000;

                // Spawn slow consumer
                let consumer = tokio::spawn(async move {
                    let mut count = 0;
                    while count < commands_to_send {
                        if rx.recv().await.is_ok() {
                            count += 1;
                            // Simulate processing delay
                            tokio::time::sleep(tokio::time::Duration::from_micros(10)).await;
                        }
                    }
                });

                // Fast producer
                for _ in 0..commands_to_send {
                    let command = create_small_command();
                    let _ = tx.send(command);
                }

                let _ = consumer.await;
            });
        });
    });

    // Mutex+VecDeque with size limit
    group.bench_function("mutex_vecdeque_limited", |b| {
        let rt = Runtime::new().unwrap();

        b.iter(|| {
            rt.block_on(async {
                let queue = LegacyCommandQueue::new();
                let commands_to_send = 1000;
                let max_queue_size = 100;

                // Producer with backpressure
                let producer = tokio::spawn({
                    let queue = queue.clone();
                    async move {
                        for _ in 0..commands_to_send {
                            // Wait if queue is full
                            while queue.len() >= max_queue_size {
                                tokio::task::yield_now().await;
                            }
                            let command = create_small_command();
                            queue.push(command);
                        }
                    }
                });

                // Consumer
                let consumer = tokio::spawn(async move {
                    let mut count = 0;
                    while count < commands_to_send {
                        if queue.pop().is_some() {
                            count += 1;
                            // Simulate processing delay
                            tokio::time::sleep(tokio::time::Duration::from_micros(10)).await;
                        } else {
                            tokio::task::yield_now().await;
                        }
                    }
                });

                let _ = tokio::join!(producer, consumer);
            });
        });
    });

    group.finish();
}

/// Benchmark Bytes clone latency (zero-copy verification)
fn bytes_clone_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("bytes_clone");

    for size in [1024, 64 * 1024, 1024 * 1024].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let data = Bytes::from(vec![0u8; size]);

            b.iter(|| {
                let cloned = black_box(data.clone());
                black_box(cloned)
            });
        });
    }

    group.finish();
}

/// Benchmark memory pool operations
fn memory_pool_operations(c: &mut Criterion) {
    use stratoswarm_core::gpu::memory::UnifiedMemoryPool;

    let mut group = c.benchmark_group("memory_pool");

    // Allocation latency
    group.bench_function("allocate_1kb", |b| {
        b.iter(|| {
            let mut pool = UnifiedMemoryPool::new(1024 * 1024 * 1024);
            let result = pool.allocate("test_buffer", 1024);
            black_box(result)
        });
    });

    group.bench_function("allocate_1mb", |b| {
        b.iter(|| {
            let mut pool = UnifiedMemoryPool::new(1024 * 1024 * 1024);
            let result = pool.allocate("test_buffer", 1024 * 1024);
            black_box(result)
        });
    });

    // Deallocation latency
    group.bench_function("deallocate_1kb", |b| {
        b.iter_batched(
            || {
                let mut pool = UnifiedMemoryPool::new(1024 * 1024 * 1024);
                pool.allocate("test_buffer", 1024).unwrap();
                pool
            },
            |mut pool| {
                let result = pool.deallocate("test_buffer");
                black_box(result)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Write operation latency
    group.bench_function("write_1kb", |b| {
        let mut pool = UnifiedMemoryPool::new(1024 * 1024 * 1024);
        pool.allocate("test_buffer", 1024 * 1024).unwrap();
        let data = Bytes::from(vec![0u8; 1024]);

        b.iter(|| {
            let result = pool.write("test_buffer", data.clone(), 0);
            black_box(result)
        });
    });

    // Read operation latency
    group.bench_function("read_1kb", |b| {
        let mut pool = UnifiedMemoryPool::new(1024 * 1024 * 1024);
        pool.allocate("test_buffer", 1024 * 1024).unwrap();
        let data = Bytes::from(vec![42u8; 1024]);
        pool.write("test_buffer", data, 0).unwrap();

        b.iter(|| {
            let result = pool.read("test_buffer", 1024, 0);
            black_box(result)
        });
    });

    // Allocation/deallocation cycle
    group.bench_function("alloc_dealloc_cycle", |b| {
        b.iter(|| {
            let mut pool = UnifiedMemoryPool::new(1024 * 1024 * 1024);
            pool.allocate("test_buffer", 1024).unwrap();
            let result = pool.deallocate("test_buffer");
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark command processing patterns
fn command_processing_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("command_processing");

    // Sequential command processing
    group.bench_function("sequential_processing_channel", |b| {
        let rt = Runtime::new().unwrap();

        b.iter(|| {
            rt.block_on(async {
                let (tx, mut rx) = broadcast::channel(100);
                let num_commands = 100;

                // Send commands
                for _ in 0..num_commands {
                    let command = create_small_command();
                    let _ = tx.send(command);
                }

                // Process sequentially
                for _ in 0..num_commands {
                    if let Ok(cmd) = rx.recv().await {
                        black_box(cmd);
                    }
                }
            });
        });
    });

    group.bench_function("sequential_processing_mutex", |b| {
        let rt = Runtime::new().unwrap();

        b.iter(|| {
            rt.block_on(async {
                let queue = LegacyCommandQueue::new();
                let num_commands = 100;

                // Send commands
                for _ in 0..num_commands {
                    let command = create_small_command();
                    queue.push(command);
                }

                // Process sequentially
                for _ in 0..num_commands {
                    if let Some(cmd) = queue.pop() {
                        black_box(cmd);
                    }
                }
            });
        });
    });

    group.finish();
}

/// Benchmark contention scenarios
fn contention_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("contention");

    // Multiple producers, single consumer (channel)
    group.bench_function("multi_producer_channel", |b| {
        let rt = Runtime::new().unwrap();

        b.iter(|| {
            rt.block_on(async {
                let (tx, mut rx) = broadcast::channel(1000);
                let num_producers = 4;
                let commands_per_producer = 250;

                // Spawn producers
                let mut producers = Vec::new();
                for _ in 0..num_producers {
                    let tx = tx.clone();
                    producers.push(tokio::spawn(async move {
                        for _ in 0..commands_per_producer {
                            let command = create_small_command();
                            let _ = tx.send(command);
                        }
                    }));
                }

                // Consumer
                let consumer = tokio::spawn(async move {
                    let mut count = 0;
                    let total = num_producers * commands_per_producer;
                    while count < total {
                        if rx.recv().await.is_ok() {
                            count += 1;
                        }
                    }
                });

                for producer in producers {
                    let _ = producer.await;
                }
                let _ = consumer.await;
            });
        });
    });

    // Multiple producers, single consumer (mutex)
    group.bench_function("multi_producer_mutex", |b| {
        let rt = Runtime::new().unwrap();

        b.iter(|| {
            rt.block_on(async {
                let queue = LegacyCommandQueue::new();
                let num_producers = 4;
                let commands_per_producer = 250;

                // Spawn producers
                let mut producers = Vec::new();
                for _ in 0..num_producers {
                    let queue = queue.clone();
                    producers.push(tokio::spawn(async move {
                        for _ in 0..commands_per_producer {
                            let command = create_small_command();
                            queue.push(command);
                        }
                    }));
                }

                // Consumer
                let consumer = tokio::spawn(async move {
                    let mut count = 0;
                    let total = num_producers * commands_per_producer;
                    while count < total {
                        if queue.pop().is_some() {
                            count += 1;
                        } else {
                            tokio::task::yield_now().await;
                        }
                    }
                });

                for producer in producers {
                    let _ = producer.await;
                }
                let _ = consumer.await;
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    channel_send_latency,
    mutex_vecdeque_latency,
    round_trip_latency,
    throughput_benchmark,
    backpressure_throughput,
    bytes_clone_latency,
    memory_pool_operations,
    command_processing_patterns,
    contention_scenarios,
);
criterion_main!(benches);
