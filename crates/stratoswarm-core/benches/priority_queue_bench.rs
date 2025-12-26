//! Benchmark comparing BinaryHeap vs VecDeque-based PrioritySchedulerQueue
//!
//! Based on mechanical sympathy principles from Brian's lecture:
//! - BinaryHeap: O(log n) comparisons per operation, hard for branch predictor
//! - Segmented VecDeque: O(1) with predictable 4-branch pattern
//!
//! Run with: cargo bench --package stratoswarm-core --bench priority_queue_bench

#![allow(dead_code)]
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::BinaryHeap;
use stratoswarm_core::{PrioritySchedulerQueue, SchedulerPriority};

/// Test item for benchmarking
#[derive(Clone, Debug)]
struct BenchItem {
    id: u64,
    priority: u32,
    data: [u8; 64], // Realistic payload size
}

impl BenchItem {
    fn new(id: u64, priority: u32) -> Self {
        Self {
            id,
            priority,
            data: [0u8; 64],
        }
    }
}

/// Wrapper for BinaryHeap comparison
#[derive(Clone, Debug)]
struct PrioritizedItem {
    item: BenchItem,
    priority_score: i64,
}

impl PartialEq for PrioritizedItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority_score == other.priority_score
    }
}

impl Eq for PrioritizedItem {}

impl PartialOrd for PrioritizedItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority_score.cmp(&other.priority_score)
    }
}

fn priority_to_scheduler(priority: u32) -> SchedulerPriority {
    match priority % 4 {
        0 => SchedulerPriority::Low,
        1 => SchedulerPriority::Normal,
        2 => SchedulerPriority::High,
        _ => SchedulerPriority::Critical,
    }
}

fn bench_enqueue(c: &mut Criterion) {
    let mut group = c.benchmark_group("priority_queue_enqueue");

    for size in [100, 1000, 10000].iter() {
        // BinaryHeap enqueue
        group.bench_with_input(BenchmarkId::new("BinaryHeap", size), size, |b, &size| {
            b.iter(|| {
                let mut heap = BinaryHeap::with_capacity(size);
                for i in 0..size {
                    let item = PrioritizedItem {
                        item: BenchItem::new(i as u64, (i % 4) as u32),
                        priority_score: (i % 4) as i64 * 1000 + i as i64,
                    };
                    heap.push(black_box(item));
                }
                heap
            });
        });

        // VecDeque-based PrioritySchedulerQueue enqueue
        group.bench_with_input(
            BenchmarkId::new("PrioritySchedulerQueue", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut queue = PrioritySchedulerQueue::with_capacity(size / 4);
                    for i in 0..size {
                        let item = BenchItem::new(i as u64, (i % 4) as u32);
                        let priority = priority_to_scheduler((i % 4) as u32);
                        queue.enqueue(black_box(item), priority);
                    }
                    queue
                });
            },
        );
    }

    group.finish();
}

fn bench_dequeue(c: &mut Criterion) {
    let mut group = c.benchmark_group("priority_queue_dequeue");

    for size in [100, 1000, 10000].iter() {
        // BinaryHeap dequeue
        group.bench_with_input(BenchmarkId::new("BinaryHeap", size), size, |b, &size| {
            b.iter_batched(
                || {
                    let mut heap = BinaryHeap::with_capacity(size);
                    for i in 0..size {
                        let item = PrioritizedItem {
                            item: BenchItem::new(i as u64, (i % 4) as u32),
                            priority_score: (i % 4) as i64 * 1000 + i as i64,
                        };
                        heap.push(item);
                    }
                    heap
                },
                |mut heap| {
                    while let Some(item) = heap.pop() {
                        black_box(item);
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // VecDeque-based PrioritySchedulerQueue dequeue
        group.bench_with_input(
            BenchmarkId::new("PrioritySchedulerQueue", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let mut queue = PrioritySchedulerQueue::with_capacity(size / 4);
                        for i in 0..size {
                            let item = BenchItem::new(i as u64, (i % 4) as u32);
                            let priority = priority_to_scheduler((i % 4) as u32);
                            queue.enqueue(item, priority);
                        }
                        queue
                    },
                    |mut queue| {
                        while let Some(item) = queue.dequeue() {
                            black_box(item);
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_mixed_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("priority_queue_mixed_ops");

    // Simulates realistic scheduler pattern: enqueue/dequeue interleaved
    for size in [100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("BinaryHeap", size), size, |b, &size| {
            b.iter(|| {
                let mut heap = BinaryHeap::with_capacity(size);

                // Fill half
                for i in 0..size / 2 {
                    let item = PrioritizedItem {
                        item: BenchItem::new(i as u64, (i % 4) as u32),
                        priority_score: (i % 4) as i64 * 1000 + i as i64,
                    };
                    heap.push(item);
                }

                // Mixed ops: dequeue one, enqueue two
                for i in (size / 2)..size {
                    let _ = heap.pop();
                    let item1 = PrioritizedItem {
                        item: BenchItem::new(i as u64, (i % 4) as u32),
                        priority_score: (i % 4) as i64 * 1000 + i as i64,
                    };
                    let item2 = PrioritizedItem {
                        item: BenchItem::new((i + 1000) as u64, ((i + 1) % 4) as u32),
                        priority_score: ((i + 1) % 4) as i64 * 1000 + (i + 1000) as i64,
                    };
                    heap.push(item1);
                    heap.push(item2);
                }

                // Drain remaining
                while let Some(_) = heap.pop() {}
            });
        });

        group.bench_with_input(
            BenchmarkId::new("PrioritySchedulerQueue", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut queue = PrioritySchedulerQueue::with_capacity(size / 4);

                    // Fill half
                    for i in 0..size / 2 {
                        let item = BenchItem::new(i as u64, (i % 4) as u32);
                        let priority = priority_to_scheduler((i % 4) as u32);
                        queue.enqueue(item, priority);
                    }

                    // Mixed ops: dequeue one, enqueue two
                    for i in (size / 2)..size {
                        let _ = queue.dequeue();
                        let item1 = BenchItem::new(i as u64, (i % 4) as u32);
                        let item2 = BenchItem::new((i + 1000) as u64, ((i + 1) % 4) as u32);
                        let priority1 = priority_to_scheduler((i % 4) as u32);
                        let priority2 = priority_to_scheduler(((i + 1) % 4) as u32);
                        queue.enqueue(item1, priority1);
                        queue.enqueue(item2, priority2);
                    }

                    // Drain remaining
                    while let Some(_) = queue.dequeue() {}
                });
            },
        );
    }

    group.finish();
}

fn bench_branch_prediction_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("priority_queue_branch_stress");

    // This test specifically stresses branch prediction by using unpredictable priority patterns
    let size = 1000;

    // Unpredictable pattern - hard for branch predictor
    group.bench_function("BinaryHeap_unpredictable", |b| {
        b.iter(|| {
            let mut heap = BinaryHeap::with_capacity(size);

            // Use LCG random pattern for unpredictable priorities
            let mut lcg = 12345u64;
            for i in 0..size {
                lcg = lcg.wrapping_mul(1103515245).wrapping_add(12345);
                let priority = (lcg % 4) as u32;
                let item = PrioritizedItem {
                    item: BenchItem::new(i as u64, priority),
                    priority_score: priority as i64 * 1000 + i as i64,
                };
                heap.push(item);
            }

            while let Some(_) = heap.pop() {}
        });
    });

    group.bench_function("PrioritySchedulerQueue_unpredictable", |b| {
        b.iter(|| {
            let mut queue = PrioritySchedulerQueue::with_capacity(size / 4);

            // Use same LCG pattern
            let mut lcg = 12345u64;
            for i in 0..size {
                lcg = lcg.wrapping_mul(1103515245).wrapping_add(12345);
                let priority = (lcg % 4) as u32;
                let item = BenchItem::new(i as u64, priority);
                let scheduler_priority = priority_to_scheduler(priority);
                queue.enqueue(item, scheduler_priority);
            }

            while let Some(_) = queue.dequeue() {}
        });
    });

    // Predictable pattern - easy for branch predictor
    group.bench_function("BinaryHeap_predictable", |b| {
        b.iter(|| {
            let mut heap = BinaryHeap::with_capacity(size);

            // Predictable pattern: all High priority
            for i in 0..size {
                let item = PrioritizedItem {
                    item: BenchItem::new(i as u64, 2),
                    priority_score: 2000 + i as i64,
                };
                heap.push(item);
            }

            while let Some(_) = heap.pop() {}
        });
    });

    group.bench_function("PrioritySchedulerQueue_predictable", |b| {
        b.iter(|| {
            let mut queue = PrioritySchedulerQueue::with_capacity(size / 4);

            // Predictable pattern: all High priority
            for i in 0..size {
                let item = BenchItem::new(i as u64, 2);
                queue.enqueue(item, SchedulerPriority::High);
            }

            while let Some(_) = queue.dequeue() {}
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_enqueue,
    bench_dequeue,
    bench_mixed_operations,
    bench_branch_prediction_stress
);
criterion_main!(benches);
