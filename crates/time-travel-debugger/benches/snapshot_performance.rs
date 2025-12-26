use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use stratoswarm_time_travel_debugger::*;
use tokio::runtime::Runtime;
use uuid::Uuid;

fn benchmark_snapshot_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let snapshot_manager = Arc::new(SnapshotManager::new(SnapshotConfig::default()));

    let mut group = c.benchmark_group("snapshot_creation");

    // Test different snapshot sizes
    let sizes = vec![100, 1000, 10000, 100000];

    for size in sizes {
        let state_data = create_test_state(size);
        let agent_id = Uuid::new_v4();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("create_snapshot", size), &size, |b, _| {
            b.to_async(&rt).iter(|| async {
                let manager = Arc::clone(&snapshot_manager);
                let data = black_box(state_data.clone());
                manager
                    .create_snapshot(agent_id, data, 1024, HashMap::new())
                    .await
                    .unwrap()
            });
        });
    }

    group.finish();
}

fn benchmark_snapshot_retrieval(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let snapshot_manager = Arc::new(SnapshotManager::new(SnapshotConfig::default()));

    // Pre-create snapshots for testing
    let agent_id = Uuid::new_v4();
    let snapshot_ids: Vec<Uuid> = (0..1000)
        .map(|i| {
            rt.block_on(async {
                snapshot_manager
                    .create_snapshot(
                        agent_id,
                        create_test_state(100),
                        1024,
                        [("index".to_string(), i.to_string())].into_iter().collect(),
                    )
                    .await
                    .unwrap()
            })
        })
        .collect();

    let mut group = c.benchmark_group("snapshot_retrieval");

    group.bench_function("get_single_snapshot", |b| {
        b.to_async(&rt).iter(|| async {
            let manager = Arc::clone(&snapshot_manager);
            let id = black_box(snapshot_ids[500]);
            manager.get_snapshot(id).await.unwrap()
        });
    });

    group.bench_function("get_agent_snapshots", |b| {
        b.to_async(&rt).iter(|| async {
            let manager = Arc::clone(&snapshot_manager);
            let id = black_box(agent_id);
            manager.get_agent_snapshots(id).await.unwrap()
        });
    });

    group.finish();
}

fn benchmark_snapshot_comparison(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let snapshot_manager = Arc::new(SnapshotManager::new(SnapshotConfig::default()));
    let comparator = Arc::new(StateComparator::new(ComparisonOptions::default()));

    let agent_id = Uuid::new_v4();

    // Create snapshots with different amounts of changes
    let change_percentages = vec![0.1, 0.3, 0.5, 0.8];
    let mut snapshot_pairs = Vec::new();

    for change_pct in change_percentages {
        let base_state = create_test_state(1000);
        let modified_state = modify_state(&base_state, change_pct);

        let snapshot1 = rt.block_on(async {
            snapshot_manager
                .create_snapshot(agent_id, base_state, 1024, HashMap::new())
                .await
                .unwrap()
        });

        let snapshot2 = rt.block_on(async {
            snapshot_manager
                .create_snapshot(agent_id, modified_state, 1024, HashMap::new())
                .await
                .unwrap()
        });

        snapshot_pairs.push((snapshot1, snapshot2, change_pct));
    }

    let mut group = c.benchmark_group("snapshot_comparison");

    for (snapshot1_id, snapshot2_id, change_pct) in snapshot_pairs {
        group.bench_with_input(
            BenchmarkId::new(
                "compare_snapshots",
                format!("{:.0}%_changes", change_pct * 100.0),
            ),
            &change_pct,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let manager = Arc::clone(&snapshot_manager);
                    let comp = Arc::clone(&comparator);

                    let snap1 = manager.get_snapshot(black_box(snapshot1_id)).await.unwrap();
                    let snap2 = manager.get_snapshot(black_box(snapshot2_id)).await.unwrap();

                    comp.compare_snapshots(&snap1, &snap2, None).await.unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_diff_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let snapshot_manager = Arc::new(SnapshotManager::new(SnapshotConfig::default()));

    let agent_id = Uuid::new_v4();
    let base_state = create_test_state(5000);

    // Create snapshots with incremental changes
    let snapshot_ids: Vec<Uuid> = (0..100)
        .map(|i| {
            let mut state = base_state.clone();
            // Add some changes to each snapshot
            state["iteration"] = json!(i);
            state["data"]["modified_fields"] = json!((0..i).collect::<Vec<i32>>());

            rt.block_on(async {
                snapshot_manager
                    .create_snapshot(agent_id, state, 1024, HashMap::new())
                    .await
                    .unwrap()
            })
        })
        .collect();

    let mut group = c.benchmark_group("diff_creation");

    group.bench_function("create_diff", |b| {
        b.to_async(&rt).iter(|| async {
            let manager = Arc::clone(&snapshot_manager);
            let from_id = black_box(snapshot_ids[25]);
            let to_id = black_box(snapshot_ids[75]);
            manager.create_diff(from_id, to_id).await.unwrap()
        });
    });

    group.finish();
}

fn benchmark_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let snapshot_manager = Arc::new(SnapshotManager::new(SnapshotConfig::default()));

    let mut group = c.benchmark_group("concurrent_operations");

    let concurrency_levels = vec![1, 4, 8, 16];

    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent_snapshot_creation", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let manager = Arc::clone(black_box(&snapshot_manager));
                    let mut handles = Vec::new();

                    for i in 0..concurrency {
                        let manager_clone = Arc::clone(&manager);
                        let agent_id = Uuid::new_v4();
                        let state = create_test_state(100);

                        let handle = tokio::spawn(async move {
                            manager_clone
                                .create_snapshot(
                                    agent_id,
                                    state,
                                    1024,
                                    [("thread".to_string(), i.to_string())]
                                        .into_iter()
                                        .collect(),
                                )
                                .await
                                .unwrap()
                        });

                        handles.push(handle);
                    }

                    futures::future::join_all(handles).await
                });
            },
        );
    }

    group.finish();
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_usage");

    // Test with different max snapshot limits
    let limits = vec![10, 50, 100, 500];

    for limit in limits {
        let config = SnapshotConfig {
            max_snapshots: limit,
            compression_enabled: true,
            ..Default::default()
        };
        let snapshot_manager = Arc::new(SnapshotManager::new(config));
        let agent_id = Uuid::new_v4();

        group.bench_with_input(
            BenchmarkId::new("memory_bounded_snapshots", limit),
            &limit,
            |b, &limit| {
                b.to_async(&rt).iter(|| async {
                    let manager = Arc::clone(black_box(&snapshot_manager));

                    // Create more snapshots than the limit to test cleanup
                    for i in 0..(limit * 2) {
                        let state = create_test_state(500);
                        manager
                            .create_snapshot(
                                agent_id,
                                state,
                                1024,
                                [("iteration".to_string(), i.to_string())]
                                    .into_iter()
                                    .collect(),
                            )
                            .await
                            .unwrap();
                    }

                    // Get memory stats
                    manager.get_memory_stats().await
                });
            },
        );
    }

    group.finish();
}

// Helper functions

fn create_test_state(size: usize) -> serde_json::Value {
    let mut data = json!({
        "id": Uuid::new_v4(),
        "timestamp": chrono::Utc::now(),
        "version": 1,
        "metadata": {
            "created_by": "benchmark",
            "size": size,
            "category": "test_data"
        }
    });

    // Add arrays and nested objects based on size
    let array_size = (size as f64).sqrt() as usize;
    let mut items = Vec::new();

    for i in 0..array_size {
        items.push(json!({
            "index": i,
            "value": format!("item_{}", i),
            "properties": {
                "x": i as f64 * 0.1,
                "y": i as f64 * 0.2,
                "active": i % 2 == 0,
                "nested": {
                    "level1": {
                        "level2": {
                            "data": vec![i, i*2, i*3]
                        }
                    }
                }
            }
        }));
    }

    data["data"] = json!({
        "items": items,
        "summary": {
            "total_items": array_size,
            "generated_at": chrono::Utc::now(),
            "checksum": format!("{:x}", array_size * 12345)
        }
    });

    data
}

fn modify_state(base_state: &serde_json::Value, change_percentage: f64) -> serde_json::Value {
    let mut modified = base_state.clone();

    // Modify version
    modified["version"] = json!(2);

    // Modify some items based on change percentage
    if let Some(items) = modified["data"]["items"].as_array_mut() {
        let num_changes = (items.len() as f64 * change_percentage) as usize;

        for i in 0..num_changes {
            if i < items.len() {
                items[i]["value"] = json!(format!("modified_item_{}", i));
                items[i]["properties"]["x"] = json!(i as f64 * 0.5);
                items[i]["properties"]["modified"] = json!(true);
            }
        }

        // Add some new items
        let new_items = (change_percentage * 10.0) as usize;
        for i in 0..new_items {
            items.push(json!({
                "index": items.len() + i,
                "value": format!("new_item_{}", i),
                "properties": {
                    "x": (items.len() + i) as f64 * 0.1,
                    "y": (items.len() + i) as f64 * 0.2,
                    "active": true,
                    "is_new": true
                }
            }));
        }
    }

    // Update summary
    modified["data"]["summary"]["last_modified"] = json!(chrono::Utc::now());
    modified["data"]["summary"]["change_percentage"] = json!(change_percentage);

    modified
}

criterion_group!(
    benches,
    benchmark_snapshot_creation,
    benchmark_snapshot_retrieval,
    benchmark_snapshot_comparison,
    benchmark_diff_creation,
    benchmark_concurrent_operations,
    benchmark_memory_usage
);

criterion_main!(benches);
