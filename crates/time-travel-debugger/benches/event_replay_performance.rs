use chrono::{Duration, Utc};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use stratoswarm_time_travel_debugger::*;
use tokio::runtime::Runtime;
use uuid::Uuid;

fn benchmark_event_recording(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let event_log = Arc::new(EventLog::new(event_log::EventLogConfig::default()));

    let mut group = c.benchmark_group("event_recording");

    // Test different batch sizes
    let batch_sizes = vec![1, 10, 100, 1000];

    for batch_size in batch_sizes {
        let agent_id = Uuid::new_v4();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("record_events", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter(|| async {
                    let log = Arc::clone(black_box(&event_log));

                    for i in 0..batch_size {
                        log.record_event(
                            agent_id,
                            EventType::ActionExecution,
                            json!({
                                "action": "test_action",
                                "index": i,
                                "data": create_event_data(i)
                            }),
                            create_event_metadata(i),
                            None,
                        )
                        .await
                        .unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_event_causality(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = event_log::EventLogConfig {
        enable_causality_tracking: true,
        ..Default::default()
    };
    let event_log = Arc::new(EventLog::new(config));

    let agent_id = Uuid::new_v4();

    // Pre-create a chain of events for causality testing
    let mut event_chain = Vec::new();
    rt.block_on(async {
        let mut previous_event = None;

        for i in 0..100 {
            let event_id = event_log
                .record_event(
                    agent_id,
                    EventType::StateChange,
                    json!({"step": i}),
                    HashMap::new(),
                    previous_event,
                )
                .await
                .unwrap();

            event_chain.push(event_id);
            previous_event = Some(event_id);
        }
    });

    let mut group = c.benchmark_group("event_causality");

    // Benchmark causality chain retrieval
    group.bench_function("get_causality_chain", |b| {
        b.to_async(&rt).iter(|| async {
            let log = Arc::clone(&event_log);
            let event_id = black_box(event_chain[50]); // Middle of chain
            log.get_causality_chain(event_id).await.unwrap()
        });
    });

    // Benchmark caused events retrieval
    group.bench_function("get_caused_events", |b| {
        b.to_async(&rt).iter(|| async {
            let log = Arc::clone(&event_log);
            let event_id = black_box(event_chain[25]); // Earlier in chain
            log.get_caused_events(event_id).await.unwrap()
        });
    });

    group.finish();
}

fn benchmark_event_replay(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let event_log = Arc::new(EventLog::new(event_log::EventLogConfig::default()));

    let agent_id = Uuid::new_v4();
    let event_count = 10000;

    // Pre-create events for replay testing
    rt.block_on(async {
        for i in 0..event_count {
            event_log
                .record_event(
                    agent_id,
                    match i % 4 {
                        0 => EventType::StateChange,
                        1 => EventType::ActionExecution,
                        2 => EventType::DecisionMade,
                        _ => EventType::MessageReceived,
                    },
                    json!({
                        "index": i,
                        "operation": format!("op_{}", i),
                        "data": create_event_data(i)
                    }),
                    create_event_metadata(i),
                    None,
                )
                .await
                .unwrap();
        }
    });

    let mut group = c.benchmark_group("event_replay");

    // Benchmark replay from time
    group.bench_function("replay_from_time", |b| {
        b.to_async(&rt).iter(|| async {
            let log = Arc::clone(&event_log);
            let start_time = black_box(Utc::now() - Duration::minutes(10));
            let mut count = 0;

            log.replay_from_time(agent_id, start_time, |_event| {
                count += 1;
                Ok(())
            })
            .await
            .unwrap()
        });
    });

    // Benchmark replay until time
    group.bench_function("replay_until_time", |b| {
        b.to_async(&rt).iter(|| async {
            let log = Arc::clone(&event_log);
            let end_time = black_box(Utc::now() - Duration::minutes(5));
            let mut count = 0;

            log.replay_until_time(agent_id, end_time, |_event| {
                count += 1;
                Ok(())
            })
            .await
            .unwrap()
        });
    });

    // Benchmark replay by index
    let replay_sizes = vec![100, 500, 1000, 5000];

    for size in replay_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("replay_events_by_index", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let log = Arc::clone(black_box(&event_log));
                    let start_index = (event_count - size) / 2;
                    let mut count = 0;

                    log.replay_events(agent_id, start_index, size, |_event| {
                        count += 1;
                        Ok(())
                    })
                    .await
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_event_queries(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let event_log = Arc::new(EventLog::new(event_log::EventLogConfig::default()));

    let agent_id = Uuid::new_v4();
    let event_count = 5000;

    // Pre-create events with different types
    let event_types = vec![
        EventType::StateChange,
        EventType::ActionExecution,
        EventType::DecisionMade,
        EventType::MessageReceived,
        EventType::ErrorOccurred,
        EventType::Custom("benchmark_event".to_string()),
    ];

    rt.block_on(async {
        for i in 0..event_count {
            let event_type = event_types[i % event_types.len()].clone();

            event_log
                .record_event(
                    agent_id,
                    event_type,
                    json!({
                        "index": i,
                        "category": match i % 3 {
                            0 => "category_a",
                            1 => "category_b",
                            _ => "category_c",
                        },
                        "priority": i % 5,
                        "data": create_event_data(i)
                    }),
                    create_event_metadata(i),
                    None,
                )
                .await
                .unwrap();
        }
    });

    let mut group = c.benchmark_group("event_queries");

    // Benchmark getting all events
    group.bench_function("get_all_events", |b| {
        b.to_async(&rt).iter(|| async {
            let log = Arc::clone(&event_log);
            log.get_agent_events(black_box(agent_id), None, None)
                .await
                .unwrap()
        });
    });

    // Benchmark getting events by type
    for event_type in &event_types {
        let type_name = format!("{:?}", event_type);
        group.bench_with_input(
            BenchmarkId::new("get_events_by_type", &type_name),
            event_type,
            |b, event_type| {
                b.to_async(&rt).iter(|| async {
                    let log = Arc::clone(&event_log);
                    log.get_events_by_type(agent_id, event_type.clone())
                        .await
                        .unwrap()
                });
            },
        );
    }

    // Benchmark time-range queries
    let time_ranges = vec![
        Duration::minutes(1),
        Duration::minutes(5),
        Duration::minutes(15),
        Duration::hours(1),
    ];

    for range in time_ranges {
        let range_name = format!("{:?}", range);
        group.bench_with_input(
            BenchmarkId::new("get_events_time_range", &range_name),
            &range,
            |b, range| {
                b.to_async(&rt).iter(|| async {
                    let log = Arc::clone(&event_log);
                    let end_time = Utc::now();
                    let start_time = end_time - *range;
                    log.get_agent_events(agent_id, Some(start_time), Some(end_time))
                        .await
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_concurrent_event_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = event_log::EventLogConfig {
        max_events_per_agent: 20000,
        enable_causality_tracking: true,
        ..Default::default()
    };
    let event_log = Arc::new(EventLog::new(config));

    let mut group = c.benchmark_group("concurrent_event_operations");

    let concurrency_levels = vec![1, 4, 8, 16];

    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent_event_recording", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let log = Arc::clone(black_box(&event_log));
                    let mut handles = Vec::new();

                    for thread_id in 0..concurrency {
                        let log_clone = Arc::clone(&log);
                        let agent_id = Uuid::new_v4();

                        let handle = tokio::spawn(async move {
                            let mut event_ids = Vec::new();

                            for i in 0..100 {
                                let event_id = log_clone
                                    .record_event(
                                        agent_id,
                                        EventType::ActionExecution,
                                        json!({
                                            "thread_id": thread_id,
                                            "index": i,
                                            "data": create_event_data(i)
                                        }),
                                        create_event_metadata(i),
                                        None,
                                    )
                                    .await
                                    .unwrap();

                                event_ids.push(event_id);
                            }

                            event_ids
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

fn benchmark_event_export_import(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let event_log = Arc::new(EventLog::new(event_log::EventLogConfig::default()));

    let agent_id = Uuid::new_v4();

    // Pre-create events for export/import testing
    let event_counts = vec![100, 500, 1000, 5000];
    let mut test_data = Vec::new();

    for &count in &event_counts {
        let test_agent_id = Uuid::new_v4();

        rt.block_on(async {
            for i in 0..count {
                event_log
                    .record_event(
                        test_agent_id,
                        EventType::StateChange,
                        json!({
                            "index": i,
                            "data": create_event_data(i)
                        }),
                        create_event_metadata(i),
                        None,
                    )
                    .await
                    .unwrap();
            }
        });

        test_data.push((test_agent_id, count));
    }

    let mut group = c.benchmark_group("event_export_import");

    // Benchmark export
    for (test_agent_id, count) in &test_data {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::new("export_events", count), count, |b, _| {
            b.to_async(&rt).iter(|| async {
                let log = Arc::clone(&event_log);
                log.export_events(black_box(*test_agent_id)).await.unwrap()
            });
        });
    }

    // Benchmark import
    let export_data: Vec<String> = test_data
        .iter()
        .map(|(test_agent_id, _)| {
            rt.block_on(async { event_log.export_events(*test_agent_id).await.unwrap() })
        })
        .collect();

    for (data, count) in export_data.into_iter().zip(event_counts.iter()) {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::new("import_events", count), count, |b, _| {
            b.to_async(&rt).iter(|| async {
                let log = Arc::clone(&event_log);
                let import_agent_id = Uuid::new_v4();
                log.import_events(import_agent_id, black_box(&data))
                    .await
                    .unwrap()
            });
        });
    }

    group.finish();
}

fn benchmark_event_statistics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let event_log = Arc::new(EventLog::new(event_log::EventLogConfig::default()));

    // Create data for multiple agents
    let num_agents = 10;
    let events_per_agent = 1000;

    rt.block_on(async {
        for agent_idx in 0..num_agents {
            let agent_id = Uuid::new_v4();

            for i in 0..events_per_agent {
                event_log
                    .record_event(
                        agent_id,
                        match i % 5 {
                            0 => EventType::StateChange,
                            1 => EventType::ActionExecution,
                            2 => EventType::DecisionMade,
                            3 => EventType::MessageReceived,
                            _ => EventType::Custom(format!("custom_{}", agent_idx)),
                        },
                        json!({
                            "agent_index": agent_idx,
                            "event_index": i,
                            "data": create_event_data(i)
                        }),
                        create_event_metadata(i),
                        None,
                    )
                    .await
                    .unwrap();
            }
        }
    });

    let mut group = c.benchmark_group("event_statistics");

    group.bench_function("get_statistics", |b| {
        b.to_async(&rt).iter(|| async {
            let log = Arc::clone(&event_log);
            log.get_statistics().await
        });
    });

    group.finish();
}

// Helper functions

fn create_event_data(index: usize) -> serde_json::Value {
    json!({
        "timestamp": chrono::Utc::now(),
        "sequence": index,
        "payload": {
            "type": "benchmark_data",
            "values": (0..10).map(|i| index + i).collect::<Vec<usize>>(),
            "metadata": {
                "processed": index % 2 == 0,
                "priority": index % 5,
                "category": match index % 3 {
                    0 => "high",
                    1 => "medium",
                    _ => "low"
                }
            }
        },
        "checksum": format!("{:x}", index * 12345)
    })
}

fn create_event_metadata(index: usize) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    metadata.insert("index".to_string(), index.to_string());
    metadata.insert("benchmark".to_string(), "true".to_string());
    metadata.insert("category".to_string(), format!("cat_{}", index % 4));
    metadata.insert("priority".to_string(), (index % 5).to_string());
    metadata
}

criterion_group!(
    benches,
    benchmark_event_recording,
    benchmark_event_causality,
    benchmark_event_replay,
    benchmark_event_queries,
    benchmark_concurrent_event_operations,
    benchmark_event_export_import,
    benchmark_event_statistics
);

criterion_main!(benches);
