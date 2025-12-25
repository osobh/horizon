//! Performance benchmarks for event publishing system

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipnet::Ipv4Net;
use std::net::Ipv4Addr;
use std::str::FromStr;
use std::sync::Arc;
use subnet_manager::events::{
    InMemoryTransport, RouteInfo, SubnetEventPublisher, SubnetInfo, SubnetMessage,
};
use subnet_manager::models::{RouteDirection, Subnet, SubnetPurpose, SubnetStatus};
use uuid::Uuid;

fn create_test_subnet(name: &str, cidr: &str) -> Subnet {
    Subnet::new(
        name,
        Ipv4Net::from_str(cidr).unwrap(),
        SubnetPurpose::Tenant,
        51820,
    )
}

fn create_test_publisher() -> (SubnetEventPublisher, Arc<InMemoryTransport>) {
    let transport = Arc::new(InMemoryTransport::new());
    let publisher = SubnetEventPublisher::with_transport(transport.clone());
    (publisher, transport)
}

/// Benchmark single event publish
fn bench_single_event_publish(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("event_publish_single", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (publisher, _transport) = create_test_publisher();
                let subnet = create_test_subnet("bench-subnet", "10.100.0.0/20");
                publisher.subnet_created(&subnet, None).await.unwrap();
                black_box(publisher.stats().messages_published)
            })
        });
    });
}

/// Benchmark batch event publishing
fn bench_batch_event_publish(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("event_publish_batch");

    for batch_size in [10u64, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*batch_size));
        group.bench_with_input(
            BenchmarkId::new("events", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    rt.block_on(async {
                        let (publisher, _transport) = create_test_publisher();
                        for i in 0..batch_size {
                            let subnet =
                                create_test_subnet(&format!("subnet-{}", i), "10.100.0.0/20");
                            publisher.subnet_created(&subnet, None).await.unwrap();
                        }
                        black_box(publisher.stats().messages_published)
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark event publishing with local subscribers
fn bench_event_with_subscribers(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("event_publish_with_subscribers");

    for subscriber_count in [1u64, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("subscribers", subscriber_count),
            subscriber_count,
            |b, &subscriber_count| {
                b.iter(|| {
                    rt.block_on(async {
                        let (publisher, _transport) = create_test_publisher();

                        // Create subscribers
                        let _subscribers: Vec<_> = (0..subscriber_count)
                            .map(|_| publisher.subscribe_local())
                            .collect();

                        // Publish events
                        for i in 0..100 {
                            let subnet =
                                create_test_subnet(&format!("subnet-{}", i), "10.100.0.0/20");
                            publisher.subnet_created(&subnet, None).await.unwrap();
                        }

                        black_box(publisher.stats().messages_published)
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different event types
fn bench_event_types(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("event_types");

    // Subnet created event
    group.bench_function("subnet_created", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (publisher, _) = create_test_publisher();
                let subnet = create_test_subnet("test", "10.100.0.0/20");
                publisher.subnet_created(&subnet, None).await.unwrap();
                black_box(())
            })
        });
    });

    // Node assigned event
    group.bench_function("node_assigned", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (publisher, _) = create_test_publisher();
                use chrono::Utc;
                use subnet_manager::models::SubnetAssignment;

                let assignment = SubnetAssignment {
                    id: Uuid::new_v4(),
                    node_id: Uuid::new_v4(),
                    subnet_id: Uuid::new_v4(),
                    assigned_ip: Ipv4Addr::new(10, 100, 0, 5),
                    wg_public_key: "test-key".to_string(),
                    assigned_at: Utc::now(),
                    assignment_method: "policy".to_string(),
                    policy_id: None,
                    is_migration_temp: false,
                };
                publisher.node_assigned(&assignment).await.unwrap();
                black_box(())
            })
        });
    });

    // Topology snapshot event (large payload)
    group.bench_function("topology_snapshot", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (publisher, _) = create_test_publisher();

                let subnets: Vec<SubnetInfo> = (0..50)
                    .map(|i| SubnetInfo {
                        id: Uuid::new_v4(),
                        name: format!("subnet-{}", i),
                        cidr: format!("10.{}.0.0/20", i),
                        purpose: SubnetPurpose::Tenant,
                        status: SubnetStatus::Active,
                        node_count: 100,
                        wg_interface: format!("wg{}", i),
                        wg_listen_port: 51820 + i as u16,
                    })
                    .collect();

                let routes: Vec<RouteInfo> = (0..25)
                    .map(|i| RouteInfo {
                        id: Uuid::new_v4(),
                        source_subnet_id: subnets[i * 2].id,
                        destination_subnet_id: subnets[i * 2 + 1].id,
                        direction: RouteDirection::Bidirectional,
                        status: "active".to_string(),
                    })
                    .collect();

                publisher.topology_snapshot(subnets, routes, 1).await.unwrap();
                black_box(())
            })
        });
    });

    // Peer config updated event
    group.bench_function("peer_config_updated", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (publisher, _) = create_test_publisher();
                publisher
                    .peer_config_updated(
                        Uuid::new_v4(),
                        Uuid::new_v4(),
                        "base64-public-key-here",
                        Some("192.168.1.100:51820".parse().unwrap()),
                        vec!["10.100.0.5/32".to_string()],
                    )
                    .await
                    .unwrap();
                black_box(())
            })
        });
    });

    group.finish();
}

/// Benchmark message buffering
fn bench_message_buffering(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("event_buffering_100", |b| {
        b.iter(|| {
            rt.block_on(async {
                let transport = Arc::new(InMemoryTransport::new());
                let publisher = SubnetEventPublisher::with_transport(transport.clone());

                // Disconnect to enable buffering
                transport.disconnect();

                // Buffer 100 messages
                for i in 0..100 {
                    let subnet = create_test_subnet(&format!("subnet-{}", i), "10.100.0.0/20");
                    let _ = publisher.subnet_created(&subnet, None).await;
                }

                black_box(publisher.stats().messages_queued)
            })
        });
    });
}

/// Benchmark buffer flush
fn bench_buffer_flush(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("event_buffer_flush_100", |b| {
        b.iter_batched(
            || {
                // Setup: buffer 100 messages
                rt.block_on(async {
                    let transport = Arc::new(InMemoryTransport::new());
                    let publisher = SubnetEventPublisher::with_transport(transport.clone());

                    transport.disconnect();

                    for i in 0..100 {
                        let subnet = create_test_subnet(&format!("subnet-{}", i), "10.100.0.0/20");
                        let _ = publisher.subnet_created(&subnet, None).await;
                    }

                    (publisher, transport)
                })
            },
            |(publisher, transport)| {
                rt.block_on(async {
                    transport.connect();
                    let flushed = publisher.flush_buffer().await.unwrap();
                    black_box(flushed)
                })
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Benchmark transport message retrieval
fn bench_transport_message_retrieval(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("transport_messages_1000", |b| {
        b.iter_batched(
            || {
                // Setup: publish 1000 messages
                rt.block_on(async {
                    let transport = Arc::new(InMemoryTransport::new());
                    let publisher = SubnetEventPublisher::with_transport(transport.clone());

                    for i in 0..1000 {
                        let subnet = create_test_subnet(&format!("subnet-{}", i), "10.100.0.0/20");
                        publisher.subnet_created(&subnet, None).await.unwrap();
                    }

                    transport
                })
            },
            |transport| {
                // Benchmark: retrieve all messages
                let messages = transport.messages();
                black_box(messages.len())
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Benchmark channel routing
fn bench_channel_routing(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("channel_routing_mixed", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (publisher, transport) = create_test_publisher();

                // Publish mix of event types to different channels
                for i in 0..100 {
                    match i % 5 {
                        0 => {
                            let subnet =
                                create_test_subnet(&format!("s-{}", i), "10.100.0.0/20");
                            publisher.subnet_created(&subnet, None).await.unwrap();
                        }
                        1 => {
                            publisher
                                .node_unassigned(
                                    Uuid::new_v4(),
                                    Uuid::new_v4(),
                                    Ipv4Addr::new(10, 100, 0, i as u8),
                                    "test",
                                )
                                .await
                                .unwrap();
                        }
                        2 => {
                            publisher
                                .route_deleted(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4())
                                .await
                                .unwrap();
                        }
                        3 => {
                            publisher
                                .interface_created(Uuid::new_v4(), "wg0", 51820, "key")
                                .await
                                .unwrap();
                        }
                        _ => {
                            publisher
                                .key_rotated(Uuid::new_v4(), "old", "new")
                                .await
                                .unwrap();
                        }
                    }
                }

                black_box(transport.messages().len())
            })
        });
    });
}

criterion_group!(
    benches,
    bench_single_event_publish,
    bench_batch_event_publish,
    bench_event_with_subscribers,
    bench_event_types,
    bench_message_buffering,
    bench_buffer_flush,
    bench_transport_message_retrieval,
    bench_channel_routing,
);

criterion_main!(benches);
