//! Performance benchmarks for migration coordinator and WireGuard operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipnet::Ipv4Net;
use std::net::Ipv4Addr;
use std::str::FromStr;
use std::sync::Arc;
use subnet_manager::events::{InMemoryTransport, SubnetEventPublisher};
use subnet_manager::migration::{Migration, MigrationReason};
use subnet_manager::models::{Subnet, SubnetPurpose};
use subnet_manager::wireguard::{
    MigrationCoordinator, ProbeConfig, SubnetAwareWireGuard, WireGuardConfigGenerator,
};
use uuid::Uuid;

fn create_test_subnet(name: &str, cidr: &str) -> Subnet {
    Subnet::new(
        name,
        Ipv4Net::from_str(cidr).unwrap(),
        SubnetPurpose::Tenant,
        51820,
    )
}

fn create_test_coordinator() -> MigrationCoordinator {
    let wireguard = Arc::new(SubnetAwareWireGuard::new());
    let transport = Arc::new(InMemoryTransport::new());
    let publisher = Arc::new(SubnetEventPublisher::with_transport(transport));
    MigrationCoordinator::new(wireguard, publisher)
}

fn create_test_migration(node_id: Uuid, source_id: Uuid, target_id: Uuid) -> Migration {
    Migration::new(
        node_id,
        source_id,
        target_id,
        Ipv4Addr::new(10, 100, 0, 5),
        MigrationReason::PolicyChange,
    )
    .with_target_ip(Ipv4Addr::new(10, 101, 0, 5))
}

/// Benchmark migration start
fn bench_migration_start(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("migration_start", |b| {
        b.iter(|| {
            rt.block_on(async {
                let coordinator = create_test_coordinator();
                let migration =
                    create_test_migration(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
                coordinator.start_migration(migration).await.unwrap();
                black_box(())
            })
        });
    });
}

/// Benchmark concurrent migrations
fn bench_concurrent_migrations(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("migration_concurrent");

    for count in [5u64, 10, 50].iter() {
        group.throughput(Throughput::Elements(*count));
        group.bench_with_input(
            BenchmarkId::new("migrations", count),
            count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let coordinator = create_test_coordinator();

                        // Start multiple migrations
                        for _ in 0..count {
                            let migration = create_test_migration(
                                Uuid::new_v4(),
                                Uuid::new_v4(),
                                Uuid::new_v4(),
                            );
                            coordinator.start_migration(migration).await.unwrap();
                        }

                        black_box(coordinator.get_active_migrations().len())
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark migration status lookup
fn bench_migration_status_lookup(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("migration_status_lookup", |b| {
        // Setup: create coordinator with active migrations
        let (coordinator, migration_ids) = rt.block_on(async {
            let coordinator = create_test_coordinator();
            let mut ids = Vec::new();

            for _ in 0..100 {
                let migration =
                    create_test_migration(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
                ids.push(migration.id);
                coordinator.start_migration(migration).await.unwrap();
            }

            (coordinator, ids)
        });

        b.iter(|| {
            // Lookup all migration statuses
            for id in &migration_ids {
                black_box(coordinator.get_status(*id));
            }
        });
    });
}

/// Benchmark migration rollback
fn bench_migration_rollback(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("migration_rollback", |b| {
        b.iter_batched(
            || {
                // Setup: start a migration
                rt.block_on(async {
                    let coordinator = create_test_coordinator();
                    let migration =
                        create_test_migration(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
                    let id = migration.id;
                    coordinator.start_migration(migration).await.unwrap();
                    (coordinator, id)
                })
            },
            |(coordinator, id)| {
                rt.block_on(async {
                    coordinator
                        .rollback_migration(id, "benchmark test")
                        .await
                        .unwrap();
                    black_box(())
                })
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Benchmark has_active_migration check
fn bench_has_active_migration(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("has_active_migration", |b| {
        // Setup: create coordinator with some active migrations
        let (coordinator, node_ids) = rt.block_on(async {
            let coordinator = create_test_coordinator();
            let mut node_ids = Vec::new();

            for _ in 0..50 {
                let node_id = Uuid::new_v4();
                node_ids.push(node_id);
                let migration = create_test_migration(node_id, Uuid::new_v4(), Uuid::new_v4());
                coordinator.start_migration(migration).await.unwrap();
            }

            // Add some non-migrating node IDs
            for _ in 0..50 {
                node_ids.push(Uuid::new_v4());
            }

            (coordinator, node_ids)
        });

        b.iter(|| {
            for node_id in &node_ids {
                black_box(coordinator.has_active_migration(*node_id));
            }
        });
    });
}

/// Benchmark get_active_migrations
fn bench_get_active_migrations(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("get_active_migrations");

    for count in [10u64, 50, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("migrations", count),
            count,
            |b, &count| {
                // Setup: create coordinator with N active migrations
                let coordinator = rt.block_on(async {
                    let coordinator = create_test_coordinator();

                    for _ in 0..count {
                        let migration = create_test_migration(
                            Uuid::new_v4(),
                            Uuid::new_v4(),
                            Uuid::new_v4(),
                        );
                        coordinator.start_migration(migration).await.unwrap();
                    }

                    coordinator
                });

                b.iter(|| {
                    black_box(coordinator.get_active_migrations())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ProbeConfig creation with custom values
fn bench_probe_config_creation(c: &mut Criterion) {
    c.bench_function("probe_config_creation", |b| {
        b.iter(|| {
            let config = ProbeConfig {
                probe_timeout_ms: 1000,
                probe_count: 5,
                required_success_rate: 0.8,
                probe_interval_ms: 200,
            };
            black_box(config)
        });
    });
}

/// Benchmark WireGuard config generator creation
fn bench_wireguard_config_generation(c: &mut Criterion) {
    c.bench_function("wireguard_config_generator_create", |b| {
        b.iter(|| {
            let generator = WireGuardConfigGenerator::new()
                .with_mtu(1420)
                .with_keepalive(25)
                .with_preshared_keys(true);
            black_box(generator)
        });
    });
}

/// Benchmark subnet config generation with peers
fn bench_wireguard_config_batch(c: &mut Criterion) {
    use std::collections::HashMap;
    use subnet_manager::models::CrossSubnetRoute;
    use chrono::Utc;
    use subnet_manager::models::SubnetAssignment;

    let mut group = c.benchmark_group("wireguard_config_batch");

    for peer_count in [10u64, 50, 100].iter() {
        group.throughput(Throughput::Elements(*peer_count));
        group.bench_with_input(
            BenchmarkId::new("peers", peer_count),
            peer_count,
            |b, &peer_count| {
                let generator = WireGuardConfigGenerator::new();
                let subnet = create_test_subnet("test-subnet", "10.100.0.0/20");

                // Create assignments
                let assignments: Vec<SubnetAssignment> = (0..peer_count)
                    .map(|i| SubnetAssignment {
                        id: Uuid::new_v4(),
                        node_id: Uuid::new_v4(),
                        subnet_id: subnet.id,
                        assigned_ip: Ipv4Addr::new(10, 100, 0, (i + 2) as u8),
                        wg_public_key: format!("key-{}", i),
                        assigned_at: Utc::now(),
                        assignment_method: "test".to_string(),
                        policy_id: None,
                        is_migration_temp: false,
                    })
                    .collect();

                // Create endpoint map
                let endpoints: HashMap<Uuid, std::net::SocketAddr> = assignments
                    .iter()
                    .enumerate()
                    .map(|(i, a)| (a.node_id, format!("192.168.1.{}:51820", i + 1).parse().unwrap()))
                    .collect();

                let routes: Vec<(CrossSubnetRoute, Subnet)> = vec![];

                b.iter(|| {
                    let config = generator.generate_subnet_config(
                        &subnet,
                        &assignments,
                        &routes,
                        &endpoints,
                    );
                    black_box(config)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SubnetAwareWireGuard interface creation
fn bench_subnet_aware_wireguard(c: &mut Criterion) {
    c.bench_function("subnet_wireguard_create_interface", |b| {
        b.iter_batched(
            || SubnetAwareWireGuard::new(),
            |wg| {
                for i in 0..100 {
                    let subnet = create_test_subnet(&format!("subnet-{}", i), "10.100.0.0/20");
                    let _ = wg.create_interface(
                        &subnet,
                        &format!("private-key-{}", i),
                        &format!("public-key-{}", i),
                    );
                }
                black_box(wg.get_all_interfaces().len())
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Benchmark SubnetPeer creation
fn bench_subnet_wireguard_add_peer(c: &mut Criterion) {
    c.bench_function("subnet_peer_creation", |b| {
        b.iter(|| {
            use subnet_manager::wireguard::SubnetPeer;
            let subnet_id = Uuid::new_v4();

            for i in 0..100u8 {
                let peer = SubnetPeer::new(
                    Uuid::new_v4(),
                    &format!("key-{}", i),
                    Ipv4Addr::new(10, 100, 0, i),
                    subnet_id,
                )
                .with_endpoint(format!("192.168.1.{}:51820", i).parse().unwrap());
                black_box(peer);
            }
        });
    });
}

/// Benchmark get_all_interfaces
fn bench_get_all_interfaces(c: &mut Criterion) {
    c.bench_function("subnet_get_all_interfaces", |b| {
        let wg = SubnetAwareWireGuard::new();

        // Create 50 interfaces
        for i in 0..50u8 {
            let subnet = Subnet::new(
                &format!("subnet-{}", i),
                Ipv4Net::from_str(&format!("10.{}.0.0/20", i)).unwrap(),
                SubnetPurpose::Tenant,
                51820 + i as u16,
            );
            let _ = wg.create_interface(
                &subnet,
                &format!("private-key-{}", i),
                &format!("public-key-{}", i),
            );
        }

        b.iter(|| {
            black_box(wg.get_all_interfaces())
        });
    });
}

/// Benchmark route operations
fn bench_route_operations(c: &mut Criterion) {
    c.bench_function("subnet_route_add_check", |b| {
        b.iter_batched(
            || SubnetAwareWireGuard::new(),
            |wg| {
                use subnet_manager::models::CrossSubnetRoute;
                // Add routes
                for _ in 0..50u32 {
                    let source_id = Uuid::new_v4();
                    let dest_id = Uuid::new_v4();
                    let route = CrossSubnetRoute::new(source_id, dest_id);
                    wg.add_route(route);
                    black_box(wg.has_route(source_id, dest_id));
                }
                black_box(())
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_migration_start,
    bench_concurrent_migrations,
    bench_migration_status_lookup,
    bench_migration_rollback,
    bench_has_active_migration,
    bench_get_active_migrations,
    bench_probe_config_creation,
    bench_wireguard_config_generation,
    bench_wireguard_config_batch,
    bench_subnet_aware_wireguard,
    bench_subnet_wireguard_add_peer,
    bench_get_all_interfaces,
    bench_route_operations,
);

criterion_main!(benches);
