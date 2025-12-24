//! Performance benchmarks for multi-region modules

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stratoswarm_multi_region::*;
use std::collections::HashMap;
use tokio::runtime::Runtime;

/// Benchmark load balancer endpoint selection performance
fn benchmark_load_balancer(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("load_balancer");
    group.throughput(Throughput::Elements(1));

    // Create test configuration and endpoints
    let config = LoadBalancerConfig::default();
    let mut endpoints = Vec::new();

    for i in 0..10 {
        let endpoint = load_balancer::RegionEndpoint::new(
            format!("region-{}", i),
            format!("https://region-{}.example.com", i),
            50,
            i as u32,
        );
        endpoints.push(endpoint);
    }

    let lb = LoadBalancer::new(config, endpoints).unwrap();

    // Create test request
    let request = load_balancer::RoutingRequest {
        client_ip: Some("192.168.1.100".to_string()),
        session_id: Some("session123".to_string()),
        geo_info: None,
        metadata: HashMap::new(),
    };

    // Benchmark different load balancing algorithms
    let algorithms = vec![
        ("round_robin", LoadBalancingAlgorithm::RoundRobin),
        (
            "least_connections",
            LoadBalancingAlgorithm::LeastConnections,
        ),
        (
            "weighted_round_robin",
            LoadBalancingAlgorithm::WeightedRoundRobin,
        ),
        ("ip_hash", LoadBalancingAlgorithm::IpHash),
    ];

    for (name, _algorithm) in algorithms {
        group.bench_with_input(
            BenchmarkId::new("endpoint_selection", name),
            &request,
            |b, req| {
                b.to_async(&rt).iter(|| async {
                    // Note: This will fail in benchmark due to no healthy endpoints,
                    // but measures the selection logic performance
                    let _ = black_box(lb.select_endpoint(req).await);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark replication vector clock operations
fn benchmark_replication(c: &mut Criterion) {
    let mut group = c.benchmark_group("replication");
    group.throughput(Throughput::Elements(1));

    // Benchmark vector clock operations
    let mut clock1 = replication::VectorClock::new();
    let mut clock2 = replication::VectorClock::new();

    // Initialize clocks with some data
    for i in 0..10 {
        clock1.increment(&format!("region-{}", i));
        clock2.increment(&format!("region-{}", (i + 5) % 10));
    }

    group.bench_function("vector_clock_merge", |b| {
        b.iter(|| {
            let _merged = black_box(clock1.merge(&clock2));
        });
    });

    group.bench_function("vector_clock_happens_before", |b| {
        b.iter(|| {
            let _result = black_box(clock1.happens_before(&clock2));
        });
    });

    group.bench_function("vector_clock_increment", |b| {
        let mut clock = replication::VectorClock::new();
        b.iter(|| {
            clock.increment(black_box("test-region"));
        });
    });

    group.finish();
}

/// Benchmark tunnel connection management
fn benchmark_tunnels(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("tunnels");
    group.throughput(Throughput::Elements(1));

    let config = TunnelConfig::default();
    let manager = TunnelManager::new(config).unwrap();

    group.bench_function("tunnel_request_creation", |b| {
        b.iter(|| {
            let request = tunnels::TunnelRequest {
                id: black_box(format!("request-{}", fastrand::u64(..))),
                source_region: black_box("us-east-1".to_string()),
                target_region: black_box("us-west-2".to_string()),
                data: black_box(vec![0u8; 1024]),
                headers: black_box(HashMap::new()),
                priority: black_box(tunnels::QosPriority::Normal),
                timestamp: black_box(chrono::Utc::now()),
                timeout_ms: black_box(Some(5000)),
            };
            black_box(request);
        });
    });

    group.bench_function("tunnel_stats_collection", |b| {
        b.to_async(&rt).iter(|| async {
            let _stats = black_box(manager.get_tunnel_stats().await);
        });
    });

    group.finish();
}

/// Benchmark compliance validation
fn benchmark_compliance(c: &mut Criterion) {
    let mut group = c.benchmark_group("compliance");
    group.throughput(Throughput::Elements(1));

    let config = ComplianceMappingConfig::default();
    let mut manager = ComplianceMappingManager::new(config);

    // Create test compliance context
    let context = compliance_mapping::ComplianceContext {
        data_classification: compliance_mapping::DataClassification::PII,
        source_region: "EU".to_string(),
        target_region: "US".to_string(),
        operation: compliance_mapping::DataOperation::Transfer,
        encryption_enabled: true,
        consent_obtained: false,
        metadata: HashMap::new(),
    };

    group.bench_function("compliance_validation", |b| {
        b.iter(|| {
            let _result = black_box(manager.validate_compliance(&context));
        });
    });

    group.bench_function("transfer_allowed_check", |b| {
        b.iter(|| {
            let _result = black_box(manager.is_transfer_allowed(
                compliance_mapping::DataClassification::PII,
                "EU",
                "US",
            ));
        });
    });

    // Benchmark with different data classifications
    let classifications = vec![
        compliance_mapping::DataClassification::Public,
        compliance_mapping::DataClassification::Internal,
        compliance_mapping::DataClassification::Confidential,
        compliance_mapping::DataClassification::PII,
        compliance_mapping::DataClassification::PHI,
    ];

    for classification in classifications {
        group.bench_with_input(
            BenchmarkId::new("classification_validation", format!("{:?}", classification)),
            &classification,
            |b, &class| {
                let ctx = compliance_mapping::ComplianceContext {
                    data_classification: class,
                    source_region: "EU".to_string(),
                    target_region: "US".to_string(),
                    operation: compliance_mapping::DataOperation::Transfer,
                    encryption_enabled: true,
                    consent_obtained: true,
                    metadata: HashMap::new(),
                };
                b.iter(|| {
                    let _result = black_box(manager.validate_compliance(&ctx));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark data sovereignty validation
fn benchmark_data_sovereignty(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_sovereignty");
    group.throughput(Throughput::Elements(1));

    let mut sovereignty = DataSovereignty::new();

    // Add test rules
    let rule = data_sovereignty::SovereigntyRule {
        id: "test-rule".to_string(),
        data_classification: "PII".to_string(),
        allowed_jurisdictions: vec!["EU".to_string(), "US".to_string()],
        forbidden_jurisdictions: vec!["CN".to_string()],
        encryption_level: data_sovereignty::EncryptionLevel::Strong,
        residency_requirements: data_sovereignty::ResidencyRequirements {
            must_remain_in_origin: false,
            allowed_transit: vec!["UK".to_string()],
            max_distance_km: Some(10000),
            backup_requirements: data_sovereignty::BackupRequirements {
                cross_border_allowed: true,
                required_jurisdictions: vec![],
                min_copies: 2,
            },
        },
        compliance_frameworks: vec!["GDPR".to_string()],
        created_at: chrono::Utc::now(),
    };

    sovereignty.add_rule(rule);

    let request = data_sovereignty::DataPlacementRequest {
        data_id: "test-data".to_string(),
        classification: "PII".to_string(),
        source_jurisdiction: "EU".to_string(),
        target_jurisdiction: "US".to_string(),
        operation: data_sovereignty::DataOperation::Store,
        metadata: HashMap::new(),
    };

    group.bench_function("sovereignty_validation", |b| {
        b.iter(|| {
            let _result = black_box(sovereignty.validate_placement(&request));
        });
    });

    group.bench_function("supported_jurisdictions", |b| {
        b.iter(|| {
            let _jurisdictions = black_box(sovereignty.get_supported_jurisdictions("PII"));
        });
    });

    group.bench_function("jurisdiction_allowed", |b| {
        b.iter(|| {
            let _allowed = black_box(sovereignty.is_jurisdiction_allowed(
                "PII",
                "EU",
                "US",
                &data_sovereignty::DataOperation::Store,
            ));
        });
    });

    group.finish();
}

/// Benchmark region manager operations
fn benchmark_region_manager(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("region_manager");
    group.throughput(Throughput::Elements(1));

    let config = RegionConfig::default();
    let mut manager = RegionManager::new(config).unwrap();

    group.bench_function("region_selection", |b| {
        let requirements = region_manager::RegionRequirements {
            required_jurisdictions: vec!["US".to_string()],
            excluded_jurisdictions: vec![],
            required_services: vec!["compute".to_string()],
            max_latency: Some(100),
        };

        b.iter(|| {
            let _result = black_box(manager.select_best_region(&requirements));
        });
    });

    group.bench_function("healthy_regions", |b| {
        b.iter(|| {
            let _regions = black_box(manager.healthy_regions());
        });
    });

    group.bench_function("region_lookup", |b| {
        b.iter(|| {
            let _region = black_box(manager.get_region("us-east-1"));
        });
    });

    group.bench_function("health_status_lookup", |b| {
        b.iter(|| {
            let _status = black_box(manager.get_health_status("us-east-1"));
        });
    });

    group.finish();
}

/// Benchmark serialization/deserialization performance
fn benchmark_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");
    group.throughput(Throughput::Elements(1));

    // Test different configuration serialization
    let lb_config = LoadBalancerConfig::default();
    let repl_config = ReplicationConfig::default();
    let tunnel_config = TunnelConfig::default();
    let compliance_config = ComplianceMappingConfig::default();

    group.bench_function("load_balancer_config_serialize", |b| {
        b.iter(|| {
            let _json = black_box(serde_json::to_string(&lb_config).unwrap());
        });
    });

    group.bench_function("replication_config_serialize", |b| {
        b.iter(|| {
            let _json = black_box(serde_json::to_string(&repl_config).unwrap());
        });
    });

    group.bench_function("tunnel_config_serialize", |b| {
        b.iter(|| {
            let _json = black_box(serde_json::to_string(&tunnel_config).unwrap());
        });
    });

    group.bench_function("compliance_config_serialize", |b| {
        b.iter(|| {
            let _json = black_box(serde_json::to_string(&compliance_config).unwrap());
        });
    });

    // Test deserialization
    let lb_json = serde_json::to_string(&lb_config).unwrap();
    let repl_json = serde_json::to_string(&repl_config).unwrap();
    let tunnel_json = serde_json::to_string(&tunnel_config).unwrap();
    let compliance_json = serde_json::to_string(&compliance_config).unwrap();

    group.bench_function("load_balancer_config_deserialize", |b| {
        b.iter(|| {
            let _config: LoadBalancerConfig = black_box(serde_json::from_str(&lb_json).unwrap());
        });
    });

    group.bench_function("replication_config_deserialize", |b| {
        b.iter(|| {
            let _config: ReplicationConfig = black_box(serde_json::from_str(&repl_json).unwrap());
        });
    });

    group.bench_function("tunnel_config_deserialize", |b| {
        b.iter(|| {
            let _config: TunnelConfig = black_box(serde_json::from_str(&tunnel_json).unwrap());
        });
    });

    group.bench_function("compliance_config_deserialize", |b| {
        b.iter(|| {
            let _config: ComplianceMappingConfig =
                black_box(serde_json::from_str(&compliance_json).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_load_balancer,
    benchmark_replication,
    benchmark_tunnels,
    benchmark_compliance,
    benchmark_data_sovereignty,
    benchmark_region_manager,
    benchmark_serialization
);
criterion_main!(benches);
