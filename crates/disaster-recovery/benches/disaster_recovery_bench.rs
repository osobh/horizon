use chrono::{Duration, Utc};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use stratoswarm_disaster_recovery::{
    backup_manager::*, data_integrity::*, failover_coordinator::*, health_monitor::*,
    recovery_planner::*, replication_manager::*, runbook_executor::*, snapshot_manager::*,
};
use tokio::runtime::Runtime;

fn bench_backup_operations(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("backup_creation", |b| {
        let config = BackupConfig::default();
        let mut manager = BackupManager::new(config)?;

        b.iter(|| {
            rt.block_on(async {
                let request = BackupRequest {
                    source_path: black_box("/data/source".to_string()),
                    backup_type: black_box(BackupType::Incremental),
                    compression: black_box(CompressionType::Lz4),
                    encryption: black_box(true),
                    metadata: HashMap::new(),
                };
                black_box(manager.create_backup(&request).await)
            })
        });
    });

    c.bench_function("backup_verification", |b| {
        let config = BackupConfig::default();
        let manager = BackupManager::new(config).unwrap();

        b.iter(|| {
            rt.block_on(async {
                let backup_id = black_box("backup-123");
                black_box(manager.verify_backup(backup_id).await)
            })
        });
    });
}

fn bench_failover_operations(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("failover_execution", |b| {
        let config = FailoverConfig::default();
        let mut coordinator = FailoverCoordinator::new(config)?;

        b.iter(|| {
            rt.block_on(async {
                let request = FailoverRequest {
                    source_site: black_box("site-a".to_string()),
                    target_site: black_box("site-b".to_string()),
                    services: vec!["service-1".to_string(), "service-2".to_string()],
                    forced: black_box(false),
                    maintenance_mode: black_box(false),
                };
                black_box(coordinator.initiate_failover(&request).await)
            })
        });
    });

    c.bench_function("health_assessment", |b| {
        let config = FailoverConfig::default();
        let coordinator = FailoverCoordinator::new(config).unwrap();

        b.iter(|| {
            rt.block_on(async {
                let site_id = black_box("site-primary");
                black_box(coordinator.assess_site_health(site_id).await)
            })
        });
    });
}

fn bench_replication_operations(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("replication_sync", |b| {
        let config = ReplicationConfig::default();
        let mut manager = ReplicationManager::new(config)?;

        b.iter(|| {
            rt.block_on(async {
                let stream_id = black_box("stream-123");
                let data = black_box(vec![1u8; 1024]);
                black_box(manager.replicate_data(stream_id, &data).await)
            })
        });
    });

    c.bench_function("lag_monitoring", |b| {
        let config = ReplicationConfig::default();
        let manager = ReplicationManager::new(config).unwrap();

        b.iter(|| {
            let stream_id = black_box("stream-456");
            black_box(manager.get_replication_lag(stream_id))
        });
    });
}

fn bench_health_monitoring(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("health_check_execution", |b| {
        let config = HealthConfig::default();
        let mut monitor = HealthMonitor::new(config)?;

        b.iter(|| {
            rt.block_on(async {
                let service_id = black_box("service-api");
                black_box(monitor.check_service_health(service_id).await)
            })
        });
    });

    c.bench_function("cascade_detection", |b| {
        let config = HealthConfig::default();
        let monitor = HealthMonitor::new(config).unwrap();

        b.iter(|| {
            let failure_id = black_box("failure-123");
            black_box(monitor.detect_cascade_failures(failure_id))
        });
    });
}

fn bench_recovery_planning(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("recovery_plan_generation", |b| {
        let config = RecoveryConfig::default();
        let mut planner = RecoveryPlanner::new(config)?;

        b.iter(|| {
            rt.block_on(async {
                let scenario = DisasterScenario {
                    id: black_box("scenario-1".to_string()),
                    name: black_box("Database Failure".to_string()),
                    affected_services: vec!["db-primary".to_string()],
                    estimated_impact: black_box(ImpactLevel::High),
                    rto_target: black_box(Duration::minutes(15)),
                    rpo_target: black_box(Duration::minutes(5)),
                };
                black_box(planner.generate_recovery_plan(&scenario).await)
            })
        });
    });

    c.bench_function("dependency_resolution", |b| {
        let config = RecoveryConfig::default();
        let planner = RecoveryPlanner::new(config).unwrap();

        b.iter(|| {
            let service_id = black_box("service-frontend");
            black_box(planner.resolve_dependencies(service_id))
        });
    });
}

fn bench_snapshot_operations(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("snapshot_creation", |b| {
        let config = SnapshotConfig::default();
        let mut manager = SnapshotManager::new(config)?;

        b.iter(|| {
            rt.block_on(async {
                let request = SnapshotRequest {
                    volume_id: black_box("vol-123".to_string()),
                    snapshot_type: black_box(SnapshotType::CrashConsistent),
                    retention_policy: RetentionPolicy {
                        keep_daily: 7,
                        keep_weekly: 4,
                        keep_monthly: 12,
                        keep_yearly: 3,
                    },
                    tags: HashMap::new(),
                };
                black_box(manager.create_snapshot(&request).await)
            })
        });
    });

    c.bench_function("snapshot_restoration", |b| {
        let config = SnapshotConfig::default();
        let manager = SnapshotManager::new(config).unwrap();

        b.iter(|| {
            rt.block_on(async {
                let snapshot_id = black_box("snap-456");
                let target_volume = black_box("vol-789");
                black_box(manager.restore_snapshot(snapshot_id, target_volume).await)
            })
        });
    });
}

fn bench_data_integrity(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("checksum_calculation", |b| {
        let config = IntegrityConfig::default();
        let validator = DataIntegrityValidator::new(config);

        b.iter(|| {
            let data = black_box(vec![42u8; 4096]);
            black_box(validator.calculate_checksum(&data, ChecksumAlgorithm::Sha256))
        });
    });

    c.bench_function("corruption_detection", |b| {
        let config = IntegrityConfig::default();
        let mut validator = DataIntegrityValidator::new(config);

        b.iter(|| {
            rt.block_on(async {
                let file_path = black_box("/data/file.txt");
                black_box(validator.detect_corruption(file_path).await)
            })
        });
    });
}

fn bench_runbook_execution(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("runbook_execution", |b| {
        let config = RunbookConfig::default();
        let mut executor = RunbookExecutor::new(config)?;

        b.iter(|| {
            rt.block_on(async {
                let runbook_id = black_box("runbook-dr-001");
                let context = HashMap::from([
                    ("site".to_string(), "primary".to_string()),
                    ("service".to_string(), "database".to_string()),
                ]);
                black_box(executor.execute_runbook(runbook_id, context).await)
            })
        });
    });

    c.bench_function("step_validation", |b| {
        let config = RunbookConfig::default();
        let executor = RunbookExecutor::new(config).unwrap();

        b.iter(|| {
            rt.block_on(async {
                let step = RunbookStep {
                    id: black_box("step-001".to_string()),
                    name: black_box("Verify Service Health".to_string()),
                    step_type: black_box(StepType::HealthCheck),
                    parameters: HashMap::new(),
                    timeout: black_box(Duration::minutes(5)),
                    retry_count: black_box(3),
                    continue_on_failure: black_box(false),
                    approval_required: black_box(false),
                };
                black_box(executor.validate_step(&step).await)
            })
        });
    });
}

criterion_group!(
    benches,
    bench_backup_operations,
    bench_failover_operations,
    bench_replication_operations,
    bench_health_monitoring,
    bench_recovery_planning,
    bench_snapshot_operations,
    bench_data_integrity,
    bench_runbook_execution
);
criterion_main!(benches);
