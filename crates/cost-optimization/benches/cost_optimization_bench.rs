use chrono::{Duration, Utc};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stratoswarm_cost_optimization::{
    budget_manager::*, cloud_pricing::*, cost_predictor::*, gpu_optimizer::*, resource_tracker::*,
    spot_manager::*, usage_analyzer::*, workload_scheduler::*,
};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use tokio::runtime::Runtime;

fn bench_resource_tracking(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("resource_metrics_recording", |b| {
        let config = ResourceTrackerConfig::default();
        let mut tracker = ResourceTracker::new(config)?;

        b.iter(|| {
            rt.block_on(async {
                let metrics = ResourceMetrics {
                    timestamp: black_box(Utc::now()),
                    cpu_usage_percent: black_box(75.5),
                    memory_usage_gb: black_box(16.0),
                    gpu_usage_percent: black_box(HashMap::from([("gpu0".to_string(), 85.0)])),
                    gpu_memory_gb: black_box(HashMap::from([("gpu0".to_string(), 8.0)])),
                    network_bandwidth_mbps: black_box(100.0),
                    disk_iops: black_box(5000),
                    cost_per_hour: black_box(OrderedFloat(2.50)),
                };
                black_box(tracker.record_metrics(metrics).await)
            })
        });
    });

    c.bench_function("resource_statistics_calculation", |b| {
        let config = ResourceTrackerConfig::default();
        let tracker = ResourceTracker::new(config).unwrap();

        b.iter(|| {
            let start = black_box(Utc::now() - Duration::hours(1));
            let end = black_box(Utc::now());
            black_box(tracker.get_statistics(start, end))
        });
    });
}

fn bench_gpu_optimization(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("gpu_allocation_best_fit", |b| {
        let config = GpuOptimizerConfig::default();
        let mut optimizer = GpuOptimizer::new(config);

        b.iter(|| {
            rt.block_on(async {
                let request = GpuRequest {
                    workload_id: black_box("workload-123".to_string()),
                    gpu_count: black_box(2),
                    memory_gb_per_gpu: black_box(16),
                    gpu_type: Some("A100".to_string()),
                    duration: Some(Duration::hours(1)),
                    priority: black_box(5),
                    preemptible: black_box(false),
                    affinity_rules: vec![],
                };
                black_box(optimizer.allocate_gpus(&request).await)
            })
        });
    });

    c.bench_function("gpu_utilization_calculation", |b| {
        let config = GpuOptimizerConfig::default();
        let optimizer = GpuOptimizer::new(config);

        b.iter(|| black_box(optimizer.get_gpu_utilization()));
    });
}

fn bench_cloud_pricing(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("pricing_calculation", |b| {
        let config = CloudPricingConfig::default();
        let mut pricing_engine = CloudPricingEngine::new(config);

        b.iter(|| {
            rt.block_on(async {
                let request = PricingRequest {
                    provider: black_box(CloudProvider::AWS),
                    region: black_box("us-east-1".to_string()),
                    instance_type: black_box("p4d.24xlarge".to_string()),
                    duration_hours: black_box(1.0),
                    pricing_model: black_box(PricingModel::OnDemand),
                    os: black_box("linux".to_string()),
                };
                black_box(pricing_engine.calculate_cost(&request).await)
            })
        });
    });

    c.bench_function("provider_comparison", |b| {
        let config = CloudPricingConfig::default();
        let mut pricing_engine = CloudPricingEngine::new(config);

        b.iter(|| {
            rt.block_on(async {
                let instance_type = black_box("gpu-large");
                let region = black_box("us-east");
                black_box(
                    pricing_engine
                        .compare_providers(instance_type, region)
                        .await,
                )
            })
        });
    });
}

fn bench_workload_scheduling(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("workload_scheduling", |b| {
        let config = SchedulerConfig::default();
        let mut scheduler = WorkloadScheduler::new(config);

        b.iter(|| {
            rt.block_on(async {
                let workload = Workload {
                    id: black_box("workload-456".to_string()),
                    priority: black_box(8),
                    cpu_cores: black_box(16),
                    memory_gb: black_box(64),
                    gpu_count: Some(4),
                    estimated_duration: black_box(Duration::hours(2)),
                    cost_limit: Some(OrderedFloat(100.0)),
                    deadline: None,
                    dependencies: vec![],
                    preemptible: black_box(true),
                    affinity_rules: vec![],
                };
                black_box(scheduler.schedule_workload(workload).await)
            })
        });
    });

    c.bench_function("queue_optimization", |b| {
        let config = SchedulerConfig::default();
        let mut scheduler = WorkloadScheduler::new(config);

        b.iter(|| rt.block_on(async { black_box(scheduler.optimize_queue().await) }));
    });
}

fn bench_spot_management(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("spot_bidding", |b| {
        let config = SpotManagerConfig::default();
        let mut spot_manager = SpotManager::new(config);

        b.iter(|| {
            rt.block_on(async {
                let request = SpotRequest {
                    instance_type: black_box("p3.2xlarge".to_string()),
                    region: black_box("us-west-2".to_string()),
                    max_price: black_box(OrderedFloat(2.0)),
                    duration_hours: black_box(4),
                    interruption_behavior: black_box("terminate".to_string()),
                };
                black_box(spot_manager.place_bid(request).await)
            })
        });
    });

    c.bench_function("interruption_prediction", |b| {
        let config = SpotManagerConfig::default();
        let spot_manager = SpotManager::new(config);

        b.iter(|| {
            let instance_id = black_box("i-1234567890abcdef0");
            black_box(spot_manager.predict_interruption(instance_id))
        });
    });
}

fn bench_cost_prediction(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("cost_forecasting", |b| {
        let config = PredictorConfig::default();
        let mut predictor = CostPredictor::new(config);

        b.iter(|| {
            rt.block_on(async {
                let resource_id = black_box("cluster-prod");
                let days_ahead = black_box(7);
                black_box(predictor.forecast_costs(resource_id, days_ahead).await)
            })
        });
    });

    c.bench_function("anomaly_detection", |b| {
        let config = PredictorConfig::default();
        let predictor = CostPredictor::new(config);

        b.iter(|| {
            let current_cost = black_box(OrderedFloat(150.0));
            let resource_id = black_box("service-api");
            black_box(predictor.detect_anomaly(resource_id, current_cost))
        });
    });
}

fn bench_budget_management(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("budget_tracking", |b| {
        let config = BudgetManagerConfig::default();
        let mut budget_manager = BudgetManager::new(config);

        b.iter(|| {
            rt.block_on(async {
                let spend = CostRecord {
                    timestamp: black_box(Utc::now()),
                    amount: black_box(OrderedFloat(50.0)),
                    currency: black_box("USD".to_string()),
                    service: black_box("compute".to_string()),
                    resource_id: black_box("instance-123".to_string()),
                    tags: HashMap::new(),
                };
                black_box(budget_manager.record_spend(spend).await)
            })
        });
    });

    c.bench_function("alert_evaluation", |b| {
        let config = BudgetManagerConfig::default();
        let budget_manager = BudgetManager::new(config);

        b.iter(|| rt.block_on(async { black_box(budget_manager.evaluate_alerts().await) }));
    });
}

fn bench_usage_analysis(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("pattern_detection", |b| {
        let config = AnalyzerConfig::default();
        let mut analyzer = UsageAnalyzer::new(config);

        b.iter(|| {
            rt.block_on(async {
                let resource_id = black_box("gpu-cluster-1");
                black_box(analyzer.detect_patterns(resource_id).await)
            })
        });
    });

    c.bench_function("optimization_recommendations", |b| {
        let config = AnalyzerConfig::default();
        let analyzer = UsageAnalyzer::new(config);

        b.iter(|| {
            rt.block_on(async {
                let resource_id = black_box("app-backend");
                black_box(analyzer.get_recommendations(resource_id).await)
            })
        });
    });
}

criterion_group!(
    benches,
    bench_resource_tracking,
    bench_gpu_optimization,
    bench_cloud_pricing,
    bench_workload_scheduling,
    bench_spot_management,
    bench_cost_prediction,
    bench_budget_management,
    bench_usage_analysis
);
criterion_main!(benches);
