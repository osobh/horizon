use chrono::{Duration, Utc};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use stratoswarm_cost_optimization::{
    budget_manager::*, cloud_pricing::*, cost_predictor::*, gpu_optimizer::*, resource_tracker::*,
    spot_manager::*, usage_analyzer::*, workload_scheduler::*, CostOptimizationError,
};
use tokio;

#[tokio::test]
async fn test_end_to_end_cost_optimization_workflow() {
    // Initialize all components
    let tracker_config = ResourceTrackerConfig::default();
    let mut resource_tracker = ResourceTracker::new(tracker_config).unwrap();

    let gpu_config = GpuOptimizerConfig::default();
    let mut gpu_optimizer = GpuOptimizer::new(gpu_config);

    let pricing_config = CloudPricingConfig::default();
    let mut pricing_engine = CloudPricingEngine::new(pricing_config);

    let scheduler_config = SchedulerConfig::default();
    let mut scheduler = WorkloadScheduler::new(scheduler_config);

    let predictor_config = PredictorConfig::default();
    let mut predictor = CostPredictor::new(predictor_config);

    let budget_config = BudgetManagerConfig::default();
    let mut budget_manager = BudgetManager::new(budget_config);

    // Step 1: Record current resource usage
    let metrics = ResourceMetrics {
        timestamp: Utc::now(),
        cpu_usage_percent: 75.0,
        memory_usage_gb: 32.0,
        gpu_usage_percent: HashMap::from([("gpu0".to_string(), 80.0), ("gpu1".to_string(), 90.0)]),
        gpu_memory_gb: HashMap::from([("gpu0".to_string(), 16.0), ("gpu1".to_string(), 16.0)]),
        network_bandwidth_mbps: 100.0,
        disk_iops: 5000,
        cost_per_hour: OrderedFloat(5.50),
    };

    resource_tracker.record_metrics(metrics).await.unwrap();

    // Step 2: Request GPU allocation
    let gpu_request = GpuRequest {
        workload_id: "ml-training-job".to_string(),
        gpu_count: 2,
        memory_gb_per_gpu: 16,
        gpu_type: Some("A100".to_string()),
        duration: Some(Duration::hours(4)),
        priority: 8,
        preemptible: false,
        affinity_rules: vec![],
    };

    let allocation = gpu_optimizer.allocate_gpus(&gpu_request).await;
    assert!(allocation.is_ok());

    // Step 3: Calculate pricing for the workload
    let pricing_request = PricingRequest {
        provider: CloudProvider::AWS,
        region: "us-east-1".to_string(),
        instance_type: "p4d.24xlarge".to_string(),
        duration_hours: 4.0,
        pricing_model: PricingModel::OnDemand,
        os: "linux".to_string(),
    };

    let cost = pricing_engine.calculate_cost(&pricing_request).await;
    assert!(cost.is_ok());

    // Step 4: Schedule the workload
    let workload = Workload {
        id: "ml-training-job".to_string(),
        priority: 8,
        cpu_cores: 96,
        memory_gb: 256,
        gpu_count: Some(8),
        estimated_duration: Duration::hours(4),
        cost_limit: Some(OrderedFloat(500.0)),
        deadline: None,
        dependencies: vec![],
        preemptible: false,
        affinity_rules: vec![],
    };

    let schedule_result = scheduler.schedule_workload(workload).await;
    assert!(schedule_result.is_ok());

    // Step 5: Predict future costs
    let forecast = predictor.forecast_costs("cluster-prod", 7).await;
    assert!(forecast.is_ok());

    // Step 6: Track budget
    let spend = CostRecord {
        timestamp: Utc::now(),
        amount: OrderedFloat(120.0),
        currency: "USD".to_string(),
        service: "compute".to_string(),
        resource_id: "ml-training-job".to_string(),
        tags: HashMap::from([
            ("project".to_string(), "ml-research".to_string()),
            ("team".to_string(), "data-science".to_string()),
        ]),
    };

    budget_manager.record_spend(spend).await.unwrap();

    // Verify end-to-end flow
    let stats = resource_tracker.get_statistics(Utc::now() - Duration::hours(1), Utc::now());
    assert!(stats.is_some());
}

#[tokio::test]
async fn test_spot_instance_management() {
    let spot_config = SpotManagerConfig::default();
    let mut spot_manager = SpotManager::new(spot_config);

    let budget_config = BudgetManagerConfig::default();
    let mut budget_manager = BudgetManager::new(budget_config);

    // Place a spot bid
    let spot_request = SpotRequest {
        instance_type: "g4dn.xlarge".to_string(),
        region: "us-west-2".to_string(),
        max_price: OrderedFloat(0.5),
        duration_hours: 2,
        interruption_behavior: "terminate".to_string(),
    };

    let bid_result = spot_manager.place_bid(spot_request).await;
    assert!(bid_result.is_ok());

    if let Ok(bid) = bid_result {
        // Check interruption risk
        let risk = spot_manager.predict_interruption(&bid.instance_id);
        assert!(risk >= 0.0 && risk <= 1.0);

        // Record spot savings
        let spot_savings = CostRecord {
            timestamp: Utc::now(),
            amount: OrderedFloat(-20.0), // Negative for savings
            currency: "USD".to_string(),
            service: "compute-spot".to_string(),
            resource_id: bid.instance_id.clone(),
            tags: HashMap::from([("type".to_string(), "spot-savings".to_string())]),
        };

        budget_manager.record_spend(spot_savings).await.unwrap();
    }
}

#[tokio::test]
async fn test_gpu_sharing_and_optimization() {
    let gpu_config = GpuOptimizerConfig {
        allocation_strategy: AllocationStrategy::BestFit,
        enable_sharing: true,
        enable_mig: true,
        rebalance_interval: Duration::minutes(5),
        utilization_threshold: 0.8,
    };
    let mut gpu_optimizer = GpuOptimizer::new(gpu_config);

    // Allocate GPUs for multiple workloads with sharing
    let workload1 = GpuRequest {
        workload_id: "inference-1".to_string(),
        gpu_count: 1,
        memory_gb_per_gpu: 8,
        gpu_type: Some("A100".to_string()),
        duration: Some(Duration::hours(1)),
        priority: 5,
        preemptible: true,
        affinity_rules: vec![],
    };

    let workload2 = GpuRequest {
        workload_id: "inference-2".to_string(),
        gpu_count: 1,
        memory_gb_per_gpu: 8,
        gpu_type: Some("A100".to_string()),
        duration: Some(Duration::hours(1)),
        priority: 5,
        preemptible: true,
        affinity_rules: vec![],
    };

    let alloc1 = gpu_optimizer.allocate_gpus(&workload1).await;
    let alloc2 = gpu_optimizer.allocate_gpus(&workload2).await;

    assert!(alloc1.is_ok());
    assert!(alloc2.is_ok());

    // Check GPU utilization
    let utilization = gpu_optimizer.get_gpu_utilization();
    for (_, util) in utilization {
        assert!(util >= 0.0 && util <= 1.0);
    }
}

#[tokio::test]
async fn test_cost_anomaly_detection() {
    let predictor_config = PredictorConfig::default();
    let mut predictor = CostPredictor::new(predictor_config);

    let analyzer_config = AnalyzerConfig::default();
    let mut analyzer = UsageAnalyzer::new(analyzer_config);

    // Record historical cost data
    for i in 0..10 {
        let cost = CostDataPoint {
            timestamp: Utc::now() - Duration::days(i),
            cost: OrderedFloat(100.0 + (i as f64 * 5.0)),
            resource_id: "production-cluster".to_string(),
            metrics: HashMap::new(),
        };
        predictor.record_cost_data(cost).await.unwrap();
    }

    // Check for anomaly with spike
    let spike_cost = OrderedFloat(500.0);
    let is_anomaly = predictor.detect_anomaly("production-cluster", spike_cost);
    assert!(is_anomaly); // Should detect anomaly

    // Get recommendations
    let recommendations = analyzer.get_recommendations("production-cluster").await;
    assert!(recommendations.is_ok());
}

#[tokio::test]
async fn test_multi_cloud_arbitrage() {
    let pricing_config = CloudPricingConfig::default();
    let mut pricing_engine = CloudPricingEngine::new(pricing_config);

    // Compare providers for same workload
    let comparisons = pricing_engine
        .compare_providers("gpu-large", "us-east")
        .await;
    assert!(comparisons.is_ok());

    if let Ok(provider_costs) = comparisons {
        // Find cheapest provider
        let cheapest = provider_costs
            .iter()
            .min_by_key(|(_, cost)| cost.total_cost)
            .map(|(provider, _)| provider);

        assert!(cheapest.is_some());

        // Get recommendation
        let recommendation = pricing_engine
            .get_recommendations(&PricingRequest {
                provider: CloudProvider::AWS,
                region: "us-east-1".to_string(),
                instance_type: "p3.2xlarge".to_string(),
                duration_hours: 24.0,
                pricing_model: PricingModel::OnDemand,
                os: "linux".to_string(),
            })
            .await;

        assert!(recommendation.is_ok());
    }
}

#[tokio::test]
async fn test_budget_alerts_and_actions() {
    let budget_config = BudgetManagerConfig::default();
    let mut budget_manager = BudgetManager::new(budget_config);

    // Create budget with alerts
    let budget = Budget {
        id: "q4-ml-budget".to_string(),
        name: "Q4 ML Research Budget".to_string(),
        amount: OrderedFloat(10000.0),
        currency: "USD".to_string(),
        period: BudgetPeriod::Monthly,
        scope: BudgetScope::Department("ml-research".to_string()),
        alerts: vec![
            BudgetAlert {
                id: "alert-80".to_string(),
                threshold_percent: 80.0,
                severity: AlertSeverity::Warning,
                actions: vec![AlertAction::Email {
                    recipients: vec!["team@example.com".to_string()],
                }],
                cooldown: Duration::hours(24),
                last_triggered: None,
            },
            BudgetAlert {
                id: "alert-95".to_string(),
                threshold_percent: 95.0,
                severity: AlertSeverity::Critical,
                actions: vec![AlertAction::ScaleDown {
                    resource_type: "gpu".to_string(),
                    percentage: 50,
                }],
                cooldown: Duration::hours(1),
                last_triggered: None,
            },
        ],
        tags: HashMap::new(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };

    budget_manager.create_budget(budget).await.unwrap();

    // Record spending that triggers alert
    let large_spend = CostRecord {
        timestamp: Utc::now(),
        amount: OrderedFloat(8500.0),
        currency: "USD".to_string(),
        service: "compute".to_string(),
        resource_id: "ml-cluster".to_string(),
        tags: HashMap::from([("department".to_string(), "ml-research".to_string())]),
    };

    budget_manager.record_spend(large_spend).await.unwrap();

    // Evaluate alerts
    let triggered = budget_manager.evaluate_alerts().await;
    assert!(triggered.is_ok());
    assert!(!triggered.unwrap().is_empty());
}

#[tokio::test]
async fn test_workload_cost_optimization() {
    let scheduler_config = SchedulerConfig::default();
    let mut scheduler = WorkloadScheduler::new(scheduler_config);

    let analyzer_config = AnalyzerConfig::default();
    let mut analyzer = UsageAnalyzer::new(analyzer_config);

    // Schedule cost-optimized workload
    let workload = Workload {
        id: "batch-job-123".to_string(),
        priority: 3, // Low priority
        cpu_cores: 32,
        memory_gb: 128,
        gpu_count: Some(2),
        estimated_duration: Duration::hours(6),
        cost_limit: Some(OrderedFloat(100.0)),
        deadline: Some(Utc::now() + Duration::hours(24)),
        dependencies: vec![],
        preemptible: true, // Allow preemption for cost savings
        affinity_rules: vec![],
    };

    let scheduled = scheduler.schedule_workload(workload).await;
    assert!(scheduled.is_ok());

    // Analyze usage patterns
    let patterns = analyzer.detect_patterns("batch-job-123").await;
    assert!(patterns.is_ok());

    // Get optimization recommendations
    let recommendations = analyzer.get_recommendations("batch-job-123").await;
    assert!(recommendations.is_ok());
}

#[tokio::test]
async fn test_resource_rightsizing() {
    let analyzer_config = AnalyzerConfig::default();
    let mut analyzer = UsageAnalyzer::new(analyzer_config);

    let tracker_config = ResourceTrackerConfig::default();
    let mut tracker = ResourceTracker::new(tracker_config)?;

    // Record underutilized resources
    for i in 0..24 {
        let metrics = ResourceMetrics {
            timestamp: Utc::now() - Duration::hours(i as i64),
            cpu_usage_percent: 20.0, // Low CPU usage
            memory_usage_gb: 8.0,    // Using only 8GB of 64GB
            gpu_usage_percent: HashMap::from([("gpu0".to_string(), 15.0)]), // Low GPU usage
            gpu_memory_gb: HashMap::from([("gpu0".to_string(), 4.0)]),
            network_bandwidth_mbps: 10.0,
            disk_iops: 100,
            cost_per_hour: OrderedFloat(10.0), // Overpaying
        };
        tracker.record_metrics(metrics).await.unwrap();
    }

    // Detect waste
    let waste = analyzer.detect_waste("overprovisioned-instance").await;
    assert!(waste.is_ok());

    if let Ok(waste_items) = waste {
        assert!(!waste_items.is_empty());

        // Check for rightsizing recommendation
        let has_rightsize = waste_items
            .iter()
            .any(|w| matches!(w.recovery_action, RecoveryAction::Rightsize { .. }));
        assert!(has_rightsize);
    }
}

#[tokio::test]
async fn test_cost_allocation_and_chargeback() {
    let budget_config = BudgetManagerConfig::default();
    let mut budget_manager = BudgetManager::new(budget_config);

    // Record costs with proper tagging
    let costs = vec![
        CostRecord {
            timestamp: Utc::now(),
            amount: OrderedFloat(150.0),
            currency: "USD".to_string(),
            service: "compute".to_string(),
            resource_id: "instance-1".to_string(),
            tags: HashMap::from([
                ("project".to_string(), "project-a".to_string()),
                ("team".to_string(), "team-1".to_string()),
                ("env".to_string(), "prod".to_string()),
            ]),
        },
        CostRecord {
            timestamp: Utc::now(),
            amount: OrderedFloat(200.0),
            currency: "USD".to_string(),
            service: "storage".to_string(),
            resource_id: "bucket-1".to_string(),
            tags: HashMap::from([
                ("project".to_string(), "project-b".to_string()),
                ("team".to_string(), "team-2".to_string()),
                ("env".to_string(), "dev".to_string()),
            ]),
        },
    ];

    for cost in costs {
        budget_manager.record_spend(cost).await.unwrap();
    }

    // Get cost allocation report
    let allocation = budget_manager
        .get_cost_allocation(Utc::now() - Duration::days(1), Utc::now(), "project")
        .await;

    assert!(allocation.is_ok());
    if let Ok(alloc_map) = allocation {
        assert!(alloc_map.contains_key("project-a"));
        assert!(alloc_map.contains_key("project-b"));
    }
}
