//! Cost optimization framework for GPU/cloud resources with predictive analysis
//!
//! This crate provides comprehensive cost optimization capabilities for:
//! - GPU resource allocation and scheduling
//! - Cloud cost prediction and budgeting
//! - Workload placement optimization
//! - Spot instance management
//! - Resource usage forecasting
//! - Cost allocation and chargeback

#![warn(missing_docs)]

pub mod budget_manager;
pub mod cloud_pricing;
pub mod cost_predictor;
pub mod error;
pub mod gpu_optimizer;
pub mod resource_tracker;
pub mod spot_manager;
pub mod usage_analyzer;
pub mod workload_scheduler;

pub use error::{CostOptimizationError, CostOptimizationResult};

// Resource tracking
pub use resource_tracker::{
    AlertSeverity, ResourceAlert, ResourceMetrics, ResourceSnapshot, ResourceStats,
    ResourceThresholds, ResourceTracker, ResourceType,
};

// GPU optimization
pub use gpu_optimizer::{
    AllocationStrategy, GpuAllocation, GpuAllocationRequest, GpuDevice, GpuOptimizer,
    GpuSharingPolicy,
};

// Cloud pricing
pub use cloud_pricing::{
    CloudPricingService, CloudProvider, CostCalculationRequest, CostCalculationResult,
    InstancePricing, PricingModel, ProviderComparison,
};

// Workload scheduling
pub use workload_scheduler::{
    ComputeNode, ScheduledWorkload, SchedulingStrategy, Workload, WorkloadPriority,
    WorkloadScheduler, WorkloadState,
};

// Spot instance management
pub use spot_manager::{
    BiddingStrategy, FallbackStrategy, SpotInstance, SpotInstanceRequest, SpotInstanceState,
    SpotManager, SpotMarketAnalysis,
};

// Cost prediction
pub use cost_predictor::{
    Anomaly, AnomalyType, BudgetForecast, CostMetricType, CostPredictor, PredictionModel,
    PredictionRequest, PredictionResult,
};

// Budget management
pub use budget_manager::{
    Budget, BudgetAlert, BudgetManager, BudgetPeriod, BudgetScope, BudgetStatus, ChargebackReport,
    CostAllocation,
};

// Usage analysis
pub use usage_analyzer::{
    AnalysisRequest, AnalysisResult, OptimizationOpportunity, OptimizationType, UsageAnalyzer,
    UsagePattern, WasteAnalysis,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_optimization_framework_creation() {
        // Test creating the main components
        use crate::budget_manager::BudgetManagerConfig;
        use crate::cloud_pricing::CloudPricingConfig;
        use crate::cost_predictor::CostPredictorConfig;
        use crate::gpu_optimizer::GpuOptimizerConfig;
        use crate::resource_tracker::ResourceTrackerConfig;
        use crate::spot_manager::SpotManagerConfig;
        use crate::usage_analyzer::UsageAnalyzerConfig;
        use crate::workload_scheduler::SchedulerConfig;

        // Resource Tracker
        let tracker_config = ResourceTrackerConfig::default();
        let tracker = ResourceTracker::new(tracker_config);
        assert!(tracker.is_ok());

        // GPU Optimizer
        let gpu_config = GpuOptimizerConfig::default();
        let gpu_optimizer = GpuOptimizer::new(gpu_config);
        assert!(gpu_optimizer.is_ok());

        // Cloud Pricing Service
        let pricing_config = CloudPricingConfig::default();
        let pricing_service = CloudPricingService::new(pricing_config);
        assert!(pricing_service.is_ok());

        // Workload Scheduler
        let scheduler_config = SchedulerConfig::default();
        let scheduler = WorkloadScheduler::new(scheduler_config);
        assert!(scheduler.is_ok());

        // Spot Manager
        let spot_config = SpotManagerConfig::default();
        let spot_manager = SpotManager::new(spot_config);
        assert!(spot_manager.is_ok());

        // Cost Predictor
        let predictor_config = CostPredictorConfig::default();
        let predictor = CostPredictor::new(predictor_config);
        assert!(predictor.is_ok());

        // Budget Manager
        let budget_config = BudgetManagerConfig::default();
        let budget_manager = BudgetManager::new(budget_config);
        assert!(budget_manager.is_ok());

        // Usage Analyzer
        let analyzer_config = UsageAnalyzerConfig::default();
        let analyzer = UsageAnalyzer::new(analyzer_config);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_module_re_exports() {
        // Verify all types are properly re-exported

        // Resource tracking types
        let _resource_type = ResourceType::Cpu;
        let _alert_severity = AlertSeverity::Low;

        // GPU optimization types
        let _allocation_strategy = AllocationStrategy::BestFit;
        let _sharing_policy = GpuSharingPolicy::Exclusive;

        // Cloud pricing types
        let _cloud_provider = CloudProvider::AWS;
        let _pricing_model = PricingModel::OnDemand;

        // Workload scheduling types
        let _workload_priority = WorkloadPriority::Normal;
        let _workload_state = WorkloadState::Pending;
        let _scheduling_strategy = SchedulingStrategy::CostOptimized;

        // Spot instance types
        let _bidding_strategy = BiddingStrategy::Conservative;
        let _fallback_strategy = FallbackStrategy::OnDemand;
        let _spot_state = SpotInstanceState::Running;

        // Cost prediction types
        let _anomaly_type = AnomalyType::Spike;
        let _cost_metric = CostMetricType::Daily;
        let _prediction_model = PredictionModel::ARIMA;

        // Budget management types
        let _budget_period = BudgetPeriod::Monthly;
        let _budget_scope = BudgetScope::Global;
        let _budget_status = BudgetStatus::Active;

        // Usage analysis types
        let _optimization_type = OptimizationType::ResourceRightsizing;
        let _usage_pattern = UsagePattern::Steady;
    }

    #[test]
    fn test_resource_metrics_creation() {
        let snapshot = ResourceSnapshot {
            timestamp: chrono::Utc::now(),
            resource_type: ResourceType::Gpu,
            resource_id: "GPU-0".to_string(),
            utilization: 85.0,
            available: 4.0,
            total: 24.0,
            metadata: std::collections::HashMap::new(),
        };

        assert_eq!(snapshot.resource_type, ResourceType::Gpu);
        assert_eq!(snapshot.resource_id, "GPU-0");
        assert_eq!(snapshot.utilization, 85.0);
        assert_eq!(snapshot.available, 4.0);
        assert_eq!(snapshot.total, 24.0);
    }

    #[test]
    fn test_gpu_allocation_request() {
        let request = GpuAllocationRequest {
            workload_id: "wl-123".to_string(),
            gpu_memory_gb: 16.0,
            gpu_count: 2,
            sharing_policy: GpuSharingPolicy::Exclusive,
            priority: WorkloadPriority::High,
            duration_hours: Some(24),
            preferred_device: Some("RTX 4090".to_string()),
        };

        assert_eq!(request.workload_id, "wl-123");
        assert_eq!(request.gpu_memory_gb, 16.0);
        assert_eq!(request.gpu_count, 2);
        assert_eq!(request.sharing_policy, GpuSharingPolicy::Exclusive);
    }

    #[test]
    fn test_workload_creation() {
        let workload = Workload {
            id: "wl-456".to_string(),
            name: "Training Job".to_string(),
            priority: WorkloadPriority::Normal,
            cpu_cores: 8.0,
            memory_gb: 32.0,
            gpu_count: Some(1),
            gpu_memory_gb: Some(12.0),
            estimated_duration_hours: 6.0,
            max_cost_per_hour: Some(10.0),
            preferred_regions: vec!["us-east-1".to_string()],
            constraints: std::collections::HashMap::new(),
            created_at: chrono::Utc::now(),
            deadline: None,
        };

        assert_eq!(workload.id, "wl-456");
        assert_eq!(workload.name, "Training Job");
        assert_eq!(workload.cpu_cores, 8.0);
        assert_eq!(workload.memory_gb, 32.0);
    }

    #[test]
    fn test_spot_instance_request() {
        let request = SpotInstanceRequest {
            instance_type: "p3.2xlarge".to_string(),
            region: "us-west-2".to_string(),
            availability_zone: Some("us-west-2a".to_string()),
            max_price_per_hour: 3.50,
            duration_hours: 12,
            workload_id: "wl-789".to_string(),
            bidding_strategy: BiddingStrategy::Aggressive,
            fallback_strategy: FallbackStrategy::Wait,
        };

        assert_eq!(request.instance_type, "p3.2xlarge");
        assert_eq!(request.region, "us-west-2");
        assert_eq!(request.max_price_per_hour, 3.50);
        assert_eq!(request.duration_hours, 12);
    }

    #[test]
    fn test_budget_creation() {
        let budget = Budget {
            id: "budget-001".to_string(),
            name: "Monthly GPU Budget".to_string(),
            amount: 10000.0,
            period: BudgetPeriod::Monthly,
            scope: BudgetScope::Department("Engineering".to_string()),
            alert_thresholds: vec![80.0, 90.0, 95.0],
            tags: std::collections::HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        assert_eq!(budget.id, "budget-001");
        assert_eq!(budget.name, "Monthly GPU Budget");
        assert_eq!(budget.amount, 10000.0);
        assert_eq!(budget.alert_thresholds, vec![80.0, 90.0, 95.0]);
    }

    #[test]
    fn test_cost_calculation_request() {
        let request = CostCalculationRequest {
            provider: CloudProvider::AWS,
            region: "eu-west-1".to_string(),
            instance_type: "g4dn.xlarge".to_string(),
            pricing_model: PricingModel::Spot,
            duration_hours: 48.0,
            data_transfer_gb: Some(100.0),
            storage_gb: Some(500.0),
            additional_services: vec![],
        };

        assert_eq!(request.provider, CloudProvider::AWS);
        assert_eq!(request.region, "eu-west-1");
        assert_eq!(request.instance_type, "g4dn.xlarge");
        assert_eq!(request.duration_hours, 48.0);
    }

    #[test]
    fn test_prediction_request() {
        let request = PredictionRequest {
            metric_type: CostMetricType::Daily,
            historical_days: 30,
            forecast_days: 7,
            model: PredictionModel::Prophet,
            confidence_level: 0.95,
            include_seasonality: true,
            include_anomalies: false,
        };

        assert_eq!(request.metric_type, CostMetricType::Daily);
        assert_eq!(request.historical_days, 30);
        assert_eq!(request.forecast_days, 7);
        assert_eq!(request.confidence_level, 0.95);
    }

    #[test]
    fn test_analysis_request() {
        let request = AnalysisRequest {
            start_date: chrono::Utc::now() - chrono::Duration::days(30),
            end_date: chrono::Utc::now(),
            resource_types: vec![ResourceType::Gpu, ResourceType::Cpu],
            group_by: vec!["department".to_string(), "project".to_string()],
            include_recommendations: true,
            minimum_savings_threshold: Some(100.0),
        };

        assert_eq!(request.resource_types.len(), 2);
        assert!(request.include_recommendations);
        assert_eq!(request.minimum_savings_threshold, Some(100.0));
    }

    #[test]
    fn test_error_result_type() {
        // Test successful result
        let success: CostOptimizationResult<String> = Ok("Success".to_string());
        assert!(success.is_ok());

        // Test error result
        let error: CostOptimizationResult<String> = Err(CostOptimizationError::AllocationFailed {
            reason: "No GPUs available".to_string(),
        });
        assert!(error.is_err());
    }

    #[test]
    fn test_resource_thresholds() {
        let thresholds = ResourceThresholds {
            cpu_percent: Some(80.0),
            memory_percent: Some(85.0),
            gpu_percent: Some(90.0),
            gpu_memory_percent: Some(95.0),
            network_mbps: Some(1000.0),
            disk_iops: Some(5000),
            disk_throughput_mbps: Some(500.0),
        };

        assert_eq!(thresholds.cpu_percent, Some(80.0));
        assert_eq!(thresholds.gpu_percent, Some(90.0));
        assert_eq!(thresholds.disk_iops, Some(5000));
    }

    #[test]
    fn test_optimization_opportunity() {
        let opportunity = OptimizationOpportunity {
            id: "opt-001".to_string(),
            optimization_type: OptimizationType::ResourceRightsizing,
            resource_id: "i-1234567890".to_string(),
            current_cost_per_month: 500.0,
            optimized_cost_per_month: 300.0,
            savings_per_month: 200.0,
            savings_percentage: 40.0,
            effort_level: "Low".to_string(),
            risk_level: "Low".to_string(),
            description: "Downsize instance".to_string(),
            recommendations: vec!["Change from m5.xlarge to m5.large".to_string()],
        };

        assert_eq!(opportunity.id, "opt-001");
        assert_eq!(opportunity.savings_per_month, 200.0);
        assert_eq!(opportunity.savings_percentage, 40.0);
    }

    #[test]
    fn test_multiple_cloud_providers() {
        let providers = vec![
            CloudProvider::AWS,
            CloudProvider::Azure,
            CloudProvider::GCP,
            CloudProvider::OnPremise,
        ];

        for provider in providers {
            let _name = format!("{:?}", provider);
            // Verify all providers can be used
        }
    }

    #[test]
    fn test_scheduling_strategies() {
        let strategies = vec![
            SchedulingStrategy::CostOptimized,
            SchedulingStrategy::PerformanceOptimized,
            SchedulingStrategy::Balanced,
            SchedulingStrategy::SpotFirst,
        ];

        for strategy in strategies {
            let _name = format!("{:?}", strategy);
            // Verify all strategies can be used
        }
    }

    #[test]
    fn test_anomaly_detection() {
        let anomaly = Anomaly {
            timestamp: chrono::Utc::now(),
            metric_type: CostMetricType::Hourly,
            anomaly_type: AnomalyType::Spike,
            severity: 0.85,
            actual_value: 150.0,
            expected_value: 50.0,
            deviation_percentage: 200.0,
            description: "Unusual spike in GPU costs".to_string(),
        };

        assert_eq!(anomaly.anomaly_type, AnomalyType::Spike);
        assert_eq!(anomaly.severity, 0.85);
        assert_eq!(anomaly.deviation_percentage, 200.0);
    }
}
