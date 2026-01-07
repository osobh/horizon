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
    use std::collections::HashMap;
    use uuid::Uuid;
    use crate::gpu_optimizer::GpuAffinity;

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

        // Spot Manager - SpotManager is an empty struct, no constructor needed
        let _spot_config = SpotManagerConfig { max_bid_price: 0.5 };
        let _spot_manager = SpotManager;

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
        let _alert_severity = AlertSeverity::Info;

        // GPU optimization types
        let _allocation_strategy = AllocationStrategy::BestFit;
        let _sharing_policy = GpuSharingPolicy::default();

        // Cloud pricing types
        let _cloud_provider = CloudProvider::Aws;
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
        let _cost_metric = CostMetricType::TotalCost;
        let _prediction_model = PredictionModel::Arima;

        // Budget management types
        let _budget_period = BudgetPeriod::Monthly;
        let _budget_scope = BudgetScope::Global;
        // BudgetStatus is a struct, not an enum - test its construction
        let _budget_status = BudgetStatus {
            budget_id: "test".to_string(),
            current_spend: 0.0,
            allocated_budget: 1000.0,
            remaining_budget: 1000.0,
            utilization_percent: 0.0,
            projected_spend: 0.0,
            period_start: chrono::Utc::now(),
            period_end: chrono::Utc::now(),
            trend: crate::budget_manager::SpendTrend::Stable,
            days_remaining: 30,
            is_over_budget: false,
            last_updated: chrono::Utc::now(),
        };

        // Usage analysis types
        let _optimization_type = OptimizationType::Rightsize;
        let _usage_pattern = UsagePattern::ConstantHigh;
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
            request_id: Uuid::new_v4(),
            workload_id: "wl-123".to_string(),
            memory_required: 16.0,
            compute_required: 80.0,
            preferred_models: vec!["RTX 4090".to_string()],
            exclusive: true,
            duration: std::time::Duration::from_secs(24 * 3600),
            priority: 10,
            affinity: GpuAffinity {
                numa_aware: true,
                nvlink_required: false,
                min_compute_capability: None,
                anti_affinity: vec![],
            },
        };

        assert_eq!(request.workload_id, "wl-123");
        assert_eq!(request.memory_required, 16.0);
        assert!(request.exclusive);
    }

    #[test]
    fn test_workload_creation() {
        use crate::workload_scheduler::{ResourceRequirements, PlacementConstraints};

        let workload = Workload {
            id: Uuid::new_v4(),
            name: "Training Job".to_string(),
            priority: WorkloadPriority::Normal,
            resources: ResourceRequirements {
                cpu_cores: 8.0,
                memory_gb: 32.0,
                gpu_count: 1,
                gpu_memory_gb: 12.0,
                storage_gb: 100.0,
                network_gbps: 1.0,
                gpu_models: vec![],
                exclusive: false,
            },
            constraints: PlacementConstraints {
                node_selector: HashMap::new(),
                node_affinity: vec![],
                pod_affinity: vec![],
                pod_anti_affinity: vec![],
                tolerations: vec![],
                topology_spread: vec![],
            },
            estimated_duration: std::time::Duration::from_secs(6 * 3600),
            max_cost_per_hour: Some(10.0),
            deadline: None,
            metadata: HashMap::new(),
        };

        assert_eq!(workload.name, "Training Job");
        assert_eq!(workload.resources.cpu_cores, 8.0);
        assert_eq!(workload.resources.memory_gb, 32.0);
    }

    #[test]
    fn test_spot_instance_request() {
        // SpotInstanceRequest only has instance_type field in current implementation
        let request = SpotInstanceRequest {
            instance_type: "p3.2xlarge".to_string(),
        };

        assert_eq!(request.instance_type, "p3.2xlarge");
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
        use crate::cloud_pricing::DataTransferEstimate;

        let request = CostCalculationRequest {
            provider: CloudProvider::Aws,
            region: "eu-west-1".to_string(),
            instance_type: "g4dn.xlarge".to_string(),
            instance_count: 1,
            pricing_model: PricingModel::Spot,
            duration_hours: 48.0,
            storage_gb: 500.0,
            data_transfer: DataTransferEstimate {
                inbound_gb: 0.0,
                outbound_internet_gb: 100.0,
                outbound_same_region_gb: 0.0,
                outbound_cross_region_gb: 0.0,
            },
        };

        assert_eq!(request.provider, CloudProvider::Aws);
        assert_eq!(request.region, "eu-west-1");
        assert_eq!(request.instance_type, "g4dn.xlarge");
        assert_eq!(request.duration_hours, 48.0);
    }

    #[test]
    fn test_prediction_request() {
        use crate::cost_predictor::Seasonality;

        let request = PredictionRequest {
            metric_type: CostMetricType::TotalCost,
            horizon: std::time::Duration::from_secs(7 * 24 * 3600), // 7 days
            model: PredictionModel::Arima,
            confidence_level: 0.95,
            seasonality: Seasonality::Weekly,
            filters: HashMap::new(),
        };

        assert_eq!(request.metric_type, CostMetricType::TotalCost);
        assert_eq!(request.confidence_level, 0.95);
    }

    #[test]
    fn test_analysis_request() {
        let request = AnalysisRequest {
            resource_id: "res-123".to_string(),
            resource_type: ResourceType::Gpu,
            period: std::time::Duration::from_secs(30 * 24 * 3600), // 30 days
            include_recommendations: true,
            confidence_threshold: 0.8,
            cost_per_hour: Some(5.0),
        };

        assert_eq!(request.resource_id, "res-123");
        assert!(request.include_recommendations);
        assert_eq!(request.cost_per_hour, Some(5.0));
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
            warning: 80.0,
            critical: 95.0,
            sustained_duration: std::time::Duration::from_secs(60),
        };

        assert_eq!(thresholds.warning, 80.0);
        assert_eq!(thresholds.critical, 95.0);
    }

    #[test]
    fn test_optimization_opportunity() {
        use crate::usage_analyzer::{ImplementationEffort, RiskLevel, RecommendationDetails};

        let opportunity = OptimizationOpportunity {
            id: Uuid::new_v4(),
            optimization_type: OptimizationType::Rightsize,
            title: "Downsize instance".to_string(),
            description: "Change from m5.xlarge to m5.large".to_string(),
            estimated_savings: 200.0,
            savings_percent: 40.0,
            effort: ImplementationEffort::Low,
            risk: RiskLevel::Low,
            recommendation: RecommendationDetails {
                current_config: HashMap::new(),
                recommended_config: HashMap::new(),
                steps: vec!["Change instance type".to_string()],
                prerequisites: vec![],
                rollback_plan: "Revert to original instance type".to_string(),
            },
        };

        assert_eq!(opportunity.optimization_type, OptimizationType::Rightsize);
        assert_eq!(opportunity.estimated_savings, 200.0);
        assert_eq!(opportunity.savings_percent, 40.0);
    }

    #[test]
    fn test_multiple_cloud_providers() {
        let providers = vec![
            CloudProvider::Aws,
            CloudProvider::Azure,
            CloudProvider::Gcp,
            CloudProvider::Other,
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
            SchedulingStrategy::LatencyOptimized,
        ];

        for strategy in strategies {
            let _name = format!("{:?}", strategy);
            // Verify all strategies can be used
        }
    }

    #[test]
    fn test_anomaly_detection() {
        let anomaly = Anomaly {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            anomaly_type: AnomalyType::Spike,
            actual_value: 150.0,
            expected_value: 50.0,
            deviation_percent: 200.0,
            score: 85.0,
        };

        assert_eq!(anomaly.anomaly_type, AnomalyType::Spike);
        assert_eq!(anomaly.score, 85.0);
        assert_eq!(anomaly.deviation_percent, 200.0);
    }
}
