//! Cost optimization types for AI Assistant Zero-Config Integration
//! Split from types.rs to keep files under 750 lines

use std::collections::HashMap;
use std::time::Duration;

/// Cost optimization configuration
#[derive(Debug, Clone)]
pub struct CostOptimization {
    pub cost_tracking: CostTrackingConfiguration,
    pub resource_optimization: ResourceOptimizationConfiguration,
    pub budget_controls: BudgetControlsConfiguration,
    pub recommendations: Vec<CostRecommendation>,
}

/// Cost tracking configuration
#[derive(Debug, Clone)]
pub struct CostTrackingConfiguration {
    pub enabled: bool,
    pub cost_allocation_tags: HashMap<String, String>,
    pub cost_centers: Vec<CostCenter>,
    pub reporting_frequency: Duration,
}

/// Cost centers
#[derive(Debug, Clone)]
pub struct CostCenter {
    pub name: String,
    pub owner: String,
    pub budget_limit: f64,
    pub alert_threshold: f64,
}

/// Resource optimization configuration
#[derive(Debug, Clone)]
pub struct ResourceOptimizationConfiguration {
    pub rightsizing_enabled: bool,
    pub spot_instances_enabled: bool,
    pub reserved_instances_enabled: bool,
    pub optimization_targets: Vec<OptimizationTarget>,
}

/// Optimization targets
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationTarget {
    Cost,
    Performance,
    Availability,
    Sustainability,
}

/// Budget controls configuration
#[derive(Debug, Clone)]
pub struct BudgetControlsConfiguration {
    pub budget_alerts: Vec<BudgetAlert>,
    pub spending_limits: Vec<SpendingLimit>,
    pub cost_anomaly_detection: bool,
}

/// Budget alerts
#[derive(Debug, Clone)]
pub struct BudgetAlert {
    pub name: String,
    pub threshold_percentage: f64,
    pub notification_channels: Vec<String>,
    pub time_period: TimePeriod,
}

/// Time periods for budgets
#[derive(Debug, Clone, PartialEq)]
pub enum TimePeriod {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
}

/// Spending limits
#[derive(Debug, Clone)]
pub struct SpendingLimit {
    pub resource_type: String,
    pub monthly_limit: f64,
    pub action_on_limit: LimitAction,
}

/// Actions when spending limits are reached
#[derive(Debug, Clone, PartialEq)]
pub enum LimitAction {
    Alert,
    Block,
    Approve,
    Scale,
}

/// Cost recommendations
#[derive(Debug, Clone)]
pub struct CostRecommendation {
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub estimated_savings: f64,
    pub implementation_effort: ImplementationEffort,
    pub confidence_level: f64,
}

/// Recommendation types
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationType {
    Rightsizing,
    SpotInstances,
    ReservedInstances,
    StorageOptimization,
    NetworkOptimization,
    SchedulingOptimization,
}

/// Implementation effort levels
#[derive(Debug, Clone, PartialEq)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
}