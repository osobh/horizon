//! Scaling-related types for AI Assistant Zero-Config Integration
//! Split from types.rs to keep files under 750 lines

use std::collections::HashMap;
use std::time::Duration;

/// Scaling configuration
#[derive(Debug, Clone)]
pub struct ScalingConfiguration {
    pub horizontal_scaling: HorizontalScalingConfiguration,
    pub vertical_scaling: VerticalScalingConfiguration,
    pub auto_scaling_policies: Vec<AutoScalingPolicy>,
    pub resource_quotas: ResourceQuotaConfiguration,
}

/// Horizontal scaling configuration
#[derive(Debug, Clone)]
pub struct HorizontalScalingConfiguration {
    pub enabled: bool,
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub target_cpu_utilization: f64,
    pub target_memory_utilization: f64,
    pub scale_up_policy: ScalePolicy,
    pub scale_down_policy: ScalePolicy,
}

/// Scaling policies
#[derive(Debug, Clone)]
pub struct ScalePolicy {
    pub stabilization_window: Duration,
    pub select_policy: SelectPolicy,
    pub policies: Vec<ScalingRule>,
}

/// Policy selection strategies
#[derive(Debug, Clone, PartialEq)]
pub enum SelectPolicy {
    Max,
    Min,
    Disabled,
}

/// Scaling rules
#[derive(Debug, Clone)]
pub struct ScalingRule {
    pub rule_type: ScalingRuleType,
    pub value: u32,
    pub period: Duration,
}

/// Scaling rule types
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingRuleType {
    Percent,
    Pods,
}

/// Vertical scaling configuration
#[derive(Debug, Clone)]
pub struct VerticalScalingConfiguration {
    pub enabled: bool,
    pub update_mode: VPAUpdateMode,
    pub resource_policy: VPAResourcePolicy,
    pub recommendation_margin: f64,
}

/// VPA update modes
#[derive(Debug, Clone, PartialEq)]
pub enum VPAUpdateMode {
    Off,
    Initial,
    Recreation,
    Auto,
}

/// VPA resource policies
#[derive(Debug, Clone)]
pub struct VPAResourcePolicy {
    pub container_policies: Vec<VPAContainerPolicy>,
}

/// VPA container policies
#[derive(Debug, Clone)]
pub struct VPAContainerPolicy {
    pub container_name: String,
    pub min_allowed: HashMap<String, String>,
    pub max_allowed: HashMap<String, String>,
    pub controlled_resources: Vec<String>,
}

/// Auto-scaling policies
#[derive(Debug, Clone)]
pub struct AutoScalingPolicy {
    pub name: String,
    pub policy_type: AutoScalingPolicyType,
    pub target_metrics: Vec<TargetMetric>,
    pub scaling_behavior: ScalingBehavior,
}

/// Auto-scaling policy types
#[derive(Debug, Clone, PartialEq)]
pub enum AutoScalingPolicyType {
    HPA,
    VPA,
    KEDA,
    Custom,
}

/// Target metrics for auto-scaling
#[derive(Debug, Clone)]
pub struct TargetMetric {
    pub metric_type: MetricSourceType,
    pub metric_name: String,
    pub target_value: TargetValue,
}

/// Metric source types
#[derive(Debug, Clone, PartialEq)]
pub enum MetricSourceType {
    Resource,
    Pods,
    Object,
    External,
}

/// Target values for metrics
#[derive(Debug, Clone)]
pub enum TargetValue {
    AverageValue(String),
    AverageUtilization(u32),
    Value(String),
}

/// Scaling behavior configuration
#[derive(Debug, Clone)]
pub struct ScalingBehavior {
    pub scale_up: Option<ScalePolicy>,
    pub scale_down: Option<ScalePolicy>,
}

/// Resource quota configuration
#[derive(Debug, Clone)]
pub struct ResourceQuotaConfiguration {
    pub enabled: bool,
    pub compute_quota: ComputeQuota,
    pub storage_quota: StorageQuota,
    pub network_quota: NetworkQuota,
    pub object_quota: ObjectQuota,
}

/// Compute resource quotas
#[derive(Debug, Clone)]
pub struct ComputeQuota {
    pub cpu_limit: Option<String>,
    pub memory_limit: Option<String>,
    pub gpu_limit: Option<u32>,
    pub pod_limit: Option<u32>,
}

/// Storage quotas
#[derive(Debug, Clone)]
pub struct StorageQuota {
    pub storage_limit: Option<String>,
    pub persistent_volume_claims_limit: Option<u32>,
    pub storage_class_limits: HashMap<String, String>,
}

/// Network quotas
#[derive(Debug, Clone)]
pub struct NetworkQuota {
    pub service_limit: Option<u32>,
    pub ingress_limit: Option<u32>,
    pub load_balancer_limit: Option<u32>,
}

/// Object quotas
#[derive(Debug, Clone)]
pub struct ObjectQuota {
    pub config_map_limit: Option<u32>,
    pub secret_limit: Option<u32>,
    pub service_account_limit: Option<u32>,
}

/// Scaling policies
#[derive(Debug, Clone)]
pub struct ScalingPolicy {
    pub policy_name: String,
    pub policy_type: ScalingPolicyType,
    pub triggers: Vec<ScalingTrigger>,
    pub actions: Vec<ScalingAction>,
    pub cooldown_period: Duration,
    pub min_capacity: u32,
    pub max_capacity: u32,
}

/// Scaling policy types
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingPolicyType {
    Reactive,
    Predictive,
    Scheduled,
    Manual,
}

/// Scaling triggers
#[derive(Debug, Clone)]
pub struct ScalingTrigger {
    pub trigger_type: TriggerType,
    pub metric_name: String,
    pub threshold: f64,
    pub comparison_operator: super::ComparisonOperator,
    pub evaluation_periods: u32,
}

/// Trigger types for scaling
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerType {
    Metric,
    Schedule,
    Event,
    Manual,
}

/// Scaling actions
#[derive(Debug, Clone)]
pub struct ScalingAction {
    pub action_type: ScalingActionType,
    pub adjustment_value: i32,
    pub adjustment_type: AdjustmentType,
}

/// Scaling action types
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingActionType {
    ScaleUp,
    ScaleDown,
    SetCapacity,
}

/// Adjustment types for scaling
#[derive(Debug, Clone, PartialEq)]
pub enum AdjustmentType {
    ChangeInCapacity,
    ExactCapacity,
    PercentChangeInCapacity,
}