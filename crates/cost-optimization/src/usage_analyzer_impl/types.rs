//! Core types and data structures for usage analysis
//!
//! This module contains all the fundamental types, enums, and data structures
//! used throughout the usage analysis system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

use crate::resource_tracker::ResourceType;

/// Usage pattern type detected by the analyzer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UsagePattern {
    /// Constant high usage
    ConstantHigh,
    /// Constant low usage
    ConstantLow,
    /// Periodic/cyclic usage
    Periodic,
    /// Spiky/bursty usage
    Spiky,
    /// Growing usage
    Growing,
    /// Declining usage
    Declining,
    /// Idle/unused
    Idle,
    /// Unpredictable
    Unpredictable,
}

impl std::fmt::Display for UsagePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UsagePattern::ConstantHigh => write!(f, "Constant High"),
            UsagePattern::ConstantLow => write!(f, "Constant Low"),
            UsagePattern::Periodic => write!(f, "Periodic"),
            UsagePattern::Spiky => write!(f, "Spiky"),
            UsagePattern::Growing => write!(f, "Growing"),
            UsagePattern::Declining => write!(f, "Declining"),
            UsagePattern::Idle => write!(f, "Idle"),
            UsagePattern::Unpredictable => write!(f, "Unpredictable"),
        }
    }
}

/// Optimization opportunity type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Rightsize resources
    Rightsize,
    /// Terminate idle resources
    TerminateIdle,
    /// Convert to spot instances
    ConvertToSpot,
    /// Enable autoscaling
    EnableAutoscaling,
    /// Consolidate resources
    Consolidate,
    /// Schedule on/off times
    ScheduleOnOff,
    /// Upgrade instance family
    UpgradeFamily,
    /// Use reserved instances
    UseReserved,
}

/// Resource waste type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WasteType {
    /// Overprovisioned resources
    Overprovisioned,
    /// Idle resources
    Idle,
    /// Orphaned resources
    Orphaned,
    /// Duplicate resources
    Duplicate,
    /// Inefficient configuration
    Inefficient,
    /// Unused reservations
    UnusedReservations,
}

/// Implementation effort level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ImplementationEffort {
    /// Minimal effort
    Minimal,
    /// Low effort
    Low,
    /// Medium effort
    Medium,
    /// High effort
    High,
}

/// Risk level for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    /// No risk
    None,
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
}

/// Waste severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WasteSeverity {
    /// Minimal waste
    Minimal,
    /// Low waste
    Low,
    /// Medium waste
    Medium,
    /// High waste
    High,
    /// Critical waste
    Critical,
}

/// Action priority for recovery
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ActionPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Urgent priority
    Urgent,
}

/// Trend direction for usage patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing usage
    Increasing,
    /// Decreasing usage
    Decreasing,
    /// Stable usage
    Stable,
}

/// Usage analysis request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisRequest {
    /// Resource ID to analyze
    pub resource_id: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Analysis period
    pub period: Duration,
    /// Include recommendations
    pub include_recommendations: bool,
    /// Confidence threshold for patterns
    pub confidence_threshold: f64,
    /// Cost data for optimization calculations
    pub cost_per_hour: Option<f64>,
}

/// Usage analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Request details
    pub request: AnalysisRequest,
    /// Detected usage pattern
    pub pattern: UsagePattern,
    /// Pattern confidence (0-1)
    pub confidence: f64,
    /// Usage statistics
    pub statistics: UsageStatistics,
    /// Time-based analysis
    pub temporal_analysis: TemporalAnalysis,
    /// Optimization opportunities
    pub opportunities: Vec<OptimizationOpportunity>,
    /// Waste detection results
    pub waste_analysis: WasteAnalysis,
    /// Generated timestamp
    pub generated_at: DateTime<Utc>,
}

/// Usage statistics calculated from snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStatistics {
    /// Average utilization
    pub avg_utilization: f64,
    /// Peak utilization
    pub peak_utilization: f64,
    /// Minimum utilization
    pub min_utilization: f64,
    /// Standard deviation
    pub std_deviation: f64,
    /// 95th percentile
    pub p95_utilization: f64,
    /// 99th percentile
    pub p99_utilization: f64,
    /// Idle time percentage
    pub idle_percentage: f64,
    /// Total samples analyzed
    pub sample_count: usize,
}

/// Temporal usage analysis with time-based patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysis {
    /// Hourly usage pattern
    pub hourly_pattern: Vec<HourlyUsage>,
    /// Daily usage pattern
    pub daily_pattern: Vec<DailyUsage>,
    /// Peak usage times
    pub peak_times: Vec<PeakTime>,
    /// Low usage periods
    pub low_periods: Vec<LowPeriod>,
    /// Usage trend
    pub trend: UsageTrend,
}

/// Hourly usage pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HourlyUsage {
    /// Hour of day (0-23)
    pub hour: u32,
    /// Average utilization
    pub avg_utilization: f64,
    /// Sample count
    pub samples: usize,
}

/// Daily usage pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyUsage {
    /// Day of week (1-7, 1=Monday)
    pub day_of_week: u32,
    /// Average utilization
    pub avg_utilization: f64,
    /// Sample count
    pub samples: usize,
}

/// Peak usage time period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakTime {
    /// Start time
    pub start: DateTime<Utc>,
    /// End time
    pub end: DateTime<Utc>,
    /// Peak utilization
    pub peak_utilization: f64,
    /// Duration
    pub duration: Duration,
}

/// Low usage period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowPeriod {
    /// Start time
    pub start: DateTime<Utc>,
    /// End time
    pub end: DateTime<Utc>,
    /// Average utilization
    pub avg_utilization: f64,
    /// Could be turned off
    pub can_shutdown: bool,
}

/// Usage trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageTrend {
    /// Trend direction
    pub direction: TrendDirection,
    /// Growth/decline rate
    pub rate: f64,
    /// Trend confidence
    pub confidence: f64,
}

/// Optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    /// Opportunity ID
    pub id: Uuid,
    /// Optimization type
    pub optimization_type: OptimizationType,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Estimated savings
    pub estimated_savings: f64,
    /// Savings percentage
    pub savings_percent: f64,
    /// Implementation effort
    pub effort: ImplementationEffort,
    /// Risk level
    pub risk: RiskLevel,
    /// Recommendation details
    pub recommendation: RecommendationDetails,
}

/// Recommendation details for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationDetails {
    /// Current configuration
    pub current_config: HashMap<String, String>,
    /// Recommended configuration
    pub recommended_config: HashMap<String, String>,
    /// Implementation steps
    pub steps: Vec<String>,
    /// Prerequisites
    pub prerequisites: Vec<String>,
    /// Rollback plan
    pub rollback_plan: String,
}

/// Waste analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasteAnalysis {
    /// Total waste detected
    pub total_waste_cost: f64,
    /// Waste by type
    pub waste_by_type: HashMap<WasteType, WasteDetail>,
    /// Waste severity
    pub severity: WasteSeverity,
    /// Recovery actions
    pub recovery_actions: Vec<RecoveryAction>,
}

/// Waste detail for a specific type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasteDetail {
    /// Waste type
    pub waste_type: WasteType,
    /// Amount wasted
    pub amount: f64,
    /// Percentage of total cost
    pub percentage: f64,
    /// Resources affected
    pub affected_resources: Vec<String>,
    /// Time wasted
    pub duration: Duration,
}

/// Recovery action for waste
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAction {
    /// Action type
    pub action: String,
    /// Resources to act on
    pub resources: Vec<String>,
    /// Potential recovery
    pub recovery_amount: f64,
    /// Priority
    pub priority: ActionPriority,
}

/// Resource profile for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceProfile {
    /// Profile ID
    pub id: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Capacity specifications
    pub capacity: ResourceCapacity,
    /// Cost per hour
    pub cost_per_hour: f64,
    /// Performance score
    pub performance_score: f64,
}

/// Resource capacity specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapacity {
    /// CPU cores/units
    pub cpu: f64,
    /// Memory in GB
    pub memory: f64,
    /// Storage in GB
    pub storage: f64,
    /// Network bandwidth
    pub network: f64,
    /// GPU count
    pub gpu_count: u32,
    /// Additional specs
    pub additional: HashMap<String, f64>,
}

/// Usage analyzer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageAnalyzerConfig {
    /// Minimum data points for analysis
    pub min_data_points: usize,
    /// Idle threshold percentage
    pub idle_threshold: f64,
    /// Overprovisioning threshold
    pub overprovision_threshold: f64,
    /// Pattern detection window
    pub pattern_window: Duration,
    /// Enable ML-based analysis
    pub enable_ml_analysis: bool,
    /// Resource profiles database
    pub resource_profiles: HashMap<String, ResourceProfile>,
}

impl Default for UsageAnalyzerConfig {
    fn default() -> Self {
        let mut profiles = HashMap::new();

        // Add some default profiles
        profiles.insert(
            "t3.micro".to_string(),
            ResourceProfile {
                id: "t3.micro".to_string(),
                resource_type: ResourceType::Cpu,
                capacity: ResourceCapacity {
                    cpu: 2.0,
                    memory: 1.0,
                    storage: 0.0,
                    network: 5.0,
                    gpu_count: 0,
                    additional: HashMap::new(),
                },
                cost_per_hour: 0.0104,
                performance_score: 1.0,
            },
        );

        profiles.insert(
            "t3.small".to_string(),
            ResourceProfile {
                id: "t3.small".to_string(),
                resource_type: ResourceType::Cpu,
                capacity: ResourceCapacity {
                    cpu: 2.0,
                    memory: 2.0,
                    storage: 0.0,
                    network: 5.0,
                    gpu_count: 0,
                    additional: HashMap::new(),
                },
                cost_per_hour: 0.0208,
                performance_score: 2.0,
            },
        );

        Self {
            min_data_points: 100,
            idle_threshold: 5.0,
            overprovision_threshold: 30.0,
            pattern_window: Duration::from_secs(86400 * 7), // 7 days
            enable_ml_analysis: false,
            resource_profiles: profiles,
        }
    }
}

/// Analyzer metrics for monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalyzerMetrics {
    /// Total analyses performed
    pub total_analyses: u64,
    /// Patterns detected
    pub patterns_detected: HashMap<UsagePattern, u64>,
    /// Opportunities identified
    pub opportunities_identified: u64,
    /// Total potential savings
    pub total_potential_savings: f64,
    /// Waste detected
    pub total_waste_detected: f64,
}

/// Internal pattern detection model
#[derive(Debug, Clone)]
pub(crate) struct PatternModel {
    /// Resource ID
    pub resource_id: String,
    /// Detected pattern
    pub pattern: UsagePattern,
    /// Pattern parameters
    pub parameters: PatternParameters,
    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Parameters for pattern detection
#[derive(Debug, Clone)]
pub(crate) struct PatternParameters {
    /// Period for cyclic patterns
    pub period: Option<Duration>,
    /// Growth rate for trending patterns
    pub growth_rate: Option<f64>,
    /// Spike threshold
    pub spike_threshold: Option<f64>,
    /// Baseline utilization
    pub baseline: Option<f64>,
}
