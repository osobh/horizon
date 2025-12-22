//! Type definitions for the analyzer module

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Analysis report containing insights and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub report_id: Uuid,
    pub analysis_type: AnalysisType,
    pub timestamp: u64,
    pub target_data: AnalysisTarget,
    pub findings: Vec<Finding>,
    pub recommendations: Vec<Recommendation>,
    pub severity: Severity,
    pub confidence: f64,
    pub metadata: AnalysisMetadata,
}

/// Type of analysis performed
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnalysisType {
    PerformanceAnalysis,
    AnomalyDetection,
    RegressionAnalysis,
    MemoryPatternAnalysis,
    KernelOptimization,
    ResourceUtilization,
    ComparisonAnalysis,
}

/// Target data for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisTarget {
    SingleSnapshot { snapshot_id: Uuid },
    SingleReplay { session_id: Uuid },
    CompareSnapshots { baseline: Uuid, comparison: Uuid },
    CompareReplays { baseline: Uuid, comparison: Uuid },
    Timeline { start_time: u64, end_time: u64 },
    Collection { item_ids: Vec<Uuid> },
}

/// Analysis finding with details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub finding_id: Uuid,
    pub category: FindingCategory,
    pub title: String,
    pub description: String,
    pub impact: Impact,
    pub evidence: HashMap<String, serde_json::Value>,
    pub location: Option<AnalysisLocation>,
}

/// Category of finding
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FindingCategory {
    Performance,
    Memory,
    Correctness,
    Security,
    Resource,
    Pattern,
    Anomaly,
}

/// Impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Impact {
    pub performance_impact: f64,
    pub memory_impact: f64,
    pub correctness_impact: f64,
    pub overall_score: f64,
}

/// Location information for findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisLocation {
    pub memory_address: Option<u64>,
    pub kernel_name: Option<String>,
    pub time_range: Option<(u64, u64)>,
    pub container_id: Option<Uuid>,
}

/// Recommendation for improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub recommendation_id: Uuid,
    pub title: String,
    pub description: String,
    pub action_type: ActionType,
    pub priority: Priority,
    pub estimated_improvement: EstimatedImprovement,
    pub implementation_effort: ImplementationEffort,
}

/// Type of action recommended
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionType {
    ParameterTuning,
    CodeChange,
    ConfigurationChange,
    ResourceAdjustment,
    AlgorithmOptimization,
    MemoryOptimization,
    ParallelizationImprovement,
}

/// Priority level for recommendations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
    Optional,
}

/// Severity level for issues
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Estimated improvement from recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatedImprovement {
    pub performance_gain_percent: f64,
    pub memory_savings_percent: f64,
    pub reliability_improvement: f64,
    pub confidence: f64,
}

/// Implementation effort assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationEffort {
    pub complexity: Complexity,
    pub estimated_hours: f64,
    pub risk_level: RiskLevel,
    pub dependencies: Vec<String>,
}

/// Complexity level
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Complexity {
    Trivial,
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

/// Risk level
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiskLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    pub analysis_duration_ms: u64,
    pub data_points_analyzed: u64,
    pub algorithms_used: Vec<String>,
    pub confidence_level: f64,
    pub notes: Vec<String>,
}

/// Trend analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub rate_of_change: f64,
    pub prediction_window: u64,
    pub confidence: f64,
    pub data_points: Vec<(u64, f64)>,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub anomaly_id: Uuid,
    pub anomaly_type: AnomalyType,
    pub timestamp: u64,
    pub metric_name: String,
    pub expected_value: f64,
    pub actual_value: f64,
    pub deviation_score: f64,
    pub confidence: f64,
    pub context: HashMap<String, serde_json::Value>,
}

/// Type of anomaly detected
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalyType {
    Spike,
    Drop,
    GradualDrift,
    PatternChange,
    MissingData,
    UnexpectedValue,
}

/// Analysis engine trait
#[async_trait::async_trait]
pub trait AnalysisEngine: Send + Sync {
    /// Analyze performance metrics
    async fn analyze_performance(
        &self,
        target: AnalysisTarget,
    ) -> Result<AnalysisReport, crate::DebugError>;

    /// Detect anomalies in data
    async fn detect_anomalies(
        &self,
        target: AnalysisTarget,
        baseline: Option<BaselineData>,
    ) -> Result<Vec<AnomalyResult>, crate::DebugError>;

    /// Compare two datasets
    async fn compare_analysis(
        &self,
        baseline: AnalysisTarget,
        comparison: AnalysisTarget,
    ) -> Result<AnalysisReport, crate::DebugError>;

    /// Analyze trends over time
    async fn analyze_trends(
        &self,
        metrics: Vec<String>,
        time_range: (u64, u64),
    ) -> Result<Vec<TrendAnalysis>, crate::DebugError>;

    /// Generate optimization recommendations
    async fn generate_recommendations(
        &self,
        findings: Vec<Finding>,
    ) -> Result<Vec<Recommendation>, crate::DebugError>;
}

/// Baseline data for comparison analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineData {
    pub baseline_id: Uuid,
    pub name: String,
    pub collected_at: u64,
    pub metrics: HashMap<String, BaselineMetric>,
    pub metadata: HashMap<String, String>,
}

/// Individual baseline metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetric {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: HashMap<u8, f64>,
    pub sample_count: u64,
}
