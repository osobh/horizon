//! Core types for orchestration benchmarks

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// TDD phase for orchestration benchmark development
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TddPhase {
    Red,      // Benchmark fails (expected behavior)
    Green,    // Minimal implementation meets targets
    Refactor, // Optimize for production performance
}

/// Orchestration benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationBenchmarkResult {
    pub benchmark_id: Uuid,
    pub benchmark_name: String,
    pub phase: TddPhase,
    pub workload_count: u32,
    pub concurrent_level: u32,
    pub target_metrics: OrchestrationTargets,
    pub actual_metrics: OrchestrationActuals,
    pub success: bool,
    pub efficiency_score: f64,
    pub bottleneck_analysis: Vec<BottleneckReport>,
    pub timestamp: DateTime<Utc>,
}

/// Target orchestration metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationTargets {
    pub max_startup_time_ms: u64,
    pub min_concurrent_capacity: u32,
    pub max_resource_overhead_percent: f32,
    pub min_throughput_workloads_per_sec: f64,
    pub max_scheduling_latency_ms: u64,
    pub min_resource_utilization_percent: f32,
}

/// Actual orchestration performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationActuals {
    pub avg_startup_time_ms: u64,
    pub peak_concurrent_workloads: u32,
    pub resource_overhead_percent: f32,
    pub achieved_throughput: f64,
    pub avg_scheduling_latency_ms: u64,
    pub resource_utilization_percent: f32,
    pub failed_workloads: u32,
    pub total_workloads: u32,
}

/// Optimization opportunity identified during benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub optimization_type: OptimizationType,
    pub expected_improvement_percent: f32,
    pub complexity: ComplexityLevel,
    pub impact: ImpactLevel,
}

/// Type of optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    ContainerCaching,
    ResourcePooling,
    SchedulingAlgorithm,
    NetworkOptimization,
    StorageAcceleration,
    GpuSharing,
    MemoryCompression,
    ProcessReuse,
    BatchScheduling,
    PredictiveScaling,
}

/// Complexity level of optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Trivial,
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

/// Impact level of optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Minimal,
    Low,
    Medium,
    High,
    Critical,
}

/// Bottleneck report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckReport {
    pub component: BottleneckComponent,
    pub severity: BottleneckSeverity,
    pub impact_percent: f32,
    pub description: String,
    pub suggested_optimizations: Vec<OptimizationOpportunity>,
}

/// Component causing bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckComponent {
    CpuScheduler,
    MemoryAllocator,
    DiskIo,
    NetworkIo,
    GpuScheduler,
    ContainerRuntime,
    ImagePull,
    VolumeMount,
    SecurityValidation,
    ResourceQuota,
}

/// Severity of bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Minor,
    Moderate,
    Major,
    Critical,
}

impl Default for OrchestrationTargets {
    fn default() -> Self {
        Self {
            max_startup_time_ms: 500,
            min_concurrent_capacity: 100,
            max_resource_overhead_percent: 10.0,
            min_throughput_workloads_per_sec: 50.0,
            max_scheduling_latency_ms: 10,
            min_resource_utilization_percent: 80.0,
        }
    }
}
