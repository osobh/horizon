//! Workload Orchestration Benchmarks Module
//!
//! Comprehensive benchmarking for swarmlet workload orchestration including
//! container startup time, concurrent workload handling, and resource utilization
//! efficiency. Implements TDD methodology for systematic optimization.

pub mod types;
pub mod metrics;
pub mod workloads;
pub mod constraints;
pub mod monitor;
pub mod benchmarks;
pub mod tests;

pub use types::{
    TddPhase, OrchestrationBenchmarkResult, OrchestrationTargets,
    OrchestrationActuals, BottleneckReport, BottleneckComponent,
    BottleneckSeverity, OptimizationOpportunity, OptimizationType,
    ComplexityLevel, ImpactLevel,
};

pub use metrics::{
    ContainerLifecycleMetrics, ConcurrentExecutionMetrics,
    ResourceUtilizationMetrics, PerformanceExpectations,
};

pub use workloads::{
    BenchmarkWorkload, WorkloadType, ContainerSpec,
    ImagePullPolicy, SecurityContext, ResourceRequirements,
    GpuRequirements,
};

pub use constraints::{
    SchedulingConstraints, AffinityRule, AffinityType,
    Toleration, TolerationOperator, TaintEffect, PriorityClass,
    DeadlineConstraints, DeadlineFailureAction, LifecycleHooks,
    Hook, Probe, ProbeType,
};

pub use monitor::{
    ResourceMonitor, ResourceSnapshot, NodeResourceState,
    WorkloadResourceState,
};

pub use benchmarks::WorkloadOrchestrationBenchmarks;