//! Workload Orchestration Benchmarks Module
//!
//! Comprehensive benchmarking for swarmlet workload orchestration including
//! container startup time, concurrent workload handling, and resource utilization
//! efficiency. Implements TDD methodology for systematic optimization.

pub mod benchmarks;
pub mod constraints;
pub mod metrics;
pub mod monitor;
pub mod tests;
pub mod types;
pub mod workloads;

pub use types::{
    BottleneckComponent, BottleneckReport, BottleneckSeverity, ComplexityLevel, ImpactLevel,
    OptimizationOpportunity, OptimizationType, OrchestrationActuals, OrchestrationBenchmarkResult,
    OrchestrationTargets, TddPhase,
};

pub use metrics::{
    ConcurrentExecutionMetrics, ContainerLifecycleMetrics, PerformanceExpectations,
    ResourceUtilizationMetrics,
};

pub use workloads::{
    BenchmarkWorkload, ContainerSpec, GpuRequirements, ImagePullPolicy, ResourceRequirements,
    SecurityContext, WorkloadType,
};

pub use constraints::{
    AffinityRule, AffinityType, DeadlineConstraints, DeadlineFailureAction, Hook, LifecycleHooks,
    PriorityClass, Probe, ProbeType, SchedulingConstraints, TaintEffect, Toleration,
    TolerationOperator,
};

pub use monitor::{NodeResourceState, ResourceMonitor, ResourceSnapshot, WorkloadResourceState};

pub use benchmarks::WorkloadOrchestrationBenchmarks;
