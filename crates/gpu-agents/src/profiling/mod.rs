//! Performance profiling modules

pub mod synthesis_profiler;

pub use synthesis_profiler::{
    Bottleneck, BottleneckAnalysis, PerformanceMetrics, SynthesisPipelineProfiler,
};
