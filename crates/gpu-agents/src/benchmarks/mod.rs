//! Benchmarking infrastructure for GPU agents
//!
//! This module provides comprehensive benchmarking capabilities for GPU agents,
//! including scalability tests, LLM integration, knowledge graphs, and evolution strategies.

// Infrastructure modules
pub mod progress_monitor;
pub mod progress_writer;
pub mod scenario_runner;

// Real benchmark implementations
pub mod evolution_benchmark;
pub mod gpu_evolution_benchmark;
pub mod gpu_integration_benchmark;
pub mod gpu_knowledge_graph_benchmark;
pub mod gpu_streaming_benchmark;
pub mod knowledge_graph_benchmark;
pub mod llm_benchmark;
pub mod scalability_benchmark;
pub mod simplified_evolution_benchmark;
pub mod storage_benchmark;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod real_gpu_test;

#[cfg(test)]
mod knowledge_graph_benchmark_tests;

#[cfg(test)]
mod knowledge_graph_tdd_tests;

#[cfg(test)]
mod arc_type_debug_tests;

// Re-export core types
pub use progress_monitor::{BenchmarkPhase, ProgressMonitor, ProgressState, ResourceSnapshot};
pub use progress_writer::ProgressWriter;
pub use scenario_runner::{
    scenario_to_benchmark_config, ComparisonMetric, ReportFormat, ScenarioBenchmarkMetrics,
    ScenarioBenchmarkReport, ScenarioBenchmarkResult, ScenarioBenchmarkRunner, ScenarioComparison,
    ScenarioSuite, ScenarioSuiteResults,
};

// Re-export benchmark results
pub use evolution_benchmark::{run_evolution_benchmark, EvolutionBenchmarkResults};
pub use gpu_evolution_benchmark::{
    run_evolution_benchmarks, EvolutionBenchmarkConfig as GpuEvolutionBenchmarkConfig,
    EvolutionBenchmarkResults as GpuEvolutionBenchmarkResults, GpuEvolutionBenchmark,
};
pub use gpu_integration_benchmark::{
    generate_performance_report, run_gpu_integration_benchmarks, GpuIntegrationBenchmarkConfig,
};
pub use gpu_knowledge_graph_benchmark::{
    run_knowledge_graph_benchmarks as run_gpu_knowledge_graph_benchmarks,
    GpuKnowledgeGraphBenchmark, KnowledgeGraphBenchmarkConfig as GpuKnowledgeGraphBenchmarkConfig,
    KnowledgeGraphBenchmarkResults as GpuKnowledgeGraphBenchmarkResults,
};
pub use gpu_streaming_benchmark::{
    run_streaming_benchmarks, GpuStreamingBenchmark, StreamingBenchmarkConfig,
    StreamingBenchmarkResults,
};
pub use knowledge_graph_benchmark::{
    run_knowledge_graph_benchmark, KnowledgeGraphBenchmarkResults,
};
pub use llm_benchmark::{run_llm_benchmark, LlmBenchmarkResults};
pub use scalability_benchmark::{run_scalability_benchmark, ScalabilityBenchmarkResults};
pub use simplified_evolution_benchmark::{
    run_simplified_evolution_benchmark, SimplifiedEvolutionResults,
};
pub use storage_benchmark::{
    StorageBenchmark, StorageBenchmarkConfig, StorageBenchmarkMetrics, StorageBenchmarkResults,
    StorageBenchmarkScenario,
};
