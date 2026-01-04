//! Adaptive consensus module with GPU acceleration and dynamic algorithm selection

pub mod algorithms;
pub mod config;
pub mod controller;
pub mod coordinator;
pub mod engine;
pub mod gpu;
pub mod monitor;
pub mod optimizer;
pub mod selector;

pub use algorithms::{AlgorithmConfig, AlgorithmRequirements, ConsensusAlgorithm};
pub use config::{
    AdaptiveConsensusConfig, AdaptiveGpuConfig, ConsensusAlgorithmType, NetworkMonitoringConfig,
    OptimizationConfig,
};
pub use controller::AdaptationController;
pub use coordinator::{ConsensusCoordinator, RoundResult};
pub use engine::AdaptiveConsensusEngine;
pub use gpu::GpuConsensusAccelerator;
pub use monitor::{NetworkConditions, NetworkMonitor};
pub use optimizer::OptimizationEngine;
pub use selector::{AlgorithmSelector, SelectionStrategy};

use crate::ConsensusError;

/// Outcome of a consensus operation
#[derive(Debug, Clone)]
pub struct ConsensusOutcome {
    /// Algorithm used
    pub algorithm: ConsensusAlgorithmType,
    /// Time taken in microseconds
    pub time_taken: u64,
    /// Whether consensus was reached
    pub consensus_reached: bool,
    /// Number of nodes that participated
    pub nodes_participated: usize,
    /// GPU utilization percentage
    pub gpu_utilization: f32,
}

/// Performance metrics for an algorithm
#[derive(Debug, Clone)]
pub struct AlgorithmPerformance {
    /// Algorithm type
    pub algorithm: ConsensusAlgorithmType,
    /// Average latency in microseconds
    pub avg_latency: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Throughput in operations per second
    pub throughput: f64,
    /// GPU utilization percentage
    pub gpu_utilization: f32,
    /// Number of samples
    pub sample_count: usize,
    /// Last update timestamp
    pub last_updated: std::time::Instant,
    /// Historical performance data
    pub history: Vec<HistoricalDataPoint>,
}

/// Historical performance data point
#[derive(Debug, Clone)]
pub struct HistoricalDataPoint {
    /// Timestamp
    pub timestamp: std::time::Instant,
    /// Latency at this point
    pub latency: f64,
    /// Success rate at this point
    pub success_rate: f64,
}

impl Default for AlgorithmPerformance {
    fn default() -> Self {
        Self {
            algorithm: ConsensusAlgorithmType::PBFT,
            avg_latency: 0.0,
            success_rate: 1.0,
            throughput: 0.0,
            gpu_utilization: 0.0,
            sample_count: 0,
            last_updated: std::time::Instant::now(),
            history: Vec::new(),
        }
    }
}
