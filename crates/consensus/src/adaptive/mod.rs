//! Adaptive consensus module with GPU acceleration and dynamic algorithm selection

pub mod config;
pub mod engine;
pub mod algorithms;
pub mod monitor;
pub mod selector;
pub mod optimizer;
pub mod coordinator;
pub mod controller;
pub mod gpu;

pub use config::{
    AdaptiveConsensusConfig, AdaptiveGpuConfig, NetworkMonitoringConfig,
    OptimizationConfig, ConsensusAlgorithmType,
};
pub use engine::AdaptiveConsensusEngine;
pub use algorithms::{ConsensusAlgorithm, AlgorithmRequirements, AlgorithmConfig};
pub use monitor::{NetworkMonitor, NetworkConditions};
pub use selector::{AlgorithmSelector, SelectionStrategy};
pub use optimizer::OptimizationEngine;
pub use coordinator::ConsensusCoordinator;
pub use controller::AdaptationController;
pub use gpu::GpuConsensusAccelerator;

use crate::ConsensusError;

/// Result of a consensus operation
#[derive(Debug, Clone)]
pub struct ConsensusResult {
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