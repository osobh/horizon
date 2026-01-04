//! Configuration types for adaptive consensus

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Main configuration for adaptive consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConsensusConfig {
    /// Enable adaptive algorithm selection
    pub enable_adaptation: bool,
    /// Initial consensus algorithm
    pub initial_algorithm: ConsensusAlgorithmType,
    /// GPU acceleration configuration
    pub gpu_config: AdaptiveGpuConfig,
    /// Network monitoring configuration
    pub network_monitoring: NetworkMonitoringConfig,
    /// Optimization configuration
    pub optimization: OptimizationConfig,
    /// Maximum nodes to support
    pub max_nodes: usize,
    /// Target latency in microseconds
    pub target_latency_us: u64,
    /// Minimum success rate threshold
    pub min_success_rate: f64,
}

/// GPU acceleration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveGpuConfig {
    /// Enable GPU acceleration
    pub enabled: bool,
    /// GPU device ID to use
    pub device_id: usize,
    /// Maximum GPU memory to use (in MB)
    pub max_memory_mb: usize,
    /// Enable multi-GPU support
    pub multi_gpu: bool,
    /// GPU kernel optimization level
    pub optimization_level: u32,
}

/// Network monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMonitoringConfig {
    /// Enable network monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Latency measurement samples
    pub latency_samples: usize,
    /// Bandwidth measurement interval
    pub bandwidth_interval: Duration,
    /// Packet loss threshold for adaptation
    pub packet_loss_threshold: f64,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable continuous optimization
    pub enabled: bool,
    /// Optimization interval
    pub optimization_interval: Duration,
    /// Performance history window
    pub history_window: Duration,
    /// Minimum samples for optimization
    pub min_samples: usize,
    /// Enable machine learning optimization
    pub ml_optimization: bool,
}

/// Types of consensus algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsensusAlgorithmType {
    /// Practical Byzantine Fault Tolerance
    PBFT,
    /// GPU-accelerated PBFT
    GpuPBFT,
    /// Fast consensus for small networks
    Fast,
    /// Streaming consensus for continuous data
    Streaming,
    /// Hybrid consensus combining multiple algorithms
    Hybrid,
    /// GPU-native consensus algorithm
    GpuNative,
    /// Machine learning optimized consensus
    MLOptimized,
    /// Raft consensus
    Raft,
    /// GPU-accelerated Raft
    GpuRaft,
    /// Custom algorithm
    Custom,
}

impl Default for AdaptiveConsensusConfig {
    fn default() -> Self {
        Self {
            enable_adaptation: true,
            initial_algorithm: ConsensusAlgorithmType::PBFT,
            gpu_config: AdaptiveGpuConfig::default(),
            network_monitoring: NetworkMonitoringConfig::default(),
            optimization: OptimizationConfig::default(),
            max_nodes: 1000,
            target_latency_us: 1000,
            min_success_rate: 0.95,
        }
    }
}

impl Default for AdaptiveGpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device_id: 0,
            max_memory_mb: 4096,
            multi_gpu: false,
            optimization_level: 2,
        }
    }
}

impl Default for NetworkMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_secs(1),
            latency_samples: 100,
            bandwidth_interval: Duration::from_secs(10),
            packet_loss_threshold: 0.05,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_interval: Duration::from_secs(30),
            history_window: Duration::from_secs(300),
            min_samples: 10,
            ml_optimization: false,
        }
    }
}
