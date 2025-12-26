//! Configuration types for causal inference

use chrono::Duration as ChronoDuration;
use serde::{Deserialize, Serialize};

/// Configuration for causal inference engine
#[derive(Debug, Clone)]
pub struct CausalInferenceConfig {
    /// Maximum causal chain length to explore
    pub max_chain_length: usize,
    /// Confidence threshold for causal relationships
    pub confidence_threshold: f64,
    /// Time window for temporal causal analysis
    pub temporal_window: ChronoDuration,
    /// GPU acceleration settings
    pub gpu_config: CausalGpuConfig,
    /// Uncertainty quantification method
    pub uncertainty_method: UncertaintyMethod,
    /// Enable real-time pattern detection
    pub enable_real_time_detection: bool,
}

/// GPU configuration for causal inference
#[derive(Debug, Clone)]
pub struct CausalGpuConfig {
    /// Enable GPU acceleration
    pub enabled: bool,
    /// Number of parallel causal inference streams
    pub parallel_streams: usize,
    /// GPU memory allocation for causal computations in GB
    pub memory_gb: usize,
    /// Batch size for causal neural network inference
    pub batch_size: usize,
    /// Device ID to use
    pub device_id: Option<i32>,
}

/// Uncertainty quantification methods
#[derive(Debug, Clone)]
pub enum UncertaintyMethod {
    Bayesian,
    Frequentist,
    EvidentialDeepLearning,
    ConformalPrediction,
}

/// Configuration for real-time detection
#[derive(Debug, Clone)]
pub struct RealTimeDetectionConfig {
    pub buffer_size: usize,
    pub pattern_window: ChronoDuration,
    pub confidence_threshold: f64,
    pub min_pattern_frequency: usize,
}

impl Default for CausalInferenceConfig {
    fn default() -> Self {
        Self {
            max_chain_length: 5,
            confidence_threshold: 0.7,
            temporal_window: ChronoDuration::hours(1),
            gpu_config: CausalGpuConfig::default(),
            uncertainty_method: UncertaintyMethod::Bayesian,
            enable_real_time_detection: false,
        }
    }
}

impl Default for CausalGpuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            parallel_streams: 4,
            memory_gb: 4,
            batch_size: 64,
            device_id: None,
        }
    }
}
