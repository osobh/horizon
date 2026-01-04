//! Compression configuration types and defaults

use serde::{Deserialize, Serialize};

/// Configuration for knowledge compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Target compression ratio (0.0 to 1.0)
    pub target_compression_ratio: f32,
    /// Quality threshold for compressed graph
    pub quality_threshold: f32,
    /// Maximum acceptable information loss
    pub max_information_loss: f32,
    /// GPU acceleration settings
    pub gpu_config: CompressionGpuConfig,
    /// Compression algorithm selection
    pub compression_algorithm: CompressionAlgorithm,
    /// Semantic preservation requirements
    pub semantic_preservation: SemanticPreservationConfig,
}

/// GPU configuration for compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionGpuConfig {
    /// Enable GPU acceleration
    pub enabled: bool,
    /// Number of GPU devices to use
    pub device_count: usize,
    /// Memory per device in GB
    pub memory_per_device: f32,
    /// GPU optimization level
    pub optimization_level: GpuOptimizationLevel,
    /// Enable multi-GPU parallel compression
    pub enable_multi_gpu: bool,
}

/// GPU optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuOptimizationLevel {
    /// No GPU optimization
    None,
    /// Basic GPU acceleration
    Basic,
    /// Advanced GPU optimization with kernel fusion
    Advanced,
    /// Maximum GPU optimization with custom kernels
    Maximum,
}

/// Compression algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// Lossless compression using graph isomorphism
    Lossless,
    /// Adaptive lossy compression with quality control
    AdaptiveLossy,
    /// Neural network-based compression
    Neural,
    /// Hierarchical graph decomposition
    Hierarchical,
    /// Spectral graph compression
    Spectral,
    /// Custom algorithm with user-defined parameters
    Custom,
}

/// Semantic preservation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPreservationConfig {
    /// Preserve entity relationships
    pub preserve_relationships: bool,
    /// Preserve attribute values
    pub preserve_attributes: bool,
    /// Preserve temporal information
    pub preserve_temporal: bool,
    /// Minimum semantic similarity threshold
    pub min_semantic_similarity: f32,
    /// Enable semantic validation
    pub enable_validation: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            target_compression_ratio: 0.5,
            quality_threshold: 0.95,
            max_information_loss: 0.05,
            gpu_config: CompressionGpuConfig::default(),
            compression_algorithm: CompressionAlgorithm::AdaptiveLossy,
            semantic_preservation: SemanticPreservationConfig::default(),
        }
    }
}

impl Default for CompressionGpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device_count: 1,
            memory_per_device: 8.0,
            optimization_level: GpuOptimizationLevel::Advanced,
            enable_multi_gpu: false,
        }
    }
}

impl Default for SemanticPreservationConfig {
    fn default() -> Self {
        Self {
            preserve_relationships: true,
            preserve_attributes: true,
            preserve_temporal: true,
            min_semantic_similarity: 0.9,
            enable_validation: true,
        }
    }
}
