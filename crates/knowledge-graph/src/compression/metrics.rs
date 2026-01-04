//! Compression metrics and performance monitoring

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Quality metrics for compressed knowledge graphs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionQualityMetrics {
    /// Structural similarity score (0.0 to 1.0)
    pub structural_similarity: f32,
    /// Semantic similarity score (0.0 to 1.0)
    pub semantic_similarity: f32,
    /// Information retention percentage
    pub information_retention: f32,
    /// Number of preserved relationships
    pub preserved_relationships: usize,
    /// Number of preserved attributes
    pub preserved_attributes: usize,
    /// Reconstruction error rate
    pub reconstruction_error: f32,
    /// Query accuracy on compressed graph
    pub query_accuracy: f32,
}

/// Performance metrics for compression operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionPerformanceMetrics {
    /// Total compression time
    pub compression_time: Duration,
    /// Decompression time
    pub decompression_time: Duration,
    /// Compression throughput (MB/s)
    pub compression_throughput: f64,
    /// GPU utilization percentage
    pub gpu_utilization: f32,
    /// Memory usage in bytes
    pub memory_usage: usize,
}

/// Statistics about the graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// Average node degree
    pub avg_degree: f32,
    /// Graph density
    pub density: f32,
    /// Number of connected components
    pub component_count: usize,
    /// Diameter of the graph
    pub diameter: usize,
}

/// GPU utilization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuUtilizationStats {
    /// GPU compute utilization percentage
    pub compute_utilization: f32,
    /// GPU memory utilization percentage
    pub memory_utilization: f32,
    /// GPU temperature in Celsius
    pub temperature: f32,
    /// Power consumption in watts
    pub power_consumption: f32,
}

/// Multi-GPU performance metrics
#[derive(Debug, Clone)]
pub struct MultiGpuMetrics {
    /// Per-device utilization
    pub device_utilization: Vec<GpuUtilizationStats>,
    /// Load balancing efficiency
    pub load_balance_efficiency: f32,
    /// Inter-GPU communication overhead
    pub communication_overhead: Duration,
}

impl Default for CompressionQualityMetrics {
    fn default() -> Self {
        Self {
            structural_similarity: 1.0,
            semantic_similarity: 1.0,
            information_retention: 1.0,
            preserved_relationships: 0,
            preserved_attributes: 0,
            reconstruction_error: 0.0,
            query_accuracy: 1.0,
        }
    }
}

impl Default for CompressionPerformanceMetrics {
    fn default() -> Self {
        Self {
            compression_time: Duration::ZERO,
            decompression_time: Duration::ZERO,
            compression_throughput: 0.0,
            gpu_utilization: 0.0,
            memory_usage: 0,
        }
    }
}
