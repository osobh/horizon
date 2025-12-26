//! Knowledge Graph Compression Module
//!
//! GPU-accelerated knowledge graph compression system using advanced graph algorithms
//! and neural compression techniques.

pub mod config;
pub mod engine;
pub mod streaming;
pub mod metrics;
pub mod gpu;
pub mod cache;

pub use config::{
    CompressionConfig, CompressionGpuConfig, CompressionAlgorithm,
    SemanticPreservationConfig, GpuOptimizationLevel,
};
pub use engine::KnowledgeCompressionEngine;
pub use streaming::{StreamingCompressionEngine, StreamingCompressionConfig};
pub use metrics::{
    CompressionQualityMetrics, CompressionPerformanceMetrics,
    GraphStatistics, GpuUtilizationStats,
};

use crate::KnowledgeGraphResult;

/// Main compression trait that all compression engines implement
pub trait CompressionEngine {
    /// Compress a knowledge graph
    async fn compress(&self, graph: &crate::KnowledgeGraph) -> KnowledgeGraphResult<CompressedKnowledgeGraph>;
    
    /// Decompress a compressed graph
    async fn decompress(&self, compressed: &CompressedKnowledgeGraph) -> KnowledgeGraphResult<crate::KnowledgeGraph>;
    
    /// Get compression metrics
    fn get_metrics(&self) -> CompressionPerformanceMetrics;
}

/// Represents a compressed knowledge graph
#[derive(Debug, Clone)]
pub struct CompressedKnowledgeGraph {
    /// Compressed node data
    pub compressed_nodes: Vec<u8>,
    /// Compressed edge data  
    pub compressed_edges: Vec<u8>,
    /// Metadata about the compression
    pub metadata: CompressionMetadata,
    /// Quality metrics
    pub quality_metrics: CompressionQualityMetrics,
}

/// Metadata about the compression process
#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    /// Original graph size
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Algorithm used
    pub algorithm: CompressionAlgorithm,
    /// Compression timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// GPU devices used
    pub gpu_devices: Vec<usize>,
}