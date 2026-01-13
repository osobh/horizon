//! Main compression engine implementation

use super::{
    CompressedKnowledgeGraph, CompressionAlgorithm, CompressionConfig, CompressionMetadata,
    CompressionPerformanceMetrics, CompressionQualityMetrics,
};
use crate::{Edge, KnowledgeGraph, KnowledgeGraphError, KnowledgeGraphResult, Node};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

/// Main knowledge compression engine
///
/// Uses cudarc 0.18 CudaContext API for GPU operations.
pub struct KnowledgeCompressionEngine {
    config: CompressionConfig,
    /// CUDA context for GPU operations (cudarc 0.18+ API)
    gpu_ctx: Option<Arc<cudarc::driver::CudaContext>>,
    metrics: Arc<RwLock<CompressionPerformanceMetrics>>,
    cache: Arc<Mutex<CompressionCache>>,
}

impl KnowledgeCompressionEngine {
    /// Create a new compression engine
    ///
    /// Uses cudarc 0.18 CudaContext::new() for GPU initialization.
    pub fn new(config: CompressionConfig) -> KnowledgeGraphResult<Self> {
        let gpu_ctx = if config.gpu_config.enabled {
            // CudaContext::new returns Arc<CudaContext>
            Some(cudarc::driver::CudaContext::new(0)?)
        } else {
            None
        };

        Ok(Self {
            config,
            gpu_ctx,
            metrics: Arc::new(RwLock::new(CompressionPerformanceMetrics::default())),
            cache: Arc::new(Mutex::new(CompressionCache::new(100))),
        })
    }

    /// Get the CUDA context if GPU is enabled
    pub fn gpu_context(&self) -> Option<&Arc<cudarc::driver::CudaContext>> {
        self.gpu_ctx.as_ref()
    }

    /// Compress a knowledge graph
    pub async fn compress(
        &self,
        graph: &KnowledgeGraph,
    ) -> KnowledgeGraphResult<CompressedKnowledgeGraph> {
        let start = Instant::now();

        // Select compression algorithm
        let compressed_data = match self.config.compression_algorithm {
            CompressionAlgorithm::Lossless => self.compress_lossless(graph).await?,
            CompressionAlgorithm::AdaptiveLossy => self.compress_adaptive(graph).await?,
            CompressionAlgorithm::Neural => self.compress_neural(graph).await?,
            CompressionAlgorithm::Hierarchical => self.compress_hierarchical(graph).await?,
            CompressionAlgorithm::Spectral => self.compress_spectral(graph).await?,
            CompressionAlgorithm::Custom => {
                return Err(KnowledgeGraphError::Compression {
                    message: "Custom compression not implemented".to_string(),
                });
            }
        };

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.compression_time = start.elapsed();

        Ok(compressed_data)
    }

    /// Decompress a compressed graph
    pub async fn decompress(
        &self,
        compressed: &CompressedKnowledgeGraph,
    ) -> KnowledgeGraphResult<KnowledgeGraph> {
        let start = Instant::now();

        // Decompress based on algorithm
        let graph = match compressed.metadata.algorithm {
            CompressionAlgorithm::Lossless => self.decompress_lossless(compressed).await?,
            CompressionAlgorithm::AdaptiveLossy => self.decompress_adaptive(compressed).await?,
            CompressionAlgorithm::Neural => self.decompress_neural(compressed).await?,
            CompressionAlgorithm::Hierarchical => self.decompress_hierarchical(compressed).await?,
            CompressionAlgorithm::Spectral => self.decompress_spectral(compressed).await?,
            CompressionAlgorithm::Custom => {
                return Err(KnowledgeGraphError::Compression {
                    message: "Custom decompression not implemented".to_string(),
                });
            }
        };

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.decompression_time = start.elapsed();

        Ok(graph)
    }

    /// Get current performance metrics
    pub async fn get_metrics(&self) -> CompressionPerformanceMetrics {
        self.metrics.read().await.clone()
    }

    // Private compression methods
    async fn compress_lossless(
        &self,
        graph: &KnowledgeGraph,
    ) -> KnowledgeGraphResult<CompressedKnowledgeGraph> {
        // Get nodes and edges, cloning for serialization
        let nodes: Vec<Node> = graph.get_all_nodes()?.into_iter().cloned().collect();
        let edges: Vec<Edge> = graph.get_all_edges()?.into_iter().cloned().collect();

        // Serialize the data
        let nodes_data = bincode::serialize(&nodes)?;
        let edges_data = bincode::serialize(&edges)?;

        let original_size = nodes_data.len() + edges_data.len();

        Ok(CompressedKnowledgeGraph {
            compressed_nodes: nodes_data,
            compressed_edges: edges_data,
            metadata: CompressionMetadata {
                original_size,
                compressed_size: original_size,
                compression_ratio: 1.0,
                algorithm: CompressionAlgorithm::Lossless,
                timestamp: chrono::Utc::now(),
                gpu_devices: vec![],
            },
            quality_metrics: CompressionQualityMetrics::default(),
        })
    }

    async fn compress_adaptive(
        &self,
        _graph: &KnowledgeGraph,
    ) -> KnowledgeGraphResult<CompressedKnowledgeGraph> {
        // Placeholder implementation
        Err(KnowledgeGraphError::Compression {
            message: "Adaptive compression not yet implemented".to_string(),
        })
    }

    async fn compress_neural(
        &self,
        _graph: &KnowledgeGraph,
    ) -> KnowledgeGraphResult<CompressedKnowledgeGraph> {
        // Placeholder implementation
        Err(KnowledgeGraphError::Compression {
            message: "Neural compression not yet implemented".to_string(),
        })
    }

    async fn compress_hierarchical(
        &self,
        _graph: &KnowledgeGraph,
    ) -> KnowledgeGraphResult<CompressedKnowledgeGraph> {
        // Placeholder implementation
        Err(KnowledgeGraphError::Compression {
            message: "Hierarchical compression not yet implemented".to_string(),
        })
    }

    async fn compress_spectral(
        &self,
        _graph: &KnowledgeGraph,
    ) -> KnowledgeGraphResult<CompressedKnowledgeGraph> {
        // Placeholder implementation
        Err(KnowledgeGraphError::Compression {
            message: "Spectral compression not yet implemented".to_string(),
        })
    }

    async fn decompress_lossless(
        &self,
        compressed: &CompressedKnowledgeGraph,
    ) -> KnowledgeGraphResult<KnowledgeGraph> {
        use crate::graph::KnowledgeGraphConfig;

        // Simplified decompression
        let nodes: Vec<Node> = bincode::deserialize(&compressed.compressed_nodes)?;
        let edges: Vec<Edge> = bincode::deserialize(&compressed.compressed_edges)?;

        // Create graph with default config but GPU disabled for decompression
        let config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(config).await?;

        for node in nodes {
            graph.add_node(node)?;
        }
        for edge in edges {
            graph.add_edge(edge)?;
        }

        Ok(graph)
    }

    async fn decompress_adaptive(
        &self,
        _compressed: &CompressedKnowledgeGraph,
    ) -> KnowledgeGraphResult<KnowledgeGraph> {
        Err(KnowledgeGraphError::Compression {
            message: "Adaptive decompression not yet implemented".to_string(),
        })
    }

    async fn decompress_neural(
        &self,
        _compressed: &CompressedKnowledgeGraph,
    ) -> KnowledgeGraphResult<KnowledgeGraph> {
        Err(KnowledgeGraphError::Compression {
            message: "Neural decompression not yet implemented".to_string(),
        })
    }

    async fn decompress_hierarchical(
        &self,
        _compressed: &CompressedKnowledgeGraph,
    ) -> KnowledgeGraphResult<KnowledgeGraph> {
        Err(KnowledgeGraphError::Compression {
            message: "Hierarchical decompression not yet implemented".to_string(),
        })
    }

    async fn decompress_spectral(
        &self,
        _compressed: &CompressedKnowledgeGraph,
    ) -> KnowledgeGraphResult<KnowledgeGraph> {
        Err(KnowledgeGraphError::Compression {
            message: "Spectral decompression not yet implemented".to_string(),
        })
    }
}

/// Simple compression cache
struct CompressionCache {
    capacity: usize,
    entries: std::collections::HashMap<Uuid, CompressedKnowledgeGraph>,
}

impl CompressionCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: std::collections::HashMap::new(),
        }
    }

    fn get(&self, id: &Uuid) -> Option<&CompressedKnowledgeGraph> {
        self.entries.get(id)
    }

    fn insert(&mut self, id: Uuid, compressed: CompressedKnowledgeGraph) {
        if self.entries.len() >= self.capacity {
            // Simple eviction - remove first entry
            if let Some(first_key) = self.entries.keys().next().cloned() {
                self.entries.remove(&first_key);
            }
        }
        self.entries.insert(id, compressed);
    }
}
