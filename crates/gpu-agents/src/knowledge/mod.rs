//! GPU-accelerated knowledge graph module

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceSlice};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub mod atomic;
pub mod enhanced;
pub mod gnn;
pub mod graph_adapter;
pub mod reasoning;
pub mod temporal;

#[cfg(test)]
mod gnn_tests;
#[cfg(test)]
mod reasoning_tests;
#[cfg(test)]
mod temporal_tests;

// Re-export enhanced types
pub use enhanced::{CsrGraph, CsrGraphPointers, EnhancedGpuKnowledgeGraph, SpatialIndex};

// Re-export atomic types
pub use atomic::{
    AtomicGraphStatistics, AtomicKnowledgeGraph, AtomicUpdate, AtomicUpdateOp, AtomicUpdateQueue,
    ConsistencyLevel,
};

// Re-export temporal types
pub use temporal::{
    AggregationType, CausalityAnalysis, NodeEvolution, TemporalAggregation, TemporalAnomaly,
    TemporalEdge, TemporalKnowledgeGraph, TemporalNode, TemporalPath, TemporalPathQuery,
    TimeWindowQuery,
};

// Re-export reasoning types
pub use reasoning::{
    Contradiction, FactObject, InferenceRule, LogicalFact, QueryType, ReasoningEngine,
    ReasoningQuery, ReasoningResult, RulePattern,
};

// Re-export GNN types
pub use gnn::{ActivationFunction, AggregationFunction, GnnConfig, GraphNeuralNetwork, Subgraph};

// Core knowledge graph types (moved from knowledge.rs)
/// Knowledge node representing a piece of information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    /// Unique identifier
    pub id: u32,
    /// Content of the knowledge
    pub content: String,
    /// Type of knowledge (location, resource, property, etc.)
    pub node_type: String,
    /// Vector embedding of the content
    pub embedding: Vec<f32>,
}

/// Edge connecting two knowledge nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEdge {
    /// Source node ID
    pub source_id: u32,
    /// Target node ID
    pub target_id: u32,
    /// Relationship type
    pub edge_type: String,
    /// Weight/strength of the relationship
    pub weight: f32,
}

/// Query for knowledge graph search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQuery {
    /// Query text
    pub query_text: String,
    /// Query embedding
    pub query_embedding: Vec<f32>,
    /// Maximum number of results
    pub max_results: usize,
    /// Similarity threshold
    pub threshold: f32,
}

impl Default for GraphQuery {
    fn default() -> Self {
        Self {
            query_text: "default query".to_string(),
            query_embedding: vec![0.0; 128],
            max_results: 10,
            threshold: 0.7,
        }
    }
}

/// Query result from knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Matching nodes
    pub nodes: Vec<KnowledgeNode>,
    /// Relevance scores
    pub scores: Vec<f32>,
    /// Query execution time
    pub execution_time_ms: f64,
}

/// CPU-side knowledge graph
#[derive(Debug)]
pub struct KnowledgeGraph {
    pub nodes: HashMap<u32, KnowledgeNode>,
    pub edges: Vec<KnowledgeEdge>,
}

impl KnowledgeGraph {
    pub fn new(_embedding_dim: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: KnowledgeNode) {
        self.nodes.insert(node.id, node);
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: KnowledgeEdge) {
        self.edges.push(edge);
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Upload the graph to GPU
    pub fn upload_to_gpu(
        &self,
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<GpuKnowledgeGraph> {
        let mut gpu_graph = GpuKnowledgeGraph::new(ctx, stream);

        // In a real implementation, we would serialize nodes and edges to GPU memory
        // For now, just track counts
        gpu_graph.node_count = self.nodes.len();
        gpu_graph.edge_count = self.edges.len();

        Ok(gpu_graph)
    }
}

/// GPU-accelerated knowledge graph
#[derive(Debug)]
pub struct GpuKnowledgeGraph {
    _ctx: Arc<cudarc::driver::CudaContext>,
    _stream: Arc<cudarc::driver::CudaStream>,
    nodes: Option<CudaSlice<u8>>,
    edges: Option<CudaSlice<u8>>,
    node_count: usize,
    edge_count: usize,
}

impl GpuKnowledgeGraph {
    pub fn new(ctx: Arc<cudarc::driver::CudaContext>, stream: Arc<cudarc::driver::CudaStream>) -> Self {
        Self {
            _ctx: ctx,
            _stream: stream,
            nodes: None,
            edges: None,
            node_count: 0,
            edge_count: 0,
        }
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        let node_memory = self.nodes.as_ref().map(|n| n.len()).unwrap_or(0);
        let edge_memory = self.edges.as_ref().map(|e| e.len()).unwrap_or(0);
        node_memory + edge_memory
    }

    /// Run similarity search on GPU
    pub fn run_similarity_search(&self, _query: &GraphQuery) -> Result<Vec<GpuQueryResult>> {
        // Placeholder implementation
        // In real implementation, this would run GPU kernels for similarity search
        Ok(vec![GpuQueryResult {
            node_id: 0,
            score: 0.95,
        }])
    }
}

/// GPU query result
pub struct GpuQueryResult {
    pub node_id: u32,
    pub score: f32,
}

/// Knowledge graph metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_degree: f32,
    pub clustering_coefficient: f32,
}

/// Embedding space configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingSpace {
    pub dimensions: usize,
    pub model_name: String,
}
