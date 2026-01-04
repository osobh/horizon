//! GPU-Native Knowledge Graph System
//!
//! High-performance knowledge graph system optimized for GPU computation
//! with semantic search, pattern discovery, and evolution tracking.

#![warn(missing_docs)]

pub mod error;
pub mod evolution_tracker;
pub mod graph;
pub mod memory_integration;
pub mod patterns;
pub mod pruning;
pub mod query;
pub mod scaling;
pub mod semantic;

// Modular causal knowledge inference system
pub mod causal_knowledge;

// Compression module with proper structure
pub mod compression;

#[cfg(test)]
mod edge_case_tests;

#[cfg(test)]
mod comprehensive_tests;

#[cfg(test)]
mod integration_tests;

pub use error::{KnowledgeGraphError, KnowledgeGraphResult};
pub use graph::{Edge, EdgeType, KnowledgeGraph, Node, NodeType};
pub use query::{Query, QueryEngine, QueryResult};
pub use scaling::{DistributedGraphStore, ScaledKnowledgeGraph, ScalingConfig};
pub use semantic::{EmbeddingVector, SemanticQuery, SemanticSearchEngine};

// Export new distributed knowledge graph components
pub use causal_knowledge::{
    CausalChain, CausalInferenceConfig, CausalKnowledgeEngine, CausalRelationship, CausalType,
    CounterfactualAnalysis, TemporalEvent,
};
pub use compression::{
    CompressedKnowledgeGraph, CompressionConfig, CompressionQualityMetrics,
    KnowledgeCompressionEngine, StreamingCompressionConfig, StreamingCompressionEngine,
};

/// Initialize the knowledge graph subsystem
pub async fn init() -> KnowledgeGraphResult<()> {
    tracing::info!("Initializing GPU-Native Knowledge Graph subsystem");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_knowledge_graph_init() {
        assert!(init().await.is_ok());
    }

    #[test]
    fn test_reexports() {
        // Test that key types are accessible through lib
        let _node = Node::new(NodeType::Agent, std::collections::HashMap::new());
        let _edge = Edge::new("a".to_string(), "b".to_string(), EdgeType::Has, 1.0);

        // Test error types
        let _error: KnowledgeGraphError = KnowledgeGraphError::Other("test".to_string());
        let _result: KnowledgeGraphResult<()> = Ok(());
    }
}
