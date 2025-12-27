//! Global knowledge graph with cross-region replication and compliance-aware data handling
//!
//! This crate provides comprehensive global knowledge graph capabilities for:
//! - Cross-region data replication with <100ms global queries
//! - Compliance-aware data handling respecting regional regulations
//! - Eventual consistency with conflict resolution
//! - High-performance distributed graph operations
//! - Integration with compliance frameworks for data sovereignty

#![warn(missing_docs)]

pub mod cache_actor;
pub mod cache_layer;
pub mod compliance_handler;
pub mod consistency_actor;
pub mod consistency_manager;
pub mod error;
pub mod graph_manager;
pub mod query_engine;
pub mod region_router;
pub mod replication;

// New global synchronization protocol
pub mod sync_protocol;

pub use error::{GlobalKnowledgeGraphError, GlobalKnowledgeGraphResult};
pub use graph_manager::{GraphConfig, GraphManager};

// Re-export actor types for new code
pub use cache_actor::{create_cache_actor, CacheActor, CacheHandle, CacheRequest};
pub use consistency_actor::{
    create_consistency_actor, ConsistencyActor, ConsistencyHandle, ConsistencyRequest,
};

// Export new global synchronization components
pub use sync_protocol::{
    GlobalSyncProtocol, GlobalSyncConfig, KnowledgeOperation, ConsensusResult,
    SyncMetrics, ClusterState, ConsensusMetrics, NetworkConfig, ConsensusConfig
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn test_global_knowledge_graph_creation() {
        let config = GraphConfig::default();
        let manager = GraphManager::new(config);
        assert!(manager.is_ok());
    }
}
