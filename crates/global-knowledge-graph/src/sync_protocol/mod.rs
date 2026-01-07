//! Global Synchronization Protocol
//!
//! Real-time knowledge graph synchronization across GPU clusters with Byzantine fault tolerance,
//! GPU-accelerated consensus mechanisms, and sub-100ms global knowledge propagation.

pub mod config;
pub mod conflict;
pub mod consensus;
pub mod gpu;
pub mod metrics;
pub mod network;
pub mod protocol;
pub mod state;
pub mod types;

// Re-export main types
pub use config::{
    ConsensusAlgorithm, ConsensusConfig, GlobalSyncConfig, GpuClusterSpec, NetworkConfig,
};
pub use conflict::{
    ConflictResolver, ConflictSeverity, ConflictType, ConflictingOperation, Resolution,
    ResolutionStrategy,
};
pub use consensus::{ConsensusEngine, ConsensusProposal, ConsensusResult, ConsensusVote, Vote};
pub use gpu::GpuConsensusMetrics;
pub use metrics::{
    AggregatedMetrics, ClusterMetrics, ConsensusMetrics, MetricsCollector, NetworkMetrics,
    SyncMetrics,
};
pub use network::{MessageQueue, SyncMessage};
pub use protocol::GlobalSyncProtocol;
pub use state::{ClusterState, ClusterSyncState, KnowledgeCluster};
pub use types::{KnowledgeOperation, OperationEvidence, OperationPriority, OperationType, VectorClock};

// Re-export performance types from metrics (used by protocol)
pub use metrics::{AnomalySeverity, AnomalyType, PerformanceAnomaly, PerformanceReport};
