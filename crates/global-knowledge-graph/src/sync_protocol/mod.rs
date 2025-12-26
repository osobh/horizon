//! Global Synchronization Protocol
//!
//! Real-time knowledge graph synchronization across GPU clusters with Byzantine fault tolerance,
//! GPU-accelerated consensus mechanisms, and sub-100ms global knowledge propagation.

pub mod config;
pub mod types;
pub mod metrics;
pub mod consensus;
pub mod conflict;
pub mod network;
pub mod protocol;
pub mod state;
pub mod gpu;

// Re-export main types
pub use config::{GlobalSyncConfig, GpuClusterSpec, NetworkConfig, ConsensusConfig, ConsensusAlgorithm};
pub use types::{KnowledgeOperation, OperationType, OperationPriority, VectorClock};
pub use metrics::{SyncMetrics, ConsensusMetrics, ClusterMetrics, NetworkMetrics, AggregatedMetrics};
pub use consensus::{ConsensusProposal, ConsensusVote, Vote, ConsensusResult, ConsensusEngine};
pub use conflict::{ConflictingOperation, ConflictType, ConflictSeverity, Resolution, ResolutionStrategy, ConflictResolver};
pub use network::{SyncMessage, MessageQueue};
pub use protocol::GlobalSyncProtocol;
pub use state::{KnowledgeCluster, ClusterSyncState, ClusterState};
pub use gpu::{GpuConsensusMetrics};

pub use protocol::{PerformanceAnomaly, AnomalyType, AnomalySeverity, PerformanceReport, OperationEvidence};
pub use metrics::MetricsCollector;