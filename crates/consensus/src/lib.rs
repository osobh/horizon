//! Distributed GPU Consensus Protocol
//!
//! This crate implements a Byzantine Fault Tolerant consensus protocol specifically
//! designed for GPU-intensive distributed computing environments. It ensures that
//! GPU computations across multiple nodes reach agreement on results and execution order.

#![warn(missing_docs)]

pub mod error;
pub mod leader;
pub mod protocol;
pub mod sync;
pub mod validator;
pub mod voting;

// New million-node consensus modules
pub mod million_node_consensus;
pub mod consensus_compression;
pub mod adaptive;

// HPC Channels integration
#[cfg(feature = "hpc-channels")]
pub mod hpc_bridge;

#[cfg(test)]
mod protocol_edge_tests;

#[cfg(test)]
mod edge_case_tests;

pub use error::{ConsensusError, ConsensusResult};
pub use leader::{LeaderElection, LeaderState};
pub use protocol::{ConsensusConfig, ConsensusMessage, ConsensusProtocol};
pub use sync::{StateSync, SyncManager};
pub use validator::{Validator, ValidatorId, ValidatorInfo};
pub use voting::{Vote, VoteType, VotingRound};

// Adaptive consensus exports
pub use adaptive::{
    AdaptiveConsensusEngine, AdaptiveConsensusConfig,
    ConsensusAlgorithmType, NetworkConditions, AlgorithmPerformance,
    ConsensusAlgorithm as AdaptiveAlgorithm, ConsensusOutcome,
    AdaptiveGpuConfig, NetworkMonitoringConfig, OptimizationConfig, SelectionStrategy,
};

// Million-node consensus exports
pub use million_node_consensus::{
    MillionNodeConsensus, MillionNodeConfig, MillionNodeConsensusResult, MillionNodeHealthMetrics,
    GpuConfig as MillionNodeGpuConfig, MemoryConfig, PartitionConfig, AlgorithmConfig,
    ConsensusAlgorithm as MillionNodeAlgorithm, VoteAggregationStrategy, PartitionRecoveryAlgorithm,
};

// Consensus compression exports
pub use consensus_compression::{
    ConsensusCompressionEngine, CompressionConfig, CompressionAlgorithm, CompressedMessage,
    CompressionStats, CompressionPerformanceMetrics, MessagePattern,
    CompressionGpuConfig, CompressionMemoryConfig, PatternAnalysisConfig,
};

pub use std::time::Duration;
/// Re-export commonly used types
pub use uuid::Uuid;

// HPC Channels bridge exports
#[cfg(feature = "hpc-channels")]
pub use hpc_bridge::{
    ConsensusChannelBridge, SharedConsensusChannelBridge, shared_channel_bridge,
    ConsensusEvent,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all public types are accessible
        let _config = ConsensusConfig::default();
        let _error = ConsensusError::ValidationFailed("test".to_string());
    }
}
