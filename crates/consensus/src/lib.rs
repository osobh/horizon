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
pub mod adaptive;
pub mod consensus_compression;
pub mod million_node_consensus;

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
    AdaptiveConsensusConfig, AdaptiveConsensusEngine, AdaptiveGpuConfig, AlgorithmPerformance,
    ConsensusAlgorithm as AdaptiveAlgorithm, ConsensusAlgorithmType, ConsensusOutcome,
    NetworkConditions, NetworkMonitoringConfig, OptimizationConfig, SelectionStrategy,
};

// Million-node consensus exports
pub use million_node_consensus::{
    AlgorithmConfig, ConsensusAlgorithm as MillionNodeAlgorithm, GpuConfig as MillionNodeGpuConfig,
    MemoryConfig, MillionNodeConfig, MillionNodeConsensus, MillionNodeConsensusResult,
    MillionNodeHealthMetrics, PartitionConfig, PartitionRecoveryAlgorithm, VoteAggregationStrategy,
};

// Consensus compression exports
pub use consensus_compression::{
    CompressedMessage, CompressionAlgorithm, CompressionConfig, CompressionGpuConfig,
    CompressionMemoryConfig, CompressionPerformanceMetrics, CompressionStats,
    ConsensusCompressionEngine, MessagePattern, PatternAnalysisConfig,
};

pub use std::time::Duration;
/// Re-export commonly used types
pub use uuid::Uuid;

// HPC Channels bridge exports
#[cfg(feature = "hpc-channels")]
pub use hpc_bridge::{
    shared_channel_bridge, ConsensusChannelBridge, ConsensusEvent, SharedConsensusChannelBridge,
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
