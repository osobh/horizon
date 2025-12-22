//! Algorithm selection logic

use super::{ConsensusAlgorithmType, NetworkConditions};

/// Algorithm selector
pub struct AlgorithmSelector {
    strategy: SelectionStrategy,
}

/// Selection strategy
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    /// Always use the fastest algorithm
    Fastest,
    /// Optimize for reliability
    MostReliable,
    /// Balance speed and reliability
    Balanced,
    /// Use machine learning
    MLBased,
    /// Custom selection logic
    Custom,
}

impl AlgorithmSelector {
    /// Create new selector
    pub fn new() -> Self {
        Self {
            strategy: SelectionStrategy::Balanced,
        }
    }
    
    /// Select algorithm based on conditions
    pub async fn select_algorithm(
        &self,
        conditions: &NetworkConditions,
        nodes: usize,
        target_latency_us: u64,
    ) -> Option<ConsensusAlgorithmType> {
        // Simple selection logic
        if conditions.packet_loss > 0.1 {
            // High packet loss - use more resilient algorithm
            Some(ConsensusAlgorithmType::PBFT)
        } else if nodes > 10000 {
            // Large scale - use GPU-native
            Some(ConsensusAlgorithmType::GpuNative)
        } else if target_latency_us < 100 {
            // Ultra-low latency requirement
            Some(ConsensusAlgorithmType::Fast)
        } else if conditions.bandwidth_mbps < 100.0 {
            // Low bandwidth - use efficient algorithm
            Some(ConsensusAlgorithmType::Streaming)
        } else {
            // Default to GPU-accelerated PBFT
            Some(ConsensusAlgorithmType::GpuPBFT)
        }
    }
}