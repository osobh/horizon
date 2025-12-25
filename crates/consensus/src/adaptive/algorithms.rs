//! Consensus algorithm trait and implementations

use super::{ConsensusAlgorithmType, RoundResult};
use crate::ConsensusError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Trait for consensus algorithms
#[async_trait]
pub trait ConsensusAlgorithm: Send + Sync {
    /// Get the algorithm type
    fn algorithm_type(&self) -> ConsensusAlgorithmType;
    
    /// Execute consensus round
    async fn execute_round(&self, proposal: &[u8], nodes: usize) -> Result<RoundResult, ConsensusError>;
    
    /// Check if GPU acceleration is supported
    fn supports_gpu(&self) -> bool;
    
    /// Get minimum nodes required
    fn min_nodes(&self) -> usize;
    
    /// Get maximum nodes supported
    fn max_nodes(&self) -> usize;
    
    /// Estimate latency for given number of nodes
    fn estimate_latency(&self, nodes: usize) -> u64;
    
    /// Get algorithm requirements
    fn requirements(&self) -> AlgorithmRequirements;
}

/// Requirements for an algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmRequirements {
    /// Minimum memory required (MB)
    pub min_memory_mb: usize,
    /// GPU required
    pub requires_gpu: bool,
    /// Network bandwidth required (Mbps)
    pub min_bandwidth_mbps: u32,
    /// CPU cores required
    pub min_cpu_cores: usize,
}

/// Configuration for algorithm instances
#[derive(Debug, Clone)]
pub struct AlgorithmConfig {
    /// Maximum message size
    pub max_message_size: usize,
    /// Timeout for consensus rounds
    pub round_timeout: std::time::Duration,
    /// Enable optimizations
    pub enable_optimizations: bool,
}

// Concrete algorithm implementations

/// PBFT Algorithm
pub struct PBFTAlgorithm {
    config: AlgorithmConfig,
}

/// GPU-accelerated PBFT
pub struct GpuPBFTAlgorithm {
    config: AlgorithmConfig,
    gpu_device: Option<u32>,
}

/// Fast consensus for small networks
pub struct FastConsensusAlgorithm {
    config: AlgorithmConfig,
}

/// Streaming consensus
pub struct StreamingConsensusAlgorithm {
    config: AlgorithmConfig,
}

/// Hybrid consensus
pub struct HybridConsensusAlgorithm {
    config: AlgorithmConfig,
    primary_algorithm: Box<dyn ConsensusAlgorithm>,
    fallback_algorithm: Box<dyn ConsensusAlgorithm>,
}

/// GPU-native consensus
pub struct GpuNativeConsensusAlgorithm {
    config: AlgorithmConfig,
    gpu_device: u32,
}

/// ML-optimized consensus
pub struct MLOptimizedConsensusAlgorithm {
    config: AlgorithmConfig,
    model_path: Option<String>,
}

// Macro to implement common algorithm methods
macro_rules! impl_consensus_algorithm {
    ($type:ty, $algo:expr, $gpu:expr, $min:expr, $max:expr) => {
        #[async_trait]
        impl ConsensusAlgorithm for $type {
            fn algorithm_type(&self) -> ConsensusAlgorithmType {
                $algo
            }
            
            async fn execute_round(&self, _proposal: &[u8], _nodes: usize) -> Result<RoundResult, ConsensusError> {
                // Simplified implementation
                Ok(RoundResult {
                    consensus_reached: true,
                    round: 1,
                    term: 1,
                })
            }
            
            fn supports_gpu(&self) -> bool {
                $gpu
            }
            
            fn min_nodes(&self) -> usize {
                $min
            }
            
            fn max_nodes(&self) -> usize {
                $max
            }
            
            fn estimate_latency(&self, nodes: usize) -> u64 {
                // Simple linear estimation
                100 + (nodes as u64 * 10)
            }
            
            fn requirements(&self) -> AlgorithmRequirements {
                AlgorithmRequirements {
                    min_memory_mb: 512,
                    requires_gpu: $gpu,
                    min_bandwidth_mbps: 100,
                    min_cpu_cores: 2,
                }
            }
        }
    };
}

impl_consensus_algorithm!(PBFTAlgorithm, ConsensusAlgorithmType::PBFT, false, 4, 1000);
impl_consensus_algorithm!(GpuPBFTAlgorithm, ConsensusAlgorithmType::GpuPBFT, true, 4, 100000);
impl_consensus_algorithm!(FastConsensusAlgorithm, ConsensusAlgorithmType::Fast, false, 3, 100);
impl_consensus_algorithm!(StreamingConsensusAlgorithm, ConsensusAlgorithmType::Streaming, false, 10, 10000);
impl_consensus_algorithm!(GpuNativeConsensusAlgorithm, ConsensusAlgorithmType::GpuNative, true, 100, 1000000);
impl_consensus_algorithm!(MLOptimizedConsensusAlgorithm, ConsensusAlgorithmType::MLOptimized, true, 10, 100000);

// Special implementation for HybridConsensusAlgorithm
#[async_trait]
impl ConsensusAlgorithm for HybridConsensusAlgorithm {
    fn algorithm_type(&self) -> ConsensusAlgorithmType {
        ConsensusAlgorithmType::Hybrid
    }
    
    async fn execute_round(&self, proposal: &[u8], nodes: usize) -> Result<RoundResult, ConsensusError> {
        // Try primary first, fallback if needed
        match self.primary_algorithm.execute_round(proposal, nodes).await {
            Ok(result) => Ok(result),
            Err(_) => self.fallback_algorithm.execute_round(proposal, nodes).await,
        }
    }
    
    fn supports_gpu(&self) -> bool {
        self.primary_algorithm.supports_gpu() || self.fallback_algorithm.supports_gpu()
    }
    
    fn min_nodes(&self) -> usize {
        self.primary_algorithm.min_nodes().min(self.fallback_algorithm.min_nodes())
    }
    
    fn max_nodes(&self) -> usize {
        self.primary_algorithm.max_nodes().max(self.fallback_algorithm.max_nodes())
    }
    
    fn estimate_latency(&self, nodes: usize) -> u64 {
        self.primary_algorithm.estimate_latency(nodes)
    }
    
    fn requirements(&self) -> AlgorithmRequirements {
        self.primary_algorithm.requirements()
    }
}