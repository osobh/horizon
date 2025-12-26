//! Configuration types for distributed swarm execution

use crate::{
    error::{EvolutionEngineError, EvolutionEngineResult},
    swarm::SwarmConfig,
    traits::EngineConfig,
};
use serde::{Deserialize, Serialize};

/// Configuration for distributed swarm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedSwarmConfig {
    /// Base swarm configuration
    pub base_config: SwarmConfig,
    /// Node ID for this instance
    pub node_id: String,
    /// Network configuration
    pub network_config: NetworkConfig,
    /// Load balancing configuration
    pub load_balance_config: LoadBalanceConfig,
    /// Fault tolerance configuration
    pub fault_tolerance_config: FaultToleranceConfig,
}

/// Network configuration for distributed nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Listen address for this node
    pub listen_addr: String,
    /// Port for communication
    pub port: u16,
    /// Known peer addresses for initial discovery
    pub bootstrap_peers: Vec<String>,
    /// Maximum number of connections
    pub max_connections: usize,
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalanceConfig {
    /// Target particles per node
    pub target_particles_per_node: usize,
    /// Load balancing strategy
    pub strategy: LoadBalanceStrategy,
    /// Rebalancing threshold (0.0 - 1.0)
    pub rebalance_threshold: f64,
    /// Migration batch size
    pub migration_batch_size: usize,
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Checkpoint interval in generations
    pub checkpoint_interval: u32,
    /// Number of backup replicas
    pub backup_replicas: u32,
    /// Failure detection timeout in milliseconds
    pub failure_timeout_ms: u64,
    /// Recovery strategy
    pub recovery_strategy: RecoveryStrategy,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalanceStrategy {
    /// Distribute particles evenly across nodes
    EvenDistribution,
    /// Balance based on computational load
    LoadBased,
    /// Balance based on network topology
    TopologyAware,
    /// Dynamic strategy that adapts over time
    Adaptive,
}

/// Recovery strategies for node failures
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RecoveryStrategy {
    /// Redistribute failed node's particles to other nodes
    Redistribute,
    /// Restore from checkpoint and continue
    Checkpoint,
    /// Hybrid approach using both redistribution and checkpoints
    Hybrid,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0".to_string(),
            port: 7001,
            bootstrap_peers: Vec::new(),
            max_connections: 100,
            heartbeat_interval_ms: 5000,
        }
    }
}

impl Default for LoadBalanceConfig {
    fn default() -> Self {
        Self {
            target_particles_per_node: 50,
            strategy: LoadBalanceStrategy::EvenDistribution,
            rebalance_threshold: 0.2,
            migration_batch_size: 10,
        }
    }
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: 10,
            backup_replicas: 2,
            failure_timeout_ms: 15000,
            recovery_strategy: RecoveryStrategy::Hybrid,
        }
    }
}

impl DistributedSwarmConfig {
    /// Create new distributed swarm configuration
    pub fn new(node_id: String) -> Self {
        Self {
            base_config: SwarmConfig::default(),
            node_id,
            network_config: NetworkConfig::default(),
            load_balance_config: LoadBalanceConfig::default(),
            fault_tolerance_config: FaultToleranceConfig::default(),
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> EvolutionEngineResult<()> {
        // Validate base configuration
        self.base_config.validate()?;

        // Validate network configuration
        if self.network_config.port == 0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Network port must be greater than 0".to_string(),
            });
        }

        if self.network_config.max_connections == 0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Max connections must be greater than 0".to_string(),
            });
        }

        // Validate load balance configuration
        if self.load_balance_config.rebalance_threshold < 0.0
            || self.load_balance_config.rebalance_threshold > 1.0
        {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Rebalance threshold must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(())
    }
}
