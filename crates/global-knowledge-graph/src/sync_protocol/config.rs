//! Configuration structures for global synchronization

use std::net::SocketAddr;
use std::time::Duration;

/// Configuration for global synchronization
#[derive(Debug, Clone)]
pub struct GlobalSyncConfig {
    /// Maximum allowable sync latency
    pub max_sync_latency: Duration,
    /// Number of GPU clusters
    pub cluster_count: usize,
    /// Regions for multi-region testing
    pub regions: Vec<String>,
    /// GPU cluster specifications
    pub gpu_specs: Vec<GpuClusterSpec>,
    /// Byzantine fault tolerance threshold (0.0 to 1.0)
    pub byzantine_threshold: f32,
    /// Network configuration
    pub network_config: NetworkConfig,
    /// Consensus algorithm configuration
    pub consensus_config: ConsensusConfig,
}

/// GPU cluster specification
#[derive(Debug, Clone)]
pub struct GpuClusterSpec {
    pub region: String,
    pub gpu_count: usize,
    pub memory_gb: usize,
    pub compute_capability: String,
    pub network_bandwidth_gbps: f32,
}

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Local bind address
    pub bind_address: SocketAddr,
    /// Peer cluster addresses
    pub peer_addresses: Vec<SocketAddr>,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Maximum message size
    pub max_message_size: usize,
}

/// Consensus algorithm configuration
#[derive(Debug, Clone)]
pub struct ConsensusConfig {
    /// Consensus algorithm type
    pub algorithm: ConsensusAlgorithm,
    /// Timeout for consensus rounds
    pub consensus_timeout: Duration,
    /// Maximum number of consensus rounds
    pub max_rounds: usize,
    /// Vote threshold for acceptance
    pub vote_threshold: f32,
}

/// Available consensus algorithms
#[derive(Debug, Clone)]
pub enum ConsensusAlgorithm {
    /// Practical Byzantine Fault Tolerance
    PBFT,
    /// Raft consensus (for non-Byzantine environments)
    Raft,
    /// GPU-accelerated consensus
    GpuAccelerated,
    /// Hybrid consensus combining multiple algorithms
    Hybrid,
}

impl Default for GlobalSyncConfig {
    fn default() -> Self {
        Self {
            max_sync_latency: Duration::from_millis(100),
            cluster_count: 3,
            regions: vec![
                "us-east".to_string(),
                "us-west".to_string(),
                "eu-central".to_string(),
            ],
            gpu_specs: vec![],
            byzantine_threshold: 0.33,
            network_config: NetworkConfig::default(),
            consensus_config: ConsensusConfig::default(),
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:8080".parse()?,
            peer_addresses: vec![],
            connection_timeout: Duration::from_secs(5),
            heartbeat_interval: Duration::from_secs(1),
            max_message_size: 1024 * 1024, // 1MB
        }
    }
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            algorithm: ConsensusAlgorithm::PBFT,
            consensus_timeout: Duration::from_secs(5),
            max_rounds: 10,
            vote_threshold: 0.67,
        }
    }
}
