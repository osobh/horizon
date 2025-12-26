//! Type definitions for fault tolerance module

use crate::swarm_distributed::{MigrationParticle, RecoveryStrategy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Node health status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealth {
    /// Node identifier
    pub node_id: String,
    /// Current health status
    pub status: HealthStatus,
    /// Last heartbeat timestamp
    pub last_heartbeat: u64,
    /// Response time history
    pub response_times: Vec<f64>,
    /// Error count
    pub error_count: u32,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Network connectivity score
    pub network_score: f64,
}

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    /// Node is healthy and responsive
    Healthy,
    /// Node is showing signs of degradation
    Degraded,
    /// Node is suspected to be failing
    Suspect,
    /// Node has failed
    Failed,
    /// Node is recovered and rejoining
    Recovering,
}

/// Failure detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureDetectionAlgorithm {
    /// Simple heartbeat-based detection
    Heartbeat,
    /// Adaptive failure detection based on response times
    Adaptive,
    /// Phi failure detector
    PhiAccrual,
    /// Network partition detection
    NetworkPartition,
}

/// Alert thresholds for failure detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Heartbeat timeout in milliseconds
    pub heartbeat_timeout: u64,
    /// Maximum response time in milliseconds
    pub max_response_time: f64,
    /// Maximum error rate (0.0 - 1.0)
    pub max_error_rate: f64,
    /// CPU utilization threshold
    pub cpu_threshold: f64,
    /// Memory utilization threshold
    pub memory_threshold: f64,
}

/// Recovery event information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryEvent {
    /// Event identifier
    pub id: String,
    /// Node that failed
    pub failed_node: String,
    /// Recovery strategy used
    pub strategy: RecoveryStrategy,
    /// Timestamp when failure was detected
    pub failure_time: u64,
    /// Timestamp when recovery started
    pub recovery_start_time: u64,
    /// Timestamp when recovery completed
    pub recovery_completion_time: Option<u64>,
    /// Recovery status
    pub status: RecoveryStatus,
    /// Number of particles affected
    pub particles_affected: usize,
    /// Recovery success rate
    pub success_rate: f64,
}

/// Recovery status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecoveryStatus {
    /// Recovery is in progress
    InProgress,
    /// Recovery completed successfully
    Completed,
    /// Recovery failed
    Failed,
    /// Recovery was cancelled
    Cancelled,
}

/// Checkpoint snapshot containing system state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSnapshot {
    /// Checkpoint identifier
    pub id: String,
    /// Generation number
    pub generation: u32,
    /// Timestamp
    pub timestamp: u64,
    /// Node states
    pub node_states: HashMap<String, NodeState>,
    /// Global best particle
    pub global_best: Option<MigrationParticle>,
    /// Global best fitness
    pub global_best_fitness: Option<f64>,
    /// Checkpoint size in bytes
    pub size_bytes: usize,
}

/// Individual node state in checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeState {
    /// Node identifier
    pub node_id: String,
    /// Local particles on this node
    pub particles: Vec<MigrationParticle>,
    /// Node-specific configuration
    pub config: serde_json::Value,
    /// Local best particle
    pub local_best: Option<MigrationParticle>,
    /// Local best fitness
    pub local_best_fitness: Option<f64>,
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Checkpoint identifier
    pub id: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Total size in bytes
    pub size_bytes: usize,
    /// Number of nodes included
    pub node_count: usize,
    /// Number of particles included
    pub particle_count: usize,
    /// Compression algorithm used
    pub compression: CompressionAlgorithm,
    /// Checksum for integrity verification
    pub checksum: String,
}

/// Compression algorithms for checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// LZ4 compression
    Lz4,
    /// Zstd compression
    Zstd,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            heartbeat_timeout: 30000,  // 30 seconds
            max_response_time: 5000.0, // 5 seconds
            max_error_rate: 0.1,       // 10% error rate
            cpu_threshold: 0.9,        // 90% CPU
            memory_threshold: 0.85,    // 85% memory
        }
    }
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::Gzip
    }
}
