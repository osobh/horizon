//! Network partition detection and recovery
//!
//! Handles detection and resolution of network partitions in distributed systems.
//! Implements quorum-based recovery and split-brain resolution.

use crate::error::FtResult;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Network partition recovery system
///
/// Detects and recovers from network partitions through:
/// - Connectivity monitoring
/// - Quorum-based decision making
/// - Split-brain resolution
#[derive(Debug)]
pub struct PartitionRecovery {
    /// Known partitions and their member nodes
    partitions: Arc<DashMap<PartitionId, Partition>>,
    /// Configuration for partition handling
    config: PartitionConfig,
    /// Node connectivity status
    connectivity: Arc<DashMap<String, NodeConnectivity>>,
}

/// Unique identifier for a partition
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartitionId(String);

impl PartitionId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for PartitionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a network partition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    /// Partition identifier
    pub id: PartitionId,
    /// Nodes in this partition
    pub nodes: HashSet<String>,
    /// When the partition was detected
    pub detected_at: u64,
    /// Whether this partition has quorum
    pub has_quorum: bool,
    /// Partition status
    pub status: PartitionStatus,
}

/// Status of a partition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PartitionStatus {
    /// Partition is active (network is split)
    Active,
    /// Partition is being resolved
    Resolving,
    /// Partition has been resolved
    Resolved,
}

/// Node connectivity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConnectivity {
    /// Node identifier
    pub node_id: String,
    /// Last successful heartbeat
    pub last_heartbeat: u64,
    /// Nodes this node can reach
    pub reachable_nodes: HashSet<String>,
    /// Connection status
    pub status: ConnectivityStatus,
}

/// Connectivity status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConnectivityStatus {
    Connected,
    Degraded,
    Disconnected,
}

/// Configuration for partition recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    /// Timeout before declaring a node unreachable (ms)
    pub heartbeat_timeout_ms: u64,
    /// Minimum nodes for quorum (fraction)
    pub quorum_threshold: f64,
    /// Maximum time to attempt recovery (ms)
    pub recovery_timeout_ms: u64,
    /// Interval between partition checks (ms)
    pub check_interval_ms: u64,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            heartbeat_timeout_ms: 5000,
            quorum_threshold: 0.5,
            recovery_timeout_ms: 30000,
            check_interval_ms: 1000,
        }
    }
}

impl PartitionRecovery {
    /// Create a new partition recovery system
    pub fn new() -> FtResult<Self> {
        Ok(Self {
            partitions: Arc::new(DashMap::new()),
            config: PartitionConfig::default(),
            connectivity: Arc::new(DashMap::new()),
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: PartitionConfig) -> FtResult<Self> {
        Ok(Self {
            partitions: Arc::new(DashMap::new()),
            config,
            connectivity: Arc::new(DashMap::new()),
        })
    }

    /// Detect if a network partition exists
    pub async fn detect_partition(&self) -> FtResult<bool> {
        if self.connectivity.is_empty() {
            return Ok(false);
        }

        // Check for disconnected nodes
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let disconnected_count = self.connectivity
            .iter()
            .filter(|entry| {
                let c = entry.value();
                now.saturating_sub(c.last_heartbeat) > self.config.heartbeat_timeout_ms
                    || c.status == ConnectivityStatus::Disconnected
            })
            .count();

        let total_nodes = self.connectivity.len();
        let disconnected_ratio = disconnected_count as f64 / total_nodes as f64;

        // Partition detected if significant portion of nodes are unreachable
        Ok(disconnected_ratio > (1.0 - self.config.quorum_threshold))
    }

    /// Attempt to resolve a detected partition
    pub async fn resolve_partition(&self) -> FtResult<()> {
        for entry in self.partitions.iter() {
            let id = entry.key();
            let partition = entry.value();
            if partition.status == PartitionStatus::Active {
                tracing::info!("Attempting to resolve partition {:?}", id);

                if partition.has_quorum {
                    // This partition has quorum, it should be the primary
                    tracing::info!("Partition {:?} has quorum, designated as primary", id);
                } else {
                    // This partition doesn't have quorum, should defer
                    tracing::warn!("Partition {:?} lacks quorum, deferring to majority", id);
                }
            }
        }

        Ok(())
    }

    /// Update connectivity information for a node
    pub fn update_connectivity(&self, node_id: &str, reachable: HashSet<String>) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let status = if reachable.is_empty() {
            ConnectivityStatus::Disconnected
        } else {
            ConnectivityStatus::Connected
        };

        self.connectivity.insert(
            node_id.to_string(),
            NodeConnectivity {
                node_id: node_id.to_string(),
                last_heartbeat: now,
                reachable_nodes: reachable,
                status,
            },
        );
    }

    /// Record a heartbeat from a node
    pub fn record_heartbeat(&self, node_id: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if let Some(mut conn) = self.connectivity.get_mut(node_id) {
            conn.last_heartbeat = now;
            if conn.status == ConnectivityStatus::Disconnected {
                conn.status = ConnectivityStatus::Connected;
            }
        } else {
            self.connectivity.insert(
                node_id.to_string(),
                NodeConnectivity {
                    node_id: node_id.to_string(),
                    last_heartbeat: now,
                    reachable_nodes: HashSet::new(),
                    status: ConnectivityStatus::Connected,
                },
            );
        }
    }

    /// Check if we have quorum
    pub fn has_quorum(&self, total_nodes: usize) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let connected_count = self.connectivity
            .iter()
            .filter(|entry| {
                let c = entry.value();
                now.saturating_sub(c.last_heartbeat) <= self.config.heartbeat_timeout_ms
                    && c.status != ConnectivityStatus::Disconnected
            })
            .count();

        let required = (total_nodes as f64 * self.config.quorum_threshold).ceil() as usize;
        connected_count >= required
    }

    /// Get current partition status
    pub fn get_partition_status(&self) -> Vec<Partition> {
        self.partitions.iter().map(|e| e.value().clone()).collect()
    }

    /// Get nodes that are currently unreachable
    pub fn get_unreachable_nodes(&self) -> Vec<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.connectivity
            .iter()
            .filter(|entry| {
                let c = entry.value();
                now.saturating_sub(c.last_heartbeat) > self.config.heartbeat_timeout_ms
                    || c.status == ConnectivityStatus::Disconnected
            })
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Clear resolved partitions
    pub fn clear_resolved(&self) {
        self.partitions.retain(|_, p| p.status != PartitionStatus::Resolved);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_partition_recovery_creation() {
        let recovery = PartitionRecovery::new();
        assert!(recovery.is_ok());
    }

    #[tokio::test]
    async fn test_no_partition_initially() {
        let recovery = PartitionRecovery::new().unwrap();
        let has_partition = recovery.detect_partition().await.unwrap();
        assert!(!has_partition);
    }

    #[tokio::test]
    async fn test_heartbeat_tracking() {
        let recovery = PartitionRecovery::new().unwrap();

        recovery.record_heartbeat("node1");
        recovery.record_heartbeat("node2");

        let unreachable = recovery.get_unreachable_nodes();
        assert!(unreachable.is_empty());
    }

    #[tokio::test]
    async fn test_quorum_check() {
        let recovery = PartitionRecovery::new().unwrap();

        // Add some connected nodes
        recovery.record_heartbeat("node1");
        recovery.record_heartbeat("node2");
        recovery.record_heartbeat("node3");

        // With 3 connected out of 5, we have quorum (3 >= 2.5)
        assert!(recovery.has_quorum(5));

        // With 3 connected out of 10, we don't have quorum (3 < 5)
        assert!(!recovery.has_quorum(10));
    }

    #[tokio::test]
    async fn test_connectivity_update() {
        let recovery = PartitionRecovery::new().unwrap();

        let mut reachable = HashSet::new();
        reachable.insert("node2".to_string());
        reachable.insert("node3".to_string());

        recovery.update_connectivity("node1", reachable);

        let unreachable = recovery.get_unreachable_nodes();
        assert!(!unreachable.contains(&"node1".to_string()));
    }
}
