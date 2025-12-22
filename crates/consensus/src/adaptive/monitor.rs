//! Network monitoring for adaptive consensus

use super::NetworkMonitoringConfig;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Network conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConditions {
    /// Average latency in milliseconds
    pub latency_ms: f64,
    /// Bandwidth in Mbps
    pub bandwidth_mbps: f64,
    /// Packet loss rate (0.0 to 1.0)
    pub packet_loss: f64,
    /// Number of active nodes
    pub active_nodes: usize,
    /// Network partition detected
    pub partition_detected: bool,
    /// Timestamp of measurement
    pub timestamp: std::time::Instant,
}

/// Network monitor
pub struct NetworkMonitor {
    config: NetworkMonitoringConfig,
    current_conditions: Arc<RwLock<NetworkConditions>>,
}

impl NetworkMonitor {
    /// Create new network monitor
    pub fn new(config: NetworkMonitoringConfig) -> Self {
        Self {
            config,
            current_conditions: Arc::new(RwLock::new(NetworkConditions::default())),
        }
    }
    
    /// Get current network conditions
    pub async fn get_current_conditions(&self) -> NetworkConditions {
        self.current_conditions.read().await.clone()
    }
    
    /// Update network conditions
    pub async fn update_conditions(&self, conditions: NetworkConditions) {
        *self.current_conditions.write().await = conditions;
    }
}

impl Default for NetworkConditions {
    fn default() -> Self {
        Self {
            latency_ms: 1.0,
            bandwidth_mbps: 1000.0,
            packet_loss: 0.0,
            active_nodes: 0,
            partition_detected: false,
            timestamp: std::time::Instant::now(),
        }
    }
}