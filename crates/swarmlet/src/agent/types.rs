//! Agent type definitions

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Health status of the swarmlet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub node_id: Uuid,
    pub status: NodeStatus,
    pub uptime_seconds: u64,
    pub workloads_active: u32,
    pub cpu_usage_percent: f32,
    pub memory_usage_gb: f32,
    pub disk_usage_gb: f32,
    pub network_rx_bytes: u64,
    pub network_tx_bytes: u64,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub errors_count: u32,
}

/// Node status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeStatus {
    Starting,
    Healthy,
    Degraded,
    Unhealthy,
    Shutting,
}

/// Work assignment from cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkAssignment {
    pub id: Uuid,
    pub workload_type: String,
    pub container_image: Option<String>,
    pub command: Option<Vec<String>>,
    pub shell_script: Option<String>,
    pub environment: std::collections::HashMap<String, String>,
    pub resource_limits: ResourceLimits,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Resource limits for workloads
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub cpu_cores: Option<f32>,
    pub memory_gb: Option<f32>,
    pub disk_gb: Option<f32>,
}
