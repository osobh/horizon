//! Error types for cluster mesh operations

use thiserror::Error;
use uuid::Uuid;

pub type Result<T> = std::result::Result<T, ClusterMeshError>;

#[derive(Error, Debug)]
pub enum ClusterMeshError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Node {0} not found")]
    NodeNotFound(Uuid),

    #[error("Node {0} already exists")]
    NodeAlreadyExists(Uuid),

    #[error("Hardware detection failed: {0}")]
    HardwareDetection(String),

    #[error("GPU detection failed: {0}")]
    GpuDetection(String),

    #[error("Invalid node class: {0}")]
    InvalidNodeClass(String),

    #[error("Scheduling failed: {0}")]
    SchedulingFailed(String),

    #[error("No suitable node found for job")]
    NoSuitableNode,

    #[error("Node {0} is offline")]
    NodeOffline(Uuid),

    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),

    #[error("Mesh formation failed: {0}")]
    MeshFormationFailed(String),

    #[error("NAT traversal failed: {0}")]
    NatTraversalFailed(String),

    #[error("Certificate error: {0}")]
    Certificate(String),

    #[error("Authentication failed: {0}")]
    Authentication(String),

    #[error("Battery level too low: {0}%")]
    BatteryLow(u8),

    #[error("Thermal limit exceeded: {0}°C")]
    ThermalLimit(f32),

    #[error("Work migration failed: {0}")]
    MigrationFailed(String),

    #[error("Heartbeat timeout for node {0}")]
    HeartbeatTimeout(Uuid),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("System info error: {0}")]
    SystemInfo(String),

    #[error("Other error: {0}")]
    Other(String),

    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let node_id = Uuid::new_v4();
        let err = ClusterMeshError::NodeNotFound(node_id);
        assert_eq!(format!("{}", err), format!("Node {} not found", node_id));

        let err = ClusterMeshError::BatteryLow(15);
        assert_eq!(format!("{}", err), "Battery level too low: 15%");

        let err = ClusterMeshError::ThermalLimit(95.5);
        assert_eq!(format!("{}", err), "Thermal limit exceeded: 95.5°C");
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let mesh_err: ClusterMeshError = io_err.into();
        assert!(matches!(mesh_err, ClusterMeshError::Io(_)));

        let json_err = serde_json::from_str::<String>("invalid json").unwrap_err();
        let mesh_err: ClusterMeshError = json_err.into();
        assert!(matches!(mesh_err, ClusterMeshError::Serialization(_)));
    }
}
