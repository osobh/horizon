//! # StratoSwarm Cluster Mesh
//!
//! This crate implements heterogeneous cluster management for StratoSwarm,
//! supporting diverse hardware from data centers to edge devices.

pub mod api;
pub mod classification;
pub mod discovery;
pub mod distribution;
pub mod install_script;
pub mod mesh;
// pub mod laptop;
// pub mod edge;
pub mod error;

#[cfg(test)]
mod classification_tests;
#[cfg(test)]
mod discovery_tests;
#[cfg(test)]
mod error_tests;

// Re-export main types
pub use api::{create_router, start_server, AppState, AppStateConfig, ServerConfig};
pub use classification::{NodeClass, NodeClassifier};
pub use discovery::{HardwareProfile, NetworkCharacteristics, NodeDiscovery};
pub use distribution::{JobRequirements, SchedulingPolicy, WorkDistributor};
pub use install_script::{generate_install_script, BinaryChecksums, InstallScriptConfig};
pub use mesh::{MeshManager, MeshTopology, NodeConnection};
// pub use laptop::{LaptopNode, PowerManagement, ThermalPolicy};
// pub use edge::{EdgeDevice, EdgeType, ResourceConstraints};
pub use error::{ClusterMeshError, Result};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Represents a node in the heterogeneous cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    pub id: Uuid,
    pub hostname: String,
    pub class: NodeClass,
    pub hardware: HardwareProfile,
    pub network: NetworkCharacteristics,
    pub status: NodeStatus,
    pub capabilities: NodeCapabilities,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    /// WireGuard public key for secure mesh communication
    pub wg_public_key: Option<String>,
    /// Subnet assignment info (populated by subnet-manager integration)
    pub subnet_info: Option<NodeSubnetAssignment>,
}

/// Subnet assignment information for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSubnetAssignment {
    /// The subnet ID this node is assigned to
    pub subnet_id: Uuid,
    /// Human-readable subnet name
    pub subnet_name: String,
    /// IP address assigned within the subnet
    pub assigned_ip: std::net::Ipv4Addr,
    /// WireGuard interface name (e.g., "wg-tenant-acme")
    pub wg_interface: String,
    /// When the assignment was made
    pub assigned_at: chrono::DateTime<chrono::Utc>,
}

/// Node status in the cluster
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    Online,
    Offline,
    Draining,
    Maintenance,
    Unknown,
}

/// Node capabilities and constraints
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeCapabilities {
    pub cpu_cores: u32,
    pub memory_gb: f32,
    pub gpu_count: u32,
    pub gpu_memory_gb: Option<f32>,
    pub storage_gb: f32,
    pub network_bandwidth_mbps: f32,
    pub supports_gpu_direct: bool,
    pub battery_powered: bool,
    pub thermal_constraints: Option<ThermalConstraints>,
}

/// Thermal constraints for power-limited devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConstraints {
    pub max_temp_celsius: f32,
    pub throttle_temp_celsius: f32,
    pub current_temp_celsius: f32,
    pub power_budget_watts: Option<f32>,
}

/// Main cluster mesh manager
#[derive(Clone)]
#[allow(dead_code)]
pub struct ClusterMesh {
    nodes: Arc<RwLock<Vec<ClusterNode>>>,
    discovery: Arc<NodeDiscovery>,
    classifier: Arc<NodeClassifier>,
    distributor: Arc<WorkDistributor>,
    mesh_manager: Arc<MeshManager>,
}

impl ClusterMesh {
    /// Create a new cluster mesh
    pub async fn new() -> Result<Self> {
        let discovery = Arc::new(NodeDiscovery::new().await?);
        let classifier = Arc::new(NodeClassifier::new());
        let distributor = Arc::new(WorkDistributor::new());
        let mesh_manager = Arc::new(MeshManager::new().await?);

        Ok(Self {
            nodes: Arc::new(RwLock::new(Vec::new())),
            discovery,
            classifier,
            distributor,
            mesh_manager,
        })
    }

    /// Start the cluster mesh
    pub async fn start(&self) -> Result<()> {
        // Start node discovery
        self.discovery.start_discovery().await?;

        // Start mesh formation
        self.mesh_manager.start_mesh_formation().await?;

        // Start work distribution
        self.distributor.start_distribution().await?;

        Ok(())
    }

    /// Add a node to the cluster
    pub async fn add_node(&self, node: ClusterNode) -> Result<()> {
        let mut nodes = self.nodes.write().await;

        // Check if node already exists
        if nodes.iter().any(|n| n.id == node.id) {
            return Err(ClusterMeshError::NodeAlreadyExists(node.id));
        }

        // Add to mesh topology
        self.mesh_manager.add_node(&node).await?;

        // Add to nodes list
        nodes.push(node);

        Ok(())
    }

    /// Remove a node from the cluster
    pub async fn remove_node(&self, node_id: Uuid) -> Result<()> {
        let mut nodes = self.nodes.write().await;

        // Find and remove node
        let index = nodes
            .iter()
            .position(|n| n.id == node_id)
            .ok_or(ClusterMeshError::NodeNotFound(node_id))?;

        let node = nodes.remove(index);

        // Remove from mesh topology
        self.mesh_manager.remove_node(&node).await?;

        // Migrate any running work
        self.distributor.migrate_work_from_node(node_id).await?;

        Ok(())
    }

    /// Get all nodes in the cluster
    pub async fn get_nodes(&self) -> Vec<ClusterNode> {
        self.nodes.read().await.clone()
    }

    /// Get nodes by class
    pub async fn get_nodes_by_class(&self, class: NodeClass) -> Vec<ClusterNode> {
        self.nodes
            .read()
            .await
            .iter()
            .filter(|n| n.class == class)
            .cloned()
            .collect()
    }

    /// Schedule a job on the cluster
    pub async fn schedule_job(&self, job: Job) -> Result<Uuid> {
        let nodes = self.nodes.read().await;
        let node = self.distributor.select_node(&job, &nodes).await?;
        self.distributor.schedule_on_node(job, node.id).await
    }

    /// Get cluster statistics
    pub async fn get_statistics(&self) -> ClusterStatistics {
        let nodes = self.nodes.read().await;

        let mut stats = ClusterStatistics::default();

        for node in nodes.iter() {
            match node.class {
                NodeClass::DataCenter { .. } => stats.datacenter_nodes += 1,
                NodeClass::Workstation { .. } => stats.workstation_nodes += 1,
                NodeClass::Laptop { .. } => stats.laptop_nodes += 1,
                NodeClass::Edge { .. } => stats.edge_nodes += 1,
            }

            stats.total_cpu_cores += node.capabilities.cpu_cores;
            stats.total_memory_gb += node.capabilities.memory_gb;
            stats.total_gpu_count += node.capabilities.gpu_count;

            if node.status == NodeStatus::Online {
                stats.online_nodes += 1;
            }
        }

        stats.total_nodes = nodes.len();
        stats
    }
}

/// Job representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: Uuid,
    pub name: String,
    pub requirements: JobRequirements,
    pub priority: JobPriority,
    pub submitted_at: chrono::DateTime<chrono::Utc>,
}

/// Job priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JobPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Cluster statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterStatistics {
    pub total_nodes: usize,
    pub online_nodes: usize,
    pub datacenter_nodes: usize,
    pub workstation_nodes: usize,
    pub laptop_nodes: usize,
    pub edge_nodes: usize,
    pub total_cpu_cores: u32,
    pub total_memory_gb: f32,
    pub total_gpu_count: u32,
}

/// Trait for node schedulers
#[async_trait]
pub trait NodeScheduler: Send + Sync {
    /// Check if a job can run on a node
    async fn can_run(&self, job: &Job, node: &ClusterNode) -> bool;

    /// Estimate job performance on a node
    async fn estimate_performance(&self, job: &Job, node: &ClusterNode) -> f32;

    /// Handle node departure
    async fn handle_node_departure(&self, node: &ClusterNode) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cluster_mesh_creation() {
        let mesh = ClusterMesh::new().await.unwrap();
        let nodes = mesh.get_nodes().await;
        assert!(nodes.is_empty());
    }

    #[tokio::test]
    async fn test_add_remove_node() {
        let mesh = ClusterMesh::new().await.unwrap();

        let node = ClusterNode {
            id: Uuid::new_v4(),
            hostname: "test-node".to_string(),
            class: NodeClass::Workstation {
                gpu: Some(crate::discovery::GpuInfo {
                    index: 0,
                    name: "RTX 4090".to_string(),
                    memory_mb: 24576,
                    compute_capability: (8, 9),
                    pci_bus_id: "0:1:0.0".to_string(),
                }),
                schedule: crate::classification::Schedule::BusinessHours,
            },
            hardware: HardwareProfile {
                cpu_model: "Intel i9-13900K".to_string(),
                cpu_cores: 24,
                memory_gb: 64.0,
                storage_gb: 2000.0,
                gpus: vec![],
            },
            network: NetworkCharacteristics {
                bandwidth_mbps: 1000.0,
                latency_ms: 1.0,
                jitter_ms: 0.1,
                packet_loss: 0.0,
                nat_type: crate::discovery::NatType::None,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities {
                cpu_cores: 24,
                memory_gb: 64.0,
                gpu_count: 1,
                gpu_memory_gb: Some(24.0),
                storage_gb: 2000.0,
                network_bandwidth_mbps: 1000.0,
                supports_gpu_direct: true,
                battery_powered: false,
                thermal_constraints: None,
            },
            last_heartbeat: chrono::Utc::now(),
        };

        // Add node
        mesh.add_node(node.clone()).await.unwrap();
        let nodes = mesh.get_nodes().await;
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].id, node.id);

        // Remove node
        mesh.remove_node(node.id).await.unwrap();
        let nodes = mesh.get_nodes().await;
        assert!(nodes.is_empty());
    }

    #[tokio::test]
    async fn test_node_classification() {
        let mesh = ClusterMesh::new().await.unwrap();

        // Add different types of nodes
        let datacenter_node = ClusterNode {
            id: Uuid::new_v4(),
            hostname: "dc-node".to_string(),
            class: NodeClass::DataCenter {
                gpus: vec![],
                bandwidth: crate::classification::Bandwidth::TenGigabit(10.0),
            },
            hardware: HardwareProfile {
                cpu_model: "AMD EPYC 7763".to_string(),
                cpu_cores: 64,
                memory_gb: 512.0,
                storage_gb: 10000.0,
                gpus: vec![],
            },
            network: NetworkCharacteristics {
                bandwidth_mbps: 10000.0,
                latency_ms: 0.1,
                jitter_ms: 0.01,
                packet_loss: 0.0,
                nat_type: crate::discovery::NatType::None,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities {
                cpu_cores: 64,
                memory_gb: 512.0,
                gpu_count: 0,
                gpu_memory_gb: None,
                storage_gb: 10000.0,
                network_bandwidth_mbps: 10000.0,
                supports_gpu_direct: false,
                battery_powered: false,
                thermal_constraints: None,
            },
            last_heartbeat: chrono::Utc::now(),
        };

        let laptop_node = ClusterNode {
            id: Uuid::new_v4(),
            hostname: "laptop-node".to_string(),
            class: NodeClass::Laptop {
                battery: true,
                mobility: crate::classification::MobilityPattern::Frequent,
            },
            hardware: HardwareProfile {
                cpu_model: "Intel i7-1280P".to_string(),
                cpu_cores: 14,
                memory_gb: 32.0,
                storage_gb: 1000.0,
                gpus: vec![],
            },
            network: NetworkCharacteristics {
                bandwidth_mbps: 100.0,
                latency_ms: 10.0,
                jitter_ms: 1.0,
                packet_loss: 0.1,
                nat_type: crate::discovery::NatType::Symmetric,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities {
                cpu_cores: 14,
                memory_gb: 32.0,
                gpu_count: 0,
                gpu_memory_gb: None,
                storage_gb: 1000.0,
                network_bandwidth_mbps: 100.0,
                supports_gpu_direct: false,
                battery_powered: true,
                thermal_constraints: Some(ThermalConstraints {
                    max_temp_celsius: 100.0,
                    throttle_temp_celsius: 85.0,
                    current_temp_celsius: 45.0,
                    power_budget_watts: Some(28.0),
                }),
            },
            last_heartbeat: chrono::Utc::now(),
        };

        mesh.add_node(datacenter_node).await.unwrap();
        mesh.add_node(laptop_node).await.unwrap();

        // Test classification queries
        let dc_nodes = mesh
            .get_nodes_by_class(NodeClass::DataCenter {
                gpus: vec![],
                bandwidth: crate::classification::Bandwidth::TenGigabit(10.0),
            })
            .await;
        assert_eq!(dc_nodes.len(), 1);

        let stats = mesh.get_statistics().await;
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.datacenter_nodes, 1);
        assert_eq!(stats.laptop_nodes, 1);
        assert_eq!(stats.total_cpu_cores, 78);
    }

    #[test]
    fn test_job_priority_ordering() {
        assert!(JobPriority::Critical > JobPriority::High);
        assert!(JobPriority::High > JobPriority::Normal);
        assert!(JobPriority::Normal > JobPriority::Low);
    }
}
