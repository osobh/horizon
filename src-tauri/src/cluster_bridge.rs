//! Cluster Bridge
//!
//! Integrates the stratoswarm cluster-mesh with Horizon using hpc-channels.
//! When the `embedded-cluster` feature is enabled, uses the real cluster-mesh API.
//! Otherwise, provides mock data for development.

use hpc_channels::{channels, ClusterMessage, NodeInfo as ChannelNodeInfo, NodeRole, NodeCapabilities as ChannelCapabilities, GpuInfo as ChannelGpuInfo};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

// Import real cluster-mesh types when feature is enabled
#[cfg(feature = "embedded-cluster")]
use stratoswarm_cluster_mesh::{
    ClusterMesh, ClusterNode as MeshClusterNode, NodeStatus as MeshNodeStatus,
    NodeClass, ClusterStatistics as MeshClusterStats, NodeCapabilities as MeshNodeCapabilities,
};

/// Bridge to the stratoswarm cluster mesh.
pub struct ClusterBridge {
    /// Cluster connection state
    state: Arc<RwLock<ClusterConnectionState>>,

    /// Real cluster mesh (when embedded-cluster feature enabled)
    #[cfg(feature = "embedded-cluster")]
    mesh: Arc<RwLock<Option<ClusterMesh>>>,

    /// Mock nodes for development (when embedded-cluster feature disabled)
    #[cfg(not(feature = "embedded-cluster"))]
    mock_nodes: Arc<RwLock<Vec<ClusterNode>>>,
}

#[derive(Debug, Clone, Default)]
struct ClusterConnectionState {
    connected: bool,
    endpoint: Option<String>,
}

/// Represents a node in the cluster (mirrors stratoswarm types for UI consumption)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    pub id: String,
    pub hostname: String,
    pub address: String,
    pub node_type: NodeType,
    pub status: NodeStatus,
    pub capabilities: NodeCapabilities,
    pub metrics: Option<NodeMetricsSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeType {
    DataCenter,
    Workstation,
    Laptop,
    Edge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeStatus {
    Online,
    Offline,
    Degraded,
    Maintenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub cpu_cores: u32,
    pub memory_gb: f32,
    pub gpu_count: u32,
    pub gpu_memory_gb: Option<f32>,
    pub storage_gb: f32,
    pub gpus: Vec<GpuInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub id: u32,
    pub name: String,
    pub memory_gb: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetricsSnapshot {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
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

// ============================================================================
// Real Implementation (embedded-cluster feature)
// ============================================================================

#[cfg(feature = "embedded-cluster")]
impl ClusterBridge {
    /// Create a new cluster bridge with real cluster-mesh.
    pub fn new() -> Self {
        tracing::info!("ClusterBridge initialized with embedded cluster-mesh");
        Self {
            state: Arc::new(RwLock::new(ClusterConnectionState::default())),
            mesh: Arc::new(RwLock::new(None)),
        }
    }

    /// Connect to a cluster endpoint using real cluster-mesh.
    pub async fn connect(&self, endpoint: &str) -> Result<(), String> {
        tracing::info!("Connecting to cluster at: {} (using cluster-mesh)", endpoint);

        // Initialize the cluster mesh
        let mesh = ClusterMesh::new()
            .await
            .map_err(|e| format!("Failed to create cluster mesh: {}", e))?;

        // Start the mesh
        mesh.start()
            .await
            .map_err(|e| format!("Failed to start cluster mesh: {}", e))?;

        // Store the mesh
        let mut mesh_guard = self.mesh.write().await;
        *mesh_guard = Some(mesh);

        // Update connection state
        let mut state = self.state.write().await;
        state.connected = true;
        state.endpoint = Some(endpoint.to_string());

        // Broadcast connection event
        if let Some(tx) = hpc_channels::sender::<ClusterMessage>(channels::CLUSTER_CONNECTION) {
            let node_count = self.get_nodes().await.len();
            let _ = tx.send(ClusterMessage::Connected { node_count }).await;
        }

        tracing::info!("Connected to cluster successfully via cluster-mesh");
        Ok(())
    }

    /// Disconnect from the cluster.
    pub async fn disconnect(&self) -> Result<(), String> {
        let mut mesh_guard = self.mesh.write().await;
        *mesh_guard = None;

        let mut state = self.state.write().await;
        state.connected = false;
        state.endpoint = None;

        tracing::info!("Disconnected from cluster");
        Ok(())
    }

    /// Check if connected to a cluster.
    pub async fn is_connected(&self) -> bool {
        self.state.read().await.connected
    }

    /// Get the current cluster endpoint.
    pub async fn endpoint(&self) -> Option<String> {
        self.state.read().await.endpoint.clone()
    }

    /// Get all nodes in the cluster from real cluster-mesh.
    pub async fn get_nodes(&self) -> Vec<ClusterNode> {
        let mesh_guard = self.mesh.read().await;
        if let Some(mesh) = mesh_guard.as_ref() {
            let mesh_nodes = mesh.get_nodes().await;
            mesh_nodes.into_iter().map(Self::convert_mesh_node).collect()
        } else {
            Vec::new()
        }
    }

    /// Get a specific node by ID.
    pub async fn get_node(&self, id: &str) -> Option<ClusterNode> {
        let nodes = self.get_nodes().await;
        nodes.into_iter().find(|n| n.id == id)
    }

    /// Get cluster statistics from real cluster-mesh.
    pub async fn get_statistics(&self) -> ClusterStats {
        let mesh_guard = self.mesh.read().await;
        if let Some(mesh) = mesh_guard.as_ref() {
            let stats = mesh.get_statistics().await;
            Self::convert_mesh_stats(stats)
        } else {
            ClusterStats {
                total_nodes: 0,
                online_nodes: 0,
                datacenter_nodes: 0,
                workstation_nodes: 0,
                laptop_nodes: 0,
                edge_nodes: 0,
                total_cpu_cores: 0,
                total_memory_gb: 0.0,
                total_gpu_count: 0,
            }
        }
    }

    /// Convert a mesh ClusterNode to our ClusterNode type.
    fn convert_mesh_node(mesh_node: MeshClusterNode) -> ClusterNode {
        let node_type = match mesh_node.class {
            NodeClass::DataCenter { .. } => NodeType::DataCenter,
            NodeClass::Workstation { .. } => NodeType::Workstation,
            NodeClass::Laptop { .. } => NodeType::Laptop,
            NodeClass::Edge { .. } => NodeType::Edge,
        };

        let status = match mesh_node.status {
            MeshNodeStatus::Online => NodeStatus::Online,
            MeshNodeStatus::Offline => NodeStatus::Offline,
            MeshNodeStatus::Draining | MeshNodeStatus::Maintenance => NodeStatus::Maintenance,
            MeshNodeStatus::Unknown => NodeStatus::Degraded,
        };

        ClusterNode {
            id: mesh_node.id.to_string(),
            hostname: mesh_node.hostname,
            address: format!("{}:50051", mesh_node.id), // Use node ID as address placeholder
            node_type,
            status,
            capabilities: NodeCapabilities {
                cpu_cores: mesh_node.capabilities.cpu_cores,
                memory_gb: mesh_node.capabilities.memory_gb,
                gpu_count: mesh_node.capabilities.gpu_count,
                gpu_memory_gb: mesh_node.capabilities.gpu_memory_gb,
                storage_gb: mesh_node.capabilities.storage_gb,
                gpus: mesh_node.hardware.gpus.iter().enumerate().map(|(i, gpu)| GpuInfo {
                    id: i as u32,
                    name: gpu.name.clone(),
                    memory_gb: gpu.memory_mb as f32 / 1024.0,
                }).collect(),
            },
            metrics: None, // TODO: Add real-time metrics collection
        }
    }

    /// Convert mesh ClusterStatistics to our ClusterStats type.
    fn convert_mesh_stats(stats: MeshClusterStats) -> ClusterStats {
        ClusterStats {
            total_nodes: stats.total_nodes,
            online_nodes: stats.online_nodes,
            datacenter_nodes: stats.datacenter_nodes,
            workstation_nodes: stats.workstation_nodes,
            laptop_nodes: stats.laptop_nodes,
            edge_nodes: stats.edge_nodes,
            total_cpu_cores: stats.total_cpu_cores,
            total_memory_gb: stats.total_memory_gb,
            total_gpu_count: stats.total_gpu_count,
        }
    }

    /// Convert to hpc-channels NodeInfo format
    #[allow(dead_code)]
    pub fn to_channel_node_info(node: &ClusterNode) -> ChannelNodeInfo {
        ChannelNodeInfo {
            id: node.id.clone(),
            hostname: node.hostname.clone(),
            address: node.address.clone(),
            role: match node.node_type {
                NodeType::DataCenter => NodeRole::Leader,
                _ => NodeRole::Worker,
            },
            capabilities: ChannelCapabilities {
                cpu_cores: node.capabilities.cpu_cores,
                memory_bytes: (node.capabilities.memory_gb * 1024.0 * 1024.0 * 1024.0) as u64,
                gpus: node.capabilities.gpus.iter().map(|g| ChannelGpuInfo {
                    id: g.id,
                    name: g.name.clone(),
                    memory_bytes: (g.memory_gb * 1024.0 * 1024.0 * 1024.0) as u64,
                    compute_capability: "8.9".to_string(),
                }).collect(),
                storage_bytes: (node.capabilities.storage_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            },
            joined_at: 0,
        }
    }
}

// ============================================================================
// Mock Implementation (no embedded-cluster feature)
// ============================================================================

#[cfg(not(feature = "embedded-cluster"))]
impl ClusterBridge {
    /// Create a new cluster bridge with mock data.
    pub fn new() -> Self {
        tracing::info!("ClusterBridge initialized with mock data (embedded-cluster feature disabled)");
        Self {
            state: Arc::new(RwLock::new(ClusterConnectionState::default())),
            mock_nodes: Arc::new(RwLock::new(Self::create_mock_nodes())),
        }
    }

    /// Create mock nodes for development/demo
    fn create_mock_nodes() -> Vec<ClusterNode> {
        vec![
            ClusterNode {
                id: "node-dc-01".to_string(),
                hostname: "gpu-cluster-01".to_string(),
                address: "10.0.1.10:50051".to_string(),
                node_type: NodeType::DataCenter,
                status: NodeStatus::Online,
                capabilities: NodeCapabilities {
                    cpu_cores: 64,
                    memory_gb: 512.0,
                    gpu_count: 8,
                    gpu_memory_gb: Some(640.0),
                    storage_gb: 10000.0,
                    gpus: vec![
                        GpuInfo { id: 0, name: "NVIDIA A100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 1, name: "NVIDIA A100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 2, name: "NVIDIA A100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 3, name: "NVIDIA A100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 4, name: "NVIDIA A100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 5, name: "NVIDIA A100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 6, name: "NVIDIA A100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 7, name: "NVIDIA A100 80GB".to_string(), memory_gb: 80.0 },
                    ],
                },
                metrics: Some(NodeMetricsSnapshot {
                    cpu_usage: 0.45,
                    memory_usage: 0.62,
                    gpu_usage: vec![0.85, 0.92, 0.78, 0.88, 0.90, 0.82, 0.75, 0.80],
                }),
            },
            ClusterNode {
                id: "node-dc-02".to_string(),
                hostname: "gpu-cluster-02".to_string(),
                address: "10.0.1.11:50051".to_string(),
                node_type: NodeType::DataCenter,
                status: NodeStatus::Online,
                capabilities: NodeCapabilities {
                    cpu_cores: 128,
                    memory_gb: 1024.0,
                    gpu_count: 8,
                    gpu_memory_gb: Some(640.0),
                    storage_gb: 20000.0,
                    gpus: vec![
                        GpuInfo { id: 0, name: "NVIDIA H100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 1, name: "NVIDIA H100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 2, name: "NVIDIA H100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 3, name: "NVIDIA H100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 4, name: "NVIDIA H100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 5, name: "NVIDIA H100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 6, name: "NVIDIA H100 80GB".to_string(), memory_gb: 80.0 },
                        GpuInfo { id: 7, name: "NVIDIA H100 80GB".to_string(), memory_gb: 80.0 },
                    ],
                },
                metrics: Some(NodeMetricsSnapshot {
                    cpu_usage: 0.38,
                    memory_usage: 0.55,
                    gpu_usage: vec![0.70, 0.65, 0.72, 0.68, 0.75, 0.60, 0.58, 0.62],
                }),
            },
            ClusterNode {
                id: "node-ws-01".to_string(),
                hostname: "dev-workstation".to_string(),
                address: "192.168.1.50:50051".to_string(),
                node_type: NodeType::Workstation,
                status: NodeStatus::Online,
                capabilities: NodeCapabilities {
                    cpu_cores: 24,
                    memory_gb: 128.0,
                    gpu_count: 1,
                    gpu_memory_gb: Some(24.0),
                    storage_gb: 4000.0,
                    gpus: vec![
                        GpuInfo { id: 0, name: "NVIDIA RTX 4090".to_string(), memory_gb: 24.0 },
                    ],
                },
                metrics: Some(NodeMetricsSnapshot {
                    cpu_usage: 0.25,
                    memory_usage: 0.40,
                    gpu_usage: vec![0.15],
                }),
            },
            ClusterNode {
                id: "node-laptop-01".to_string(),
                hostname: "macbook-m3".to_string(),
                address: "192.168.1.100:50051".to_string(),
                node_type: NodeType::Laptop,
                status: NodeStatus::Online,
                capabilities: NodeCapabilities {
                    cpu_cores: 12,
                    memory_gb: 36.0,
                    gpu_count: 1,
                    gpu_memory_gb: Some(36.0),
                    storage_gb: 1000.0,
                    gpus: vec![
                        GpuInfo { id: 0, name: "Apple M3 Max GPU".to_string(), memory_gb: 36.0 },
                    ],
                },
                metrics: Some(NodeMetricsSnapshot {
                    cpu_usage: 0.12,
                    memory_usage: 0.35,
                    gpu_usage: vec![0.05],
                }),
            },
            ClusterNode {
                id: "node-edge-01".to_string(),
                hostname: "jetson-orin".to_string(),
                address: "192.168.1.200:50051".to_string(),
                node_type: NodeType::Edge,
                status: NodeStatus::Degraded,
                capabilities: NodeCapabilities {
                    cpu_cores: 12,
                    memory_gb: 32.0,
                    gpu_count: 1,
                    gpu_memory_gb: Some(32.0),
                    storage_gb: 256.0,
                    gpus: vec![
                        GpuInfo { id: 0, name: "NVIDIA Orin GPU".to_string(), memory_gb: 32.0 },
                    ],
                },
                metrics: Some(NodeMetricsSnapshot {
                    cpu_usage: 0.65,
                    memory_usage: 0.78,
                    gpu_usage: vec![0.45],
                }),
            },
        ]
    }

    /// Connect to a cluster endpoint (mock).
    pub async fn connect(&self, endpoint: &str) -> Result<(), String> {
        tracing::info!("Connecting to cluster at: {} (mock mode)", endpoint);

        // Simulate connection delay
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let mut state = self.state.write().await;
        state.connected = true;
        state.endpoint = Some(endpoint.to_string());

        // Broadcast connection event
        if let Some(tx) = hpc_channels::sender::<ClusterMessage>(channels::CLUSTER_CONNECTION) {
            let _ = tx.send(ClusterMessage::Connected {
                node_count: self.mock_nodes.read().await.len(),
            }).await;
        }

        tracing::info!("Connected to cluster successfully (mock)");
        Ok(())
    }

    /// Disconnect from the cluster.
    pub async fn disconnect(&self) -> Result<(), String> {
        let mut state = self.state.write().await;
        state.connected = false;
        state.endpoint = None;

        tracing::info!("Disconnected from cluster");
        Ok(())
    }

    /// Check if connected to a cluster.
    pub async fn is_connected(&self) -> bool {
        self.state.read().await.connected
    }

    /// Get the current cluster endpoint.
    pub async fn endpoint(&self) -> Option<String> {
        self.state.read().await.endpoint.clone()
    }

    /// Get all nodes in the cluster (mock).
    pub async fn get_nodes(&self) -> Vec<ClusterNode> {
        self.mock_nodes.read().await.clone()
    }

    /// Get a specific node by ID.
    pub async fn get_node(&self, id: &str) -> Option<ClusterNode> {
        self.mock_nodes.read().await.iter().find(|n| n.id == id).cloned()
    }

    /// Get cluster statistics (mock).
    pub async fn get_statistics(&self) -> ClusterStats {
        let nodes = self.mock_nodes.read().await;

        let mut stats = ClusterStats {
            total_nodes: nodes.len(),
            online_nodes: 0,
            datacenter_nodes: 0,
            workstation_nodes: 0,
            laptop_nodes: 0,
            edge_nodes: 0,
            total_cpu_cores: 0,
            total_memory_gb: 0.0,
            total_gpu_count: 0,
        };

        for node in nodes.iter() {
            match node.node_type {
                NodeType::DataCenter => stats.datacenter_nodes += 1,
                NodeType::Workstation => stats.workstation_nodes += 1,
                NodeType::Laptop => stats.laptop_nodes += 1,
                NodeType::Edge => stats.edge_nodes += 1,
            }

            if matches!(node.status, NodeStatus::Online) {
                stats.online_nodes += 1;
            }

            stats.total_cpu_cores += node.capabilities.cpu_cores;
            stats.total_memory_gb += node.capabilities.memory_gb;
            stats.total_gpu_count += node.capabilities.gpu_count;
        }

        stats
    }

    /// Convert to hpc-channels NodeInfo format
    #[allow(dead_code)]
    pub fn to_channel_node_info(node: &ClusterNode) -> ChannelNodeInfo {
        ChannelNodeInfo {
            id: node.id.clone(),
            hostname: node.hostname.clone(),
            address: node.address.clone(),
            role: match node.node_type {
                NodeType::DataCenter => NodeRole::Leader,
                _ => NodeRole::Worker,
            },
            capabilities: ChannelCapabilities {
                cpu_cores: node.capabilities.cpu_cores,
                memory_bytes: (node.capabilities.memory_gb * 1024.0 * 1024.0 * 1024.0) as u64,
                gpus: node.capabilities.gpus.iter().map(|g| ChannelGpuInfo {
                    id: g.id,
                    name: g.name.clone(),
                    memory_bytes: (g.memory_gb * 1024.0 * 1024.0 * 1024.0) as u64,
                    compute_capability: "8.9".to_string(),
                }).collect(),
                storage_bytes: (node.capabilities.storage_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            },
            joined_at: 0,
        }
    }
}

impl Default for ClusterBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Start the cluster message handler task.
#[allow(dead_code)]
pub async fn start_cluster_handler(bridge: Arc<ClusterBridge>) {
    // Create cluster channels
    let (_tx, mut rx) = hpc_channels::channel::<ClusterMessage>(channels::CLUSTER_CONNECTION);
    let nodes_tx = hpc_channels::broadcast::<ClusterMessage>(channels::CLUSTER_NODES, 256);
    let _health_tx = hpc_channels::broadcast::<ClusterMessage>(channels::CLUSTER_HEALTH, 256);

    tracing::info!("Cluster handler started");

    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match msg {
                ClusterMessage::Connect { endpoint } => {
                    if let Err(e) = bridge.connect(&endpoint).await {
                        tracing::error!("Failed to connect: {}", e);
                    }
                }
                ClusterMessage::GetTopology => {
                    let nodes = bridge.get_nodes().await;
                    let node_infos: Vec<ChannelNodeInfo> = nodes
                        .iter()
                        .map(ClusterBridge::to_channel_node_info)
                        .collect();

                    let _ = nodes_tx.send(ClusterMessage::Topology { nodes: node_infos });
                }
                _ => {}
            }
        }
    });
}
