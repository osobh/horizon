//! Application State Management
//!
//! Centralized state for the Horizon application.

use crate::cluster_bridge::ClusterBridge;
use crate::gpu_compiler_bridge::GpuCompilerBridge;
use crate::kernel_bridge::KernelBridge;
use crate::storage_bridge::StorageBridge;
use crate::training_bridge::TrainingBridge;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Global application state shared across all Tauri commands.
pub struct AppState {
    /// Cluster connection state (reserved for future use)
    #[allow(dead_code)]
    pub cluster_state: Arc<RwLock<ClusterState>>,
    /// Notebook kernel state
    pub notebook: Arc<RwLock<NotebookState>>,
    /// Kernel bridge for executing code
    pub kernel: Arc<KernelBridge>,
    /// Cluster bridge for cluster management
    pub cluster: Arc<ClusterBridge>,
    /// Training bridge for ML training jobs
    pub training: Arc<TrainingBridge>,
    /// Storage bridge for file transfers
    pub storage: Arc<StorageBridge>,
    /// GPU compiler bridge for accelerated Rust compilation
    pub gpu_compiler: Arc<GpuCompilerBridge>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            cluster_state: Arc::new(RwLock::new(ClusterState::default())),
            notebook: Arc::new(RwLock::new(NotebookState::default())),
            kernel: Arc::new(KernelBridge::new()),
            cluster: Arc::new(ClusterBridge::new()),
            training: Arc::new(TrainingBridge::new()),
            storage: Arc::new(StorageBridge::new()),
            gpu_compiler: Arc::new(GpuCompilerBridge::new()),
        }
    }

    /// Initialize the kernel bridge.
    #[allow(dead_code)]
    pub async fn initialize_kernel(&self) -> Result<(), String> {
        self.kernel.initialize().await
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Cluster connection and node state (reserved for future state synchronization).
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct ClusterState {
    /// Whether connected to a StratoSwarm cluster
    pub connected: bool,
    /// Cluster endpoint (if connected)
    pub endpoint: Option<String>,
    /// Discovered nodes
    pub nodes: Vec<NodeInfo>,
}

/// Information about a cluster node.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub hostname: String,
    pub node_type: NodeType,
    pub status: NodeStatus,
    pub gpu_count: u32,
    pub gpu_memory_gb: u32,
    pub cpu_cores: u32,
    pub memory_gb: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeType {
    DataCenter,
    Workstation,
    Laptop,
    Edge,
    Storage,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeStatus {
    Online,
    Offline,
    Degraded,
    Starting,
}

/// Notebook kernel state.
#[derive(Debug, Default)]
pub struct NotebookState {
    /// Whether kernel is running
    pub kernel_running: bool,
    /// Current execution count
    pub execution_count: u64,
    /// Variables in scope
    pub variables: Vec<VariableInfo>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VariableInfo {
    pub name: String,
    pub var_type: String,
    pub size_bytes: u64,
    pub preview: String,
}

/// Training job state (legacy - see training_bridge for active implementation).
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct TrainingState {
    /// Active training jobs
    pub jobs: Vec<TrainingJob>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct TrainingJob {
    pub id: String,
    pub name: String,
    pub status: TrainingStatus,
    pub progress: f32,
    pub epoch: u32,
    pub total_epochs: u32,
    pub loss: f64,
    pub metrics: std::collections::HashMap<String, f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
#[allow(dead_code)]
pub enum TrainingStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
}
