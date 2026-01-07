//! Inventory node types
//!
//! Core data structures for node inventory management.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Node deployment mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum NodeMode {
    /// Deploy agent via Docker container
    #[default]
    Docker,
    /// Deploy agent as native binary
    Binary,
}

impl NodeMode {
    /// Get string representation
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Docker => "docker",
            Self::Binary => "binary",
        }
    }
}

impl std::fmt::Display for NodeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Operating system type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OsType {
    /// Linux
    Linux,
    /// macOS
    Darwin,
    /// Windows
    Windows,
}

impl OsType {
    /// Get string representation
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Linux => "linux",
            Self::Darwin => "darwin",
            Self::Windows => "windows",
        }
    }
}

impl std::fmt::Display for OsType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// CPU architecture
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Architecture {
    /// x86_64/amd64
    Amd64,
    /// aarch64/arm64
    Arm64,
    /// Unknown architecture
    #[serde(other)]
    Unknown,
}

impl Architecture {
    /// Get string representation
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Amd64 => "amd64",
            Self::Arm64 => "arm64",
            Self::Unknown => "unknown",
        }
    }

    /// Parse from uname output
    #[must_use]
    pub fn from_uname(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "x86_64" | "amd64" => Self::Amd64,
            "aarch64" | "arm64" => Self::Arm64,
            _ => Self::Unknown,
        }
    }
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Node connection status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum NodeStatus {
    /// Initial state, not yet connected
    #[default]
    Pending,
    /// SSH/WinRM connection established
    Connecting,
    /// Bootstrap in progress
    Bootstrapping,
    /// Agent installed and responding
    Connected,
    /// Agent not responding (missed heartbeats)
    Unreachable,
    /// Explicitly marked offline
    Offline,
    /// Bootstrap or connection failed
    Failed,
}

impl NodeStatus {
    /// Get string representation
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Connecting => "connecting",
            Self::Bootstrapping => "bootstrapping",
            Self::Connected => "connected",
            Self::Unreachable => "unreachable",
            Self::Offline => "offline",
            Self::Failed => "failed",
        }
    }

    /// Get status symbol for display
    #[must_use]
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Pending => "○",
            Self::Connecting => "◐",
            Self::Bootstrapping => "◑",
            Self::Connected => "●",
            Self::Unreachable => "◌",
            Self::Offline => "○",
            Self::Failed => "✗",
        }
    }

    /// Check if status represents a healthy node
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Connected)
    }
}

impl std::fmt::Display for NodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU index
    pub index: u32,
    /// GPU name/model
    pub name: String,
    /// GPU memory in MB
    pub memory_mb: u64,
    /// GPU vendor (nvidia, amd, intel, apple)
    pub vendor: String,
}

/// Hardware profile for a node
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareProfile {
    /// CPU model name
    pub cpu_model: String,
    /// Number of CPU cores
    pub cpu_cores: u32,
    /// Total memory in GB
    pub memory_gb: f32,
    /// Total storage in GB
    pub storage_gb: f32,
    /// List of GPUs
    pub gpus: Vec<GpuInfo>,
}

impl HardwareProfile {
    /// Get GPU count
    #[must_use]
    pub fn gpu_count(&self) -> usize {
        self.gpus.len()
    }

    /// Get total GPU memory in GB
    #[must_use]
    pub fn total_gpu_memory_gb(&self) -> f32 {
        self.gpus.iter().map(|g| g.memory_mb as f32 / 1024.0).sum()
    }
}

/// Credential reference type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CredentialRef {
    /// Path to SSH private key file
    SshKey {
        /// Path to private key
        path: PathBuf,
    },
    /// Password stored in secure keyring
    Password {
        /// Key ID in keyring
        key_id: String,
    },
    /// Use SSH agent
    SshAgent,
}

impl Default for CredentialRef {
    fn default() -> Self {
        Self::SshAgent
    }
}

/// Complete node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Unique node identifier (UUID)
    pub id: String,
    /// User-friendly node name
    pub name: String,
    /// IP address or hostname
    pub address: String,
    /// SSH/WinRM port
    pub port: u16,
    /// SSH username
    pub username: String,
    /// Reference to credential (key path or password)
    pub credential_ref: CredentialRef,
    /// Deployment mode (docker or binary)
    pub mode: NodeMode,
    /// Operating system (detected after connection)
    pub os: Option<OsType>,
    /// Architecture (detected after connection)
    pub arch: Option<Architecture>,
    /// Current status
    pub status: NodeStatus,
    /// Hardware profile (populated after bootstrap)
    pub hardware: Option<HardwareProfile>,
    /// Last successful heartbeat
    pub last_heartbeat: Option<DateTime<Utc>>,
    /// QUIC mesh endpoint (after bootstrap)
    pub quic_endpoint: Option<String>,
    /// When node was added
    pub created_at: DateTime<Utc>,
    /// Last status update
    pub updated_at: DateTime<Utc>,
    /// Error message if failed
    pub error: Option<String>,
    /// Custom tags/labels
    pub tags: Vec<String>,
}

impl NodeInfo {
    /// Create a new pending node
    #[must_use]
    pub fn new(
        name: String,
        address: String,
        port: u16,
        username: String,
        credential_ref: CredentialRef,
        mode: NodeMode,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            address,
            port,
            username,
            credential_ref,
            mode,
            os: None,
            arch: None,
            status: NodeStatus::Pending,
            hardware: None,
            last_heartbeat: None,
            quic_endpoint: None,
            created_at: now,
            updated_at: now,
            error: None,
            tags: Vec::new(),
        }
    }

    /// Get OS/arch string (e.g., "linux/amd64")
    #[must_use]
    pub fn platform_str(&self) -> String {
        match (&self.os, &self.arch) {
            (Some(os), Some(arch)) => format!("{os}/{arch}"),
            (Some(os), None) => os.to_string(),
            (None, Some(arch)) => format!("?/{arch}"),
            (None, None) => "unknown".to_string(),
        }
    }

    /// Update status and timestamp
    pub fn set_status(&mut self, status: NodeStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }

    /// Set error message and mark as failed
    pub fn set_error(&mut self, error: impl Into<String>) {
        self.error = Some(error.into());
        self.set_status(NodeStatus::Failed);
    }

    /// Record a heartbeat
    pub fn record_heartbeat(&mut self) {
        self.last_heartbeat = Some(Utc::now());
        if self.status == NodeStatus::Unreachable {
            self.set_status(NodeStatus::Connected);
        }
    }
}

/// Inventory summary statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InventorySummary {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of connected nodes
    pub connected: usize,
    /// Number of unreachable nodes
    pub unreachable: usize,
    /// Number of pending nodes
    pub pending: usize,
    /// Number of failed nodes
    pub failed: usize,
    /// Total CPU cores across all nodes
    pub total_cpus: u32,
    /// Total GPU count across all nodes
    pub total_gpus: usize,
    /// Total memory in GB across all nodes
    pub total_memory_gb: f32,
}

impl InventorySummary {
    /// Calculate summary from a list of nodes
    #[must_use]
    pub fn from_nodes(nodes: &[NodeInfo]) -> Self {
        let mut summary = Self::default();
        summary.total_nodes = nodes.len();

        for node in nodes {
            match node.status {
                NodeStatus::Connected => summary.connected += 1,
                NodeStatus::Unreachable => summary.unreachable += 1,
                NodeStatus::Pending | NodeStatus::Connecting | NodeStatus::Bootstrapping => {
                    summary.pending += 1;
                }
                NodeStatus::Failed => summary.failed += 1,
                NodeStatus::Offline => {}
            }

            if let Some(hw) = &node.hardware {
                summary.total_cpus += hw.cpu_cores;
                summary.total_gpus += hw.gpu_count();
                summary.total_memory_gb += hw.memory_gb;
            }
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_mode() {
        assert_eq!(NodeMode::Docker.as_str(), "docker");
        assert_eq!(NodeMode::Binary.as_str(), "binary");
    }

    #[test]
    fn test_os_type() {
        assert_eq!(OsType::Linux.as_str(), "linux");
        assert_eq!(OsType::Darwin.as_str(), "darwin");
        assert_eq!(OsType::Windows.as_str(), "windows");
    }

    #[test]
    fn test_architecture_from_uname() {
        assert_eq!(Architecture::from_uname("x86_64"), Architecture::Amd64);
        assert_eq!(Architecture::from_uname("amd64"), Architecture::Amd64);
        assert_eq!(Architecture::from_uname("aarch64"), Architecture::Arm64);
        assert_eq!(Architecture::from_uname("arm64"), Architecture::Arm64);
        assert_eq!(Architecture::from_uname("ppc64"), Architecture::Unknown);
    }

    #[test]
    fn test_node_status() {
        assert!(NodeStatus::Connected.is_healthy());
        assert!(!NodeStatus::Pending.is_healthy());
        assert!(!NodeStatus::Failed.is_healthy());
    }

    #[test]
    fn test_node_info_creation() {
        let node = NodeInfo::new(
            "test-node".to_string(),
            "192.168.1.100".to_string(),
            22,
            "admin".to_string(),
            CredentialRef::SshAgent,
            NodeMode::Docker,
        );

        assert_eq!(node.name, "test-node");
        assert_eq!(node.status, NodeStatus::Pending);
        assert!(node.os.is_none());
        assert!(!node.id.is_empty());
    }

    #[test]
    fn test_node_platform_str() {
        let mut node = NodeInfo::new(
            "test".to_string(),
            "127.0.0.1".to_string(),
            22,
            "user".to_string(),
            CredentialRef::SshAgent,
            NodeMode::Binary,
        );

        assert_eq!(node.platform_str(), "unknown");

        node.os = Some(OsType::Linux);
        node.arch = Some(Architecture::Amd64);
        assert_eq!(node.platform_str(), "linux/amd64");
    }

    #[test]
    fn test_inventory_summary() {
        let nodes = vec![
            {
                let mut n = NodeInfo::new(
                    "n1".to_string(),
                    "1.1.1.1".to_string(),
                    22,
                    "u".to_string(),
                    CredentialRef::SshAgent,
                    NodeMode::Docker,
                );
                n.status = NodeStatus::Connected;
                n.hardware = Some(HardwareProfile {
                    cpu_cores: 8,
                    memory_gb: 32.0,
                    ..Default::default()
                });
                n
            },
            {
                let mut n = NodeInfo::new(
                    "n2".to_string(),
                    "2.2.2.2".to_string(),
                    22,
                    "u".to_string(),
                    CredentialRef::SshAgent,
                    NodeMode::Docker,
                );
                n.status = NodeStatus::Pending;
                n
            },
        ];

        let summary = InventorySummary::from_nodes(&nodes);
        assert_eq!(summary.total_nodes, 2);
        assert_eq!(summary.connected, 1);
        assert_eq!(summary.pending, 1);
        assert_eq!(summary.total_cpus, 8);
        assert!((summary.total_memory_gb - 32.0).abs() < f32::EPSILON);
    }
}
