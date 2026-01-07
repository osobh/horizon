//! Inventory management module
//!
//! Provides node inventory management for the HPC-AI platform.
//! This module handles:
//! - Node registration and tracking
//! - SSH credential management
//! - Remote command execution
//! - Node status monitoring
//! - Hardware profile detection
//! - Heartbeat health monitoring
//! - Async bootstrap with progress reporting

pub mod bootstrap;
pub mod credentials;
pub mod detection;
pub mod heartbeat;
pub mod ssh;
pub mod store;

// Re-export types from hpc-inventory crate
pub use hpc_inventory::{
    Architecture, CredentialRef, GpuInfo, HardwareProfile, InventorySummary, NodeInfo, NodeMode,
    NodeStatus, OsType,
};

// Re-export CLI-specific modules
pub use bootstrap::{bootstrap_node_async, BootstrapParams, BootstrapProgress, BootstrapStage};
pub use credentials::{CredentialStore, KeyInfo};
pub use detection::{NodeDetector, RequirementCheck};
pub use heartbeat::{HeartbeatMonitor, HeartbeatState, NodeHealthResult};
pub use ssh::{CommandOutput, SshAuth, SshClient, SshSession};
pub use store::InventoryStore;
