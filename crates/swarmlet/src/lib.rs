//! StratoSwarm Swarmlet Library
//!
//! A lightweight library for creating StratoSwarm node agents that can
//! easily join existing clusters.

pub mod agent;
pub mod command;
pub mod config;
pub mod discovery;
pub mod error;
pub mod join;
pub mod profile;
pub mod security;
pub mod workload;

// Re-export main types
pub use agent::SwarmletAgent;
pub use command::{CommandExecutor, CommandRequest, CommandResult, CommandStatus};
pub use config::Config;
pub use discovery::{ClusterDiscovery, ClusterInfo};
pub use error::{Result, SwarmletError};
pub use join::{JoinProtocol, JoinResult};
pub use profile::{HardwareProfile, HardwareProfiler, NodeCapabilities};

/// Swarmlet version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration values
pub mod defaults {
    /// Default discovery port for cluster mesh
    pub const DISCOVERY_PORT: u16 = 7946;

    /// Default API port for swarmlet
    pub const API_PORT: u16 = 8080;

    /// Default metrics port
    pub const METRICS_PORT: u16 = 9090;

    /// Default heartbeat interval in seconds
    pub const HEARTBEAT_INTERVAL: u64 = 30;

    /// Default join timeout in seconds
    pub const JOIN_TIMEOUT: u64 = 60;

    /// Default data directory
    pub const DATA_DIR: &str = "/data";

    /// Default configuration file
    pub const CONFIG_FILE: &str = "/config/swarmlet.toml";
}
