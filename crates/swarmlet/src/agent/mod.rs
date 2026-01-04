//! Swarmlet agent runtime

mod api;
mod build;
mod core;
mod deploy;
mod lifecycle;
mod state;
mod types;

#[cfg(test)]
mod tests;

pub use state::SavedAgentState;
pub use types::{HealthStatus, NodeStatus, ResourceLimits, WorkAssignment};

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::{
    build_job_manager::BuildJobManager, build_log_stream::BuildLogStreamer,
    command::CommandExecutor, config::Config, join::JoinResult, security::NodeCertificate,
    wireguard::WireGuardManager, workload::WorkloadManager,
};

#[cfg(feature = "hpc-channels")]
use crate::{
    SharedAgentChannelBridge, SharedArtifactTransferBridge, SharedBuildChannelBridge,
    SharedDeployChannelBridge,
};

/// Main swarmlet agent that manages node lifecycle
pub struct SwarmletAgent {
    pub(crate) config: Arc<Config>,
    pub(crate) join_result: JoinResult,
    #[allow(dead_code)]
    pub(crate) node_certificate: NodeCertificate,
    pub(crate) workload_manager: Arc<WorkloadManager>,
    pub(crate) command_executor: Arc<CommandExecutor>,
    pub(crate) wireguard_manager: Arc<WireGuardManager>,
    pub(crate) build_job_manager: Arc<BuildJobManager>,
    pub(crate) log_streamer: Arc<BuildLogStreamer>,
    pub(crate) health_status: Arc<RwLock<HealthStatus>>,
    pub(crate) shutdown_signal: tokio::sync::watch::Receiver<bool>,
    pub(crate) shutdown_sender: tokio::sync::watch::Sender<bool>,
    #[cfg(feature = "hpc-channels")]
    pub(crate) event_bridge: SharedAgentChannelBridge,
    #[cfg(feature = "hpc-channels")]
    pub(crate) build_bridge: SharedBuildChannelBridge,
    #[cfg(feature = "hpc-channels")]
    pub(crate) deploy_bridge: SharedDeployChannelBridge,
    #[cfg(feature = "hpc-channels")]
    pub(crate) artifact_bridge: SharedArtifactTransferBridge,
}
