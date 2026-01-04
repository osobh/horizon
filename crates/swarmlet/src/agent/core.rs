//! Core SwarmletAgent implementation

use base64::Engine;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::{
    build_job_manager::BuildJobManager, build_log_stream::BuildLogStreamer,
    command::CommandExecutor, config::Config, join::JoinResult, security::NodeCertificate,
    wireguard::WireGuardManager, workload::WorkloadManager, Result, SwarmletError,
};

#[cfg(feature = "hpc-channels")]
use crate::{
    SharedAgentChannelBridge, SharedArtifactTransferBridge, SharedBuildChannelBridge,
    SharedDeployChannelBridge,
};

#[cfg(feature = "hpc-channels")]
use crate::hpc_bridge::{
    shared_artifact_bridge, shared_build_bridge, shared_channel_bridge, shared_deploy_bridge,
};

use super::{HealthStatus, NodeStatus, SavedAgentState, SwarmletAgent};

impl SwarmletAgent {
    pub async fn new(join_result: JoinResult, data_dir: String) -> Result<Self> {
        let config = Config::default_with_data_dir(PathBuf::from(data_dir));
        let config = Arc::new(config);

        let node_certificate = NodeCertificate::from_pem(&join_result.node_certificate)?;
        let workload_manager =
            Arc::new(WorkloadManager::new(config.clone(), join_result.node_id).await?);
        let command_executor = Arc::new(CommandExecutor::new(config.data_dir.clone()));
        let wireguard_manager = Arc::new(WireGuardManager::new());
        let build_job_manager = Arc::new(
            BuildJobManager::new(config.clone(), join_result.node_id, config.data_dir.clone())
                .await?,
        );
        let log_streamer = Arc::new(BuildLogStreamer::new());

        // Connect log streamer to build job manager for WebSocket broadcasting
        build_job_manager
            .set_log_streamer(log_streamer.clone())
            .await;

        let health_status = Arc::new(RwLock::new(HealthStatus {
            node_id: join_result.node_id,
            status: NodeStatus::Starting,
            uptime_seconds: 0,
            workloads_active: 0,
            cpu_usage_percent: 0.0,
            memory_usage_gb: 0.0,
            disk_usage_gb: 0.0,
            network_rx_bytes: 0,
            network_tx_bytes: 0,
            last_heartbeat: chrono::Utc::now(),
            errors_count: 0,
        }));

        let (shutdown_sender, shutdown_signal) = tokio::sync::watch::channel(false);

        #[cfg(feature = "hpc-channels")]
        let node_id_str = join_result.node_id.to_string();

        Ok(Self {
            config,
            join_result,
            node_certificate,
            workload_manager,
            command_executor,
            wireguard_manager,
            build_job_manager,
            log_streamer,
            health_status,
            shutdown_signal,
            shutdown_sender,
            #[cfg(feature = "hpc-channels")]
            event_bridge: shared_channel_bridge(),
            #[cfg(feature = "hpc-channels")]
            build_bridge: shared_build_bridge(node_id_str.clone()),
            #[cfg(feature = "hpc-channels")]
            deploy_bridge: shared_deploy_bridge(node_id_str.clone()),
            #[cfg(feature = "hpc-channels")]
            artifact_bridge: shared_artifact_bridge(node_id_str),
        })
    }

    /// Get the event bridge for subscribing to agent lifecycle events
    #[cfg(feature = "hpc-channels")]
    pub fn event_bridge(&self) -> &SharedAgentChannelBridge {
        &self.event_bridge
    }

    /// Get the build bridge for hpc-ci integration
    #[cfg(feature = "hpc-channels")]
    pub fn build_bridge(&self) -> &SharedBuildChannelBridge {
        &self.build_bridge
    }

    /// Get the deploy bridge for deployment requests
    #[cfg(feature = "hpc-channels")]
    pub fn deploy_bridge(&self) -> &SharedDeployChannelBridge {
        &self.deploy_bridge
    }

    /// Get the artifact bridge for warp transfers
    #[cfg(feature = "hpc-channels")]
    pub fn artifact_bridge(&self) -> &SharedArtifactTransferBridge {
        &self.artifact_bridge
    }

    /// Create a swarmlet agent from configuration
    ///
    /// This loads previously saved state from the data directory, allowing
    /// the agent to resume after a restart without re-joining the cluster.
    pub async fn from_config(config: Config) -> Result<Self> {
        // Load saved state from disk
        let saved_state = SavedAgentState::load(&config.data_dir).await?;

        // Recreate the agent from saved state
        let config = Arc::new(config);
        let node_certificate =
            NodeCertificate::from_pem(&saved_state.join_result.node_certificate)?;
        let workload_manager =
            Arc::new(WorkloadManager::new(config.clone(), saved_state.join_result.node_id).await?);
        let command_executor = Arc::new(CommandExecutor::new(config.data_dir.clone()));
        let build_job_manager = Arc::new(
            BuildJobManager::new(
                config.clone(),
                saved_state.join_result.node_id,
                config.data_dir.clone(),
            )
            .await?,
        );
        let log_streamer = Arc::new(BuildLogStreamer::new());

        // Connect log streamer to build job manager for WebSocket broadcasting
        build_job_manager
            .set_log_streamer(log_streamer.clone())
            .await;

        // Create WireGuard manager and restore keys
        let wireguard_manager = Arc::new(WireGuardManager::new());
        wireguard_manager
            .restore_keypair(
                saved_state.wireguard_private_key.clone(),
                saved_state.wireguard_public_key.clone(),
            )
            .await;

        // Restore cluster public key if available
        if let Some(ref cluster_key_b64) = saved_state.cluster_public_key {
            if let Err(e) = wireguard_manager
                .set_cluster_public_key_b64(cluster_key_b64)
                .await
            {
                warn!("Failed to restore cluster public key: {}", e);
            }
        }

        let health_status = Arc::new(RwLock::new(HealthStatus {
            node_id: saved_state.join_result.node_id,
            status: NodeStatus::Starting,
            uptime_seconds: 0,
            workloads_active: 0,
            cpu_usage_percent: 0.0,
            memory_usage_gb: 0.0,
            disk_usage_gb: 0.0,
            network_rx_bytes: 0,
            network_tx_bytes: 0,
            last_heartbeat: chrono::Utc::now(),
            errors_count: 0,
        }));

        let (shutdown_sender, shutdown_signal) = tokio::sync::watch::channel(false);

        info!(
            "Agent restored from saved state for cluster '{}' (node {})",
            saved_state.join_result.cluster_name, saved_state.join_result.node_id
        );

        #[cfg(feature = "hpc-channels")]
        let node_id_str = saved_state.join_result.node_id.to_string();

        Ok(Self {
            config,
            join_result: saved_state.join_result,
            node_certificate,
            workload_manager,
            command_executor,
            wireguard_manager,
            build_job_manager,
            log_streamer,
            health_status,
            shutdown_signal,
            shutdown_sender,
            #[cfg(feature = "hpc-channels")]
            event_bridge: shared_channel_bridge(),
            #[cfg(feature = "hpc-channels")]
            build_bridge: shared_build_bridge(node_id_str.clone()),
            #[cfg(feature = "hpc-channels")]
            deploy_bridge: shared_deploy_bridge(node_id_str.clone()),
            #[cfg(feature = "hpc-channels")]
            artifact_bridge: shared_artifact_bridge(node_id_str),
        })
    }

    /// Save the current agent state for later restoration
    pub async fn save_state(&self) -> Result<()> {
        let wireguard_private_key =
            self.wireguard_manager
                .get_private_key()
                .await
                .ok_or_else(|| {
                    SwarmletError::Configuration("No WireGuard private key available".to_string())
                })?;
        let wireguard_public_key =
            self.wireguard_manager
                .get_public_key()
                .await
                .ok_or_else(|| {
                    SwarmletError::Configuration("No WireGuard public key available".to_string())
                })?;

        let cluster_public_key = self
            .wireguard_manager
            .get_cluster_public_key()
            .await
            .map(|key| base64::engine::general_purpose::STANDARD.encode(key));

        let state = SavedAgentState {
            join_result: self.join_result.clone(),
            wireguard_private_key,
            wireguard_public_key,
            cluster_public_key,
            saved_at: chrono::Utc::now(),
            version: SavedAgentState::CURRENT_VERSION,
        };

        state.save(&self.config.data_dir).await
    }

    /// Check if saved state exists for a data directory
    pub async fn has_saved_state(data_dir: &Path) -> bool {
        SavedAgentState::state_path(data_dir).exists()
    }

    /// Run the swarmlet agent (main event loop)
    pub async fn run(self) -> Result<()> {
        info!(
            "Starting swarmlet agent for node {}",
            self.join_result.node_id
        );

        // Update status to healthy
        {
            let mut health = self.health_status.write().await;
            health.status = NodeStatus::Healthy;
            health.last_heartbeat = chrono::Utc::now();
        }

        // Publish agent started event
        #[cfg(feature = "hpc-channels")]
        self.event_bridge
            .publish_agent_started(&self.join_result.node_id.to_string());

        // Publish agent healthy event
        #[cfg(feature = "hpc-channels")]
        self.event_bridge
            .publish_agent_healthy(&self.join_result.node_id.to_string(), "Healthy");

        // Start background tasks
        let agent = Arc::new(self);
        let mut tasks = Vec::new();

        // Start heartbeat task
        tasks.push(tokio::spawn({
            let agent = agent.clone();
            async move { agent.heartbeat_loop().await }
        }));

        // Start health monitoring task
        tasks.push(tokio::spawn({
            let agent = agent.clone();
            async move { agent.health_monitor_loop().await }
        }));

        // Start workload management task
        tasks.push(tokio::spawn({
            let agent = agent.clone();
            async move { agent.workload_loop().await }
        }));

        // Start build job management task
        tasks.push(tokio::spawn({
            let agent = agent.clone();
            async move { agent.build_job_loop().await }
        }));

        // Start API server task
        tasks.push(tokio::spawn({
            let agent = agent.clone();
            async move { agent.api_server_loop().await }
        }));

        // Start hpc-channels build job listener (1-5µs latency from hpc-ci)
        #[cfg(feature = "hpc-channels")]
        tasks.push(tokio::spawn({
            let agent = agent.clone();
            async move { agent.hpc_channels_build_listener().await }
        }));

        // Start hpc-channels deploy listener (1-5µs latency from hpc-ci)
        #[cfg(feature = "hpc-channels")]
        tasks.push(tokio::spawn({
            let agent = agent.clone();
            async move { agent.hpc_channels_deploy_listener().await }
        }));

        // Start hpc-channels cancel listener (1-5µs latency from hpc-ci)
        #[cfg(feature = "hpc-channels")]
        tasks.push(tokio::spawn({
            let agent = agent.clone();
            async move { agent.hpc_channels_cancel_listener().await }
        }));

        // Wait for shutdown signal
        info!("Swarmlet agent running, waiting for shutdown signal...");

        let mut shutdown_signal = agent.shutdown_signal.clone();
        tokio::select! {
            _ = shutdown_signal.changed() => {
                info!("Shutdown signal received");
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Ctrl+C received, shutting down");
                let _ = agent.shutdown_sender.send(true);
            }
        }

        // Update status to shutting down
        {
            let mut health = agent.health_status.write().await;
            health.status = NodeStatus::Shutting;
        }

        // Publish agent shutdown event
        #[cfg(feature = "hpc-channels")]
        agent
            .event_bridge
            .publish_agent_shutdown(&agent.join_result.node_id.to_string(), "graceful shutdown");

        // Send final heartbeat
        if let Err(e) = agent.send_heartbeat().await {
            warn!("Failed to send final heartbeat: {}", e);
        }

        // Wait for tasks to complete with timeout
        let shutdown_timeout = Duration::from_secs(30);

        match tokio::time::timeout(shutdown_timeout, futures::future::join_all(tasks)).await {
            Ok(_) => info!("All tasks completed successfully"),
            Err(_) => warn!("Shutdown timeout reached, some tasks may not have completed"),
        }

        info!("Swarmlet agent shutdown complete");
        Ok(())
    }

    /// Send shutdown signal to the agent
    pub fn shutdown(&self) -> Result<()> {
        info!("Requesting agent shutdown");
        self.shutdown_sender.send(true).map_err(|_| {
            SwarmletError::AgentRuntime("Failed to send shutdown signal".to_string())
        })?;
        Ok(())
    }
}
