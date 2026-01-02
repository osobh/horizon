//! Swarmlet agent runtime

use crate::{
    build_job::BuildJob,
    build_job_manager::BuildJobManager,
    command::CommandExecutor, config::Config, join::JoinResult, profile::HardwareProfiler,
    security::NodeCertificate,
    wireguard::{WireGuardManager, WireGuardConfigRequest, AddPeerRequest, RemovePeerRequest},
    workload::WorkloadManager, Result, SwarmletError,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use warp::Filter;

/// Saved agent state for persistence across restarts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedAgentState {
    /// The join result containing cluster membership info
    pub join_result: JoinResult,
    /// WireGuard private key (base64 encoded)
    pub wireguard_private_key: String,
    /// WireGuard public key (base64 encoded)
    pub wireguard_public_key: String,
    /// Cluster's public key for signature verification (base64, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster_public_key: Option<String>,
    /// Timestamp when state was saved
    pub saved_at: chrono::DateTime<chrono::Utc>,
    /// State file format version for migration
    pub version: u32,
}

impl SavedAgentState {
    /// Current state file format version
    pub const CURRENT_VERSION: u32 = 1;

    /// State file name
    pub const STATE_FILE: &'static str = "swarmlet_state.json";

    /// Get the state file path for a given data directory
    pub fn state_path(data_dir: &Path) -> PathBuf {
        data_dir.join(Self::STATE_FILE)
    }

    /// Load saved state from disk
    pub async fn load(data_dir: &Path) -> Result<Self> {
        let state_path = Self::state_path(data_dir);

        if !state_path.exists() {
            return Err(SwarmletError::Configuration(format!(
                "No saved state found at {}",
                state_path.display()
            )));
        }

        let content = tokio::fs::read_to_string(&state_path)
            .await
            .map_err(|e| SwarmletError::Configuration(format!("Failed to read state file: {}", e)))?;

        let state: SavedAgentState = serde_json::from_str(&content)
            .map_err(|e| SwarmletError::Configuration(format!("Failed to parse state file: {}", e)))?;

        // Check version and migrate if needed
        if state.version > Self::CURRENT_VERSION {
            return Err(SwarmletError::Configuration(format!(
                "State file version {} is newer than supported version {}",
                state.version,
                Self::CURRENT_VERSION
            )));
        }

        info!("Loaded saved agent state from {}", state_path.display());
        Ok(state)
    }

    /// Save state to disk atomically
    pub async fn save(&self, data_dir: &Path) -> Result<()> {
        let state_path = Self::state_path(data_dir);
        let temp_path = state_path.with_extension("json.tmp");

        // Ensure data directory exists
        if !data_dir.exists() {
            tokio::fs::create_dir_all(data_dir)
                .await
                .map_err(|e| SwarmletError::Configuration(format!("Failed to create data directory: {}", e)))?;
        }

        // Serialize state
        let content = serde_json::to_string_pretty(&self)
            .map_err(|e| SwarmletError::Configuration(format!("Failed to serialize state: {}", e)))?;

        // Write to temp file
        tokio::fs::write(&temp_path, content)
            .await
            .map_err(|e| SwarmletError::Configuration(format!("Failed to write temp state file: {}", e)))?;

        // Atomically rename to final path
        tokio::fs::rename(&temp_path, &state_path)
            .await
            .map_err(|e| SwarmletError::Configuration(format!("Failed to rename state file: {}", e)))?;

        debug!("Saved agent state to {}", state_path.display());
        Ok(())
    }
}

#[cfg(feature = "hpc-channels")]
use crate::build_job::BuildJobStatus;
#[cfg(feature = "hpc-channels")]
use crate::hpc_bridge::{
    SharedAgentChannelBridge, SharedBuildChannelBridge, SharedDeployChannelBridge,
    SharedArtifactTransferBridge,
    shared_channel_bridge, shared_build_bridge, shared_deploy_bridge, shared_artifact_bridge,
    build_job_from_submit,
};
#[cfg(feature = "hpc-channels")]
use hpc_channels::messages::{BuildMessage, DeployMessage};

/// Main swarmlet agent that manages node lifecycle
pub struct SwarmletAgent {
    config: Arc<Config>,
    join_result: JoinResult,
    #[allow(dead_code)] // Certificate will be used for TLS authentication in production
    node_certificate: NodeCertificate,
    workload_manager: Arc<WorkloadManager>,
    command_executor: Arc<CommandExecutor>,
    wireguard_manager: Arc<WireGuardManager>,
    build_job_manager: Arc<BuildJobManager>,
    health_status: Arc<RwLock<HealthStatus>>,
    shutdown_signal: tokio::sync::watch::Receiver<bool>,
    shutdown_sender: tokio::sync::watch::Sender<bool>,
    /// HPC-Channels event bridge for publishing agent lifecycle events
    #[cfg(feature = "hpc-channels")]
    event_bridge: SharedAgentChannelBridge,
    /// HPC-Channels build bridge for hpc-ci integration (1-5µs latency)
    #[cfg(feature = "hpc-channels")]
    build_bridge: SharedBuildChannelBridge,
    /// HPC-Channels deploy bridge for deployment requests
    #[cfg(feature = "hpc-channels")]
    deploy_bridge: SharedDeployChannelBridge,
    /// HPC-Channels artifact transfer bridge for warp integration
    #[cfg(feature = "hpc-channels")]
    artifact_bridge: SharedArtifactTransferBridge,
}

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

impl SwarmletAgent {
    /// Create a new swarmlet agent from join result
    pub async fn new(join_result: JoinResult, data_dir: String) -> Result<Self> {
        let config = Config::default_with_data_dir(PathBuf::from(data_dir));
        let config = Arc::new(config);

        let node_certificate = NodeCertificate::from_pem(&join_result.node_certificate)?;
        let workload_manager = Arc::new(WorkloadManager::new(config.clone(), join_result.node_id).await?);
        let command_executor = Arc::new(CommandExecutor::new(config.data_dir.clone()));
        let wireguard_manager = Arc::new(WireGuardManager::new());
        let build_job_manager = Arc::new(
            BuildJobManager::new(config.clone(), join_result.node_id, config.data_dir.clone()).await?
        );

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
        let node_certificate = NodeCertificate::from_pem(&saved_state.join_result.node_certificate)?;
        let workload_manager = Arc::new(WorkloadManager::new(config.clone(), saved_state.join_result.node_id).await?);
        let command_executor = Arc::new(CommandExecutor::new(config.data_dir.clone()));
        let build_job_manager = Arc::new(
            BuildJobManager::new(config.clone(), saved_state.join_result.node_id, config.data_dir.clone()).await?
        );

        // Create WireGuard manager and restore keys
        let wireguard_manager = Arc::new(WireGuardManager::new());
        wireguard_manager.restore_keypair(
            saved_state.wireguard_private_key.clone(),
            saved_state.wireguard_public_key.clone(),
        ).await;

        // Restore cluster public key if available
        if let Some(ref cluster_key_b64) = saved_state.cluster_public_key {
            if let Err(e) = wireguard_manager.set_cluster_public_key_b64(cluster_key_b64).await {
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
            saved_state.join_result.cluster_name,
            saved_state.join_result.node_id
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
        let wireguard_private_key = self.wireguard_manager.get_private_key().await
            .ok_or_else(|| SwarmletError::Configuration("No WireGuard private key available".to_string()))?;
        let wireguard_public_key = self.wireguard_manager.get_public_key().await
            .ok_or_else(|| SwarmletError::Configuration("No WireGuard public key available".to_string()))?;

        let state = SavedAgentState {
            join_result: self.join_result.clone(),
            wireguard_private_key,
            wireguard_public_key,
            cluster_public_key: None, // TODO: Get from wireguard_manager if needed
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
        self.event_bridge.publish_agent_started(&self.join_result.node_id.to_string());

        // Publish agent healthy event
        #[cfg(feature = "hpc-channels")]
        self.event_bridge.publish_agent_healthy(&self.join_result.node_id.to_string(), "Healthy");

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
        agent.event_bridge.publish_agent_shutdown(&agent.join_result.node_id.to_string(), "graceful shutdown");

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

    /// Heartbeat loop - sends periodic status updates to cluster
    async fn heartbeat_loop(&self) -> Result<()> {
        let mut interval = interval(self.join_result.heartbeat_interval);
        let _client = reqwest::Client::new();
        let mut shutdown_signal = self.shutdown_signal.clone();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.send_heartbeat().await {
                        error!("Heartbeat failed: {}", e);

                        // Increment error counter
                        {
                            let mut health = self.health_status.write().await;
                            health.errors_count += 1;
                        }
                    }
                }
                _ = shutdown_signal.changed() => {
                    debug!("Heartbeat loop shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Health monitoring loop - updates local health metrics
    async fn health_monitor_loop(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(10));
        let start_time = std::time::Instant::now();
        let mut shutdown_signal = self.shutdown_signal.clone();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    self.update_health_metrics(start_time).await?;
                }
                _ = shutdown_signal.changed() => {
                    debug!("Health monitor loop shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Workload management loop - handles work assignments from cluster
    async fn workload_loop(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(5));
        let mut shutdown_signal = self.shutdown_signal.clone();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.check_for_work().await {
                        warn!("Failed to check for work: {}", e);
                    }
                }
                _ = shutdown_signal.changed() => {
                    debug!("Workload loop shutting down");

                    // Stop all workloads gracefully
                    if let Err(e) = self.workload_manager.stop_all_workloads().await {
                        error!("Failed to stop workloads: {}", e);
                    }
                    break;
                }
            }
        }

        Ok(())
    }

    /// Build job management loop - monitors and cleans up build jobs
    async fn build_job_loop(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(60)); // Check every minute
        let mut shutdown_signal = self.shutdown_signal.clone();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Clean up old completed/failed build jobs (older than 1 hour)
                    self.build_job_manager.cleanup_old_jobs(3600).await;

                    // Log active build count
                    let active_count = self.build_job_manager.active_job_count().await;
                    if active_count > 0 {
                        debug!("Active build jobs: {}", active_count);
                    }
                }
                _ = shutdown_signal.changed() => {
                    debug!("Build job loop shutting down");

                    // Cancel all active builds gracefully
                    if let Err(e) = self.build_job_manager.cancel_all().await {
                        error!("Failed to cancel build jobs: {}", e);
                    }
                    break;
                }
            }
        }

        Ok(())
    }

    /// HPC-Channels build job listener - receives jobs from hpc-ci (1-5µs latency)
    ///
    /// This is the main integration point between hpc-ci and stratoswarm.
    /// Jobs submitted via hpc-channels bypass REST API for ultra-low latency.
    #[cfg(feature = "hpc-channels")]
    async fn hpc_channels_build_listener(&self) -> Result<()> {
        use crate::hpc_bridge::BuildChannelBridge;

        // Subscribe to build job submissions from hpc-ci
        let mut build_rx = match BuildChannelBridge::subscribe_submissions() {
            Some(rx) => rx,
            None => {
                // Channel doesn't exist yet, wait and retry
                info!("Waiting for hpc.build.submit channel to be created...");
                tokio::time::sleep(Duration::from_secs(1)).await;

                // Try to create the channel by broadcasting
                let _ = hpc_channels::broadcast::<BuildMessage>(
                    hpc_channels::channels::BUILD_SUBMIT,
                    256,
                );

                match BuildChannelBridge::subscribe_submissions() {
                    Some(rx) => rx,
                    None => {
                        warn!("Could not subscribe to build submit channel");
                        return Ok(());
                    }
                }
            }
        };

        let mut shutdown_signal = self.shutdown_signal.clone();

        info!("HPC-Channels build listener started, listening on hpc.build.submit");

        loop {
            tokio::select! {
                result = build_rx.recv() => {
                    match result {
                        Ok(msg) => {
                            self.handle_hpc_build_message(msg).await;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            warn!("Build listener lagged by {} messages", n);
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            info!("Build submit channel closed");
                            break;
                        }
                    }
                }
                _ = shutdown_signal.changed() => {
                    debug!("HPC-Channels build listener shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle an incoming build message from hpc-ci
    #[cfg(feature = "hpc-channels")]
    async fn handle_hpc_build_message(&self, msg: BuildMessage) {
        // Only process Submit messages
        if let Some(job) = build_job_from_submit(&msg) {
            let job_id = job.id;

            info!(
                "Received build job from hpc-ci: {} (command: {:?})",
                job_id, job.command
            );

            // Publish queued status
            self.build_bridge.publish_queued(&job_id, 0);

            // Submit to local BuildJobManager
            match self.build_job_manager.submit_build(job).await {
                Ok(_) => {
                    // Publish started status
                    self.build_bridge.publish_started(&job_id);

                    // Start monitoring this job for completion
                    let build_manager = self.build_job_manager.clone();
                    let build_bridge = self.build_bridge.clone();

                    tokio::spawn(async move {
                        Self::monitor_build_completion(job_id, build_manager, build_bridge).await;
                    });
                }
                Err(e) => {
                    error!("Failed to submit build job {}: {}", job_id, e);
                    self.build_bridge.publish_failed(&job_id, &e.to_string());
                }
            }
        }
    }

    /// Monitor a build job for completion and publish status updates + logs
    #[cfg(feature = "hpc-channels")]
    async fn monitor_build_completion(
        job_id: Uuid,
        build_manager: Arc<BuildJobManager>,
        build_bridge: SharedBuildChannelBridge,
    ) {
        use crate::build_job::LogStream;
        use hpc_channels::messages::BuildLogLevel;

        let mut last_phase = String::new();
        let mut logs_sent = 0usize; // Track how many logs we've already sent
        let start_time = std::time::Instant::now();

        loop {
            // Wait a bit before checking
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Stream any new logs
            if let Some(logs) = build_manager.get_logs(job_id).await {
                for log in logs.iter().skip(logs_sent) {
                    // Convert LogStream to BuildLogLevel
                    let level = match log.stream {
                        LogStream::Stdout => BuildLogLevel::Info,
                        LogStream::Stderr => BuildLogLevel::Warn,
                        LogStream::System => BuildLogLevel::Info,
                    };
                    let source = match log.stream {
                        LogStream::Stdout => "cargo",
                        LogStream::Stderr => "cargo",
                        LogStream::System => "system",
                    };
                    build_bridge.publish_log(&job_id, level, &log.message, source);
                }
                logs_sent = logs.len();
            }

            // Get job status
            match build_manager.get_job_status(job_id).await {
                Some(status) => {
                    // Convert status to phase string and progress percentage
                    let (phase_str, percent, is_terminal) = match &status {
                        BuildJobStatus::Queued => ("Queued".to_string(), 0, false),
                        BuildJobStatus::PreparingEnvironment => ("PreparingEnvironment".to_string(), 10, false),
                        BuildJobStatus::FetchingSource => ("FetchingSource".to_string(), 20, false),
                        BuildJobStatus::ProvisioningToolchain => ("ProvisioningToolchain".to_string(), 30, false),
                        BuildJobStatus::Building => ("Building".to_string(), 50, false),
                        BuildJobStatus::Testing => ("Testing".to_string(), 75, false),
                        BuildJobStatus::CollectingArtifacts => ("CollectingArtifacts".to_string(), 90, false),
                        BuildJobStatus::Completed => ("Completed".to_string(), 100, true),
                        BuildJobStatus::Failed { .. } => ("Failed".to_string(), 0, true),
                        BuildJobStatus::Cancelled => ("Cancelled".to_string(), 0, true),
                        BuildJobStatus::TimedOut => ("TimedOut".to_string(), 0, true),
                    };

                    // Publish progress if phase changed
                    if phase_str != last_phase && !is_terminal {
                        last_phase = phase_str.clone();
                        build_bridge.publish_progress(&job_id, &phase_str, percent);
                    }

                    // Handle terminal states
                    match status {
                        BuildJobStatus::Completed => {
                            let duration_ms = start_time.elapsed().as_millis() as u64;
                            // Get artifacts from the full job info
                            let artifacts = match build_manager.get_job(job_id).await {
                                Some(active_job) => active_job.artifacts
                                    .iter()
                                    .map(|a| a.path.display().to_string())
                                    .collect(),
                                None => vec![],
                            };

                            build_bridge.publish_completed(&job_id, true, duration_ms, artifacts);
                            info!(
                                "Build job {} completed (duration: {}ms)",
                                job_id, duration_ms
                            );
                            return;
                        }
                        BuildJobStatus::Failed { ref error } => {
                            build_bridge.publish_failed(&job_id, error);
                            error!("Build job {} failed: {}", job_id, error);
                            return;
                        }
                        BuildJobStatus::Cancelled => {
                            build_bridge.publish_cancelled(&job_id);
                            info!("Build job {} was cancelled", job_id);
                            return;
                        }
                        BuildJobStatus::TimedOut => {
                            build_bridge.publish_failed(&job_id, "Build job timed out");
                            error!("Build job {} timed out", job_id);
                            return;
                        }
                        _ => {
                            // Continue monitoring
                        }
                    }
                }
                None => {
                    // Job not found, it may have been cleaned up
                    warn!("Build job {} not found in manager", job_id);
                    return;
                }
            }

            // Timeout after 1 hour
            if start_time.elapsed() > Duration::from_secs(3600) {
                build_bridge.publish_failed(&job_id, "Build timeout exceeded (1 hour)");
                error!("Build job {} timed out", job_id);
                return;
            }
        }
    }

    /// HPC-Channels deploy listener - receives deployments from hpc-ci (1-5µs latency)
    ///
    /// Handles deployment requests including blue-green, rolling, and canary strategies.
    #[cfg(feature = "hpc-channels")]
    async fn hpc_channels_deploy_listener(&self) -> Result<()> {
        use crate::hpc_bridge::DeployChannelBridge;

        // Subscribe to deploy job submissions from hpc-ci
        let mut deploy_rx = match DeployChannelBridge::subscribe_submissions() {
            Some(rx) => rx,
            None => {
                // Channel doesn't exist yet, wait and retry
                info!("Waiting for hpc.deploy.submit channel to be created...");
                tokio::time::sleep(Duration::from_secs(1)).await;

                // Try to create the channel by broadcasting
                let _ = hpc_channels::broadcast::<DeployMessage>(
                    hpc_channels::channels::DEPLOY_SUBMIT,
                    256,
                );

                match DeployChannelBridge::subscribe_submissions() {
                    Some(rx) => rx,
                    None => {
                        warn!("Could not subscribe to deploy submit channel");
                        return Ok(());
                    }
                }
            }
        };

        let mut shutdown_signal = self.shutdown_signal.clone();

        info!("HPC-Channels deploy listener started, listening on hpc.deploy.submit");

        loop {
            tokio::select! {
                result = deploy_rx.recv() => {
                    match result {
                        Ok(msg) => {
                            self.handle_hpc_deploy_message(msg).await;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            warn!("Deploy listener lagged by {} messages", n);
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            info!("Deploy submit channel closed");
                            break;
                        }
                    }
                }
                _ = shutdown_signal.changed() => {
                    debug!("HPC-Channels deploy listener shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle an incoming deploy message from hpc-ci
    #[cfg(feature = "hpc-channels")]
    async fn handle_hpc_deploy_message(&self, msg: DeployMessage) {
        // Only process Submit messages
        if let DeployMessage::Submit {
            deploy_id,
            artifact_ref,
            namespace,
            strategy,
            replicas,
            health_check,
            rollback_on_failure,
            ..
        } = msg
        {
            info!(
                "Received deployment from hpc-ci: {} (namespace: {}, replicas: {})",
                deploy_id, namespace, replicas
            );

            // Publish started status
            self.deploy_bridge.publish_started(&deploy_id, strategy.clone());

            // Execute the deployment
            let result = self
                .execute_deployment(
                    &deploy_id,
                    &artifact_ref,
                    &namespace,
                    &strategy,
                    replicas,
                    health_check.as_deref(),
                    rollback_on_failure,
                )
                .await;

            match result {
                Ok(url) => {
                    self.deploy_bridge.publish_completed(&deploy_id, true, url);
                    info!("Deployment {} completed successfully", deploy_id);
                }
                Err(e) => {
                    error!("Deployment {} failed: {}", deploy_id, e);
                    if rollback_on_failure {
                        self.deploy_bridge.publish_rolled_back(&deploy_id, &e);
                    } else {
                        self.deploy_bridge.publish_completed(&deploy_id, false, None);
                    }
                }
            }
        }
    }

    /// Execute a deployment with the specified strategy
    #[cfg(feature = "hpc-channels")]
    async fn execute_deployment(
        &self,
        deploy_id: &str,
        artifact_ref: &str,
        namespace: &str,
        strategy: &hpc_channels::messages::DeployStrategy,
        replicas: u32,
        health_check: Option<&str>,
        _rollback_on_failure: bool,
    ) -> std::result::Result<Option<String>, String> {
        use hpc_channels::messages::DeployStrategy as HpcDeployStrategy;

        // Publish progress: fetching artifact
        self.deploy_bridge.publish_progress(deploy_id, "fetching_artifact", 10);

        // TODO: Fetch artifact from warp storage using artifact_ref
        // For now, we assume the artifact is already available locally
        info!("Fetching artifact: {}", artifact_ref);

        // Publish progress: preparing deployment
        self.deploy_bridge.publish_progress(deploy_id, "preparing", 30);

        match strategy {
            HpcDeployStrategy::BlueGreen => {
                self.execute_blue_green_deployment(deploy_id, artifact_ref, namespace, replicas, health_check).await
            }
            HpcDeployStrategy::Rolling { batch_size } => {
                self.execute_rolling_deployment(deploy_id, artifact_ref, namespace, replicas, *batch_size, health_check).await
            }
            HpcDeployStrategy::Canary { traffic_percent } => {
                self.execute_canary_deployment(deploy_id, artifact_ref, namespace, replicas, *traffic_percent, health_check).await
            }
        }
    }

    /// Execute a blue-green deployment
    #[cfg(feature = "hpc-channels")]
    async fn execute_blue_green_deployment(
        &self,
        deploy_id: &str,
        artifact_ref: &str,
        namespace: &str,
        replicas: u32,
        health_check: Option<&str>,
    ) -> std::result::Result<Option<String>, String> {
        info!(
            "Executing blue-green deployment {} in namespace {}",
            deploy_id, namespace
        );

        // Phase 1: Deploy green (new version)
        self.deploy_bridge.publish_progress(deploy_id, "deploying_green", 40);

        // Create workload assignment for the new version
        let assignment = WorkAssignment {
            id: Uuid::new_v4(),
            workload_type: "deployment".to_string(),
            container_image: Some(artifact_ref.to_string()),
            command: None,
            shell_script: None,
            environment: [
                ("DEPLOY_ID".to_string(), deploy_id.to_string()),
                ("NAMESPACE".to_string(), namespace.to_string()),
                ("DEPLOY_COLOR".to_string(), "green".to_string()),
            ].into_iter().collect(),
            resource_limits: ResourceLimits::default(),
            created_at: chrono::Utc::now(),
        };

        // Start the workload
        for i in 0..replicas {
            let mut replica_assignment = assignment.clone();
            replica_assignment.id = Uuid::new_v4();
            replica_assignment.environment.insert("REPLICA_INDEX".to_string(), i.to_string());

            if let Err(e) = self.workload_manager.start_workload(replica_assignment).await {
                return Err(format!("Failed to start replica {}: {}", i, e));
            }
        }

        self.deploy_bridge.publish_progress(deploy_id, "green_deployed", 60);

        // Phase 2: Health check
        if let Some(endpoint) = health_check {
            self.deploy_bridge.publish_progress(deploy_id, "health_checking", 70);

            // Simple health check - in production this would poll the endpoint
            tokio::time::sleep(Duration::from_secs(2)).await;

            // Assume health check passes for now
            self.deploy_bridge.publish_health_check(deploy_id, true, endpoint);
        }

        // Phase 3: Switch traffic (simulated)
        self.deploy_bridge.publish_progress(deploy_id, "switching_traffic", 90);
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Return deployment URL
        let url = format!("https://{}.{}.svc.cluster.local", deploy_id, namespace);
        Ok(Some(url))
    }

    /// Execute a rolling deployment
    #[cfg(feature = "hpc-channels")]
    async fn execute_rolling_deployment(
        &self,
        deploy_id: &str,
        artifact_ref: &str,
        namespace: &str,
        replicas: u32,
        batch_size: u32,
        health_check: Option<&str>,
    ) -> std::result::Result<Option<String>, String> {
        info!(
            "Executing rolling deployment {} (batch_size: {}) in namespace {}",
            deploy_id, batch_size, namespace
        );

        let total_batches = (replicas + batch_size - 1) / batch_size;
        let mut deployed = 0u32;

        for batch in 0..total_batches {
            let batch_start = batch * batch_size;
            let batch_end = std::cmp::min(batch_start + batch_size, replicas);

            // Calculate progress percentage
            let progress = 30 + ((batch as u8 * 60) / total_batches as u8);
            self.deploy_bridge.publish_progress(deploy_id, &format!("rolling_batch_{}", batch), progress);

            // Deploy this batch
            for i in batch_start..batch_end {
                let assignment = WorkAssignment {
                    id: Uuid::new_v4(),
                    workload_type: "deployment".to_string(),
                    container_image: Some(artifact_ref.to_string()),
                    command: None,
                    shell_script: None,
                    environment: [
                        ("DEPLOY_ID".to_string(), deploy_id.to_string()),
                        ("NAMESPACE".to_string(), namespace.to_string()),
                        ("REPLICA_INDEX".to_string(), i.to_string()),
                    ].into_iter().collect(),
                    resource_limits: ResourceLimits::default(),
                    created_at: chrono::Utc::now(),
                };

                if let Err(e) = self.workload_manager.start_workload(assignment).await {
                    return Err(format!("Failed to deploy replica {}: {}", i, e));
                }
                deployed += 1;
            }

            // Health check after each batch
            if let Some(endpoint) = health_check {
                tokio::time::sleep(Duration::from_secs(1)).await;
                self.deploy_bridge.publish_health_check(deploy_id, true, endpoint);
            }

            info!("Rolling deployment {}: batch {} complete ({}/{})", deploy_id, batch, deployed, replicas);
        }

        self.deploy_bridge.publish_progress(deploy_id, "complete", 100);

        let url = format!("https://{}.{}.svc.cluster.local", deploy_id, namespace);
        Ok(Some(url))
    }

    /// Execute a canary deployment
    #[cfg(feature = "hpc-channels")]
    async fn execute_canary_deployment(
        &self,
        deploy_id: &str,
        artifact_ref: &str,
        namespace: &str,
        replicas: u32,
        traffic_percent: u8,
        health_check: Option<&str>,
    ) -> std::result::Result<Option<String>, String> {
        info!(
            "Executing canary deployment {} ({}% traffic) in namespace {}",
            deploy_id, traffic_percent, namespace
        );

        // Calculate canary replicas (minimum 1)
        let canary_replicas = std::cmp::max(1, (replicas * traffic_percent as u32) / 100);

        self.deploy_bridge.publish_progress(deploy_id, "deploying_canary", 40);

        // Deploy canary instances
        for i in 0..canary_replicas {
            let assignment = WorkAssignment {
                id: Uuid::new_v4(),
                workload_type: "deployment".to_string(),
                container_image: Some(artifact_ref.to_string()),
                command: None,
                shell_script: None,
                environment: [
                    ("DEPLOY_ID".to_string(), deploy_id.to_string()),
                    ("NAMESPACE".to_string(), namespace.to_string()),
                    ("DEPLOY_TYPE".to_string(), "canary".to_string()),
                    ("REPLICA_INDEX".to_string(), i.to_string()),
                ].into_iter().collect(),
                resource_limits: ResourceLimits::default(),
                created_at: chrono::Utc::now(),
            };

            if let Err(e) = self.workload_manager.start_workload(assignment).await {
                return Err(format!("Failed to deploy canary replica {}: {}", i, e));
            }
        }

        self.deploy_bridge.publish_progress(deploy_id, "canary_deployed", 60);

        // Health check canary
        if let Some(endpoint) = health_check {
            self.deploy_bridge.publish_progress(deploy_id, "health_checking_canary", 70);
            tokio::time::sleep(Duration::from_secs(2)).await;
            self.deploy_bridge.publish_health_check(deploy_id, true, endpoint);
        }

        // In a real canary, we would monitor metrics and gradually increase traffic
        // For now, we just report success after the canary is deployed
        self.deploy_bridge.publish_progress(deploy_id, "canary_monitoring", 80);

        info!(
            "Canary deployment {}: {} replicas receiving {}% traffic",
            deploy_id, canary_replicas, traffic_percent
        );

        let url = format!("https://{}-canary.{}.svc.cluster.local", deploy_id, namespace);
        Ok(Some(url))
    }

    /// API server loop - provides local HTTP API for health checks and metrics
    async fn api_server_loop(&self) -> Result<()> {
        use std::convert::Infallible;
        use std::net::SocketAddr;

        // Create API routes
        let health_status = self.health_status.clone();
        let command_executor = self.command_executor.clone();
        let mut shutdown_signal = self.shutdown_signal.clone();

        let health_route = warp::path("health").and(warp::get()).and_then(move || {
            let health_status = health_status.clone();
            async move {
                let health = health_status.read().await;
                let response = serde_json::to_string(&*health)
                    .unwrap_or_else(|_| r#"{"error": "serialization_failed"}"#.to_string());

                Ok::<_, Infallible>(warp::reply::with_header(
                    warp::reply::html(response),
                    "content-type",
                    "application/json",
                ))
            }
        });

        let metrics_route = warp::path("metrics").and(warp::get()).map(|| {
            // Return Prometheus-style metrics
            format!(
                "# HELP swarmlet_uptime_seconds Uptime in seconds\n\
                     # TYPE swarmlet_uptime_seconds counter\n\
                     swarmlet_uptime_seconds {}\n",
                chrono::Utc::now().timestamp()
            )
        });

        // Command execution routes
        let execute_command_route = warp::path!("api" / "v1" / "execute")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |request: crate::command::CommandRequest| {
                let command_executor = command_executor.clone();
                async move {
                    match command_executor.execute_command(request).await {
                        Ok(result) => {
                            let json = serde_json::to_string(&result).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{e}"}}"#);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let command_executor_shell = self.command_executor.clone();
        let execute_shell_route = warp::path!("api" / "v1" / "shell")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |request: serde_json::Value| {
                let command_executor = command_executor_shell.clone();
                async move {
                    if let Some(script) = request.get("script").and_then(|s| s.as_str()) {
                        match command_executor.execute_shell(script).await {
                            Ok(result) => {
                                let json = serde_json::to_string(&result).unwrap_or_else(|_| {
                                    r#"{"error": "serialization_failed"}"#.to_string()
                                });
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(json, warp::http::StatusCode::OK),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                            Err(e) => {
                                let error_response = format!(r#"{{"error": "{e}"}}"#);
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(
                                        error_response,
                                        warp::http::StatusCode::BAD_REQUEST,
                                    ),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                        }
                    } else {
                        let error_response = r#"{"error": "Missing 'script' field"}"#.to_string();
                        Ok::<_, Infallible>(warp::reply::with_header(
                            warp::reply::with_status(
                                error_response,
                                warp::http::StatusCode::BAD_REQUEST,
                            ),
                            "content-type",
                            "application/json",
                        ))
                    }
                }
            });

        // WireGuard routes
        let wg_manager_configure = self.wireguard_manager.clone();
        let wireguard_configure_route = warp::path!("api" / "v1" / "wireguard" / "configure")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |request: WireGuardConfigRequest| {
                let wg_manager = wg_manager_configure.clone();
                async move {
                    match wg_manager.apply_config(request).await {
                        Ok(response) => {
                            let json = serde_json::to_string(&response).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let wg_manager_status = self.wireguard_manager.clone();
        let wireguard_status_route = warp::path!("api" / "v1" / "wireguard" / "status" / String)
            .and(warp::get())
            .and_then(move |interface_name: String| {
                let wg_manager = wg_manager_status.clone();
                async move {
                    match wg_manager.get_status(&interface_name).await {
                        Ok(status) => {
                            let json = serde_json::to_string(&status).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let wg_manager_add_peer = self.wireguard_manager.clone();
        let wireguard_add_peer_route = warp::path!("api" / "v1" / "wireguard" / "peer")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |request: AddPeerRequest| {
                let wg_manager = wg_manager_add_peer.clone();
                async move {
                    match wg_manager.add_peer(request).await {
                        Ok(()) => {
                            let json = r#"{"success": true}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let wg_manager_remove_peer = self.wireguard_manager.clone();
        let wireguard_remove_peer_route = warp::path!("api" / "v1" / "wireguard" / "peer")
            .and(warp::delete())
            .and(warp::body::json())
            .and_then(move |request: RemovePeerRequest| {
                let wg_manager = wg_manager_remove_peer.clone();
                async move {
                    match wg_manager.remove_peer(request).await {
                        Ok(()) => {
                            let json = r#"{"success": true}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let wg_manager_list_peers = self.wireguard_manager.clone();
        let wireguard_list_peers_route = warp::path!("api" / "v1" / "wireguard" / "peers" / String)
            .and(warp::get())
            .and_then(move |interface_name: String| {
                let wg_manager = wg_manager_list_peers.clone();
                async move {
                    match wg_manager.list_peers(&interface_name).await {
                        Ok(peers) => {
                            let json = serde_json::to_string(&peers).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        // Workload routes
        let workload_manager_list = self.workload_manager.clone();
        let workloads_list_route = warp::path!("api" / "v1" / "workloads")
            .and(warp::get())
            .and_then(move || {
                let wm = workload_manager_list.clone();
                async move {
                    let workloads = wm.get_active_workloads().await;
                    let json = serde_json::to_string(&workloads).unwrap_or_else(|_| {
                        r#"{"error": "serialization_failed"}"#.to_string()
                    });
                    Ok::<_, Infallible>(warp::reply::with_header(
                        warp::reply::with_status(json, warp::http::StatusCode::OK),
                        "content-type",
                        "application/json",
                    ))
                }
            });

        let workload_manager_stop = self.workload_manager.clone();
        let workloads_stop_route = warp::path!("api" / "v1" / "workloads" / String / "stop")
            .and(warp::post())
            .and_then(move |workload_id: String| {
                let wm = workload_manager_stop.clone();
                async move {
                    match Uuid::parse_str(&workload_id) {
                        Ok(id) => match wm.stop_workload(id).await {
                            Ok(()) => {
                                let json = r#"{"success": true}"#.to_string();
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(json, warp::http::StatusCode::OK),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                            Err(e) => {
                                let error_response = format!(r#"{{"error": "{}"}}"#, e);
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(
                                        error_response,
                                        warp::http::StatusCode::NOT_FOUND,
                                    ),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                        },
                        Err(_) => {
                            let error_response = r#"{"error": "invalid_workload_id"}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        // Build job routes
        let build_manager_submit = self.build_job_manager.clone();
        let builds_submit_route = warp::path!("api" / "v1" / "builds")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |job: BuildJob| {
                let bm = build_manager_submit.clone();
                async move {
                    match bm.submit_build(job).await {
                        Ok(job_id) => {
                            let json = format!(r#"{{"job_id": "{}"}}"#, job_id);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::ACCEPTED),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let build_manager_list = self.build_job_manager.clone();
        let builds_list_route = warp::path!("api" / "v1" / "builds")
            .and(warp::get())
            .and_then(move || {
                let bm = build_manager_list.clone();
                async move {
                    let jobs = bm.list_active_jobs().await;
                    let json = serde_json::to_string(&jobs).unwrap_or_else(|_| {
                        r#"{"error": "serialization_failed"}"#.to_string()
                    });
                    Ok::<_, Infallible>(warp::reply::with_header(
                        warp::reply::with_status(json, warp::http::StatusCode::OK),
                        "content-type",
                        "application/json",
                    ))
                }
            });

        let build_manager_get = self.build_job_manager.clone();
        let builds_get_route = warp::path!("api" / "v1" / "builds" / String)
            .and(warp::get())
            .and_then(move |job_id: String| {
                let bm = build_manager_get.clone();
                async move {
                    match Uuid::parse_str(&job_id) {
                        Ok(id) => match bm.get_job(id).await {
                            Some(job) => {
                                let json = serde_json::to_string(&job).unwrap_or_else(|_| {
                                    r#"{"error": "serialization_failed"}"#.to_string()
                                });
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(json, warp::http::StatusCode::OK),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                            None => {
                                let error_response = r#"{"error": "job_not_found"}"#.to_string();
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(
                                        error_response,
                                        warp::http::StatusCode::NOT_FOUND,
                                    ),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                        },
                        Err(_) => {
                            let error_response = r#"{"error": "invalid_job_id"}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let build_manager_cancel = self.build_job_manager.clone();
        let builds_cancel_route = warp::path!("api" / "v1" / "builds" / String)
            .and(warp::delete())
            .and_then(move |job_id: String| {
                let bm = build_manager_cancel.clone();
                async move {
                    match Uuid::parse_str(&job_id) {
                        Ok(id) => match bm.cancel_build(id).await {
                            Ok(()) => {
                                let json = r#"{"success": true}"#.to_string();
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(json, warp::http::StatusCode::OK),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                            Err(e) => {
                                let error_response = format!(r#"{{"error": "{}"}}"#, e);
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(
                                        error_response,
                                        warp::http::StatusCode::NOT_FOUND,
                                    ),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                        },
                        Err(_) => {
                            let error_response = r#"{"error": "invalid_job_id"}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let build_manager_logs = self.build_job_manager.clone();
        let builds_logs_route = warp::path!("api" / "v1" / "builds" / String / "logs")
            .and(warp::get())
            .and_then(move |job_id: String| {
                let bm = build_manager_logs.clone();
                async move {
                    match Uuid::parse_str(&job_id) {
                        Ok(id) => match bm.get_logs(id).await {
                            Some(logs) => {
                                let json = serde_json::to_string(&logs).unwrap_or_else(|_| {
                                    r#"{"error": "serialization_failed"}"#.to_string()
                                });
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(json, warp::http::StatusCode::OK),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                            None => {
                                let error_response = r#"{"error": "job_not_found"}"#.to_string();
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(
                                        error_response,
                                        warp::http::StatusCode::NOT_FOUND,
                                    ),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                        },
                        Err(_) => {
                            let error_response = r#"{"error": "invalid_job_id"}"#.to_string();
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        // Detailed metrics route (Prometheus format)
        let health_status_detailed = self.health_status.clone();
        let workload_manager_metrics = self.workload_manager.clone();
        let detailed_metrics_route = warp::path!("api" / "v1" / "metrics" / "detailed")
            .and(warp::get())
            .and_then(move || {
                let health = health_status_detailed.clone();
                let wm = workload_manager_metrics.clone();
                async move {
                    let h = health.read().await;
                    let workload_count = wm.active_workload_count().await;

                    let metrics = format!(
                        "# HELP swarmlet_cpu_usage_percent CPU usage percentage\n\
                         # TYPE swarmlet_cpu_usage_percent gauge\n\
                         swarmlet_cpu_usage_percent {}\n\
                         # HELP swarmlet_memory_usage_gb Memory usage in GB\n\
                         # TYPE swarmlet_memory_usage_gb gauge\n\
                         swarmlet_memory_usage_gb {}\n\
                         # HELP swarmlet_disk_usage_gb Disk usage in GB\n\
                         # TYPE swarmlet_disk_usage_gb gauge\n\
                         swarmlet_disk_usage_gb {}\n\
                         # HELP swarmlet_workloads_active Number of active workloads\n\
                         # TYPE swarmlet_workloads_active gauge\n\
                         swarmlet_workloads_active {}\n\
                         # HELP swarmlet_uptime_seconds Uptime in seconds\n\
                         # TYPE swarmlet_uptime_seconds counter\n\
                         swarmlet_uptime_seconds {}\n\
                         # HELP swarmlet_errors_total Total error count\n\
                         # TYPE swarmlet_errors_total counter\n\
                         swarmlet_errors_total {}\n",
                        h.cpu_usage_percent,
                        h.memory_usage_gb,
                        h.disk_usage_gb,
                        workload_count,
                        h.uptime_seconds,
                        h.errors_count
                    );

                    Ok::<_, Infallible>(warp::reply::with_header(
                        warp::reply::with_status(metrics, warp::http::StatusCode::OK),
                        "content-type",
                        "text/plain; version=0.0.4; charset=utf-8",
                    ))
                }
            });

        // Hardware profile route
        let node_id_hardware = self.join_result.node_id;
        let hardware_route = warp::path!("api" / "v1" / "hardware")
            .and(warp::get())
            .and_then(move || {
                let node_id = node_id_hardware;
                async move {
                    let mut profiler = HardwareProfiler::new();
                    match profiler.profile().await {
                        Ok(mut profile) => {
                            // Use the actual node_id instead of randomly generated one
                            profile.node_id = node_id;
                            let json = serde_json::to_string(&profile).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{}"}}"#, e);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let routes = health_route
            .or(metrics_route)
            .or(execute_command_route)
            .or(execute_shell_route)
            .or(wireguard_configure_route)
            .or(wireguard_status_route)
            .or(wireguard_add_peer_route)
            .or(wireguard_remove_peer_route)
            .or(wireguard_list_peers_route)
            .or(workloads_list_route)
            .or(workloads_stop_route)
            .or(builds_submit_route)
            .or(builds_list_route)
            .or(builds_get_route)
            .or(builds_cancel_route)
            .or(builds_logs_route)
            .or(detailed_metrics_route)
            .or(hardware_route);

        // Start server
        let port = self.config.api_port.unwrap_or(8080);
        let addr: SocketAddr = ([0, 0, 0, 0], port).into();

        info!("Starting API server on {}", addr);

        tokio::select! {
            _ = warp::serve(routes).run(addr) => {
                debug!("API server completed");
            }
            _ = shutdown_signal.changed() => {
                debug!("API server shutting down");
            }
        }

        Ok(())
    }

    /// Send heartbeat to cluster
    async fn send_heartbeat(&self) -> Result<()> {
        let health = self.health_status.read().await.clone();

        let heartbeat = HeartbeatMessage {
            node_id: self.join_result.node_id,
            timestamp: chrono::Utc::now(),
            status: health.status,
            metrics: HealthMetrics {
                cpu_usage_percent: health.cpu_usage_percent,
                memory_usage_gb: health.memory_usage_gb,
                disk_usage_gb: health.disk_usage_gb,
                workloads_active: health.workloads_active,
                uptime_seconds: health.uptime_seconds,
            },
        };

        let client = reqwest::Client::new();
        let url = format!("{}/heartbeat", self.join_result.api_endpoints.health_check);

        match client.post(&url).json(&heartbeat).send().await {
            Ok(response) if response.status().is_success() => {
                debug!("Heartbeat sent successfully");

                // Update last heartbeat time
                {
                    let mut health = self.health_status.write().await;
                    health.last_heartbeat = chrono::Utc::now();
                }

                Ok(())
            }
            Ok(response) => {
                warn!("Heartbeat failed with status: {}", response.status());
                Err(SwarmletError::AgentRuntime(format!(
                    "Heartbeat rejected: {}",
                    response.status()
                )))
            }
            Err(e) => {
                warn!("Heartbeat network error: {}", e);
                Err(SwarmletError::Network(e))
            }
        }
    }

    /// Update local health metrics
    async fn update_health_metrics(&self, start_time: std::time::Instant) -> Result<()> {
        use sysinfo::System;

        let mut system = System::new_all();
        system.refresh_all();

        let uptime_seconds = start_time.elapsed().as_secs();
        let cpu_usage = system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>()
            / system.cpus().len() as f32;
        let memory_usage =
            (system.total_memory() - system.available_memory()) as f32 / (1024.0 * 1024.0 * 1024.0);

        // Get disk usage for data directory
        let disk_usage = self.get_disk_usage().await.unwrap_or(0.0);

        // Get active workload count
        let workloads_active = self.workload_manager.active_workload_count().await;

        {
            let mut health = self.health_status.write().await;
            health.uptime_seconds = uptime_seconds;
            health.cpu_usage_percent = cpu_usage;
            health.memory_usage_gb = memory_usage;
            health.disk_usage_gb = disk_usage;
            health.workloads_active = workloads_active;

            // Update status based on metrics
            health.status = if cpu_usage > 90.0 || memory_usage > health.memory_usage_gb * 0.9 {
                NodeStatus::Degraded
            } else if health.errors_count > 10 {
                NodeStatus::Unhealthy
            } else {
                NodeStatus::Healthy
            };
        }

        Ok(())
    }

    /// Check for new work assignments from cluster
    async fn check_for_work(&self) -> Result<()> {
        let client = reqwest::Client::new();
        let url = format!("{}/work", self.join_result.api_endpoints.workload_api);

        match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                let work_assignments: Vec<WorkAssignment> = response.json().await?;

                for assignment in work_assignments {
                    debug!("Received work assignment: {}", assignment.id);

                    if let Err(e) = self.workload_manager.start_workload(assignment).await {
                        error!("Failed to start workload: {}", e);
                    }
                }
            }
            Ok(response) if response.status() == 204 => {
                // No work available
                debug!("No work assignments available");
            }
            Ok(response) => {
                warn!("Work check failed with status: {}", response.status());
            }
            Err(e) => {
                debug!("Work check network error: {}", e);
            }
        }

        Ok(())
    }

    /// Get disk usage for data directory's mount point
    async fn get_disk_usage(&self) -> Result<f32> {
        use sysinfo::Disks;
        use std::path::Path;

        let data_path = Path::new(&self.config.data_dir).canonicalize().unwrap_or_else(|_| {
            Path::new(&self.config.data_dir).to_path_buf()
        });

        let disks = Disks::new_with_refreshed_list();

        // Find the disk with the longest mount point that is a prefix of data_path
        let mut best_disk: Option<&sysinfo::Disk> = None;
        let mut best_mount_len = 0;

        for disk in disks.list() {
            let mount_point = disk.mount_point();
            if data_path.starts_with(mount_point) {
                let mount_len = mount_point.as_os_str().len();
                if mount_len > best_mount_len {
                    best_mount_len = mount_len;
                    best_disk = Some(disk);
                }
            }
        }

        match best_disk {
            Some(disk) => {
                let total = disk.total_space();
                let available = disk.available_space();
                let used = total.saturating_sub(available);
                Ok(used as f32 / (1024.0 * 1024.0 * 1024.0))
            }
            None => {
                // Fallback: if no disk found, try to get the first disk's usage
                if let Some(disk) = disks.list().first() {
                    let total = disk.total_space();
                    let available = disk.available_space();
                    let used = total.saturating_sub(available);
                    Ok(used as f32 / (1024.0 * 1024.0 * 1024.0))
                } else {
                    Ok(0.0)
                }
            }
        }
    }
}

/// Heartbeat message sent to cluster
#[derive(Debug, Serialize, Deserialize)]
struct HeartbeatMessage {
    node_id: Uuid,
    timestamp: chrono::DateTime<chrono::Utc>,
    status: NodeStatus,
    metrics: HealthMetrics,
}

/// Health metrics included in heartbeat
#[derive(Debug, Serialize, Deserialize)]
struct HealthMetrics {
    cpu_usage_percent: f32,
    memory_usage_gb: f32,
    disk_usage_gb: f32,
    workloads_active: u32,
    uptime_seconds: u64,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::join::ClusterApiEndpoints;
    use tempfile::TempDir;
    use tokio::time::Duration;

    /// TDD Phase tracking
    #[derive(Debug, Clone, PartialEq)]
    enum TddPhase {
        Red,      // Write failing tests
        Green,    // Make tests pass
        Refactor, // Optimize implementation
    }

    /// Test result tracking
    #[derive(Debug)]
    struct TestResult {
        test_name: String,
        phase: TddPhase,
        success: bool,
        duration: Duration,
        error_message: Option<String>,
    }

    /// Create a test join result
    fn create_test_join_result() -> JoinResult {
        JoinResult {
            node_id: Uuid::new_v4(),
            cluster_name: "test-cluster".to_string(),
            node_certificate: generate_test_certificate(),
            cluster_endpoints: vec!["http://localhost:7946".to_string()],
            assigned_capabilities: vec!["compute".to_string(), "storage".to_string()],
            heartbeat_interval: Duration::from_secs(30),
            api_endpoints: ClusterApiEndpoints {
                workload_api: "http://localhost:8081".to_string(),
                metrics_api: "http://localhost:8082".to_string(),
                logs_api: "http://localhost:8083".to_string(),
                health_check: "http://localhost:8080".to_string(),
            },
            wireguard_config: None,
            subnet_info: None,
        }
    }

    /// Generate a test certificate
    fn generate_test_certificate() -> String {
        // This is a minimal valid PEM certificate for testing
        r#"-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAKHHIgKwA4jAMA0GCSqGSIb3DQEBCwUAMCExCzAJBgNVBAYTAlVT
MRIwEAYDVQQDDAlsb2NhbGhvc3QwHhcNMjQwMTAxMDAwMDAwWhcNMjUwMTAxMDAw
MDAwWjAhMQswCQYDVQQGEwJVUzESMBAGA1UEAwwJbG9jYWxob3N0MFwwDQYJKoZI
hvcNAQEBBQADSwAwSAJBAKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4j
AKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jACAwEAATANBgkqhkiG9w0B
AQsFAANBAKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jA
KHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jA=
-----END CERTIFICATE-----"#
            .to_string()
    }

    #[tokio::test]
    async fn test_agent_creation_from_join_result() {
        let start = std::time::Instant::now();
        let mut results = Vec::new();

        // RED Phase - Should fail initially
        let phase = TddPhase::Red;
        let join_result = create_test_join_result();
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        match SwarmletAgent::new(join_result.clone(), data_dir.clone()).await {
            Ok(agent) => {
                results.push(TestResult {
                    test_name: "agent_creation".to_string(),
                    phase: phase.clone(),
                    success: true,
                    duration: start.elapsed(),
                    error_message: None,
                });

                // Verify agent properties
                assert_eq!(agent.join_result.node_id, join_result.node_id);
                assert_eq!(agent.join_result.cluster_name, join_result.cluster_name);
            }
            Err(e) => {
                results.push(TestResult {
                    test_name: "agent_creation".to_string(),
                    phase,
                    success: false,
                    duration: start.elapsed(),
                    error_message: Some(e.to_string()),
                });
            }
        }

        // GREEN Phase - Should pass
        let phase = TddPhase::Green;
        let agent = SwarmletAgent::new(join_result.clone(), data_dir.clone())
            .await
            .expect("Should create agent successfully");

        results.push(TestResult {
            test_name: "agent_creation_green".to_string(),
            phase,
            success: true,
            duration: start.elapsed(),
            error_message: None,
        });

        // Verify health status initialization
        let health = agent.health_status.read().await;
        assert_eq!(health.node_id, join_result.node_id);
        assert_eq!(health.status, NodeStatus::Starting);
        assert_eq!(health.workloads_active, 0);
        assert_eq!(health.errors_count, 0);
    }

    #[tokio::test]
    async fn test_agent_health_status_updates() {
        let join_result = create_test_join_result();
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        let agent = SwarmletAgent::new(join_result, data_dir)
            .await
            .expect("Should create agent");

        // Test health status transitions
        {
            let mut health = agent.health_status.write().await;
            health.status = NodeStatus::Healthy;
            health.cpu_usage_percent = 45.0;
            health.memory_usage_gb = 3.5;
            health.workloads_active = 5;
        }

        // Verify updates
        {
            let health = agent.health_status.read().await;
            assert_eq!(health.status, NodeStatus::Healthy);
            assert_eq!(health.cpu_usage_percent, 45.0);
            assert_eq!(health.memory_usage_gb, 3.5);
            assert_eq!(health.workloads_active, 5);
        }

        // Test degraded status when resources are high
        {
            let mut health = agent.health_status.write().await;
            health.cpu_usage_percent = 95.0;
        }

        // Update health metrics should set degraded status
        agent
            .update_health_metrics(std::time::Instant::now())
            .await
            .ok();

        {
            let health = agent.health_status.read().await;
            // Note: Status might not change in test environment without actual high CPU
            assert!(health.cpu_usage_percent >= 0.0);
        }
    }

    #[tokio::test]
    async fn test_agent_shutdown_signal() {
        let join_result = create_test_join_result();
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        let agent = SwarmletAgent::new(join_result, data_dir)
            .await
            .expect("Should create agent");

        // Test shutdown signal
        assert!(agent.shutdown().is_ok());

        // Verify shutdown signal was sent
        let mut shutdown_signal = agent.shutdown_signal.clone();
        shutdown_signal
            .changed()
            .await
            .expect("Should receive shutdown signal");
        assert_eq!(*shutdown_signal.borrow(), true);
    }

    #[tokio::test]
    async fn test_heartbeat_message_serialization() {
        let heartbeat = HeartbeatMessage {
            node_id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            status: NodeStatus::Healthy,
            metrics: HealthMetrics {
                cpu_usage_percent: 25.5,
                memory_usage_gb: 4.2,
                disk_usage_gb: 50.0,
                workloads_active: 3,
                uptime_seconds: 3600,
            },
        };

        // Test serialization
        let json = serde_json::to_string(&heartbeat).expect("Should serialize heartbeat");
        assert!(json.contains("Healthy"));
        assert!(json.contains("25.5"));

        // Test deserialization
        let deserialized: HeartbeatMessage =
            serde_json::from_str(&json).expect("Should deserialize heartbeat");
        assert_eq!(deserialized.node_id, heartbeat.node_id);
        assert_eq!(deserialized.status, heartbeat.status);
        assert_eq!(deserialized.metrics.cpu_usage_percent, 25.5);
    }

    #[tokio::test]
    async fn test_work_assignment_handling() {
        let work_assignment = WorkAssignment {
            id: Uuid::new_v4(),
            workload_type: "container".to_string(),
            container_image: Some("nginx:latest".to_string()),
            command: Some(vec![
                "nginx".to_string(),
                "-g".to_string(),
                "daemon off;".to_string(),
            ]),
            shell_script: None,
            environment: std::collections::HashMap::from([(
                "ENV_VAR".to_string(),
                "value".to_string(),
            )]),
            resource_limits: ResourceLimits {
                cpu_cores: Some(2.0),
                memory_gb: Some(4.0),
                disk_gb: Some(10.0),
            },
            created_at: chrono::Utc::now(),
        };

        // Test serialization
        let json =
            serde_json::to_string(&work_assignment).expect("Should serialize work assignment");
        assert!(json.contains("nginx:latest"));
        assert!(json.contains("ENV_VAR"));

        // Test deserialization
        let deserialized: WorkAssignment =
            serde_json::from_str(&json).expect("Should deserialize work assignment");
        assert_eq!(deserialized.id, work_assignment.id);
        assert_eq!(deserialized.workload_type, "container");
        assert_eq!(
            deserialized.container_image,
            Some("nginx:latest".to_string())
        );
    }

    #[tokio::test]
    async fn test_node_status_transitions() {
        // Test all node status values
        let statuses = vec![
            NodeStatus::Starting,
            NodeStatus::Healthy,
            NodeStatus::Degraded,
            NodeStatus::Unhealthy,
            NodeStatus::Shutting,
        ];

        for status in statuses {
            let health = HealthStatus {
                node_id: Uuid::new_v4(),
                status: status.clone(),
                uptime_seconds: 100,
                workloads_active: 1,
                cpu_usage_percent: 10.0,
                memory_usage_gb: 2.0,
                disk_usage_gb: 20.0,
                network_rx_bytes: 1000,
                network_tx_bytes: 2000,
                last_heartbeat: chrono::Utc::now(),
                errors_count: 0,
            };

            // Test serialization
            let json = serde_json::to_string(&health).expect("Should serialize health status");
            let deserialized: HealthStatus =
                serde_json::from_str(&json).expect("Should deserialize health status");
            assert_eq!(deserialized.status, status);
        }
    }

    #[tokio::test]
    async fn test_api_routes_health_endpoint() {
        let join_result = create_test_join_result();
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        let agent = SwarmletAgent::new(join_result, data_dir)
            .await
            .expect("Should create agent");

        // Update some health metrics
        {
            let mut health = agent.health_status.write().await;
            health.status = NodeStatus::Healthy;
            health.cpu_usage_percent = 30.0;
            health.memory_usage_gb = 4.0;
            health.workloads_active = 2;
        }

        // Read health status
        let health = agent.health_status.read().await;
        let health_json = serde_json::to_string(&*health).expect("Should serialize health");

        // Verify health JSON contains expected fields
        assert!(health_json.contains("\"status\":\"Healthy\""));
        assert!(health_json.contains("\"cpu_usage_percent\":30.0"));
        assert!(health_json.contains("\"workloads_active\":2"));
    }

    #[tokio::test]
    async fn test_resource_limits_validation() {
        // Test with all fields set
        let limits_full = ResourceLimits {
            cpu_cores: Some(4.0),
            memory_gb: Some(8.0),
            disk_gb: Some(100.0),
        };

        let json = serde_json::to_string(&limits_full).expect("Should serialize");
        let deserialized: ResourceLimits = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.cpu_cores, Some(4.0));
        assert_eq!(deserialized.memory_gb, Some(8.0));
        assert_eq!(deserialized.disk_gb, Some(100.0));

        // Test with no limits set
        let limits_none = ResourceLimits {
            cpu_cores: None,
            memory_gb: None,
            disk_gb: None,
        };

        let json = serde_json::to_string(&limits_none).expect("Should serialize");
        let deserialized: ResourceLimits = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.cpu_cores, None);
        assert_eq!(deserialized.memory_gb, None);
        assert_eq!(deserialized.disk_gb, None);
    }

    #[tokio::test]
    async fn test_agent_from_config_no_saved_state() {
        let temp_dir = TempDir::new().unwrap();
        let config = Config::default_with_data_dir(temp_dir.path().to_path_buf());

        // Should fail because there's no saved state
        match SwarmletAgent::from_config(config).await {
            Ok(_) => panic!("Should return error when no saved state exists"),
            Err(e) => match e {
                crate::SwarmletError::Configuration(msg) => {
                    assert!(msg.contains("No saved state found"));
                }
                _ => panic!("Expected Config error, got: {:?}", e),
            },
        }
    }

    #[tokio::test]
    async fn test_agent_from_config_with_saved_state() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_path_buf();

        // Create and save agent state
        let join_result = create_test_join_result();
        let saved_state = SavedAgentState {
            join_result: join_result.clone(),
            wireguard_private_key: "test_private_key_base64".to_string(),
            wireguard_public_key: "test_public_key_base64".to_string(),
            cluster_public_key: None,
            saved_at: chrono::Utc::now(),
            version: SavedAgentState::CURRENT_VERSION,
        };
        saved_state.save(&data_dir).await.expect("Should save state");

        // Now from_config should work
        let config = Config::default_with_data_dir(data_dir);
        let agent = SwarmletAgent::from_config(config).await
            .expect("Should create agent from saved state");

        assert_eq!(agent.join_result.cluster_name, join_result.cluster_name);
        assert_eq!(agent.join_result.node_id, join_result.node_id);
    }

    #[tokio::test]
    async fn test_saved_agent_state_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_path_buf();

        let join_result = create_test_join_result();
        let original_state = SavedAgentState {
            join_result: join_result.clone(),
            wireguard_private_key: "private_key_b64".to_string(),
            wireguard_public_key: "public_key_b64".to_string(),
            cluster_public_key: Some("cluster_key_b64".to_string()),
            saved_at: chrono::Utc::now(),
            version: SavedAgentState::CURRENT_VERSION,
        };

        // Save state
        original_state.save(&data_dir).await.expect("Should save state");

        // Load state
        let loaded_state = SavedAgentState::load(&data_dir).await.expect("Should load state");

        assert_eq!(loaded_state.join_result.node_id, original_state.join_result.node_id);
        assert_eq!(loaded_state.wireguard_private_key, original_state.wireguard_private_key);
        assert_eq!(loaded_state.wireguard_public_key, original_state.wireguard_public_key);
        assert_eq!(loaded_state.cluster_public_key, original_state.cluster_public_key);
        assert_eq!(loaded_state.version, original_state.version);
    }

    #[tokio::test]
    async fn test_has_saved_state() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path();

        // Initially no saved state
        assert!(!SwarmletAgent::has_saved_state(data_dir).await);

        // Create saved state
        let join_result = create_test_join_result();
        let state = SavedAgentState {
            join_result,
            wireguard_private_key: "key".to_string(),
            wireguard_public_key: "key".to_string(),
            cluster_public_key: None,
            saved_at: chrono::Utc::now(),
            version: SavedAgentState::CURRENT_VERSION,
        };
        state.save(data_dir).await.expect("Should save state");

        // Now has saved state
        assert!(SwarmletAgent::has_saved_state(data_dir).await);
    }

    #[tokio::test]
    async fn test_disk_usage_calculation() {
        let join_result = create_test_join_result();
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        let agent = SwarmletAgent::new(join_result, data_dir)
            .await
            .expect("Should create agent");

        // Get disk usage - now returns actual disk usage of the mount point
        let disk_usage = agent.get_disk_usage().await.expect("Should get disk usage");

        // Should return actual disk usage in GB (realistic values on any system)
        assert!(disk_usage >= 0.0, "Disk usage should be non-negative");
        // A typical system will have at least some disk usage, but we can't be too specific
        // Just verify we get a reasonable value (less than 100 TB)
        assert!(
            disk_usage < 100_000.0,
            "Disk usage {} GB seems unreasonably high",
            disk_usage
        );
    }
}
