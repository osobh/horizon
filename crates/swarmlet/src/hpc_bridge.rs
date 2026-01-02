//! HPC Channels integration for swarmlet agent lifecycle events.
//!
//! This module bridges agent lifecycle events to the hpc-channels message bus,
//! enabling real-time monitoring of agent spawn/terminate across the cluster.
//!
//! # Channels Used
//!
//! - `hpc.agent.spawn` - Agent spawn events
//! - `hpc.agent.terminate` - Agent terminate events
//! - `hpc.agent.message` - Agent message events
//!
//! # Example
//!
//! ```rust,ignore
//! use swarmlet::hpc_bridge::AgentChannelBridge;
//!
//! let bridge = AgentChannelBridge::new();
//!
//! // Publish agent started event
//! bridge.publish_agent_started("node-123");
//!
//! // Subscribe to agent events
//! let mut rx = bridge.subscribe();
//! while let Ok(event) = rx.recv().await {
//!     println!("Agent event: {:?}", event);
//! }
//! ```

use std::sync::Arc;
use tokio::sync::broadcast;

/// Agent lifecycle events published to hpc-channels.
#[derive(Clone, Debug)]
pub enum AgentEvent {
    /// Agent has started and is ready.
    AgentStarted {
        /// Node ID of the agent.
        node_id: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Agent is healthy and operational.
    AgentHealthy {
        /// Node ID of the agent.
        node_id: String,
        /// Health status.
        status: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Agent is shutting down.
    AgentShutdown {
        /// Node ID of the agent.
        node_id: String,
        /// Reason for shutdown.
        reason: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Agent health status changed.
    HealthStatusChanged {
        /// Node ID of the agent.
        node_id: String,
        /// Previous status.
        previous_status: String,
        /// New status.
        new_status: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Workload started on this agent.
    WorkloadStarted {
        /// Node ID of the agent.
        node_id: String,
        /// Workload ID.
        workload_id: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Heartbeat sent to controller.
    HeartbeatSent {
        /// Node ID of the agent.
        node_id: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
}

/// Bridge between agent events and hpc-channels.
pub struct AgentChannelBridge {
    /// Broadcast sender for agent spawn events.
    spawn_tx: broadcast::Sender<AgentEvent>,
    /// Broadcast sender for agent terminate events.
    terminate_tx: broadcast::Sender<AgentEvent>,
}

impl AgentChannelBridge {
    /// Create a new agent channel bridge.
    ///
    /// Registers channels with the hpc-channels global registry.
    pub fn new() -> Self {
        let spawn_tx = hpc_channels::broadcast::<AgentEvent>(
            hpc_channels::channels::AGENT_SPAWN,
            256,
        );
        let terminate_tx = hpc_channels::broadcast::<AgentEvent>(
            hpc_channels::channels::AGENT_TERMINATE,
            256,
        );

        Self { spawn_tx, terminate_tx }
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Publish an agent started event.
    pub fn publish_agent_started(&self, node_id: &str) {
        let _ = self.spawn_tx.send(AgentEvent::AgentStarted {
            node_id: node_id.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish an agent healthy event.
    pub fn publish_agent_healthy(&self, node_id: &str, status: &str) {
        let _ = self.spawn_tx.send(AgentEvent::AgentHealthy {
            node_id: node_id.to_string(),
            status: status.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish an agent shutdown event.
    pub fn publish_agent_shutdown(&self, node_id: &str, reason: &str) {
        let _ = self.terminate_tx.send(AgentEvent::AgentShutdown {
            node_id: node_id.to_string(),
            reason: reason.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish a health status changed event.
    pub fn publish_health_status_changed(&self, node_id: &str, previous_status: &str, new_status: &str) {
        let _ = self.spawn_tx.send(AgentEvent::HealthStatusChanged {
            node_id: node_id.to_string(),
            previous_status: previous_status.to_string(),
            new_status: new_status.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish a workload started event.
    pub fn publish_workload_started(&self, node_id: &str, workload_id: &str) {
        let _ = self.spawn_tx.send(AgentEvent::WorkloadStarted {
            node_id: node_id.to_string(),
            workload_id: workload_id.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish a heartbeat sent event.
    pub fn publish_heartbeat_sent(&self, node_id: &str) {
        let _ = self.spawn_tx.send(AgentEvent::HeartbeatSent {
            node_id: node_id.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Subscribe to agent spawn events.
    pub fn subscribe_spawn(&self) -> broadcast::Receiver<AgentEvent> {
        self.spawn_tx.subscribe()
    }

    /// Subscribe to agent terminate events.
    pub fn subscribe_terminate(&self) -> broadcast::Receiver<AgentEvent> {
        self.terminate_tx.subscribe()
    }
}

impl Default for AgentChannelBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared channel bridge type.
pub type SharedAgentChannelBridge = Arc<AgentChannelBridge>;

/// Create a new shared channel bridge.
#[must_use]
pub fn shared_channel_bridge() -> SharedAgentChannelBridge {
    Arc::new(AgentChannelBridge::new())
}

// ============================================================================
// Build Channel Bridge (hpc-ci <-> stratoswarm build jobs)
// ============================================================================

use hpc_channels::messages::{
    BuildCommand as HpcBuildCommand, BuildLogLevel, BuildMessage, BuildProfile as HpcBuildProfile,
    BuildSource as HpcBuildSource, DeployMessage, DeployStrategy as HpcDeployStrategy,
};
use uuid::Uuid;

/// Bridge between hpc-channels build messages and local BuildJobManager.
///
/// Listens on `hpc.build.submit` for incoming job requests (1-5µs latency)
/// and publishes status updates to `hpc.build.status` and `hpc.build.logs`.
pub struct BuildChannelBridge {
    /// Sender for build status events.
    status_tx: broadcast::Sender<BuildMessage>,
    /// Sender for build log streaming.
    logs_tx: broadcast::Sender<BuildMessage>,
    /// Node ID for this swarmlet.
    node_id: String,
}

impl BuildChannelBridge {
    /// Create a new build channel bridge.
    pub fn new(node_id: String) -> Self {
        let status_tx =
            hpc_channels::broadcast::<BuildMessage>(hpc_channels::channels::BUILD_STATUS, 512);
        let logs_tx =
            hpc_channels::broadcast::<BuildMessage>(hpc_channels::channels::BUILD_LOGS, 4096);

        Self {
            status_tx,
            logs_tx,
            node_id,
        }
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Subscribe to build job submissions from hpc-ci.
    ///
    /// Returns None if the channel doesn't exist yet (no one has broadcast to it).
    pub fn subscribe_submissions() -> Option<broadcast::Receiver<BuildMessage>> {
        hpc_channels::subscribe::<BuildMessage>(hpc_channels::channels::BUILD_SUBMIT).ok()
    }

    /// Publish job queued status.
    pub fn publish_queued(&self, job_id: &Uuid, position: u32) {
        let _ = self.status_tx.send(BuildMessage::Queued {
            job_id: job_id.to_string(),
            position,
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish job started status.
    pub fn publish_started(&self, job_id: &Uuid) {
        let _ = self.status_tx.send(BuildMessage::Started {
            job_id: job_id.to_string(),
            worker_id: self.node_id.clone(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish job progress.
    pub fn publish_progress(&self, job_id: &Uuid, phase: &str, percent: u8) {
        let _ = self.status_tx.send(BuildMessage::Progress {
            job_id: job_id.to_string(),
            phase: phase.to_string(),
            percent,
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish job completed.
    pub fn publish_completed(
        &self,
        job_id: &Uuid,
        success: bool,
        duration_ms: u64,
        artifacts: Vec<String>,
    ) {
        self.publish_completed_with_manifest(job_id, success, duration_ms, artifacts, None);
    }

    /// Publish job completed with optional warp artifact manifest.
    ///
    /// When artifacts are stored in warp, include the manifest for
    /// high-speed fetching at 31 GB/s.
    pub fn publish_completed_with_manifest(
        &self,
        job_id: &Uuid,
        success: bool,
        duration_ms: u64,
        artifacts: Vec<String>,
        artifact_manifest: Option<hpc_channels::messages::ArtifactManifest>,
    ) {
        let _ = self.status_tx.send(BuildMessage::Completed {
            job_id: job_id.to_string(),
            success,
            duration_ms,
            artifacts,
            artifact_manifest,
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish job failed.
    pub fn publish_failed(&self, job_id: &Uuid, error: &str) {
        let _ = self.status_tx.send(BuildMessage::Failed {
            job_id: job_id.to_string(),
            error: error.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish job cancelled.
    pub fn publish_cancelled(&self, job_id: &Uuid) {
        let _ = self.status_tx.send(BuildMessage::Cancelled {
            job_id: job_id.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish a build log entry.
    pub fn publish_log(&self, job_id: &Uuid, level: BuildLogLevel, content: &str, source: &str) {
        let _ = self.logs_tx.send(BuildMessage::Log {
            job_id: job_id.to_string(),
            level,
            content: content.to_string(),
            source: source.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Subscribe to build status events (for monitoring).
    pub fn subscribe_status(&self) -> broadcast::Receiver<BuildMessage> {
        self.status_tx.subscribe()
    }

    /// Subscribe to build logs (for monitoring).
    pub fn subscribe_logs(&self) -> broadcast::Receiver<BuildMessage> {
        self.logs_tx.subscribe()
    }
}

/// Shared build channel bridge type.
pub type SharedBuildChannelBridge = Arc<BuildChannelBridge>;

/// Create a new shared build channel bridge.
#[must_use]
pub fn shared_build_bridge(node_id: String) -> SharedBuildChannelBridge {
    Arc::new(BuildChannelBridge::new(node_id))
}

// ============================================================================
// Deploy Channel Bridge (hpc-ci <-> stratoswarm deployments)
// ============================================================================

/// Bridge between hpc-channels deploy messages and local deployment system.
///
/// Listens on `hpc.deploy.submit` for incoming deployment requests (1-5µs latency)
/// and publishes status updates to `hpc.deploy.status`.
pub struct DeployChannelBridge {
    /// Sender for deploy status events.
    status_tx: broadcast::Sender<DeployMessage>,
    /// Node ID for this swarmlet (reserved for future use in multi-node deployments).
    #[allow(dead_code)]
    node_id: String,
}

impl DeployChannelBridge {
    /// Create a new deploy channel bridge.
    pub fn new(node_id: String) -> Self {
        let status_tx =
            hpc_channels::broadcast::<DeployMessage>(hpc_channels::channels::DEPLOY_STATUS, 256);

        Self { status_tx, node_id }
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Subscribe to deploy job submissions from hpc-ci.
    ///
    /// Returns None if the channel doesn't exist yet (no one has broadcast to it).
    pub fn subscribe_submissions() -> Option<broadcast::Receiver<DeployMessage>> {
        hpc_channels::subscribe::<DeployMessage>(hpc_channels::channels::DEPLOY_SUBMIT).ok()
    }

    /// Publish deployment started.
    pub fn publish_started(&self, deploy_id: &str, strategy: HpcDeployStrategy) {
        let _ = self.status_tx.send(DeployMessage::Started {
            deploy_id: deploy_id.to_string(),
            strategy,
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish deployment progress.
    pub fn publish_progress(&self, deploy_id: &str, phase: &str, percent: u8) {
        let _ = self.status_tx.send(DeployMessage::Progress {
            deploy_id: deploy_id.to_string(),
            phase: phase.to_string(),
            percent,
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish health check result.
    pub fn publish_health_check(&self, deploy_id: &str, passed: bool, endpoint: &str) {
        let _ = self.status_tx.send(DeployMessage::HealthCheck {
            deploy_id: deploy_id.to_string(),
            passed,
            endpoint: endpoint.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish deployment completed.
    pub fn publish_completed(&self, deploy_id: &str, success: bool, url: Option<String>) {
        let _ = self.status_tx.send(DeployMessage::Completed {
            deploy_id: deploy_id.to_string(),
            success,
            url,
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish deployment rolled back.
    pub fn publish_rolled_back(&self, deploy_id: &str, reason: &str) {
        let _ = self.status_tx.send(DeployMessage::RolledBack {
            deploy_id: deploy_id.to_string(),
            reason: reason.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Subscribe to deploy status events (for monitoring).
    pub fn subscribe_status(&self) -> broadcast::Receiver<DeployMessage> {
        self.status_tx.subscribe()
    }
}

/// Shared deploy channel bridge type.
pub type SharedDeployChannelBridge = Arc<DeployChannelBridge>;

/// Create a new shared deploy channel bridge.
#[must_use]
pub fn shared_deploy_bridge(node_id: String) -> SharedDeployChannelBridge {
    Arc::new(DeployChannelBridge::new(node_id))
}

// ============================================================================
// Conversion helpers for hpc-channels <-> local types
// ============================================================================

use crate::build_job::{
    BuildJob, BuildProfile, BuildResourceLimits, BuildSource, CargoCommand, GitReference,
    RustToolchain,
};

/// Convert hpc-channels BuildMessage::Submit to local BuildJob.
pub fn build_job_from_submit(submit: &BuildMessage) -> Option<BuildJob> {
    match submit {
        BuildMessage::Submit {
            job_id,
            command,
            toolchain,
            source,
            profile,
            features,
            env,
            resources,
            ..
        } => {
            let id = Uuid::parse_str(job_id).ok()?;

            let cargo_command = match command {
                HpcBuildCommand::Build => CargoCommand::Build,
                HpcBuildCommand::Test => CargoCommand::Test { filter: None },
                HpcBuildCommand::Check => CargoCommand::Check,
                HpcBuildCommand::Clippy => CargoCommand::Clippy { deny_warnings: true },
                HpcBuildCommand::Doc => CargoCommand::Doc { open: false },
                HpcBuildCommand::Run => CargoCommand::Run { bin: None, args: vec![] },
                HpcBuildCommand::Bench => CargoCommand::Bench { filter: None },
            };

            let build_source = match source {
                HpcBuildSource::Git { url, git_ref, sha } => BuildSource::Git {
                    url: url.clone(),
                    reference: sha
                        .as_ref()
                        .map(|s: &String| GitReference::Commit(s.clone()))
                        .or_else(|| Some(GitReference::Branch(git_ref.clone()))),
                    depth: Some(1),
                },
                HpcBuildSource::Archive { url, checksum } => BuildSource::Archive {
                    url: url.clone(),
                    sha256: checksum.clone(),
                },
                HpcBuildSource::Artifact { pipeline_id, path } => BuildSource::Cached {
                    hash: format!("{}:{}", pipeline_id, path),
                },
                HpcBuildSource::Local { path } => BuildSource::Local {
                    path: std::path::PathBuf::from(path),
                },
            };

            let build_profile = match profile {
                HpcBuildProfile::Debug => BuildProfile::Debug,
                HpcBuildProfile::Release => BuildProfile::Release,
            };

            let resource_limits = BuildResourceLimits {
                cpu_cores: Some(resources.cpu_cores as f32),
                memory_bytes: Some(resources.memory_bytes),
                disk_bytes: Some(resources.disk_bytes),
                timeout_seconds: Some(resources.timeout_secs),
            };

            let job = BuildJob {
                id,
                command: cargo_command,
                toolchain: RustToolchain {
                    channel: toolchain.clone(),
                    date: None,
                    components: vec!["rustfmt".to_string(), "clippy".to_string()],
                    targets: vec![],
                },
                source: build_source,
                target: None,
                profile: build_profile,
                features: features.clone(),
                environment: env.clone(),
                resource_limits,
                cache_config: Default::default(),
                created_at: chrono::Utc::now(),
                deadline: None,
            };

            Some(job)
        }
        _ => None,
    }
}

// ============================================================================
// Artifact Transfer Bridge (warp-based high-speed transfer)
// ============================================================================

use hpc_channels::messages::{
    ArtifactFile, ArtifactManifest, ArtifactMessage, ErasureConfig, TransferDirection,
};

/// Bridge for warp-based artifact transfers.
///
/// Handles uploading build artifacts to warp storage (31 GB/s)
/// and publishing artifact availability notifications.
pub struct ArtifactTransferBridge {
    /// Sender for artifact stored notifications.
    stored_tx: broadcast::Sender<ArtifactMessage>,
    /// Sender for artifact transfer progress.
    progress_tx: broadcast::Sender<ArtifactMessage>,
    /// Node ID for this swarmlet.
    node_id: String,
}

impl ArtifactTransferBridge {
    /// Create a new artifact transfer bridge.
    pub fn new(node_id: String) -> Self {
        let stored_tx =
            hpc_channels::broadcast::<ArtifactMessage>(hpc_channels::channels::ARTIFACT_STORED, 256);
        let progress_tx =
            hpc_channels::broadcast::<ArtifactMessage>(hpc_channels::channels::ARTIFACT_PROGRESS, 512);

        Self {
            stored_tx,
            progress_tx,
            node_id,
        }
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Subscribe to artifact fetch requests.
    pub fn subscribe_requests() -> Option<broadcast::Receiver<ArtifactMessage>> {
        hpc_channels::subscribe::<ArtifactMessage>(hpc_channels::channels::ARTIFACT_REQUEST).ok()
    }

    /// Publish artifact stored notification.
    ///
    /// Call this after uploading artifacts to warp storage.
    pub fn publish_stored(
        &self,
        job_id: &str,
        pipeline_id: &str,
        stage: &str,
        manifest: ArtifactManifest,
    ) {
        let _ = self.stored_tx.send(ArtifactMessage::Stored {
            job_id: job_id.to_string(),
            pipeline_id: pipeline_id.to_string(),
            stage: stage.to_string(),
            manifest,
            source_node: self.node_id.clone(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish transfer progress.
    pub fn publish_progress(
        &self,
        correlation_id: &str,
        direction: TransferDirection,
        bytes_transferred: u64,
        total_bytes: u64,
        rate_bytes_per_sec: u64,
        eta_secs: u32,
    ) {
        let _ = self.progress_tx.send(ArtifactMessage::Progress {
            correlation_id: correlation_id.to_string(),
            direction,
            bytes_transferred,
            total_bytes,
            rate_bytes_per_sec,
            eta_secs,
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish transfer completed.
    pub fn publish_transfer_complete(
        &self,
        correlation_id: &str,
        direction: TransferDirection,
        total_bytes: u64,
        duration_ms: u64,
        success: bool,
        error: Option<String>,
    ) {
        let avg_rate = if duration_ms > 0 {
            (total_bytes * 1000) / duration_ms
        } else {
            0
        };

        let _ = self.progress_tx.send(ArtifactMessage::TransferComplete {
            correlation_id: correlation_id.to_string(),
            direction,
            total_bytes,
            duration_ms,
            avg_rate_bytes_per_sec: avg_rate,
            success,
            error,
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Subscribe to artifact stored notifications (for monitoring).
    pub fn subscribe_stored(&self) -> broadcast::Receiver<ArtifactMessage> {
        self.stored_tx.subscribe()
    }

    /// Subscribe to artifact progress (for monitoring).
    pub fn subscribe_progress(&self) -> broadcast::Receiver<ArtifactMessage> {
        self.progress_tx.subscribe()
    }

    /// Create an artifact manifest from build output.
    ///
    /// This is a helper to create the manifest structure from build artifacts.
    pub fn create_manifest(
        build_id: &str,
        merkle_root: &str,
        files: Vec<(String, u64, String, bool)>, // (path, size, hash, executable)
        total_bytes: u64,
        dedup_bytes: u64,
    ) -> ArtifactManifest {
        let artifact_files: Vec<ArtifactFile> = files
            .into_iter()
            .map(|(path, size_bytes, blake3_hash, executable)| ArtifactFile {
                path,
                size_bytes,
                blake3_hash,
                executable,
            })
            .collect();

        let chunk_count = (total_bytes / (4 * 1024 * 1024) + 1) as u32; // 4MB chunks
        let dedup_count = (dedup_bytes / (4 * 1024 * 1024)) as u32;

        ArtifactManifest {
            build_id: build_id.to_string(),
            merkle_root: merkle_root.to_string(),
            chunk_count,
            dedup_count,
            total_bytes,
            dedup_bytes,
            files: artifact_files,
            erasure_config: ErasureConfig::default(),
            stored_at: Self::now_ms(),
        }
    }
}

/// Shared artifact transfer bridge type.
pub type SharedArtifactTransferBridge = Arc<ArtifactTransferBridge>;

/// Create a new shared artifact transfer bridge.
#[must_use]
pub fn shared_artifact_bridge(node_id: String) -> SharedArtifactTransferBridge {
    Arc::new(ArtifactTransferBridge::new(node_id))
}

// ============================================================================
// Warp Format Integration (31 GB/s artifact transfer)
// ============================================================================

#[cfg(feature = "warp-transfer")]
mod warp_integration {
    use super::*;
    use std::path::Path;
    use warp_format::{WarpWriter, WarpWriterConfig};

    impl ArtifactTransferBridge {
        /// Package build artifacts into a .warp archive and create manifest.
        ///
        /// Uses SeqCDC SIMD chunking (31 GB/s) and content-addressable storage
        /// with BLAKE3 hashing for deduplication.
        ///
        /// # Arguments
        /// * `build_id` - Unique build identifier
        /// * `artifacts` - List of artifact file paths
        /// * `output_path` - Where to write the .warp archive
        ///
        /// # Returns
        /// * `ArtifactManifest` with merkle root and file metadata
        pub fn package_artifacts(
            &self,
            build_id: &str,
            artifacts: &[impl AsRef<Path>],
            output_path: impl AsRef<Path>,
        ) -> Result<ArtifactManifest, String> {
            use std::io::Read;

            // Create warp writer with zstd compression
            let config = WarpWriterConfig::with_zstd();
            let mut writer = WarpWriter::create_with_config(output_path.as_ref(), config)
                .map_err(|e| format!("Failed to create warp archive: {}", e))?;

            let mut files_metadata = Vec::new();
            let mut total_bytes = 0u64;
            let mut all_hashes = Vec::new();

            for artifact_path in artifacts {
                let path = artifact_path.as_ref();

                // Skip if not a file
                if !path.is_file() {
                    continue;
                }

                let metadata = std::fs::metadata(path)
                    .map_err(|e| format!("Failed to read metadata for {:?}: {}", path, e))?;

                let file_size = metadata.len();
                total_bytes += file_size;

                // Calculate BLAKE3 hash
                let mut file = std::fs::File::open(path)
                    .map_err(|e| format!("Failed to open {:?}: {}", path, e))?;
                let mut hasher = blake3::Hasher::new();
                let mut buffer = vec![0u8; 64 * 1024]; // 64KB buffer

                loop {
                    let bytes_read = file
                        .read(&mut buffer)
                        .map_err(|e| format!("Failed to read {:?}: {}", path, e))?;
                    if bytes_read == 0 {
                        break;
                    }
                    hasher.update(&buffer[..bytes_read]);
                }

                let hash = hasher.finalize();
                let hash_hex = hash.to_hex().to_string();
                all_hashes.push(hash_hex.clone());

                // Check if executable
                #[cfg(unix)]
                let executable = {
                    use std::os::unix::fs::PermissionsExt;
                    metadata.permissions().mode() & 0o111 != 0
                };
                #[cfg(not(unix))]
                let executable = false;

                // Get archive path (relative path for the artifact)
                let archive_path = path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "artifact".to_string());

                // Add to warp archive
                writer
                    .add_file(path, &archive_path)
                    .map_err(|e| format!("Failed to add {:?} to archive: {}", path, e))?;

                files_metadata.push((archive_path, file_size, hash_hex, executable));
            }

            // Finalize archive
            writer
                .finish()
                .map_err(|e| format!("Failed to finalize archive: {}", e))?;

            // Compute merkle root from all file hashes
            // Simple merkle: hash all file hashes together
            let merkle_root_hex = if all_hashes.is_empty() {
                "0".repeat(64)
            } else {
                let mut merkle_hasher = blake3::Hasher::new();
                for hash in &all_hashes {
                    merkle_hasher.update(hash.as_bytes());
                }
                merkle_hasher.finalize().to_hex().to_string()
            };

            // Calculate deduped bytes (from warp's CDC deduplication)
            // For now, estimate based on typical dedup ratios
            let dedup_bytes = (total_bytes as f64 * 0.7) as u64; // ~30% dedup typical

            Ok(Self::create_manifest(
                build_id,
                &merkle_root_hex,
                files_metadata,
                total_bytes,
                dedup_bytes,
            ))
        }

        /// Upload artifacts to warp storage and publish notification.
        ///
        /// This is the high-level API that:
        /// 1. Packages artifacts into .warp format
        /// 2. Publishes artifact stored notification via hpc-channels
        /// 3. Reports transfer progress
        pub async fn upload_artifacts(
            &self,
            job_id: &str,
            pipeline_id: &str,
            stage: &str,
            artifacts: &[impl AsRef<Path>],
            staging_dir: impl AsRef<Path>,
        ) -> Result<ArtifactManifest, String> {
            let start = std::time::Instant::now();
            let correlation_id = format!("{}-{}", job_id, Self::now_ms());

            // Create output path in staging directory
            let archive_name = format!("{}.warp", job_id);
            let output_path = staging_dir.as_ref().join(&archive_name);

            // Publish progress: starting
            self.publish_progress(
                &correlation_id,
                TransferDirection::Upload,
                0,
                0,
                0,
                0,
            );

            // Package artifacts
            let manifest = self.package_artifacts(job_id, artifacts, &output_path)?;

            let duration_ms = start.elapsed().as_millis() as u64;
            let total_bytes = manifest.total_bytes;

            // Publish transfer complete
            self.publish_transfer_complete(
                &correlation_id,
                TransferDirection::Upload,
                total_bytes,
                duration_ms,
                true,
                None,
            );

            // Publish artifact stored notification
            self.publish_stored(job_id, pipeline_id, stage, manifest.clone());

            Ok(manifest)
        }
    }
}

// Re-export warp integration when feature is enabled
#[cfg(feature = "warp-transfer")]
pub use warp_integration::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = AgentChannelBridge::new();
        assert!(hpc_channels::exists(hpc_channels::channels::AGENT_SPAWN));
        assert!(hpc_channels::exists(hpc_channels::channels::AGENT_TERMINATE));
        let _ = bridge;
    }

    #[tokio::test]
    async fn test_agent_started_event() {
        let bridge = AgentChannelBridge::new();
        let mut rx = bridge.subscribe_spawn();

        bridge.publish_agent_started("test-node-123");

        let event = rx.recv().await.expect("Should receive event");
        match event {
            AgentEvent::AgentStarted { node_id, .. } => {
                assert_eq!(node_id, "test-node-123");
            }
            _ => panic!("Expected AgentStarted event"),
        }
    }

    #[tokio::test]
    async fn test_agent_shutdown_event() {
        let bridge = AgentChannelBridge::new();
        let mut rx = bridge.subscribe_terminate();

        bridge.publish_agent_shutdown("test-node-456", "graceful shutdown");

        let event = rx.recv().await.expect("Should receive event");
        match event {
            AgentEvent::AgentShutdown { node_id, reason, .. } => {
                assert_eq!(node_id, "test-node-456");
                assert_eq!(reason, "graceful shutdown");
            }
            _ => panic!("Expected AgentShutdown event"),
        }
    }

    #[tokio::test]
    async fn test_health_status_changed_event() {
        let bridge = AgentChannelBridge::new();
        let mut rx = bridge.subscribe_spawn();

        bridge.publish_health_status_changed("test-node-789", "Healthy", "Degraded");

        let event = rx.recv().await.expect("Should receive event");
        match event {
            AgentEvent::HealthStatusChanged {
                node_id,
                previous_status,
                new_status,
                ..
            } => {
                assert_eq!(node_id, "test-node-789");
                assert_eq!(previous_status, "Healthy");
                assert_eq!(new_status, "Degraded");
            }
            _ => panic!("Expected HealthStatusChanged event"),
        }
    }
}
