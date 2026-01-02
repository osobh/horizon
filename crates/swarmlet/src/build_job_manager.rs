//! Build job manager
//!
//! This module handles the orchestration of Rust/cargo build jobs,
//! including job submission, execution, and status tracking.

use crate::build_backend::{BuildBackend, BuildContext, detect_backend};
use crate::build_job::{
    ActiveBuildJob, BuildJob, BuildJobStatus, BuildLogEntry, BuildSource, LogStream,
};
use crate::build_log_stream::BuildLogStreamer;
use crate::build_metrics::{
    BuildOutcome, BuildRecord, CommandType, ProfileType, SharedBuildMetrics,
};
use crate::cache_manager::CacheManager;
use crate::config::Config;
use crate::toolchain_manager::ToolchainManager;
use crate::error::BuildPhase;
use crate::{Result, SwarmletError};
use chrono::Utc;
use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use tar::Archive;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Manages build job execution
pub struct BuildJobManager {
    /// Configuration
    config: Arc<Config>,
    /// Active build jobs
    active_jobs: Arc<RwLock<HashMap<Uuid, ActiveBuildJob>>>,
    /// Node ID for this swarmlet
    node_id: Uuid,
    /// Toolchain manager
    toolchain_manager: Arc<ToolchainManager>,
    /// Cache manager
    cache_manager: Arc<CacheManager>,
    /// Build backend
    backend: Arc<dyn BuildBackend>,
    /// Maximum concurrent builds
    max_concurrent_builds: usize,
    /// Optional log streamer for WebSocket broadcasting
    log_streamer: Arc<RwLock<Option<Arc<BuildLogStreamer>>>>,
    /// Build metrics collector
    metrics: SharedBuildMetrics,
}

impl BuildJobManager {
    /// Create a new build job manager
    pub async fn new(config: Arc<Config>, node_id: Uuid, data_dir: PathBuf) -> Result<Self> {
        // Detect and create appropriate backend
        let backend = detect_backend().await?;
        info!("Using build backend: {}", backend.name());

        // Create toolchain manager
        let toolchains_dir = data_dir.join("toolchains");
        let toolchain_manager = Arc::new(ToolchainManager::new(toolchains_dir).await?);

        // Create cache manager
        let cache_dir = data_dir.join("cache");
        let cache_manager = Arc::new(CacheManager::new(cache_dir).await?);

        // Create metrics collector
        let metrics = crate::build_metrics::create_metrics_collector(1000);

        // Get max concurrent builds from config or default
        let max_concurrent_builds = 4; // TODO: get from config

        Ok(Self {
            config,
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            node_id,
            toolchain_manager,
            cache_manager,
            backend: Arc::from(backend),
            max_concurrent_builds,
            log_streamer: Arc::new(RwLock::new(None)),
            metrics,
        })
    }

    /// Set the log streamer for WebSocket broadcasting
    pub async fn set_log_streamer(&self, streamer: Arc<BuildLogStreamer>) {
        let mut guard = self.log_streamer.write().await;
        *guard = Some(streamer);
    }

    /// Get a reference to the cache manager
    pub fn cache_manager(&self) -> &Arc<CacheManager> {
        &self.cache_manager
    }

    /// Get a reference to the metrics collector
    pub fn metrics(&self) -> &SharedBuildMetrics {
        &self.metrics
    }

    /// Get current build metrics snapshot
    pub async fn get_metrics_snapshot(&self) -> crate::build_metrics::MetricsSnapshot {
        self.metrics.get_snapshot().await
    }

    /// Get aggregated build statistics
    pub async fn get_build_stats(&self) -> crate::build_metrics::AggregatedStats {
        self.metrics.get_stats().await
    }

    /// Get build summary for a time window (in hours)
    pub async fn get_build_summary(&self, hours: i64) -> crate::build_metrics::BuildSummary {
        self.metrics.get_summary(chrono::Duration::hours(hours)).await
    }

    /// Submit a new build job
    pub async fn submit_build(&self, job: BuildJob) -> Result<Uuid> {
        // Check capacity
        let current_count = self.active_job_count().await;
        if current_count >= self.max_concurrent_builds {
            return Err(SwarmletError::WorkloadExecution(
                "Maximum build job limit reached".to_string(),
            ));
        }

        let job_id = job.id;
        info!("Submitting build job: {}", job_id);

        let active_job = ActiveBuildJob {
            id: job_id,
            job: job.clone(),
            status: BuildJobStatus::Queued,
            started_at: Utc::now(),
            pid: None,
            container_id: None,
            output_log: vec![BuildLogEntry {
                timestamp: Utc::now(),
                stream: LogStream::System,
                message: "Job queued".to_string(),
            }],
            artifacts: Vec::new(),
            resource_usage: Default::default(),
        };

        // Add to active jobs
        {
            let mut jobs = self.active_jobs.write().await;
            jobs.insert(job_id, active_job);
        }

        // Spawn build execution
        let this = BuildJobManager {
            config: self.config.clone(),
            active_jobs: self.active_jobs.clone(),
            node_id: self.node_id,
            toolchain_manager: self.toolchain_manager.clone(),
            cache_manager: self.cache_manager.clone(),
            backend: self.backend.clone(),
            max_concurrent_builds: self.max_concurrent_builds,
            log_streamer: self.log_streamer.clone(),
            metrics: self.metrics.clone(),
        };

        tokio::spawn(async move {
            if let Err(e) = this.execute_build(job_id, job).await {
                error!("Build {} failed: {}", job_id, e);
                let error_msg = e.to_string();
                this.update_status(
                    job_id,
                    BuildJobStatus::Failed {
                        error: error_msg.clone(),
                    },
                )
                .await;

                // Broadcast error to WebSocket clients
                let streamer_guard = this.log_streamer.read().await;
                if let Some(ref streamer) = *streamer_guard {
                    streamer.broadcast_error(job_id, error_msg).await;
                }
            }
        });

        Ok(job_id)
    }

    /// Execute a build job
    async fn execute_build(&self, job_id: Uuid, job: BuildJob) -> Result<()> {
        // Track build start time for metrics
        let build_started_at = Utc::now();

        // Phase 1: Prepare environment
        self.update_status(job_id, BuildJobStatus::PreparingEnvironment)
            .await;
        self.log(job_id, LogStream::System, "Preparing build environment")
            .await;

        let workspace = self.backend.create_workspace(&job_id.to_string()).await
            .map_err(|e| {
                SwarmletError::build_error_with_source(
                    BuildPhase::PreparingEnvironment,
                    "Failed to create workspace",
                    e,
                )
            })?;

        // Phase 2: Fetch source (with retry for recoverable errors)
        self.update_status(job_id, BuildJobStatus::FetchingSource)
            .await;
        self.log(job_id, LogStream::System, "Fetching source code")
            .await;

        if let Err(e) = self.fetch_source_with_retry(&job.source, &workspace, job_id, 3).await {
            // Cleanup workspace on failure
            self.cleanup_on_error(job_id, &workspace).await;
            return Err(SwarmletError::build_error_with_source(
                BuildPhase::FetchingSource,
                "Failed to fetch source code",
                e,
            ));
        }

        // Phase 3: Provision toolchain
        self.update_status(job_id, BuildJobStatus::ProvisioningToolchain)
            .await;
        self.log(
            job_id,
            LogStream::System,
            &format!("Provisioning toolchain: {}", job.toolchain.toolchain_string()),
        )
        .await;

        let toolchain_path = match self.toolchain_manager.ensure_toolchain(&job.toolchain).await {
            Ok(path) => path,
            Err(e) => {
                self.cleanup_on_error(job_id, &workspace).await;
                return Err(SwarmletError::build_error_with_source(
                    BuildPhase::ProvisioningToolchain,
                    format!("Failed to provision toolchain: {}", job.toolchain.toolchain_string()),
                    e,
                ));
            }
        };

        // Phase 4: Execute build
        self.update_status(job_id, BuildJobStatus::Building).await;
        self.log(
            job_id,
            LogStream::System,
            &format!("Running: cargo {}", job.cargo_args().join(" ")),
        )
        .await;

        // Get cache mounts
        let cache_mounts = self.cache_manager.get_mounts(&job.cache_config).await?;

        // Build context
        let context = BuildContext::new(workspace.clone(), toolchain_path)
            .with_resource_limits(job.resource_limits.clone())
            .with_cargo_args(job.cargo_args())
            .with_environment(job.environment.clone());

        // Add cache mounts
        let context = cache_mounts.into_iter().fold(context, |ctx, mount| {
            ctx.with_cache_mount(mount)
        });

        // Execute
        let result = self.backend.execute_cargo(&job.command, &context).await?;

        // Phase 5: Collect artifacts
        self.update_status(job_id, BuildJobStatus::CollectingArtifacts)
            .await;

        // Update final status and broadcast completion
        let duration_secs = result.duration.as_secs_f64();
        if result.is_success() {
            self.update_status(job_id, BuildJobStatus::Completed).await;
            self.log(
                job_id,
                LogStream::System,
                &format!("Build completed in {:.2}s", duration_secs),
            )
            .await;
        } else {
            self.update_status(
                job_id,
                BuildJobStatus::Failed {
                    error: format!("Exit code: {}", result.exit_code),
                },
            )
            .await;
        }

        // Broadcast completion to WebSocket clients
        {
            let streamer_guard = self.log_streamer.read().await;
            if let Some(ref streamer) = *streamer_guard {
                streamer
                    .broadcast_completed(job_id, result.exit_code, duration_secs)
                    .await;
            }
        }

        // Record artifact cache usage if enabled
        if job.cache_config.cache_target {
            if let Some(ref cache_key) = job.cache_config.cache_key {
                if let Err(e) = self
                    .cache_manager
                    .record_artifact_usage(cache_key, &job_id.to_string())
                    .await
                {
                    warn!("Failed to record artifact cache usage: {}", e);
                }
            }
        }

        // Prepare metrics data before moving result
        let resource_usage = result.resource_usage.clone();
        let build_succeeded = result.is_success();

        // Update resource usage
        {
            let mut jobs = self.active_jobs.write().await;
            if let Some(active_job) = jobs.get_mut(&job_id) {
                active_job.resource_usage = result.resource_usage;
                active_job.artifacts = result.artifacts;
            }
        }

        // Record build metrics
        let build_completed_at = Utc::now();
        let build_record = BuildRecord {
            job_id,
            command: CommandType::from(&job.command),
            profile: ProfileType::from(&job.profile),
            status: if build_succeeded {
                BuildOutcome::Success
            } else {
                BuildOutcome::Failed
            },
            started_at: build_started_at,
            completed_at: build_completed_at,
            duration_seconds: duration_secs,
            resource_usage,
            cache_enabled: job.cache_config.use_sccache || job.cache_config.cache_target,
            source_cache_hit: matches!(&job.source, BuildSource::Cached { .. }),
            toolchain: job.toolchain.toolchain_string(),
        };
        self.metrics.record_build(build_record).await;

        // Cleanup workspace
        if let Err(e) = self.backend.cleanup_workspace(&workspace).await {
            warn!("Failed to cleanup workspace: {}", e);
        }

        Ok(())
    }

    /// Fetch source with retry logic for recoverable errors
    async fn fetch_source_with_retry(
        &self,
        source: &BuildSource,
        workspace: &PathBuf,
        job_id: Uuid,
        max_retries: u32,
    ) -> Result<()> {
        let mut last_error = None;

        for attempt in 1..=max_retries {
            match self.fetch_source(source, workspace).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let is_last_attempt = attempt == max_retries;

                    if e.is_recoverable() && !is_last_attempt {
                        warn!(
                            "Build {}: Source fetch attempt {}/{} failed (recoverable): {}",
                            job_id, attempt, max_retries, e
                        );
                        self.log(
                            job_id,
                            LogStream::System,
                            &format!(
                                "Source fetch failed (attempt {}/{}), retrying in 2s: {}",
                                attempt, max_retries, e
                            ),
                        )
                        .await;

                        // Clean up partial workspace before retry
                        if let Err(cleanup_err) = self.cleanup_workspace_contents(workspace).await {
                            warn!("Failed to cleanup workspace before retry: {}", cleanup_err);
                        }

                        // Wait before retry with exponential backoff
                        tokio::time::sleep(std::time::Duration::from_secs(2_u64.pow(attempt - 1))).await;
                        last_error = Some(e);
                    } else {
                        // Non-recoverable error or last attempt
                        if is_last_attempt {
                            error!(
                                "Build {}: Source fetch failed after {} attempts: {}",
                                job_id, max_retries, e
                            );
                            self.log(
                                job_id,
                                LogStream::System,
                                &format!("Source fetch failed after {} attempts: {}", max_retries, e),
                            )
                            .await;
                        }
                        return Err(e);
                    }
                }
            }
        }

        // Should not reach here, but return last error if we do
        Err(last_error.unwrap_or_else(|| {
            SwarmletError::WorkloadExecution("Source fetch failed with unknown error".to_string())
        }))
    }

    /// Clean up workspace contents (for retry)
    async fn cleanup_workspace_contents(&self, workspace: &PathBuf) -> Result<()> {
        if workspace.exists() {
            let mut entries = tokio::fs::read_dir(workspace).await?;
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.is_dir() {
                    tokio::fs::remove_dir_all(&path).await?;
                } else {
                    tokio::fs::remove_file(&path).await?;
                }
            }
        }
        Ok(())
    }

    /// Clean up on build error
    async fn cleanup_on_error(&self, job_id: Uuid, workspace: &PathBuf) {
        self.log(
            job_id,
            LogStream::System,
            "Cleaning up after build failure...",
        )
        .await;

        if let Err(e) = self.backend.cleanup_workspace(workspace).await {
            warn!("Failed to cleanup workspace after error: {}", e);
        }
    }

    /// Fetch source code to workspace
    async fn fetch_source(&self, source: &BuildSource, workspace: &PathBuf) -> Result<()> {
        match source {
            BuildSource::Git {
                url,
                reference,
                depth,
            } => {
                let mut cmd = tokio::process::Command::new("git");
                cmd.arg("clone");

                if let Some(d) = depth {
                    cmd.arg("--depth").arg(d.to_string());
                }

                if let Some(reference) = reference {
                    match reference {
                        crate::build_job::GitReference::Branch(b) => {
                            cmd.arg("--branch").arg(b);
                        }
                        crate::build_job::GitReference::Tag(t) => {
                            cmd.arg("--branch").arg(t);
                        }
                        crate::build_job::GitReference::Commit(_) => {
                            // Clone then checkout specific commit
                        }
                    }
                }

                cmd.arg(url).arg(workspace);

                let output = cmd.output().await.map_err(|e| {
                    SwarmletError::WorkloadExecution(format!("Git clone failed: {e}"))
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(SwarmletError::WorkloadExecution(format!(
                        "Git clone failed: {}",
                        stderr
                    )));
                }

                // If specific commit, checkout now
                if let Some(crate::build_job::GitReference::Commit(commit)) = reference {
                    let output = tokio::process::Command::new("git")
                        .current_dir(workspace)
                        .args(["checkout", commit])
                        .output()
                        .await?;

                    if !output.status.success() {
                        return Err(SwarmletError::WorkloadExecution(
                            "Git checkout failed".to_string(),
                        ));
                    }
                }

                Ok(())
            }
            BuildSource::Archive { url, sha256 } => {
                debug!("Downloading archive from: {}", url);

                // Download archive
                let response = reqwest::get(url).await.map_err(SwarmletError::Network)?;

                if !response.status().is_success() {
                    return Err(SwarmletError::WorkloadExecution(format!(
                        "Failed to download archive: HTTP {}",
                        response.status()
                    )));
                }

                let bytes = response.bytes().await.map_err(SwarmletError::Network)?;

                // Verify SHA256 hash if provided
                if let Some(expected_hash) = sha256 {
                    let mut hasher = Sha256::new();
                    hasher.update(&bytes);
                    let actual_hash = format!("{:x}", hasher.finalize());

                    if actual_hash != *expected_hash {
                        return Err(SwarmletError::WorkloadExecution(format!(
                            "SHA256 mismatch: expected {}, got {}",
                            expected_hash, actual_hash
                        )));
                    }
                    debug!("SHA256 verification passed: {}", actual_hash);
                }

                // Extract archive based on URL extension
                let is_gzip = url.ends_with(".tar.gz") || url.ends_with(".tgz");
                let is_tar = url.ends_with(".tar") || is_gzip;

                if !is_tar {
                    return Err(SwarmletError::WorkloadExecution(
                        "Unsupported archive format. Only .tar, .tar.gz, and .tgz are supported"
                            .to_string(),
                    ));
                }

                // Extract tar archive (with optional gzip decompression)
                let bytes_slice = bytes.as_ref();

                if is_gzip {
                    debug!("Extracting gzipped tar archive to {:?}", workspace);
                    let decoder = GzDecoder::new(bytes_slice);
                    let mut archive = Archive::new(decoder);
                    extract_tar_archive(&mut archive, workspace)?;
                } else {
                    debug!("Extracting tar archive to {:?}", workspace);
                    let mut archive = Archive::new(bytes_slice);
                    extract_tar_archive(&mut archive, workspace)?;
                }

                info!("Archive extracted successfully to {:?}", workspace);
                Ok(())
            }
            BuildSource::Cached { hash } => {
                // Retrieve cached source by hash
                let cache_path = self
                    .cache_manager
                    .get_cached_source(hash)
                    .await
                    .ok_or_else(|| {
                        SwarmletError::WorkloadExecution(format!(
                            "Cached source not found: {}",
                            hash
                        ))
                    })?;

                let archive_path = cache_path.join("source.tar.gz");
                if !archive_path.exists() {
                    return Err(SwarmletError::WorkloadExecution(format!(
                        "Cached source archive not found: {}",
                        archive_path.display()
                    )));
                }

                debug!(
                    "Extracting cached source {} from {:?}",
                    hash, archive_path
                );

                // Extract the cached archive
                let file = std::fs::File::open(&archive_path).map_err(|e| {
                    SwarmletError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to open cached archive: {}", e),
                    ))
                })?;
                let decoder = GzDecoder::new(file);
                let mut archive = Archive::new(decoder);

                // Extract directly (no stripping needed - cached sources are stored flat)
                archive.unpack(workspace).map_err(|e| {
                    SwarmletError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to extract cached archive: {}", e),
                    ))
                })?;

                // Update last_used timestamp
                self.cache_manager.touch_source(hash).await?;

                info!("Cached source {} extracted to {:?}", hash, workspace);
                Ok(())
            }
            BuildSource::Local { path } => {
                // Copy local files to workspace
                copy_dir_recursive(path, workspace).await
            }
        }
    }

    /// Update job status
    async fn update_status(&self, job_id: Uuid, status: BuildJobStatus) {
        {
            let mut jobs = self.active_jobs.write().await;
            if let Some(job) = jobs.get_mut(&job_id) {
                debug!("Job {} status: {:?}", job_id, status);
                job.status = status.clone();
            }
        }

        // Broadcast status update to WebSocket clients
        let streamer_guard = self.log_streamer.read().await;
        if let Some(ref streamer) = *streamer_guard {
            streamer.broadcast_status(job_id, status).await;
        }
    }

    /// Add a log entry
    async fn log(&self, job_id: Uuid, stream: LogStream, message: &str) {
        let entry = BuildLogEntry {
            timestamp: Utc::now(),
            stream,
            message: message.to_string(),
        };

        {
            let mut jobs = self.active_jobs.write().await;
            if let Some(job) = jobs.get_mut(&job_id) {
                job.output_log.push(entry.clone());
            }
        }

        // Broadcast log entry to WebSocket clients
        let streamer_guard = self.log_streamer.read().await;
        if let Some(ref streamer) = *streamer_guard {
            streamer.broadcast_log(job_id, entry).await;
        }
    }

    /// Get the number of active jobs
    pub async fn active_job_count(&self) -> usize {
        let jobs = self.active_jobs.read().await;
        jobs.values()
            .filter(|j| !j.status.is_terminal())
            .count()
    }

    /// Get a job's status
    pub async fn get_job_status(&self, job_id: Uuid) -> Option<BuildJobStatus> {
        let jobs = self.active_jobs.read().await;
        jobs.get(&job_id).map(|j| j.status.clone())
    }

    /// Get a job's full info
    pub async fn get_job(&self, job_id: Uuid) -> Option<ActiveBuildJob> {
        let jobs = self.active_jobs.read().await;
        jobs.get(&job_id).cloned()
    }

    /// List all active jobs
    pub async fn list_active_jobs(&self) -> Vec<ActiveBuildJob> {
        let jobs = self.active_jobs.read().await;
        jobs.values().cloned().collect()
    }

    /// Cancel a build job
    ///
    /// This will:
    /// 1. Mark the job status as Cancelled
    /// 2. Kill the running process (if native Linux isolation)
    /// 3. Stop the container (if Docker backend)
    pub async fn cancel_build(&self, job_id: Uuid) -> Result<()> {
        let mut jobs = self.active_jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            if !job.status.is_terminal() {
                // Kill process if running (native Linux isolation)
                if let Some(pid) = job.pid {
                    #[cfg(unix)]
                    {
                        use nix::sys::signal::{kill, Signal};
                        use nix::unistd::Pid;
                        info!("Killing build process {} for job {}", pid, job_id);
                        // Send SIGTERM first for graceful shutdown
                        if let Err(e) = kill(Pid::from_raw(pid as i32), Signal::SIGTERM) {
                            warn!("Failed to send SIGTERM to process {}: {}", pid, e);
                            // Try SIGKILL if SIGTERM fails
                            let _ = kill(Pid::from_raw(pid as i32), Signal::SIGKILL);
                        }
                    }
                    #[cfg(not(unix))]
                    {
                        warn!("Process termination not supported on this platform");
                    }
                }

                // Stop container if running (Docker backend)
                #[cfg(feature = "docker")]
                if let Some(ref container_id) = job.container_id {
                    info!("Stopping container {} for job {}", container_id, job_id);
                    // Container stopping is handled asynchronously by the Docker backend
                    // The BuildBackend trait has cancel_build which we can't call from here
                    // since we don't have access to the backend. Instead, we store the
                    // container_id and the caller is responsible for stopping it.
                }

                job.status = BuildJobStatus::Cancelled;
                job.output_log.push(BuildLogEntry {
                    timestamp: Utc::now(),
                    stream: LogStream::System,
                    message: "Job cancelled by user".to_string(),
                });
                info!("Cancelled build job: {}", job_id);
                return Ok(());
            }
        }
        Err(SwarmletError::WorkloadExecution(
            "Job not found or already finished".to_string(),
        ))
    }

    /// Cancel all active builds
    pub async fn cancel_all(&self) -> Result<()> {
        let mut jobs = self.active_jobs.write().await;
        for job in jobs.values_mut() {
            if !job.status.is_terminal() {
                job.status = BuildJobStatus::Cancelled;
            }
        }
        Ok(())
    }

    /// Get build logs for a job
    pub async fn get_logs(&self, job_id: Uuid) -> Option<Vec<BuildLogEntry>> {
        let jobs = self.active_jobs.read().await;
        jobs.get(&job_id).map(|j| j.output_log.clone())
    }

    /// Clean up completed jobs older than the specified duration
    pub async fn cleanup_old_jobs(&self, max_age_secs: u64) {
        let mut jobs = self.active_jobs.write().await;
        let now = Utc::now();

        jobs.retain(|_, job| {
            if job.status.is_terminal() {
                let age = now.signed_duration_since(job.started_at);
                age.num_seconds() < max_age_secs as i64
            } else {
                true
            }
        });
    }
}

/// Recursively copy a directory
async fn copy_dir_recursive(src: &PathBuf, dst: &PathBuf) -> Result<()> {
    tokio::fs::create_dir_all(dst).await?;

    let mut entries = tokio::fs::read_dir(src).await?;
    while let Some(entry) = entries.next_entry().await? {
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if entry.file_type().await?.is_dir() {
            Box::pin(copy_dir_recursive(&src_path, &dst_path)).await?;
        } else {
            tokio::fs::copy(&src_path, &dst_path).await?;
        }
    }

    Ok(())
}

/// Extract a tar archive to the destination directory
///
/// Handles archives that may contain a single top-level directory (common for GitHub releases).
/// If the archive contains only one directory at the root, extracts its contents directly
/// to the workspace instead of creating a nested directory.
fn extract_tar_archive<R: Read>(archive: &mut Archive<R>, dest: &PathBuf) -> Result<()> {
    // First, collect all entries to analyze structure
    let mut entries_data: Vec<(PathBuf, Vec<u8>, bool, u32)> = Vec::new();
    let mut top_level_dirs: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

    for entry_result in archive.entries().map_err(|e| {
        SwarmletError::WorkloadExecution(format!("Failed to read archive entries: {}", e))
    })? {
        let mut entry = entry_result.map_err(|e| {
            SwarmletError::WorkloadExecution(format!("Failed to read archive entry: {}", e))
        })?;

        let path = entry.path().map_err(|e| {
            SwarmletError::WorkloadExecution(format!("Invalid path in archive: {}", e))
        })?;
        let path = path.to_path_buf();

        // Track top-level directories
        if let Some(first_component) = path.components().next() {
            let first = PathBuf::from(first_component.as_os_str());
            top_level_dirs.insert(first);
        }

        let is_dir = entry.header().entry_type().is_dir();
        let mode = entry.header().mode().unwrap_or(0o644);

        let mut data = Vec::new();
        if !is_dir {
            entry.read_to_end(&mut data).map_err(|e| {
                SwarmletError::WorkloadExecution(format!("Failed to read archive content: {}", e))
            })?;
        }

        entries_data.push((path, data, is_dir, mode));
    }

    // Determine if we should strip the top-level directory
    // (common pattern: archive contains single dir like "project-1.0.0/")
    let strip_prefix = if top_level_dirs.len() == 1 {
        top_level_dirs.into_iter().next()
    } else {
        None
    };

    // Create destination directory
    std::fs::create_dir_all(dest).map_err(|e| {
        SwarmletError::WorkloadExecution(format!("Failed to create destination directory: {}", e))
    })?;

    // Extract entries
    for (path, data, is_dir, mode) in entries_data {
        // Apply prefix stripping if needed
        let relative_path = if let Some(ref prefix) = strip_prefix {
            match path.strip_prefix(prefix) {
                Ok(stripped) => stripped.to_path_buf(),
                Err(_) => path,
            }
        } else {
            path
        };

        // Skip empty paths (happens when stripping the root directory itself)
        if relative_path.as_os_str().is_empty() {
            continue;
        }

        let full_path = dest.join(&relative_path);

        if is_dir {
            std::fs::create_dir_all(&full_path).map_err(|e| {
                SwarmletError::WorkloadExecution(format!(
                    "Failed to create directory {:?}: {}",
                    full_path, e
                ))
            })?;
        } else {
            // Ensure parent directory exists
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    SwarmletError::WorkloadExecution(format!(
                        "Failed to create parent directory {:?}: {}",
                        parent, e
                    ))
                })?;
            }

            // Write file
            std::fs::write(&full_path, &data).map_err(|e| {
                SwarmletError::WorkloadExecution(format!(
                    "Failed to write file {:?}: {}",
                    full_path, e
                ))
            })?;

            // Set file permissions on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let permissions = std::fs::Permissions::from_mode(mode);
                std::fs::set_permissions(&full_path, permissions).ok(); // Ignore permission errors
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_job::CargoCommand;

    #[test]
    fn test_cargo_args() {
        let job = BuildJob::new(
            CargoCommand::Build,
            BuildSource::Local {
                path: PathBuf::from("."),
            },
        );
        let args = job.cargo_args();
        assert_eq!(args, vec!["build"]);
    }

    #[test]
    fn test_extract_tar_archive_simple() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use tar::Builder;

        // Create a tar.gz archive in memory
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        {
            let mut builder = Builder::new(&mut encoder);

            // Add a file
            let data = b"Hello, World!";
            let mut header = tar::Header::new_gnu();
            header.set_path("test.txt").unwrap();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append(&header, &data[..]).unwrap();

            // Add a directory and file inside it
            let mut dir_header = tar::Header::new_gnu();
            dir_header.set_path("subdir/").unwrap();
            dir_header.set_size(0);
            dir_header.set_mode(0o755);
            dir_header.set_entry_type(tar::EntryType::Directory);
            dir_header.set_cksum();
            builder.append(&dir_header, &[][..]).unwrap();

            let nested_data = b"Nested content";
            let mut nested_header = tar::Header::new_gnu();
            nested_header.set_path("subdir/nested.txt").unwrap();
            nested_header.set_size(nested_data.len() as u64);
            nested_header.set_mode(0o644);
            nested_header.set_cksum();
            builder.append(&nested_header, &nested_data[..]).unwrap();

            builder.finish().unwrap();
        }
        let compressed = encoder.finish().unwrap();

        // Extract to temp directory
        let temp_dir = tempfile::tempdir().unwrap();
        let dest = temp_dir.path().to_path_buf();

        let decoder = GzDecoder::new(&compressed[..]);
        let mut archive = Archive::new(decoder);
        extract_tar_archive(&mut archive, &dest).unwrap();

        // Verify extraction
        assert!(dest.join("test.txt").exists());
        assert!(dest.join("subdir").is_dir());
        assert!(dest.join("subdir/nested.txt").exists());

        let content = std::fs::read_to_string(dest.join("test.txt")).unwrap();
        assert_eq!(content, "Hello, World!");

        let nested_content = std::fs::read_to_string(dest.join("subdir/nested.txt")).unwrap();
        assert_eq!(nested_content, "Nested content");
    }

    #[test]
    fn test_extract_tar_archive_strips_single_root_dir() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use tar::Builder;

        // Create archive with single top-level directory (like GitHub releases)
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        {
            let mut builder = Builder::new(&mut encoder);

            // Add root directory
            let mut root_header = tar::Header::new_gnu();
            root_header.set_path("project-1.0.0/").unwrap();
            root_header.set_size(0);
            root_header.set_mode(0o755);
            root_header.set_entry_type(tar::EntryType::Directory);
            root_header.set_cksum();
            builder.append(&root_header, &[][..]).unwrap();

            // Add file inside root directory
            let data = b"fn main() {}";
            let mut header = tar::Header::new_gnu();
            header.set_path("project-1.0.0/main.rs").unwrap();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append(&header, &data[..]).unwrap();

            builder.finish().unwrap();
        }
        let compressed = encoder.finish().unwrap();

        let temp_dir = tempfile::tempdir().unwrap();
        let dest = temp_dir.path().to_path_buf();

        let decoder = GzDecoder::new(&compressed[..]);
        let mut archive = Archive::new(decoder);
        extract_tar_archive(&mut archive, &dest).unwrap();

        // The single root directory should be stripped
        // So main.rs should be directly in dest, not in dest/project-1.0.0/
        assert!(dest.join("main.rs").exists());
        assert!(!dest.join("project-1.0.0").exists());
    }

    #[test]
    fn test_sha256_verification() {
        use sha2::{Digest, Sha256};

        let data = b"test data for hashing";
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = format!("{:x}", hasher.finalize());

        // SHA256 produces 64 hex chars
        assert_eq!(hash.len(), 64);
        // Verify the hash is computed correctly (actual hash of "test data for hashing")
        assert_eq!(
            hash,
            "f7eb7961d8a233e6256d3a6257548bbb9293c3a08fb3574c88c7d6b429dbb9f5"
        );
    }
}
