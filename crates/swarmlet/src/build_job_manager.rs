//! Build job manager
//!
//! This module handles the orchestration of Rust/cargo build jobs,
//! including job submission, execution, and status tracking.

use crate::build_backend::{BuildBackend, BuildContext, detect_backend};
use crate::build_job::{
    ActiveBuildJob, BuildJob, BuildJobStatus, BuildLogEntry, BuildSource, LogStream,
};
use crate::cache_manager::CacheManager;
use crate::config::Config;
use crate::toolchain_manager::ToolchainManager;
use crate::{Result, SwarmletError};
use chrono::Utc;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
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
        })
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
        };

        tokio::spawn(async move {
            if let Err(e) = this.execute_build(job_id, job).await {
                error!("Build {} failed: {}", job_id, e);
                this.update_status(
                    job_id,
                    BuildJobStatus::Failed {
                        error: e.to_string(),
                    },
                )
                .await;
            }
        });

        Ok(job_id)
    }

    /// Execute a build job
    async fn execute_build(&self, job_id: Uuid, job: BuildJob) -> Result<()> {
        // Phase 1: Prepare environment
        self.update_status(job_id, BuildJobStatus::PreparingEnvironment)
            .await;
        self.log(job_id, LogStream::System, "Preparing build environment")
            .await;

        let workspace = self.backend.create_workspace(&job_id.to_string()).await?;

        // Phase 2: Fetch source
        self.update_status(job_id, BuildJobStatus::FetchingSource)
            .await;
        self.log(job_id, LogStream::System, "Fetching source code")
            .await;

        self.fetch_source(&job.source, &workspace).await?;

        // Phase 3: Provision toolchain
        self.update_status(job_id, BuildJobStatus::ProvisioningToolchain)
            .await;
        self.log(
            job_id,
            LogStream::System,
            &format!("Provisioning toolchain: {}", job.toolchain.toolchain_string()),
        )
        .await;

        let toolchain_path = self
            .toolchain_manager
            .ensure_toolchain(&job.toolchain)
            .await?;

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

        // Update final status
        if result.is_success() {
            self.update_status(job_id, BuildJobStatus::Completed).await;
            self.log(
                job_id,
                LogStream::System,
                &format!(
                    "Build completed in {:.2}s",
                    result.duration.as_secs_f64()
                ),
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

        // Update resource usage
        {
            let mut jobs = self.active_jobs.write().await;
            if let Some(active_job) = jobs.get_mut(&job_id) {
                active_job.resource_usage = result.resource_usage;
                active_job.artifacts = result.artifacts;
            }
        }

        // Cleanup workspace
        if let Err(e) = self.backend.cleanup_workspace(&workspace).await {
            warn!("Failed to cleanup workspace: {}", e);
        }

        Ok(())
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
                // Download archive
                let response = reqwest::get(url).await.map_err(|e| {
                    SwarmletError::Network(e)
                })?;

                let bytes = response.bytes().await.map_err(|e| {
                    SwarmletError::Network(e)
                })?;

                // Verify hash if provided
                if let Some(_expected_hash) = sha256 {
                    // TODO: Verify SHA256
                }

                // Extract to workspace
                // TODO: Handle tar.gz extraction
                let _ = bytes;

                Err(SwarmletError::NotImplemented(
                    "Archive extraction not yet implemented".to_string(),
                ))
            }
            BuildSource::Cached { hash: _ } => {
                Err(SwarmletError::NotImplemented(
                    "Cached source not yet implemented".to_string(),
                ))
            }
            BuildSource::Local { path } => {
                // Copy local files to workspace
                copy_dir_recursive(path, workspace).await
            }
        }
    }

    /// Update job status
    async fn update_status(&self, job_id: Uuid, status: BuildJobStatus) {
        let mut jobs = self.active_jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            debug!("Job {} status: {:?}", job_id, status);
            job.status = status;
        }
    }

    /// Add a log entry
    async fn log(&self, job_id: Uuid, stream: LogStream, message: &str) {
        let mut jobs = self.active_jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.output_log.push(BuildLogEntry {
                timestamp: Utc::now(),
                stream,
                message: message.to_string(),
            });
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
    pub async fn cancel_build(&self, job_id: Uuid) -> Result<()> {
        let mut jobs = self.active_jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            if !job.status.is_terminal() {
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
}
