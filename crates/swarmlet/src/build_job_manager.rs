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
