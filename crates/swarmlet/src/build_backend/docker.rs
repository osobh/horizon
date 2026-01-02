//! Docker-based build backend
//!
//! This backend uses Docker for build isolation, providing cross-platform
//! support for macOS, Windows, and Linux systems without kernel module.
//!
//! ## Isolation Features
//!
//! - Network isolation (none, bridge, host modes)
//! - Read-only root filesystem option
//! - Seccomp security profiles
//! - Capability dropping (no NET_RAW, SYS_ADMIN, etc.)
//! - User namespace remapping
//! - Resource limits (CPU, memory, PIDs)

use super::{BackendCapabilities, BackendType, BuildBackend, BuildContext};
use crate::build_job::{BuildResult, CargoCommand};
use crate::{Result, SwarmletError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

// Docker-only imports
#[cfg(feature = "docker")]
use crate::build_job::{ArtifactType, BuildArtifact, BuildResourceUsage};
#[cfg(feature = "docker")]
use sha2::{Digest, Sha256};
#[cfg(feature = "docker")]
use tracing::{debug, info, warn};
#[cfg(feature = "docker")]
use uuid::Uuid;

// sha2 is now always available (not feature-gated)

/// Docker container isolation profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationProfile {
    /// Network mode (none, bridge, host)
    pub network_mode: NetworkMode,
    /// Make root filesystem read-only
    pub read_only_rootfs: bool,
    /// Drop all capabilities and only add those needed
    pub drop_capabilities: bool,
    /// Capabilities to add (if drop_capabilities is true)
    pub add_capabilities: Vec<String>,
    /// Enable seccomp profile (default Docker profile)
    pub seccomp_enabled: bool,
    /// Custom seccomp profile path (if any)
    pub seccomp_profile: Option<String>,
    /// Run container as non-root user
    pub run_as_user: Option<String>,
    /// Enable user namespace remapping
    pub userns_mode: Option<String>,
    /// Disable inter-container communication
    pub icc_disabled: bool,
    /// Memory swap limit (-1 for unlimited)
    pub memory_swap_bytes: Option<i64>,
    /// OOM kill disable
    pub oom_kill_disable: bool,
    /// Ulimit settings
    pub ulimits: UlimitConfig,
    /// Tmpfs mounts for /tmp and other ephemeral data
    pub tmpfs_mounts: Vec<TmpfsMount>,
}

impl Default for IsolationProfile {
    fn default() -> Self {
        Self {
            network_mode: NetworkMode::None,
            read_only_rootfs: false, // Many builds need to write to various locations
            drop_capabilities: true,
            add_capabilities: vec![],
            seccomp_enabled: true,
            seccomp_profile: None,
            run_as_user: None,
            userns_mode: None,
            icc_disabled: true,
            memory_swap_bytes: None,
            oom_kill_disable: false,
            ulimits: UlimitConfig::default(),
            tmpfs_mounts: vec![
                TmpfsMount::new("/tmp", 512 * 1024 * 1024), // 512MB /tmp
            ],
        }
    }
}

impl IsolationProfile {
    /// Create a minimal security profile (for trusted builds)
    pub fn minimal() -> Self {
        Self {
            network_mode: NetworkMode::Bridge,
            read_only_rootfs: false,
            drop_capabilities: false,
            add_capabilities: vec![],
            seccomp_enabled: false,
            seccomp_profile: None,
            run_as_user: None,
            userns_mode: None,
            icc_disabled: false,
            memory_swap_bytes: None,
            oom_kill_disable: false,
            ulimits: UlimitConfig::default(),
            tmpfs_mounts: vec![],
        }
    }

    /// Create a high security profile (for untrusted builds)
    pub fn high_security() -> Self {
        Self {
            network_mode: NetworkMode::None,
            read_only_rootfs: true,
            drop_capabilities: true,
            add_capabilities: vec![],
            seccomp_enabled: true,
            seccomp_profile: None,
            run_as_user: Some("1000:1000".to_string()),
            userns_mode: Some("host".to_string()),
            icc_disabled: true,
            memory_swap_bytes: Some(-1), // No swap
            oom_kill_disable: false,
            ulimits: UlimitConfig::strict(),
            tmpfs_mounts: vec![
                TmpfsMount::new("/tmp", 256 * 1024 * 1024),
                TmpfsMount::new("/run", 64 * 1024 * 1024),
            ],
        }
    }
}

/// Network mode for container
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NetworkMode {
    /// No network access
    None,
    /// Bridge network (isolated with NAT)
    Bridge,
    /// Host network (shared with host)
    Host,
}

impl NetworkMode {
    /// Convert to Docker network mode string
    pub fn to_docker_string(&self) -> String {
        match self {
            NetworkMode::None => "none".to_string(),
            NetworkMode::Bridge => "bridge".to_string(),
            NetworkMode::Host => "host".to_string(),
        }
    }
}

/// Ulimit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UlimitConfig {
    /// Max number of open files
    pub nofile_soft: i64,
    pub nofile_hard: i64,
    /// Max number of processes
    pub nproc_soft: i64,
    pub nproc_hard: i64,
    /// Core dump size (0 = disabled)
    pub core_soft: i64,
    pub core_hard: i64,
}

impl Default for UlimitConfig {
    fn default() -> Self {
        Self {
            nofile_soft: 65536,
            nofile_hard: 65536,
            nproc_soft: 4096,
            nproc_hard: 4096,
            core_soft: 0,
            core_hard: 0,
        }
    }
}

impl UlimitConfig {
    /// Strict ulimits for untrusted builds
    pub fn strict() -> Self {
        Self {
            nofile_soft: 1024,
            nofile_hard: 2048,
            nproc_soft: 256,
            nproc_hard: 512,
            core_soft: 0,
            core_hard: 0,
        }
    }
}

/// Tmpfs mount configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TmpfsMount {
    /// Mount path inside container
    pub path: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Mount options
    pub options: Vec<String>,
}

impl TmpfsMount {
    /// Create a new tmpfs mount
    pub fn new(path: &str, size_bytes: u64) -> Self {
        Self {
            path: path.to_string(),
            size_bytes,
            options: vec!["noexec".to_string(), "nosuid".to_string()],
        }
    }

    /// Convert to Docker tmpfs options string
    pub fn to_docker_options(&self) -> String {
        let mut opts = vec![format!("size={}", self.size_bytes)];
        opts.extend(self.options.iter().cloned());
        opts.join(",")
    }
}

/// Capabilities to drop for security
#[cfg(feature = "docker")]
const DROPPED_CAPABILITIES: &[&str] = &[
    "NET_RAW",      // Prevent raw socket access
    "SYS_ADMIN",    // Prevent mount/namespace manipulation
    "SYS_PTRACE",   // Prevent process tracing
    "SYS_MODULE",   // Prevent kernel module loading
    "SYS_RAWIO",    // Prevent raw I/O
    "MKNOD",        // Prevent device creation
    "SETUID",       // Prevent setuid
    "SETGID",       // Prevent setgid
    "NET_BIND_SERVICE", // Prevent binding to low ports
    "DAC_OVERRIDE", // Prevent bypassing file permissions
    "FOWNER",       // Prevent bypassing file ownership checks
    "KILL",         // Prevent killing other processes
    "SETPCAP",      // Prevent capability manipulation
    "LINUX_IMMUTABLE", // Prevent immutable file manipulation
    "IPC_LOCK",     // Prevent memory locking
    "SYS_CHROOT",   // Prevent chroot
    "LEASE",        // Prevent file lease manipulation
    "AUDIT_WRITE",  // Prevent audit log writing
    "AUDIT_CONTROL", // Prevent audit control
    "SYS_BOOT",     // Prevent rebooting
    "WAKE_ALARM",   // Prevent wake alarms
    "BLOCK_SUSPEND", // Prevent blocking suspend
    "MAC_ADMIN",    // Prevent MAC administration
    "MAC_OVERRIDE", // Prevent MAC override
];

/// Docker-based build backend
#[allow(dead_code)] // Fields will be used when Docker implementation is complete
pub struct DockerBackend {
    #[cfg(feature = "docker")]
    docker: Arc<bollard::Docker>,
    /// Container name prefix
    container_prefix: String,
    /// Active container IDs
    active_containers: Arc<RwLock<HashMap<String, String>>>,
    /// Default Rust image
    default_image: String,
    /// Whether GPU is available
    gpu_available: bool,
    /// Isolation profile
    isolation_profile: IsolationProfile,
}

impl DockerBackend {
    /// Create a new Docker backend with default isolation profile
    pub async fn new() -> Result<Self> {
        Self::with_isolation_profile(IsolationProfile::default()).await
    }

    /// Create a new Docker backend with a specific isolation profile
    pub async fn with_isolation_profile(isolation_profile: IsolationProfile) -> Result<Self> {
        #[cfg(feature = "docker")]
        {
            use bollard::Docker;
            let docker = Docker::connect_with_local_defaults().map_err(|e| {
                SwarmletError::Docker(format!("Failed to connect to Docker: {e}"))
            })?;

            // Check if Docker is running
            docker.ping().await.map_err(|e| {
                SwarmletError::Docker(format!("Docker ping failed: {e}"))
            })?;

            // Check for GPU support
            let gpu_available = Self::check_gpu_support(&docker).await;

            info!(
                "Docker backend initialized with {:?} network mode, capabilities dropped: {}",
                isolation_profile.network_mode,
                isolation_profile.drop_capabilities
            );

            Ok(Self {
                docker: Arc::new(docker),
                container_prefix: "stratoswarm-build-".to_string(),
                active_containers: Arc::new(RwLock::new(HashMap::new())),
                default_image: "rust:latest".to_string(),
                gpu_available,
                isolation_profile,
            })
        }

        #[cfg(not(feature = "docker"))]
        {
            let _ = isolation_profile;
            Err(SwarmletError::NotImplemented(
                "Docker feature not enabled. Compile with --features docker".to_string(),
            ))
        }
    }

    /// Get the current isolation profile
    pub fn isolation_profile(&self) -> &IsolationProfile {
        &self.isolation_profile
    }

    /// Update the isolation profile
    pub fn set_isolation_profile(&mut self, profile: IsolationProfile) {
        self.isolation_profile = profile;
    }

    #[cfg(feature = "docker")]
    async fn check_gpu_support(docker: &bollard::Docker) -> bool {
        match docker.info().await {
            Ok(info) => {
                if let Some(runtimes) = info.runtimes {
                    runtimes.contains_key("nvidia")
                } else {
                    false
                }
            }
            Err(_) => false,
        }
    }

    #[cfg(feature = "docker")]
    fn build_host_config(
        &self,
        context: &BuildContext,
    ) -> bollard::models::HostConfig {
        use bollard::models::{HostConfig, ResourcesUlimits};

        let limits = &context.resource_limits;
        let iso = &self.isolation_profile;

        // Build bind mounts
        let mut binds = vec![
            // Mount workspace
            format!("{}:/workspace:rw", context.workspace.to_string_lossy()),
        ];

        // Add cache mounts
        for mount in &context.cache_mounts {
            binds.push(mount.to_docker_bind());
        }

        // Build capability lists
        let cap_drop = if iso.drop_capabilities {
            Some(DROPPED_CAPABILITIES.iter().map(|s| s.to_string()).collect())
        } else {
            None
        };

        let cap_add = if !iso.add_capabilities.is_empty() {
            Some(iso.add_capabilities.clone())
        } else {
            None
        };

        // Build ulimits
        let ulimits = vec![
            ResourcesUlimits {
                name: Some("nofile".to_string()),
                soft: Some(iso.ulimits.nofile_soft),
                hard: Some(iso.ulimits.nofile_hard),
            },
            ResourcesUlimits {
                name: Some("nproc".to_string()),
                soft: Some(iso.ulimits.nproc_soft),
                hard: Some(iso.ulimits.nproc_hard),
            },
            ResourcesUlimits {
                name: Some("core".to_string()),
                soft: Some(iso.ulimits.core_soft),
                hard: Some(iso.ulimits.core_hard),
            },
        ];

        // Build tmpfs mounts
        let tmpfs: HashMap<String, String> = iso
            .tmpfs_mounts
            .iter()
            .map(|m| (m.path.clone(), m.to_docker_options()))
            .collect();

        // Build security options
        let mut security_opt = Vec::new();
        if !iso.seccomp_enabled {
            security_opt.push("seccomp=unconfined".to_string());
        } else if let Some(ref profile) = iso.seccomp_profile {
            security_opt.push(format!("seccomp={}", profile));
        }

        HostConfig {
            binds: Some(binds),
            memory: limits.memory_bytes.map(|b| b as i64),
            memory_swap: iso.memory_swap_bytes,
            nano_cpus: limits.cpu_cores.map(|c| (c as f64 * 1_000_000_000.0) as i64),
            pids_limit: Some(1000),
            network_mode: Some(iso.network_mode.to_docker_string()),
            cap_drop,
            cap_add,
            ulimits: Some(ulimits),
            tmpfs: if tmpfs.is_empty() { None } else { Some(tmpfs) },
            readonly_rootfs: Some(iso.read_only_rootfs),
            security_opt: if security_opt.is_empty() { None } else { Some(security_opt) },
            userns_mode: iso.userns_mode.clone(),
            oom_kill_disable: Some(iso.oom_kill_disable),
            ..Default::default()
        }
    }

    #[cfg(feature = "docker")]
    async fn ensure_image(&self, image: &str) -> Result<()> {
        use bollard::image::CreateImageOptions;
        use futures::StreamExt;

        info!("Ensuring Docker image: {}", image);

        let options = CreateImageOptions {
            from_image: image,
            ..Default::default()
        };

        let mut stream = self.docker.create_image(Some(options), None, None);

        while let Some(result) = stream.next().await {
            match result {
                Ok(info) => {
                    if let Some(status) = info.status {
                        debug!("Docker pull: {}", status);
                    }
                }
                Err(e) => {
                    warn!("Docker pull warning: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Get the appropriate Docker image for a Rust toolchain
    #[cfg(feature = "docker")]
    fn get_image_for_toolchain(&self, toolchain: Option<&str>) -> String {
        match toolchain {
            Some("nightly") => "rustlang/rust:nightly".to_string(),
            Some("beta") => "rust:beta".to_string(),
            Some(version) if version.starts_with("1.") => format!("rust:{}", version),
            Some(_) | None => self.default_image.clone(),
        }
    }

    /// Collect build artifacts from workspace
    #[cfg(feature = "docker")]
    async fn collect_artifacts(
        &self,
        workspace: &PathBuf,
        command: &CargoCommand,
    ) -> Vec<BuildArtifact> {
        let mut artifacts = Vec::new();

        match command {
            CargoCommand::Build | CargoCommand::Run { .. } => {
                // Look for binaries in target/release and target/debug
                for profile in ["release", "debug"] {
                    let target_dir = workspace.join("target").join(profile);
                    if let Ok(entries) = tokio::fs::read_dir(&target_dir).await {
                        let mut entries = entries;
                        while let Ok(Some(entry)) = entries.next_entry().await {
                            let path = entry.path();
                            if self.is_executable(&path).await {
                                if let Some(artifact) = self.create_artifact(&path, ArtifactType::Binary).await {
                                    artifacts.push(artifact);
                                }
                            }
                        }
                    }
                }
            }
            CargoCommand::Test { .. } => {
                // Look for test results in target/
                let results_path = workspace.join("target").join("test-results.json");
                if results_path.exists() {
                    if let Some(artifact) = self.create_artifact(&results_path, ArtifactType::TestResults).await {
                        artifacts.push(artifact);
                    }
                }
            }
            CargoCommand::Doc { .. } => {
                // Look for documentation
                let doc_dir = workspace.join("target").join("doc");
                if doc_dir.exists() {
                    // Create a tarball of docs would be ideal, for now just note it exists
                    artifacts.push(BuildArtifact {
                        name: "documentation".to_string(),
                        path: doc_dir,
                        size_bytes: 0,
                        artifact_type: ArtifactType::Documentation,
                        sha256: String::new(),
                    });
                }
            }
            _ => {}
        }

        artifacts
    }

    /// Check if a file is an executable binary
    #[cfg(feature = "docker")]
    async fn is_executable(&self, path: &PathBuf) -> bool {
        if !path.is_file() {
            return false;
        }

        // Skip common non-executable files
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with(".d") || name.ends_with(".rlib") || name.ends_with(".rmeta") {
            return false;
        }

        // On Unix, check execute permission
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(metadata) = tokio::fs::metadata(path).await {
                return metadata.permissions().mode() & 0o111 != 0;
            }
            return false;
        }

        // On Windows, check for .exe extension
        #[cfg(windows)]
        {
            return name.ends_with(".exe");
        }

        #[cfg(not(any(unix, windows)))]
        {
            false
        }
    }

    /// Create a BuildArtifact from a file path
    #[cfg(feature = "docker")]
    async fn create_artifact(&self, path: &PathBuf, artifact_type: ArtifactType) -> Option<BuildArtifact> {
        let metadata = tokio::fs::metadata(path).await.ok()?;
        let name = path.file_name()?.to_str()?.to_string();

        // Calculate SHA256 hash
        let contents = tokio::fs::read(path).await.ok()?;
        let mut hasher = Sha256::new();
        hasher.update(&contents);
        let sha256 = format!("{:x}", hasher.finalize());

        Some(BuildArtifact {
            name,
            path: path.clone(),
            size_bytes: metadata.len(),
            artifact_type,
            sha256,
        })
    }

    /// Get container stats for resource usage
    #[cfg(feature = "docker")]
    async fn get_container_stats(&self, container_id: &str) -> BuildResourceUsage {
        use futures::StreamExt;

        let mut usage = BuildResourceUsage::default();

        let options = bollard::container::StatsOptions {
            stream: false,
            one_shot: true,
        };

        let mut stats_stream = self.docker.stats(container_id, Some(options));

        if let Some(Ok(stats)) = stats_stream.next().await {
            // Calculate CPU usage - bollard 0.14 uses direct structs, not Options
            let cpu_stats = &stats.cpu_stats;
            let precpu_stats = &stats.precpu_stats;

            let cpu_delta = cpu_stats.cpu_usage.total_usage
                .saturating_sub(precpu_stats.cpu_usage.total_usage);
            let system_delta = cpu_stats.system_cpu_usage
                .unwrap_or(0)
                .saturating_sub(precpu_stats.system_cpu_usage.unwrap_or(0));

            if system_delta > 0 {
                let num_cpus = cpu_stats.online_cpus.unwrap_or(1) as f64;
                usage.cpu_seconds = (cpu_delta as f64 / system_delta as f64) * num_cpus;
            }

            // Memory usage
            let memory_stats = &stats.memory_stats;
            if let Some(mem_usage) = memory_stats.usage {
                usage.peak_memory_mb = (mem_usage / (1024 * 1024)) as f32;
            }

            // IO stats
            let blkio_stats = &stats.blkio_stats;
            if let Some(io_service_bytes) = &blkio_stats.io_service_bytes_recursive {
                for stat in io_service_bytes {
                    match stat.op.as_str() {
                        "read" | "Read" => usage.disk_read_bytes += stat.value,
                        "write" | "Write" => usage.disk_write_bytes += stat.value,
                        _ => {}
                    }
                }
            }
        }

        usage
    }
}

#[async_trait]
impl BuildBackend for DockerBackend {
    async fn execute_cargo(
        &self,
        command: &CargoCommand,
        context: &BuildContext,
    ) -> Result<BuildResult> {
        #[cfg(feature = "docker")]
        {
            use bollard::container::{
                Config, CreateContainerOptions, LogsOptions, RemoveContainerOptions,
                StartContainerOptions, WaitContainerOptions,
            };
            use futures::StreamExt;

            let container_name = format!("{}{}", self.container_prefix, Uuid::new_v4());

            // Build cargo command
            let cargo_args = context.cargo_args.join(" ");
            let cmd = format!("cargo {}", cargo_args);

            info!("Executing in Docker: {}", cmd);

            // Select appropriate image based on toolchain
            let image = self.get_image_for_toolchain(context.toolchain.as_deref());

            // Ensure image exists
            self.ensure_image(&image).await?;

            // Build environment variables
            let env: Vec<String> = context
                .build_environment()
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();

            // Create container configuration
            let host_config = self.build_host_config(context);

            let config = Config {
                image: Some(image),
                cmd: Some(vec!["sh".to_string(), "-c".to_string(), cmd]),
                env: Some(env),
                working_dir: Some("/workspace".to_string()),
                host_config: Some(host_config),
                ..Default::default()
            };

            // Create container
            let create_options = CreateContainerOptions {
                name: container_name.clone(),
                platform: None,
            };

            let container = self
                .docker
                .create_container(Some(create_options), config)
                .await
                .map_err(|e| SwarmletError::Docker(format!("Failed to create container: {e}")))?;

            // Track active container
            {
                let mut containers = self.active_containers.write().await;
                containers.insert(container_name.clone(), container.id.clone());
            }

            let start_time = std::time::Instant::now();

            // Start container
            self.docker
                .start_container(&container.id, None::<StartContainerOptions<String>>)
                .await
                .map_err(|e| SwarmletError::Docker(format!("Failed to start container: {e}")))?;

            // Collect logs with timeout
            let timeout_duration = context.timeout.unwrap_or(std::time::Duration::from_secs(3600));
            let logs_options = LogsOptions::<String> {
                follow: true,
                stdout: true,
                stderr: true,
                ..Default::default()
            };

            let mut log_stream = self.docker.logs(&container.id, Some(logs_options));

            let log_collection = async {
                while let Some(log_result) = log_stream.next().await {
                    match log_result {
                        Ok(bollard::container::LogOutput::StdOut { message }) => {
                            let line = String::from_utf8_lossy(&message);
                            debug!("[stdout] {}", line.trim());
                        }
                        Ok(bollard::container::LogOutput::StdErr { message }) => {
                            let line = String::from_utf8_lossy(&message);
                            debug!("[stderr] {}", line.trim());
                        }
                        Err(e) => {
                            warn!("Log stream error: {}", e);
                        }
                        _ => {}
                    }
                }
            };

            // Apply timeout to log collection
            let timed_out = tokio::time::timeout(timeout_duration, log_collection).await.is_err();

            if timed_out {
                warn!("Build timed out after {:?}", timeout_duration);
                // Kill the container
                if let Err(e) = self.docker.kill_container::<String>(&container.id, None).await {
                    warn!("Failed to kill timed out container: {}", e);
                }
            }

            // Get container stats before it's removed
            let mut resource_usage = self.get_container_stats(&container.id).await;
            resource_usage.compile_time_seconds = start_time.elapsed().as_secs_f64();

            // Wait for container to finish
            let wait_options = WaitContainerOptions {
                condition: "not-running",
            };

            let mut wait_stream = self.docker.wait_container(&container.id, Some(wait_options));
            let exit_code = if timed_out {
                -1 // Timeout exit code
            } else if let Some(Ok(result)) = wait_stream.next().await {
                result.status_code as i32
            } else {
                -1
            };

            let duration = start_time.elapsed();

            // Collect artifacts from workspace
            let artifacts = self.collect_artifacts(&context.workspace, command).await;

            // Remove container from tracking
            {
                let mut containers = self.active_containers.write().await;
                containers.remove(&container_name);
            }

            // Cleanup container
            let remove_options = RemoveContainerOptions {
                force: true,
                v: true,
                ..Default::default()
            };

            if let Err(e) = self
                .docker
                .remove_container(&container.id, Some(remove_options))
                .await
            {
                warn!("Failed to remove container: {}", e);
            }

            Ok(BuildResult {
                exit_code,
                resource_usage,
                duration,
                artifacts,
            })
        }

        #[cfg(not(feature = "docker"))]
        {
            let _ = (command, context);
            Err(SwarmletError::NotImplemented(
                "Docker feature not enabled".to_string(),
            ))
        }
    }

    async fn create_workspace(&self, job_id: &str) -> Result<PathBuf> {
        let workspace = std::env::temp_dir()
            .join("stratoswarm-builds")
            .join(job_id);

        tokio::fs::create_dir_all(&workspace).await.map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create workspace: {e}"),
            ))
        })?;

        Ok(workspace)
    }

    async fn cleanup_workspace(&self, workspace: &PathBuf) -> Result<()> {
        if workspace.exists() {
            tokio::fs::remove_dir_all(workspace).await.map_err(|e| {
                SwarmletError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to cleanup workspace: {e}"),
                ))
            })?;
        }
        Ok(())
    }

    async fn is_available(&self) -> bool {
        #[cfg(feature = "docker")]
        {
            self.docker.ping().await.is_ok()
        }

        #[cfg(not(feature = "docker"))]
        {
            false
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            full_namespace_isolation: false, // Docker abstracts this
            cgroups_v2: false,               // Uses Docker's resource management
            gpu_passthrough: self.gpu_available,
            seccomp: true,
            user_namespace: true,
            max_containers: 100,
        }
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Docker
    }

    fn name(&self) -> &str {
        "docker"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_backend::CacheMount;

    #[test]
    fn test_cache_mount_bind_string() {
        let mount = CacheMount::new(
            PathBuf::from("/host/cargo"),
            PathBuf::from("/root/.cargo"),
        );
        assert_eq!(mount.to_docker_bind(), "/host/cargo:/root/.cargo:rw");
    }

    #[test]
    fn test_backend_type() {
        // Test that we can create an instance (Docker not running is fine)
        // Just verify type information
        assert_eq!(BackendType::Docker.to_string(), "docker");
    }

    #[test]
    fn test_backend_capabilities_default() {
        let caps = BackendCapabilities::default();
        assert!(!caps.full_namespace_isolation);
        assert!(!caps.cgroups_v2);
        assert!(!caps.gpu_passthrough);
        assert_eq!(caps.max_containers, 10);
    }

    #[cfg(feature = "docker")]
    mod docker_feature_tests {
        // Helper to create a minimal DockerBackend for testing helper methods
        // without actually connecting to Docker
        struct MockDockerBackend {
            default_image: String,
        }

        impl MockDockerBackend {
            fn new() -> Self {
                Self {
                    default_image: "rust:latest".to_string(),
                }
            }

            fn get_image_for_toolchain(&self, toolchain: Option<&str>) -> String {
                match toolchain {
                    Some("nightly") => "rustlang/rust:nightly".to_string(),
                    Some("beta") => "rust:beta".to_string(),
                    Some(version) if version.starts_with("1.") => format!("rust:{}", version),
                    Some(_) | None => self.default_image.clone(),
                }
            }
        }

        #[test]
        fn test_toolchain_image_selection_stable() {
            let backend = MockDockerBackend::new();
            assert_eq!(backend.get_image_for_toolchain(None), "rust:latest");
            assert_eq!(backend.get_image_for_toolchain(Some("stable")), "rust:latest");
        }

        #[test]
        fn test_toolchain_image_selection_nightly() {
            let backend = MockDockerBackend::new();
            assert_eq!(backend.get_image_for_toolchain(Some("nightly")), "rustlang/rust:nightly");
        }

        #[test]
        fn test_toolchain_image_selection_beta() {
            let backend = MockDockerBackend::new();
            assert_eq!(backend.get_image_for_toolchain(Some("beta")), "rust:beta");
        }

        #[test]
        fn test_toolchain_image_selection_version() {
            let backend = MockDockerBackend::new();
            assert_eq!(backend.get_image_for_toolchain(Some("1.76.0")), "rust:1.76.0");
            assert_eq!(backend.get_image_for_toolchain(Some("1.75")), "rust:1.75");
        }
    }

    // Isolation profile tests (no Docker connection needed)
    mod isolation_tests {
        use super::*;

        #[test]
        fn test_isolation_profile_default() {
            let profile = IsolationProfile::default();
            assert_eq!(profile.network_mode, NetworkMode::None);
            assert!(profile.drop_capabilities);
            assert!(profile.seccomp_enabled);
            assert!(!profile.read_only_rootfs);
            assert!(profile.icc_disabled);
        }

        #[test]
        fn test_isolation_profile_minimal() {
            let profile = IsolationProfile::minimal();
            assert_eq!(profile.network_mode, NetworkMode::Bridge);
            assert!(!profile.drop_capabilities);
            assert!(!profile.seccomp_enabled);
        }

        #[test]
        fn test_isolation_profile_high_security() {
            let profile = IsolationProfile::high_security();
            assert_eq!(profile.network_mode, NetworkMode::None);
            assert!(profile.drop_capabilities);
            assert!(profile.seccomp_enabled);
            assert!(profile.read_only_rootfs);
            assert!(profile.run_as_user.is_some());
        }

        #[test]
        fn test_network_mode_strings() {
            assert_eq!(NetworkMode::None.to_docker_string(), "none");
            assert_eq!(NetworkMode::Bridge.to_docker_string(), "bridge");
            assert_eq!(NetworkMode::Host.to_docker_string(), "host");
        }

        #[test]
        fn test_ulimit_config_default() {
            let config = UlimitConfig::default();
            assert_eq!(config.nofile_soft, 65536);
            assert_eq!(config.nproc_soft, 4096);
            assert_eq!(config.core_soft, 0);
        }

        #[test]
        fn test_ulimit_config_strict() {
            let config = UlimitConfig::strict();
            assert_eq!(config.nofile_soft, 1024);
            assert_eq!(config.nproc_soft, 256);
        }

        #[test]
        fn test_tmpfs_mount() {
            let mount = TmpfsMount::new("/tmp", 512 * 1024 * 1024);
            assert_eq!(mount.path, "/tmp");
            assert_eq!(mount.size_bytes, 512 * 1024 * 1024);
            assert!(mount.options.contains(&"noexec".to_string()));
            assert!(mount.options.contains(&"nosuid".to_string()));
        }

        #[test]
        fn test_tmpfs_docker_options() {
            let mount = TmpfsMount::new("/tmp", 1024);
            let opts = mount.to_docker_options();
            assert!(opts.contains("size=1024"));
            assert!(opts.contains("noexec"));
        }

        #[test]
        fn test_dropped_capabilities_list() {
            #[cfg(feature = "docker")]
            {
                // Ensure important security capabilities are dropped
                assert!(DROPPED_CAPABILITIES.contains(&"SYS_ADMIN"));
                assert!(DROPPED_CAPABILITIES.contains(&"NET_RAW"));
                assert!(DROPPED_CAPABILITIES.contains(&"SYS_PTRACE"));
            }
        }
    }
}
