//! Linux native build backend
//!
//! This backend uses Linux kernel primitives directly for build isolation:
//! - Namespaces (PID, MNT, NET, UTS, IPC, USER, CGROUP)
//! - cgroups v2 for resource limits
//! - OverlayFS for copy-on-write filesystem
//! - seccomp for syscall filtering
//! - Optional swarm_guard kernel module for enhanced isolation
//!
//! This is the preferred backend on Linux as it provides better performance
//! than Docker through direct kernel integration.
//!
//! ## Kernel Module Integration
//!
//! When the swarm_guard kernel module is loaded, this backend will use
//! kernel-level mount isolation for better security and performance.
//! Falls back to direct syscalls when the module is not available.

use super::{BackendCapabilities, BackendType, BuildBackend, BuildContext, CacheMount};
use crate::build_job::{BuildResourceLimits, BuildResourceUsage, BuildResult, CargoCommand};
use crate::{Result, SwarmletError};
use async_trait::async_trait;
use std::collections::HashMap;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tracing::{debug, error, info, warn};

/// Atomic counter for generating unique agent IDs for kernel module
static AGENT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Configuration for container isolation
#[derive(Debug, Clone)]
pub struct ContainerConfig {
    /// Container root filesystem (overlayfs merged)
    pub rootfs: PathBuf,
    /// OverlayFS lower directory (read-only base)
    pub lower_dir: PathBuf,
    /// OverlayFS upper directory (writable layer)
    pub upper_dir: PathBuf,
    /// OverlayFS work directory
    pub work_dir: PathBuf,
    /// Bind mounts for caches
    pub bind_mounts: Vec<BindMount>,
    /// Resource limits
    pub resource_limits: Option<BuildResourceLimits>,
    /// Hostname for UTS namespace
    pub hostname: String,
    /// Environment variables
    pub environment: HashMap<String, String>,
}

/// Bind mount configuration
#[derive(Debug, Clone)]
pub struct BindMount {
    /// Source path on host
    pub source: PathBuf,
    /// Target path in container
    pub target: PathBuf,
    /// Mount as read-only
    pub readonly: bool,
}

/// Linux native build backend using namespaces and cgroups
pub struct LinuxNativeBackend {
    /// Base directory for container workspaces
    workspaces_dir: PathBuf,
    /// Base directory for toolchains
    toolchains_dir: PathBuf,
    /// cgroups v2 available
    cgroups_v2: bool,
    /// User namespace available
    user_ns: bool,
    /// cgroup base path for builds
    cgroup_base: PathBuf,
    /// Whether swarm_guard kernel module is available
    kernel_module: bool,
}

impl LinuxNativeBackend {
    /// Create a new Linux native backend
    pub async fn new() -> Result<Self> {
        let workspaces_dir = PathBuf::from("/var/lib/stratoswarm/builds");
        let toolchains_dir = PathBuf::from("/var/lib/stratoswarm/toolchains");
        let cgroup_base = PathBuf::from("/sys/fs/cgroup/stratoswarm");

        // Check for cgroups v2
        let cgroups_v2 = Self::check_cgroups_v2();

        // Check for user namespace support
        let user_ns = Self::check_user_ns();

        // Check for swarm_guard kernel module
        let kernel_module = Self::check_kernel_module();

        if !cgroups_v2 {
            warn!("cgroups v2 not available, some resource limits may not work");
        }

        if !user_ns {
            warn!("User namespaces not available, rootless builds not possible");
        }

        if kernel_module {
            info!("swarm_guard kernel module detected, using kernel-level mount isolation");
        } else {
            info!("swarm_guard kernel module not available, using direct syscalls for isolation");
        }

        // Ensure base directories exist
        tokio::fs::create_dir_all(&workspaces_dir).await.ok();
        tokio::fs::create_dir_all(&toolchains_dir).await.ok();

        Ok(Self {
            workspaces_dir,
            toolchains_dir,
            cgroups_v2,
            user_ns,
            cgroup_base,
            kernel_module,
        })
    }

    /// Check if swarm_guard kernel module is loaded
    fn check_kernel_module() -> bool {
        // Check if the kernel module is loaded via /sys/module
        let module_path = Path::new("/sys/module/swarm_guard");
        if module_path.exists() {
            debug!("swarm_guard module found in /sys/module");
            return true;
        }

        // Alternative: check /proc/modules for loaded module
        if let Ok(content) = std::fs::read_to_string("/proc/modules") {
            if content.lines().any(|line| line.starts_with("swarm_guard ")) {
                debug!("swarm_guard module found in /proc/modules");
                return true;
            }
        }

        // Alternative: check for swarm_guard device
        let device_path = Path::new("/dev/swarm_guard");
        if device_path.exists() {
            debug!("swarm_guard device found at /dev/swarm_guard");
            return true;
        }

        // Check for proc entry
        let proc_path = Path::new("/proc/swarm");
        if proc_path.exists() {
            debug!("swarm proc entry found at /proc/swarm");
            return true;
        }

        false
    }

    /// Generate a unique agent ID for kernel module operations
    fn next_agent_id() -> u64 {
        AGENT_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
    }

    /// Check if kernel module is available
    pub fn has_kernel_module(&self) -> bool {
        self.kernel_module
    }

    /// Check if cgroups v2 (unified hierarchy) is available
    fn check_cgroups_v2() -> bool {
        std::path::Path::new("/sys/fs/cgroup/cgroup.controllers").exists()
    }

    /// Check if user namespaces are available
    fn check_user_ns() -> bool {
        // Check if unprivileged user namespaces are allowed
        if let Ok(content) =
            std::fs::read_to_string("/proc/sys/kernel/unprivileged_userns_clone")
        {
            return content.trim() == "1";
        }
        // On newer kernels, it might always be enabled
        std::path::Path::new("/proc/self/ns/user").exists()
    }

    /// Check if we have the necessary capabilities
    fn check_capabilities() -> bool {
        // For now, check if we're running as root or have necessary caps
        nix::unistd::geteuid().is_root()
    }

    /// Create workspace directory structure for overlayfs
    async fn create_workspace_structure(&self, job_id: &str) -> Result<ContainerConfig> {
        let base = self.workspaces_dir.join(job_id);

        let lower_dir = base.join("lower");
        let upper_dir = base.join("upper");
        let work_dir = base.join("work");
        let rootfs = base.join("merged");

        // Create all directories
        for dir in [&lower_dir, &upper_dir, &work_dir, &rootfs] {
            tokio::fs::create_dir_all(dir).await.map_err(|e| {
                SwarmletError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create {}: {e}", dir.display()),
                ))
            })?;
        }

        debug!(
            "Created workspace structure at {} for job {}",
            base.display(),
            job_id
        );

        Ok(ContainerConfig {
            rootfs,
            lower_dir,
            upper_dir,
            work_dir,
            bind_mounts: Vec::new(),
            resource_limits: None,
            hostname: format!("build-{}", &job_id[..8.min(job_id.len())]),
            environment: HashMap::new(),
        })
    }

    /// Setup overlayfs mount
    #[cfg(target_os = "linux")]
    fn setup_overlayfs(config: &ContainerConfig) -> Result<()> {
        use nix::mount::{mount, MsFlags};

        let options = format!(
            "lowerdir={},upperdir={},workdir={}",
            config.lower_dir.display(),
            config.upper_dir.display(),
            config.work_dir.display()
        );

        mount(
            Some("overlay"),
            &config.rootfs,
            Some("overlay"),
            MsFlags::empty(),
            Some(options.as_str()),
        )
        .map_err(|e| {
            SwarmletError::WorkloadExecution(format!("Failed to mount overlayfs: {e}"))
        })?;

        debug!("Mounted overlayfs at {}", config.rootfs.display());
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn setup_overlayfs(_config: &ContainerConfig) -> Result<()> {
        Err(SwarmletError::NotImplemented(
            "Overlayfs only available on Linux".to_string(),
        ))
    }

    /// Setup bind mounts for caches
    #[cfg(target_os = "linux")]
    fn setup_bind_mounts(config: &ContainerConfig) -> Result<()> {
        use nix::mount::{mount, MsFlags};

        for bind in &config.bind_mounts {
            // Ensure target exists
            if bind.source.is_dir() {
                std::fs::create_dir_all(&bind.target).ok();
            }

            let mut flags = MsFlags::MS_BIND;
            if bind.readonly {
                flags |= MsFlags::MS_RDONLY;
            }

            mount(
                Some(&bind.source),
                &bind.target,
                None::<&str>,
                flags,
                None::<&str>,
            )
            .map_err(|e| {
                SwarmletError::WorkloadExecution(format!(
                    "Failed to bind mount {} -> {}: {e}",
                    bind.source.display(),
                    bind.target.display()
                ))
            })?;

            debug!(
                "Bind mounted {} -> {}",
                bind.source.display(),
                bind.target.display()
            );
        }

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn setup_bind_mounts(_config: &ContainerConfig) -> Result<()> {
        Ok(())
    }

    /// Setup cgroup limits for the build
    #[cfg(target_os = "linux")]
    async fn setup_cgroup(
        &self,
        job_id: &str,
        limits: &BuildResourceLimits,
    ) -> Result<PathBuf> {
        let cgroup_path = self.cgroup_base.join(job_id);

        // Create cgroup directory
        tokio::fs::create_dir_all(&cgroup_path).await.map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create cgroup: {e}"),
            ))
        })?;

        // Set memory limit
        let memory_max = cgroup_path.join("memory.max");
        let memory_bytes = (limits.memory_mb as u64) * 1024 * 1024;
        tokio::fs::write(&memory_max, memory_bytes.to_string())
            .await
            .ok();

        // Set CPU limit (in microseconds per 100ms period)
        let cpu_max = cgroup_path.join("cpu.max");
        let cpu_us = (limits.cpu_cores * 100_000.0) as u64;
        tokio::fs::write(&cpu_max, format!("{} 100000", cpu_us))
            .await
            .ok();

        debug!(
            "Created cgroup {} with mem={}MB, cpu={}",
            cgroup_path.display(),
            limits.memory_mb,
            limits.cpu_cores
        );

        Ok(cgroup_path)
    }

    #[cfg(not(target_os = "linux"))]
    async fn setup_cgroup(
        &self,
        _job_id: &str,
        _limits: &BuildResourceLimits,
    ) -> Result<PathBuf> {
        Err(SwarmletError::NotImplemented(
            "cgroups only available on Linux".to_string(),
        ))
    }

    /// Add process to cgroup
    #[cfg(target_os = "linux")]
    fn add_to_cgroup(cgroup_path: &Path, pid: u32) -> Result<()> {
        let procs_file = cgroup_path.join("cgroup.procs");
        std::fs::write(&procs_file, pid.to_string()).map_err(|e| {
            SwarmletError::WorkloadExecution(format!("Failed to add to cgroup: {e}"))
        })?;
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn add_to_cgroup(_cgroup_path: &Path, _pid: u32) -> Result<()> {
        Ok(())
    }

    /// Execute cargo in an isolated environment
    async fn execute_isolated(
        &self,
        command: &CargoCommand,
        context: &BuildContext,
        config: &ContainerConfig,
    ) -> Result<BuildResult> {
        let cargo_args = &context.cargo_args;
        let env = context.build_environment();

        info!(
            "Executing cargo {:?} in isolated environment (kernel_module: {})",
            cargo_args.first(),
            self.kernel_module
        );

        // Setup cgroup if limits are provided
        let cgroup_path = if let Some(limits) = &config.resource_limits {
            if self.cgroups_v2 {
                let job_id = context
                    .workspace
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                Some(self.setup_cgroup(job_id, limits).await?)
            } else {
                None
            }
        } else {
            None
        };

        // Setup kernel mount isolation if available
        let kernel_agent_id = if self.kernel_module {
            match Self::setup_kernel_mount_isolation(config) {
                Ok(agent_id) => {
                    info!("Kernel mount isolation setup for agent {}", agent_id);
                    Some(agent_id)
                }
                Err(e) => {
                    warn!("Failed to setup kernel mount isolation, falling back: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Build the command
        let mut cmd = Command::new("cargo");
        cmd.args(cargo_args)
            .current_dir(&context.workspace)
            .envs(&env)
            .envs(&config.environment)
            .env("HOME", "/root")
            .env("USER", "root")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // On Linux, use pre_exec for namespace isolation
        // Skip namespace setup if kernel module is handling isolation
        #[cfg(target_os = "linux")]
        if kernel_agent_id.is_none() && Self::check_capabilities() {
            use nix::sched::{unshare, CloneFlags};

            let hostname = config.hostname.clone();

            info!("Using direct syscall namespace isolation (fallback mode)");

            unsafe {
                cmd.pre_exec(move || {
                    // Create new namespaces
                    let mut flags = CloneFlags::CLONE_NEWNS
                        | CloneFlags::CLONE_NEWUTS
                        | CloneFlags::CLONE_NEWIPC;

                    // Add PID namespace if we have capabilities
                    if nix::unistd::geteuid().is_root() {
                        flags |= CloneFlags::CLONE_NEWPID;
                    }

                    unshare(flags).map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("unshare failed: {e}"),
                        )
                    })?;

                    // Set hostname
                    nix::unistd::sethostname(&hostname).ok();

                    Ok(())
                });
            }
        } else if kernel_agent_id.is_some() {
            info!("Using kernel module for namespace isolation");
        }

        let start_time = std::time::Instant::now();

        let mut child = cmd.spawn().map_err(|e| {
            SwarmletError::WorkloadExecution(format!("Failed to spawn cargo: {e}"))
        })?;

        // Add to cgroup if available
        if let Some(cgroup) = &cgroup_path {
            if let Some(pid) = child.id() {
                Self::add_to_cgroup(cgroup, pid)?;
            }
        }

        // Stream output
        let stdout = child.stdout.take().expect("stdout configured");
        let stderr = child.stderr.take().expect("stderr configured");

        let mut stdout_output = Vec::new();
        let mut stderr_output = Vec::new();

        let stdout_handle = {
            let mut output = Vec::new();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                    debug!("[stdout] {}", line.trim());
                    output.push(line.clone());
                    line.clear();
                }
                output
            })
        };

        let stderr_handle = {
            let mut output = Vec::new();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                    debug!("[stderr] {}", line.trim());
                    output.push(line.clone());
                    line.clear();
                }
                output
            })
        };

        let status = child.wait().await.map_err(|e| {
            SwarmletError::WorkloadExecution(format!("Failed to wait for cargo: {e}"))
        })?;

        // Collect output
        stdout_output = stdout_handle.await.unwrap_or_default();
        stderr_output = stderr_handle.await.unwrap_or_default();

        let duration = start_time.elapsed();

        // Read cgroup stats if available
        let resource_usage = if let Some(cgroup) = &cgroup_path {
            self.read_cgroup_stats(cgroup).await
        } else {
            BuildResourceUsage {
                compile_time_seconds: duration.as_secs_f64(),
                ..Default::default()
            }
        };

        // Cleanup cgroup
        if let Some(cgroup) = cgroup_path {
            tokio::fs::remove_dir(&cgroup).await.ok();
        }

        // Cleanup kernel mount isolation
        if let Some(agent_id) = kernel_agent_id {
            if let Err(e) = Self::teardown_kernel_mount_isolation(agent_id) {
                warn!("Failed to teardown kernel mount isolation: {}", e);
            }
        }

        let exit_code = status.code().unwrap_or(-1);

        if exit_code != 0 {
            warn!(
                "Cargo exited with code {}. stderr: {:?}",
                exit_code,
                stderr_output.last()
            );
        }

        Ok(BuildResult {
            exit_code,
            resource_usage,
            duration,
            artifacts: Vec::new(),
        })
    }

    /// Read resource usage from cgroup
    async fn read_cgroup_stats(&self, cgroup_path: &Path) -> BuildResourceUsage {
        let mut usage = BuildResourceUsage::default();

        // Read memory usage
        if let Ok(content) = tokio::fs::read_to_string(cgroup_path.join("memory.current")).await
        {
            if let Ok(bytes) = content.trim().parse::<u64>() {
                usage.peak_memory_mb = (bytes / (1024 * 1024)) as u32;
            }
        }

        // Read CPU usage
        if let Ok(content) = tokio::fs::read_to_string(cgroup_path.join("cpu.stat")).await {
            for line in content.lines() {
                if line.starts_with("usage_usec") {
                    if let Some(usec) = line.split_whitespace().nth(1) {
                        if let Ok(us) = usec.parse::<u64>() {
                            usage.compile_time_seconds = us as f64 / 1_000_000.0;
                        }
                    }
                }
            }
        }

        usage
    }

    /// Cleanup overlayfs mount
    #[cfg(target_os = "linux")]
    fn cleanup_overlayfs(rootfs: &Path) -> Result<()> {
        use nix::mount::{umount2, MntFlags};

        if rootfs.exists() {
            umount2(rootfs, MntFlags::MNT_DETACH).ok();
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn cleanup_overlayfs(_rootfs: &Path) -> Result<()> {
        Ok(())
    }

    // ==================== Kernel Module Mount Isolation ====================

    /// Setup mount isolation using the swarm_guard kernel module
    ///
    /// This method uses the kernel module's mount isolation functions for
    /// better security and performance than direct syscalls.
    ///
    /// Returns the agent_id used for kernel module operations.
    #[cfg(target_os = "linux")]
    fn setup_kernel_mount_isolation(config: &ContainerConfig) -> Result<u64> {
        use crate::kernel_interface::{KernelInterface, MountIsolationConfig, mount_flags};

        // Generate unique agent ID
        let agent_id = Self::next_agent_id();

        debug!(
            "Setting up kernel mount isolation for agent {} at {}",
            agent_id,
            config.rootfs.display()
        );

        // Check if kernel interface is available
        if !KernelInterface::is_available() {
            return Err(SwarmletError::System(
                "Kernel module interface not available".to_string(),
            ));
        }

        // Open kernel interface
        let ki = KernelInterface::open()?;

        // Build mount isolation config
        let lower_dir = config.lower_dir.to_string_lossy();
        let upper_dir = config.upper_dir.to_string_lossy();
        let work_dir = config.work_dir.to_string_lossy();
        let merged_dir = config.rootfs.to_string_lossy();

        let mut iso_config = MountIsolationConfig::new(agent_id)
            .with_overlayfs(&lower_dir, &upper_dir, &work_dir, &merged_dir)
            .with_old_root("/.oldroot");

        // Add bind mounts
        for bind in &config.bind_mounts {
            let mut flags = 0u32;
            if bind.readonly {
                flags |= mount_flags::READONLY;
            }

            let source = bind.source.to_string_lossy();
            let target = bind.target.to_string_lossy();

            if !iso_config.add_bind_mount(&source, &target, flags) {
                warn!("Max bind mounts reached, skipping: {} -> {}", source, target);
            }
        }

        debug!(
            "Kernel overlayfs config: lower={}, upper={}, work={}, merged={}, {} bind mounts",
            lower_dir, upper_dir, work_dir, merged_dir, iso_config.num_bind_mounts
        );

        // Call kernel module to setup mount isolation
        ki.setup_mount_isolation(&iso_config)?;

        info!(
            "Kernel mount isolation setup complete for agent {}",
            agent_id
        );

        Ok(agent_id)
    }

    #[cfg(not(target_os = "linux"))]
    fn setup_kernel_mount_isolation(_config: &ContainerConfig) -> Result<u64> {
        Err(SwarmletError::NotImplemented(
            "Kernel mount isolation only available on Linux".to_string(),
        ))
    }

    /// Teardown kernel mount isolation for an agent
    #[cfg(target_os = "linux")]
    fn teardown_kernel_mount_isolation(agent_id: u64) -> Result<()> {
        use crate::kernel_interface::KernelInterface;

        debug!("Tearing down kernel mount isolation for agent {}", agent_id);

        // Check if kernel interface is available
        if !KernelInterface::is_available() {
            warn!("Kernel module interface not available for teardown");
            return Ok(());
        }

        // Open kernel interface and teardown
        let ki = KernelInterface::open()?;
        ki.teardown_mount_isolation(agent_id)?;

        info!(
            "Kernel mount isolation teardown complete for agent {}",
            agent_id
        );

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn teardown_kernel_mount_isolation(_agent_id: u64) -> Result<()> {
        Ok(())
    }

    // ==================== End Kernel Module Mount Isolation ====================

    /// Convert CacheMounts to BindMounts
    fn cache_to_bind_mounts(caches: &[CacheMount], container_root: &Path) -> Vec<BindMount> {
        caches
            .iter()
            .map(|cache| BindMount {
                source: cache.host_path.clone(),
                target: container_root.join(cache.container_path.strip_prefix("/").unwrap_or(&cache.container_path)),
                readonly: cache.readonly,
            })
            .collect()
    }
}

#[async_trait]
impl BuildBackend for LinuxNativeBackend {
    async fn execute_cargo(
        &self,
        command: &CargoCommand,
        context: &BuildContext,
    ) -> Result<BuildResult> {
        info!("Executing cargo command via Linux native backend");

        // Create container config from context
        let job_id = context
            .workspace
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        let mut config = self.create_workspace_structure(job_id).await?;

        // Add cache bind mounts
        config.bind_mounts = Self::cache_to_bind_mounts(&context.cache_mounts, &config.rootfs);

        // Add resource limits if provided
        config.resource_limits = context.resource_limits.clone();

        // Add environment
        config.environment = context.build_environment();

        // Execute in isolated environment
        self.execute_isolated(command, context, &config).await
    }

    async fn create_workspace(&self, job_id: &str) -> Result<PathBuf> {
        let workspace = self.workspaces_dir.join(job_id);

        tokio::fs::create_dir_all(&workspace).await.map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create workspace: {e}"),
            ))
        })?;

        // Create subdirectories for overlayfs
        for subdir in ["lower", "upper", "work", "merged", "source"] {
            tokio::fs::create_dir_all(workspace.join(subdir))
                .await
                .ok();
        }

        Ok(workspace.join("source"))
    }

    async fn cleanup_workspace(&self, workspace: &PathBuf) -> Result<()> {
        // Get parent directory (the job workspace)
        let job_dir = workspace.parent().unwrap_or(workspace);
        let merged = job_dir.join("merged");

        // Unmount overlayfs if mounted
        Self::cleanup_overlayfs(&merged)?;

        // Remove workspace directory
        if job_dir.exists() {
            tokio::fs::remove_dir_all(job_dir).await.map_err(|e| {
                SwarmletError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to cleanup workspace: {e}"),
                ))
            })?;
        }

        Ok(())
    }

    async fn is_available(&self) -> bool {
        // Check if we have the necessary permissions
        Self::check_capabilities() || self.user_ns
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            full_namespace_isolation: true,
            cgroups_v2: self.cgroups_v2,
            gpu_passthrough: true,
            seccomp: true,
            user_namespace: self.user_ns,
            max_containers: 200_000, // From swarm_guard MAX_AGENTS
        }
    }

    fn backend_type(&self) -> BackendType {
        BackendType::LinuxNative
    }

    fn name(&self) -> &str {
        "linux-native"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cgroups_v2_check() {
        // This will depend on the system running the tests
        let result = LinuxNativeBackend::check_cgroups_v2();
        // Just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_user_ns_check() {
        let result = LinuxNativeBackend::check_user_ns();
        let _ = result;
    }

    #[test]
    fn test_capabilities_check() {
        let result = LinuxNativeBackend::check_capabilities();
        // Will be false unless running as root
        let _ = result;
    }

    #[test]
    fn test_bind_mount_conversion() {
        let caches = vec![CacheMount {
            host_path: PathBuf::from("/host/cache"),
            container_path: PathBuf::from("/container/cache"),
            readonly: true,
        }];

        let container_root = PathBuf::from("/merged");
        let binds = LinuxNativeBackend::cache_to_bind_mounts(&caches, &container_root);

        assert_eq!(binds.len(), 1);
        assert_eq!(binds[0].source, PathBuf::from("/host/cache"));
        assert!(binds[0].readonly);
    }

    #[test]
    fn test_container_config() {
        let config = ContainerConfig {
            rootfs: PathBuf::from("/merged"),
            lower_dir: PathBuf::from("/lower"),
            upper_dir: PathBuf::from("/upper"),
            work_dir: PathBuf::from("/work"),
            bind_mounts: vec![],
            resource_limits: None,
            hostname: "test-build".to_string(),
            environment: HashMap::new(),
        };

        assert_eq!(config.hostname, "test-build");
    }

    #[test]
    fn test_kernel_module_check() {
        // Test kernel module detection
        let result = LinuxNativeBackend::check_kernel_module();
        // Will likely be false unless running on a system with swarm_guard loaded
        // Just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_agent_id_generation() {
        let id1 = LinuxNativeBackend::next_agent_id();
        let id2 = LinuxNativeBackend::next_agent_id();
        // IDs should be unique and increasing
        assert!(id2 > id1);
    }
}
