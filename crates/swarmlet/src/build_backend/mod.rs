//! Build backend abstraction layer
//!
//! This module provides a platform-agnostic interface for executing build jobs
//! in isolated containers. It supports:
//! - Linux native isolation (namespaces, cgroups, overlayfs)
//! - Docker fallback (for macOS, Windows, or when kernel module unavailable)

pub mod docker;
#[cfg(target_os = "linux")]
pub mod linux;

use crate::build_job::{BuildResourceLimits, CargoCommand, BuildResult, CacheConfig};
use crate::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;

/// Platform abstraction for build execution
#[async_trait]
pub trait BuildBackend: Send + Sync {
    /// Execute a cargo command in an isolated environment
    async fn execute_cargo(
        &self,
        command: &CargoCommand,
        context: &BuildContext,
    ) -> Result<BuildResult>;

    /// Create an isolated workspace for a build
    async fn create_workspace(&self, job_id: &str) -> Result<PathBuf>;

    /// Clean up a workspace after build completion
    async fn cleanup_workspace(&self, workspace: &PathBuf) -> Result<()>;

    /// Check if the backend is available and ready
    async fn is_available(&self) -> bool;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Get backend type identifier
    fn backend_type(&self) -> BackendType;

    /// Get the name of this backend for logging
    fn name(&self) -> &str;
}

/// Build execution context
#[derive(Debug, Clone)]
pub struct BuildContext {
    /// Workspace directory containing source code
    pub workspace: PathBuf,
    /// Path to the Rust toolchain
    pub toolchain_path: PathBuf,
    /// Cache mount configurations
    pub cache_mounts: Vec<CacheMount>,
    /// Environment variables for the build
    pub environment: HashMap<String, String>,
    /// Resource limits
    pub resource_limits: BuildResourceLimits,
    /// Cargo arguments
    pub cargo_args: Vec<String>,
}

impl BuildContext {
    /// Create a new build context
    pub fn new(workspace: PathBuf, toolchain_path: PathBuf) -> Self {
        Self {
            workspace,
            toolchain_path,
            cache_mounts: Vec::new(),
            environment: HashMap::new(),
            resource_limits: BuildResourceLimits::default(),
            cargo_args: Vec::new(),
        }
    }

    /// Add a cache mount
    pub fn with_cache_mount(mut self, mount: CacheMount) -> Self {
        self.cache_mounts.push(mount);
        self
    }

    /// Set environment variables
    pub fn with_environment(mut self, env: HashMap<String, String>) -> Self {
        self.environment = env;
        self
    }

    /// Set resource limits
    pub fn with_resource_limits(mut self, limits: BuildResourceLimits) -> Self {
        self.resource_limits = limits;
        self
    }

    /// Set cargo arguments
    pub fn with_cargo_args(mut self, args: Vec<String>) -> Self {
        self.cargo_args = args;
        self
    }

    /// Get environment variables with toolchain paths configured
    pub fn build_environment(&self) -> HashMap<String, String> {
        let mut env = self.environment.clone();

        // Set toolchain paths
        env.insert(
            "RUSTUP_HOME".to_string(),
            self.toolchain_path.join("rustup").to_string_lossy().to_string(),
        );
        env.insert(
            "CARGO_HOME".to_string(),
            self.toolchain_path.join("cargo").to_string_lossy().to_string(),
        );

        // Set sccache if configured
        for mount in &self.cache_mounts {
            if mount.container_path.to_string_lossy().contains("sccache") {
                env.insert(
                    "SCCACHE_DIR".to_string(),
                    mount.container_path.to_string_lossy().to_string(),
                );
                env.insert("RUSTC_WRAPPER".to_string(), "sccache".to_string());
            }
        }

        // Ensure color output
        env.insert("CARGO_TERM_COLOR".to_string(), "always".to_string());

        env
    }
}

/// Cache mount specification
#[derive(Debug, Clone)]
pub struct CacheMount {
    /// Path on the host
    pub host_path: PathBuf,
    /// Path inside the container
    pub container_path: PathBuf,
    /// Whether the mount is read-only
    pub readonly: bool,
}

impl CacheMount {
    /// Create a new cache mount
    pub fn new(host_path: PathBuf, container_path: PathBuf) -> Self {
        Self {
            host_path,
            container_path,
            readonly: false,
        }
    }

    /// Create a read-only cache mount
    pub fn readonly(host_path: PathBuf, container_path: PathBuf) -> Self {
        Self {
            host_path,
            container_path,
            readonly: true,
        }
    }

    /// Convert to Docker bind mount string
    pub fn to_docker_bind(&self) -> String {
        let mode = if self.readonly { "ro" } else { "rw" };
        format!(
            "{}:{}:{}",
            self.host_path.to_string_lossy(),
            self.container_path.to_string_lossy(),
            mode
        )
    }
}

/// Backend type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Linux native isolation using namespaces/cgroups
    LinuxNative,
    /// Docker-based isolation
    Docker,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::LinuxNative => write!(f, "linux-native"),
            BackendType::Docker => write!(f, "docker"),
        }
    }
}

/// Backend capabilities for feature detection
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Supports full namespace isolation
    pub full_namespace_isolation: bool,
    /// Supports cgroups v2
    pub cgroups_v2: bool,
    /// Supports GPU passthrough
    pub gpu_passthrough: bool,
    /// Supports seccomp filtering
    pub seccomp: bool,
    /// Supports user namespace remapping
    pub user_namespace: bool,
    /// Maximum concurrent containers
    pub max_containers: usize,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            full_namespace_isolation: false,
            cgroups_v2: false,
            gpu_passthrough: false,
            seccomp: false,
            user_namespace: false,
            max_containers: 10,
        }
    }
}

/// Output handler for streaming build output
#[async_trait]
pub trait OutputHandler: Send + Sync {
    /// Handle a line from stdout
    async fn handle_stdout(&mut self, line: &str);
    /// Handle a line from stderr
    async fn handle_stderr(&mut self, line: &str);
    /// Handle a status update
    async fn handle_status(&mut self, status: &str);
}

/// Simple output handler that collects output
#[derive(Debug, Default)]
pub struct CollectingOutputHandler {
    pub stdout: Vec<String>,
    pub stderr: Vec<String>,
    pub status: Vec<String>,
}

#[async_trait]
impl OutputHandler for CollectingOutputHandler {
    async fn handle_stdout(&mut self, line: &str) {
        self.stdout.push(line.to_string());
    }

    async fn handle_stderr(&mut self, line: &str) {
        self.stderr.push(line.to_string());
    }

    async fn handle_status(&mut self, status: &str) {
        self.status.push(status.to_string());
    }
}

/// Detect and create the appropriate backend for the current platform
pub async fn detect_backend() -> Result<Box<dyn BuildBackend>> {
    // Try Docker first (available on all platforms)
    let docker = docker::DockerBackend::new().await;
    if let Ok(backend) = docker {
        if backend.is_available().await {
            tracing::info!("Using Docker backend for build isolation");
            return Ok(Box::new(backend));
        }
    }

    // On Linux, try native backend
    #[cfg(target_os = "linux")]
    {
        let native = linux::LinuxNativeBackend::new().await;
        if let Ok(backend) = native {
            if backend.is_available().await {
                tracing::info!("Using Linux native backend for build isolation");
                return Ok(Box::new(backend));
            }
        }
    }

    Err(crate::SwarmletError::NotImplemented(
        "No build backend available. Please install Docker or run on Linux with appropriate permissions.".to_string()
    ))
}

/// Get available backends in order of preference
pub fn available_backends() -> Vec<BackendType> {
    let mut backends = vec![BackendType::Docker];

    #[cfg(target_os = "linux")]
    backends.insert(0, BackendType::LinuxNative);

    backends
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_mount_docker_bind() {
        let mount = CacheMount::new(
            PathBuf::from("/host/cache"),
            PathBuf::from("/container/cache"),
        );
        assert_eq!(mount.to_docker_bind(), "/host/cache:/container/cache:rw");

        let readonly_mount = CacheMount::readonly(
            PathBuf::from("/host/cache"),
            PathBuf::from("/container/cache"),
        );
        assert_eq!(readonly_mount.to_docker_bind(), "/host/cache:/container/cache:ro");
    }

    #[test]
    fn test_build_context_environment() {
        let ctx = BuildContext::new(
            PathBuf::from("/workspace"),
            PathBuf::from("/toolchain"),
        );
        let env = ctx.build_environment();

        assert_eq!(env.get("RUSTUP_HOME"), Some(&"/toolchain/rustup".to_string()));
        assert_eq!(env.get("CARGO_HOME"), Some(&"/toolchain/cargo".to_string()));
    }

    #[test]
    fn test_backend_type_display() {
        assert_eq!(BackendType::LinuxNative.to_string(), "linux-native");
        assert_eq!(BackendType::Docker.to_string(), "docker");
    }
}
