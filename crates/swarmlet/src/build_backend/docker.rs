//! Docker-based build backend
//!
//! This backend uses Docker for build isolation, providing cross-platform
//! support for macOS, Windows, and Linux systems without kernel module.

use super::{BackendCapabilities, BackendType, BuildBackend, BuildContext};
use crate::build_job::{BuildResult, CargoCommand};
use crate::{Result, SwarmletError};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

// Docker-only imports
#[cfg(feature = "docker")]
use crate::build_job::BuildResourceUsage;
#[cfg(feature = "docker")]
use tracing::{debug, info, warn};
#[cfg(feature = "docker")]
use uuid::Uuid;

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
}

impl DockerBackend {
    /// Create a new Docker backend
    pub async fn new() -> Result<Self> {
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

            Ok(Self {
                docker: Arc::new(docker),
                container_prefix: "stratoswarm-build-".to_string(),
                active_containers: Arc::new(RwLock::new(HashMap::new())),
                default_image: "rust:latest".to_string(),
                gpu_available,
            })
        }

        #[cfg(not(feature = "docker"))]
        {
            Err(SwarmletError::NotImplemented(
                "Docker feature not enabled. Compile with --features docker".to_string(),
            ))
        }
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
        use bollard::models::HostConfig;

        let limits = &context.resource_limits;

        // Build bind mounts
        let mut binds = vec![
            // Mount workspace
            format!("{}:/workspace:rw", context.workspace.to_string_lossy()),
        ];

        // Add cache mounts
        for mount in &context.cache_mounts {
            binds.push(mount.to_docker_bind());
        }

        HostConfig {
            binds: Some(binds),
            memory: limits.memory_bytes.map(|b| b as i64),
            nano_cpus: limits.cpu_cores.map(|c| (c as f64 * 1_000_000_000.0) as i64),
            pids_limit: Some(1000),
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
}

#[async_trait]
impl BuildBackend for DockerBackend {
    async fn execute_cargo(
        &self,
        command: &CargoCommand,
        context: &BuildContext,
    ) -> Result<BuildResult> {
        // Suppress unused warning when docker feature is enabled
        let _ = command;
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

            // Ensure image exists
            self.ensure_image(&self.default_image).await?;

            // Build environment variables
            let env: Vec<String> = context
                .build_environment()
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();

            // Create container configuration
            let host_config = self.build_host_config(context);

            let config = Config {
                image: Some(self.default_image.clone()),
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

            // Collect logs
            let logs_options = LogsOptions::<String> {
                follow: true,
                stdout: true,
                stderr: true,
                ..Default::default()
            };

            let mut log_stream = self.docker.logs(&container.id, Some(logs_options));
            let mut stdout_lines = Vec::new();
            let mut stderr_lines = Vec::new();

            while let Some(log_result) = log_stream.next().await {
                match log_result {
                    Ok(bollard::container::LogOutput::StdOut { message }) => {
                        let line = String::from_utf8_lossy(&message);
                        debug!("[stdout] {}", line.trim());
                        stdout_lines.push(line.to_string());
                    }
                    Ok(bollard::container::LogOutput::StdErr { message }) => {
                        let line = String::from_utf8_lossy(&message);
                        debug!("[stderr] {}", line.trim());
                        stderr_lines.push(line.to_string());
                    }
                    Err(e) => {
                        warn!("Log stream error: {}", e);
                    }
                    _ => {}
                }
            }

            // Wait for container to finish
            let wait_options = WaitContainerOptions {
                condition: "not-running",
            };

            let mut wait_stream = self.docker.wait_container(&container.id, Some(wait_options));
            let exit_code = if let Some(Ok(result)) = wait_stream.next().await {
                result.status_code as i32
            } else {
                -1
            };

            let duration = start_time.elapsed();

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
                resource_usage: BuildResourceUsage {
                    compile_time_seconds: duration.as_secs_f64(),
                    ..Default::default()
                },
                duration,
                artifacts: Vec::new(), // TODO: collect artifacts
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
}
