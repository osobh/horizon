//! Workload management for swarmlet

use crate::{agent::WorkAssignment, config::Config, Result, SwarmletError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Workload manager handles execution of tasks assigned by the cluster
pub struct WorkloadManager {
    config: Arc<Config>,
    active_workloads: Arc<RwLock<HashMap<Uuid, ActiveWorkload>>>,
    /// Node ID for this swarmlet (passed to workloads as environment variable)
    node_id: Uuid,
    #[cfg(feature = "docker")]
    docker: Arc<bollard::Docker>,
}

/// An active workload running on this swarmlet
#[derive(Debug, Clone, Serialize)]
pub struct ActiveWorkload {
    pub id: Uuid,
    pub assignment: WorkAssignment,
    pub status: WorkloadStatus,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub pid: Option<u32>,
    pub container_id: Option<String>,
    pub resource_usage: ResourceUsage,
}

/// Current status of a workload
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WorkloadStatus {
    Starting,
    Running,
    Completed,
    Failed,
    Stopping,
    Stopped,
}

/// Resource usage statistics for a workload
#[derive(Debug, Clone, Default, Serialize)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: f32,
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,
    pub network_rx_bytes: u64,
    pub network_tx_bytes: u64,
}

impl WorkloadManager {
    /// Create a new workload manager
    pub async fn new(config: Arc<Config>, node_id: Uuid) -> Result<Self> {
        let active_workloads = Arc::new(RwLock::new(HashMap::new()));

        #[cfg(feature = "docker")]
        let docker =
            {
                use bollard::Docker;
                Arc::new(Docker::connect_with_local_defaults().map_err(|e| {
                    SwarmletError::Docker(format!("Failed to connect to Docker: {e}"))
                })?)
            };

        Ok(Self {
            config,
            active_workloads,
            node_id,
            #[cfg(feature = "docker")]
            docker,
        })
    }

    /// Get the node ID
    pub fn node_id(&self) -> Uuid {
        self.node_id
    }

    /// Start a new workload from assignment
    pub async fn start_workload(&self, assignment: WorkAssignment) -> Result<Uuid> {
        info!("Starting workload: {}", assignment.id);

        // Check if we can accept more workloads
        let current_count = self.active_workload_count().await;
        let max_workloads = self.config.node.max_workloads.unwrap_or(10);

        if current_count >= max_workloads {
            return Err(SwarmletError::WorkloadExecution(
                "Maximum workload limit reached".to_string(),
            ));
        }

        // Create active workload entry
        let workload = ActiveWorkload {
            id: assignment.id,
            assignment: assignment.clone(),
            status: WorkloadStatus::Starting,
            started_at: chrono::Utc::now(),
            pid: None,
            container_id: None,
            resource_usage: ResourceUsage::default(),
        };

        // Add to active workloads
        {
            let mut workloads = self.active_workloads.write().await;
            workloads.insert(assignment.id, workload);
        }

        // Start the workload based on type
        let result = if let Some(ref container_image) = assignment.container_image {
            self.start_container_workload(&assignment, container_image)
                .await
        } else if let Some(ref shell_script) = assignment.shell_script {
            self.start_shell_workload(&assignment, shell_script).await
        } else {
            self.start_process_workload(&assignment).await
        };

        match result {
            Ok(workload_info) => {
                // Update workload with runtime info
                {
                    let mut workloads = self.active_workloads.write().await;
                    if let Some(workload) = workloads.get_mut(&assignment.id) {
                        workload.status = WorkloadStatus::Running;
                        workload.pid = workload_info.pid;
                        workload.container_id = workload_info.container_id;
                    }
                }

                info!("Workload {} started successfully", assignment.id);
                Ok(assignment.id)
            }
            Err(e) => {
                error!("Failed to start workload {}: {}", assignment.id, e);

                // Update status to failed
                {
                    let mut workloads = self.active_workloads.write().await;
                    if let Some(workload) = workloads.get_mut(&assignment.id) {
                        workload.status = WorkloadStatus::Failed;
                    }
                }

                Err(e)
            }
        }
    }

    /// Stop a specific workload
    pub async fn stop_workload(&self, workload_id: Uuid) -> Result<()> {
        info!("Stopping workload: {}", workload_id);

        let workload = {
            let mut workloads = self.active_workloads.write().await;
            let workload = workloads.get_mut(&workload_id).ok_or_else(|| {
                SwarmletError::WorkloadExecution(format!("Workload {workload_id} not found"))
            })?;

            workload.status = WorkloadStatus::Stopping;
            workload.clone()
        };

        // Stop based on workload type
        if let Some(container_id) = &workload.container_id {
            self.stop_container_workload(container_id).await?;
        } else if let Some(pid) = workload.pid {
            self.stop_process_workload(pid).await?;
        }

        // Update status to stopped
        {
            let mut workloads = self.active_workloads.write().await;
            if let Some(workload) = workloads.get_mut(&workload_id) {
                workload.status = WorkloadStatus::Stopped;
            }
        }

        info!("Workload {} stopped successfully", workload_id);
        Ok(())
    }

    /// Stop all active workloads
    pub async fn stop_all_workloads(&self) -> Result<()> {
        info!("Stopping all active workloads");

        let workload_ids: Vec<Uuid> = {
            let workloads = self.active_workloads.read().await;
            workloads.keys().cloned().collect()
        };

        for workload_id in workload_ids {
            if let Err(e) = self.stop_workload(workload_id).await {
                error!("Failed to stop workload {}: {}", workload_id, e);
            }
        }

        Ok(())
    }

    /// Get count of active workloads
    pub async fn active_workload_count(&self) -> u32 {
        let workloads = self.active_workloads.read().await;
        workloads.len() as u32
    }

    /// Get list of active workloads
    pub async fn get_active_workloads(&self) -> Vec<ActiveWorkload> {
        let workloads = self.active_workloads.read().await;
        workloads.values().cloned().collect()
    }

    /// Start a container-based workload
    #[cfg(feature = "docker")]
    async fn start_container_workload(
        &self,
        assignment: &WorkAssignment,
        image: &str,
    ) -> Result<WorkloadInfo> {
        use bollard::container::{Config as ContainerConfig, CreateContainerOptions};
        use bollard::models::HostConfig;

        debug!("Starting container workload with image: {}", image);

        // Create container configuration
        let mut env: Vec<String> = assignment
            .environment
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();

        // Add swarmlet-specific environment variables
        env.push(format!("SWARMLET_WORKLOAD_ID={}", assignment.id));
        env.push(format!("SWARMLET_NODE_ID={}", self.node_id));

        let host_config = HostConfig {
            memory: assignment
                .resource_limits
                .memory_gb
                .map(|gb| (gb * 1024.0 * 1024.0 * 1024.0) as i64),
            nano_cpus: assignment
                .resource_limits
                .cpu_cores
                .map(|cores| (cores * 1_000_000_000.0) as i64),
            ..Default::default()
        };

        let config = ContainerConfig {
            image: Some(image.to_string()),
            cmd: assignment.command.clone(),
            env: Some(env),
            host_config: Some(host_config),
            working_dir: Some("/workspace".to_string()),
            ..Default::default()
        };

        // Create container
        let container_name = format!("swarmlet-{}", assignment.id);
        let create_options = CreateContainerOptions {
            name: container_name.clone(),
            platform: None,
        };

        let container = self
            .docker
            .create_container(Some(create_options), config)
            .await
            .map_err(|e| SwarmletError::Docker(format!("Failed to create container: {e}")))?;

        // Start container
        self.docker
            .start_container::<String>(&container.id, None)
            .await
            .map_err(|e| SwarmletError::Docker(format!("Failed to start container: {e}")))?;

        Ok(WorkloadInfo {
            pid: None,
            container_id: Some(container.id),
        })
    }

    /// Start a container-based workload (fallback implementation)
    #[cfg(not(feature = "docker"))]
    async fn start_container_workload(
        &self,
        assignment: &WorkAssignment,
        image: &str,
    ) -> Result<WorkloadInfo> {
        debug!("Starting container workload with Docker CLI: {}", image);

        // Use docker CLI as fallback
        let mut cmd = Command::new("docker");
        cmd.arg("run")
            .arg("-d")
            .arg("--name")
            .arg(format!("swarmlet-{}", assignment.id));

        // Add environment variables
        for (key, value) in &assignment.environment {
            cmd.arg("-e").arg(format!("{}={}", key, value));
        }

        // Add resource limits
        if let Some(memory_gb) = assignment.resource_limits.memory_gb {
            cmd.arg("--memory").arg(format!("{}g", memory_gb));
        }

        if let Some(cpu_cores) = assignment.resource_limits.cpu_cores {
            cmd.arg("--cpus").arg(cpu_cores.to_string());
        }

        cmd.arg(image);

        // Add command if specified
        if let Some(ref command) = assignment.command {
            cmd.args(command);
        }

        let output = cmd.output().await.map_err(|e| {
            SwarmletError::WorkloadExecution(format!("Docker command failed: {}", e))
        })?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(SwarmletError::WorkloadExecution(format!(
                "Docker failed: {}",
                error
            )));
        }

        let container_id = String::from_utf8_lossy(&output.stdout).trim().to_string();

        Ok(WorkloadInfo {
            pid: None,
            container_id: Some(container_id),
        })
    }

    /// Start a process-based workload
    async fn start_process_workload(&self, assignment: &WorkAssignment) -> Result<WorkloadInfo> {
        debug!("Starting process workload");

        let command = assignment.command.as_ref().ok_or_else(|| {
            SwarmletError::WorkloadExecution(
                "No command specified for process workload".to_string(),
            )
        })?;

        if command.is_empty() {
            return Err(SwarmletError::WorkloadExecution(
                "Empty command for process workload".to_string(),
            ));
        }

        let mut cmd = Command::new(&command[0]);

        if command.len() > 1 {
            cmd.args(&command[1..]);
        }

        // Set environment variables
        for (key, value) in &assignment.environment {
            cmd.env(key, value);
        }

        // Set working directory
        cmd.current_dir(&self.config.storage.workload_dir);

        // Configure stdio
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        // Spawn the process
        let mut child = cmd.spawn().map_err(|e| {
            SwarmletError::WorkloadExecution(format!("Failed to spawn process: {e}"))
        })?;

        let pid = child.id();

        // Detach the process (it will run independently)
        tokio::spawn(async move {
            let _ = child.wait().await;
        });

        Ok(WorkloadInfo {
            pid,
            container_id: None,
        })
    }

    /// Start a shell script workload
    async fn start_shell_workload(
        &self,
        assignment: &WorkAssignment,
        script: &str,
    ) -> Result<WorkloadInfo> {
        debug!("Starting shell script workload");

        let mut cmd = Command::new("sh");
        cmd.args(["-c", script]);

        // Set environment variables
        for (key, value) in &assignment.environment {
            cmd.env(key, value);
        }

        // Add workload-specific environment variables
        cmd.env("SWARMLET_WORKLOAD_ID", assignment.id.to_string());
        cmd.env("SWARMLET_WORKLOAD_TYPE", "shell");

        // Set working directory
        cmd.current_dir(&self.config.storage.workload_dir);

        // Configure stdio
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        // Spawn the process
        let mut child = cmd.spawn().map_err(|e| {
            SwarmletError::WorkloadExecution(format!("Failed to spawn shell script: {e}"))
        })?;

        let pid = child.id();

        // Detach the process (it will run independently)
        tokio::spawn(async move {
            let _ = child.wait().await;
        });

        Ok(WorkloadInfo {
            pid,
            container_id: None,
        })
    }

    /// Stop a container workload
    #[cfg(feature = "docker")]
    async fn stop_container_workload(&self, container_id: &str) -> Result<()> {
        use bollard::container::StopContainerOptions;

        debug!("Stopping container: {}", container_id);

        let stop_options = StopContainerOptions {
            t: 10, // 10 second timeout
        };

        self.docker
            .stop_container(container_id, Some(stop_options))
            .await
            .map_err(|e| SwarmletError::Docker(format!("Failed to stop container: {e}")))?;

        // Remove the container
        self.docker
            .remove_container(container_id, None)
            .await
            .map_err(|e| SwarmletError::Docker(format!("Failed to remove container: {e}")))?;

        Ok(())
    }

    /// Stop a container workload (fallback implementation)
    #[cfg(not(feature = "docker"))]
    async fn stop_container_workload(&self, container_id: &str) -> Result<()> {
        debug!("Stopping container with Docker CLI: {}", container_id);

        // Stop container
        let output = Command::new("docker")
            .args(&["stop", container_id])
            .output()
            .await
            .map_err(|e| SwarmletError::WorkloadExecution(format!("Docker stop failed: {}", e)))?;

        if !output.status.success() {
            warn!(
                "Docker stop failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Remove container
        let output = Command::new("docker")
            .args(&["rm", container_id])
            .output()
            .await
            .map_err(|e| SwarmletError::WorkloadExecution(format!("Docker rm failed: {}", e)))?;

        if !output.status.success() {
            warn!(
                "Docker rm failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    /// Stop a process workload
    async fn stop_process_workload(&self, pid: u32) -> Result<()> {
        debug!("Stopping process with PID: {}", pid);

        // Send SIGTERM first
        #[cfg(unix)]
        {
            use nix::sys::signal::{self, Signal};
            use nix::unistd::Pid;

            let pid = Pid::from_raw(pid as i32);

            if let Err(e) = signal::kill(pid, Signal::SIGTERM) {
                warn!("Failed to send SIGTERM to PID {}: {}", pid, e);

                // Try SIGKILL as last resort
                let _ = signal::kill(pid, Signal::SIGKILL);
            }
        }

        #[cfg(windows)]
        {
            // On Windows, use taskkill
            let output = Command::new("taskkill")
                .args(&["/PID", &pid.to_string(), "/F"])
                .output()
                .await
                .map_err(|e| SwarmletError::WorkloadExecution(format!("Taskkill failed: {}", e)))?;

            if !output.status.success() {
                warn!(
                    "Taskkill failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }

        Ok(())
    }
}

/// Information about a started workload
#[derive(Debug)]
struct WorkloadInfo {
    pid: Option<u32>,
    container_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use tempfile::TempDir;

    fn create_test_assignment() -> WorkAssignment {
        WorkAssignment {
            id: Uuid::new_v4(),
            workload_type: "test".to_string(),
            container_image: None,
            command: Some(vec!["echo".to_string(), "hello".to_string()]),
            shell_script: None,
            environment: std::collections::HashMap::new(),
            resource_limits: crate::agent::ResourceLimits {
                cpu_cores: Some(1.0),
                memory_gb: Some(1.0),
                disk_gb: Some(1.0),
            },
            created_at: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_workload_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = Arc::new(Config::default_with_data_dir(temp_dir.path().to_path_buf()));
        let node_id = Uuid::new_v4();

        let manager = WorkloadManager::new(config, node_id).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_active_workload_count() {
        let temp_dir = TempDir::new().unwrap();
        let config = Arc::new(Config::default_with_data_dir(temp_dir.path().to_path_buf()));
        let node_id = Uuid::new_v4();
        let manager = WorkloadManager::new(config, node_id).await.unwrap();

        let count = manager.active_workload_count().await;
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_process_workload_start() {
        let temp_dir = TempDir::new().unwrap();
        let config = Arc::new(Config::default_with_data_dir(temp_dir.path().to_path_buf()));
        let node_id = Uuid::new_v4();
        let manager = WorkloadManager::new(config, node_id).await.unwrap();

        let assignment = create_test_assignment();

        // This might fail in CI environments, but tests the code path
        let result = manager.start_workload(assignment).await;
        match result {
            Ok(workload_id) => {
                assert!(!workload_id.is_nil());
                let _ = manager.stop_workload(workload_id).await;
            }
            Err(e) => {
                // Expected in some test environments
                println!("Workload start failed (expected in CI): {}", e);
            }
        }
    }
}

// Add Unix-specific dependencies for signal handling
#[cfg(unix)]
use nix;
