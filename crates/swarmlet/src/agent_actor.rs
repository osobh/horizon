//! Actor-based swarmlet agent for cancel-safe async operations
//!
//! This module provides an actor-based implementation of the swarmlet agent
//! that eliminates spawned task management issues and provides cancel-safe operations.
//!
//! # Cancel Safety
//!
//! The actor model ensures cancel safety by:
//! 1. Actor owns all mutable state exclusively - no Arc<RwLock<...>> for health
//! 2. All internal timers (heartbeat, health, workload) are integrated into
//!    the actor's select! loop
//! 3. Graceful shutdown via explicit Shutdown message
//! 4. No orphaned spawned tasks - everything runs in the actor's loop

use crate::{
    command::CommandExecutor, config::Config, join::JoinResult, security::NodeCertificate,
    workload::WorkloadManager, Result, SwarmletError,
};
use crate::agent::{HealthStatus, NodeStatus, WorkAssignment};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[cfg(feature = "hpc-channels")]
use crate::hpc_bridge::{SharedAgentChannelBridge, shared_channel_bridge};

/// Requests that can be sent to the swarmlet agent actor
#[derive(Debug)]
pub enum SwarmletRequest {
    /// Get current health status
    GetHealth {
        reply: oneshot::Sender<HealthStatus>,
    },
    /// Get node ID
    GetNodeId {
        reply: oneshot::Sender<Uuid>,
    },
    /// Execute a command
    ExecuteCommand {
        request: crate::command::CommandRequest,
        reply: oneshot::Sender<Result<crate::command::CommandResult>>,
    },
    /// Execute a shell script
    ExecuteShell {
        script: String,
        reply: oneshot::Sender<Result<crate::command::CommandResult>>,
    },
    /// Start a workload
    StartWorkload {
        assignment: WorkAssignment,
        reply: oneshot::Sender<Result<()>>,
    },
    /// Stop all workloads
    StopAllWorkloads {
        reply: oneshot::Sender<Result<()>>,
    },
    /// Graceful shutdown
    Shutdown,
}

/// Swarmlet agent actor that owns all mutable state
///
/// This actor integrates all internal loops (heartbeat, health monitoring,
/// workload checking) into a single message processing loop, eliminating
/// the need for spawned tasks and providing inherent cancel safety.
pub struct SwarmletAgentActor {
    /// Configuration
    config: Arc<Config>,
    /// Join result from cluster
    join_result: JoinResult,
    /// Node certificate
    #[allow(dead_code)]
    node_certificate: NodeCertificate,
    /// Workload manager
    workload_manager: Arc<WorkloadManager>,
    /// Command executor
    command_executor: Arc<CommandExecutor>,
    /// Health status - owned, not shared
    health_status: HealthStatus,
    /// Start time for uptime calculation
    start_time: Instant,
    /// HTTP client for heartbeats
    http_client: reqwest::Client,
    /// Request receiver
    inbox: mpsc::Receiver<SwarmletRequest>,
    /// HPC-Channels event bridge
    #[cfg(feature = "hpc-channels")]
    event_bridge: SharedAgentChannelBridge,
}

impl SwarmletAgentActor {
    /// Create a new actor from join result
    pub async fn new(
        join_result: JoinResult,
        data_dir: String,
        inbox: mpsc::Receiver<SwarmletRequest>,
    ) -> Result<Self> {
        let config = Config::default_with_data_dir(PathBuf::from(data_dir));
        let config = Arc::new(config);

        let node_certificate = NodeCertificate::from_pem(&join_result.node_certificate)?;
        let workload_manager = Arc::new(WorkloadManager::new(config.clone()).await?);
        let command_executor = Arc::new(CommandExecutor::new(config.data_dir.clone()));

        let health_status = HealthStatus {
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
        };

        Ok(Self {
            config,
            join_result,
            node_certificate,
            workload_manager,
            command_executor,
            health_status,
            start_time: Instant::now(),
            http_client: reqwest::Client::new(),
            inbox,
            #[cfg(feature = "hpc-channels")]
            event_bridge: shared_channel_bridge(),
        })
    }

    /// Run the actor's message processing loop
    ///
    /// This method integrates all internal timers (heartbeat, health monitoring,
    /// workload checking) into a single select! loop. No spawned tasks needed.
    pub async fn run(mut self) -> Result<()> {
        info!(
            "Starting swarmlet agent actor for node {}",
            self.join_result.node_id
        );

        // Update status to healthy
        self.health_status.status = NodeStatus::Healthy;
        self.health_status.last_heartbeat = chrono::Utc::now();

        // Publish agent started event
        #[cfg(feature = "hpc-channels")]
        self.event_bridge.publish_agent_started(&self.join_result.node_id.to_string());

        // Publish agent healthy event
        #[cfg(feature = "hpc-channels")]
        self.event_bridge.publish_agent_healthy(&self.join_result.node_id.to_string(), "Healthy");

        // Create internal timers
        let mut heartbeat_interval = interval(self.join_result.heartbeat_interval);
        let mut health_interval = interval(Duration::from_secs(10));
        let mut workload_interval = interval(Duration::from_secs(5));

        loop {
            tokio::select! {
                // Process incoming requests - highest priority
                Some(request) = self.inbox.recv() => {
                    match request {
                        SwarmletRequest::GetHealth { reply } => {
                            let _ = reply.send(self.health_status.clone());
                        }
                        SwarmletRequest::GetNodeId { reply } => {
                            let _ = reply.send(self.join_result.node_id);
                        }
                        SwarmletRequest::ExecuteCommand { request, reply } => {
                            let result = self.command_executor.execute_command(request).await;
                            let _ = reply.send(result);
                        }
                        SwarmletRequest::ExecuteShell { script, reply } => {
                            let result = self.command_executor.execute_shell(&script).await;
                            let _ = reply.send(result);
                        }
                        SwarmletRequest::StartWorkload { assignment, reply } => {
                            let result = self.workload_manager.start_workload(assignment).await.map(|_| ());
                            let _ = reply.send(result);
                        }
                        SwarmletRequest::StopAllWorkloads { reply } => {
                            let result = self.workload_manager.stop_all_workloads().await;
                            let _ = reply.send(result);
                        }
                        SwarmletRequest::Shutdown => {
                            info!("Shutdown request received");
                            break;
                        }
                    }
                }
                // Heartbeat timer
                _ = heartbeat_interval.tick() => {
                    if let Err(e) = self.send_heartbeat().await {
                        error!("Heartbeat failed: {}", e);
                        self.health_status.errors_count += 1;
                    }
                }
                // Health monitoring timer
                _ = health_interval.tick() => {
                    if let Err(e) = self.update_health_metrics().await {
                        warn!("Health metrics update failed: {}", e);
                    }
                }
                // Workload checking timer
                _ = workload_interval.tick() => {
                    if let Err(e) = self.check_for_work().await {
                        debug!("Work check failed: {}", e);
                    }
                }
            }
        }

        // Graceful shutdown
        self.health_status.status = NodeStatus::Shutting;

        // Publish agent shutdown event
        #[cfg(feature = "hpc-channels")]
        self.event_bridge.publish_agent_shutdown(&self.join_result.node_id.to_string(), "graceful shutdown");

        // Send final heartbeat
        if let Err(e) = self.send_heartbeat().await {
            warn!("Failed to send final heartbeat: {}", e);
        }

        // Stop all workloads
        if let Err(e) = self.workload_manager.stop_all_workloads().await {
            error!("Failed to stop workloads: {}", e);
        }

        info!("Swarmlet agent actor shutdown complete");
        Ok(())
    }

    /// Send heartbeat to cluster (internal implementation)
    async fn send_heartbeat(&mut self) -> Result<()> {
        let heartbeat = HeartbeatMessage {
            node_id: self.join_result.node_id,
            timestamp: chrono::Utc::now(),
            status: self.health_status.status.clone(),
            metrics: HealthMetrics {
                cpu_usage_percent: self.health_status.cpu_usage_percent,
                memory_usage_gb: self.health_status.memory_usage_gb,
                disk_usage_gb: self.health_status.disk_usage_gb,
                workloads_active: self.health_status.workloads_active,
                uptime_seconds: self.health_status.uptime_seconds,
            },
        };

        let url = format!("{}/heartbeat", self.join_result.api_endpoints.health_check);

        match self.http_client.post(&url).json(&heartbeat).send().await {
            Ok(response) if response.status().is_success() => {
                debug!("Heartbeat sent successfully");
                self.health_status.last_heartbeat = chrono::Utc::now();
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

    /// Update local health metrics (internal implementation)
    async fn update_health_metrics(&mut self) -> Result<()> {
        use sysinfo::System;

        let mut system = System::new_all();
        system.refresh_all();

        self.health_status.uptime_seconds = self.start_time.elapsed().as_secs();
        self.health_status.cpu_usage_percent = system.cpus().iter()
            .map(|cpu| cpu.cpu_usage())
            .sum::<f32>() / system.cpus().len() as f32;
        self.health_status.memory_usage_gb =
            (system.total_memory() - system.available_memory()) as f32 / (1024.0 * 1024.0 * 1024.0);

        // Get disk usage
        self.health_status.disk_usage_gb = self.get_disk_usage().await.unwrap_or(0.0);

        // Get active workload count
        self.health_status.workloads_active = self.workload_manager.active_workload_count().await;

        // Update status based on metrics
        self.health_status.status = if self.health_status.cpu_usage_percent > 90.0
            || self.health_status.memory_usage_gb > self.health_status.memory_usage_gb * 0.9
        {
            NodeStatus::Degraded
        } else if self.health_status.errors_count > 10 {
            NodeStatus::Unhealthy
        } else {
            NodeStatus::Healthy
        };

        Ok(())
    }

    /// Check for new work assignments from cluster (internal implementation)
    async fn check_for_work(&mut self) -> Result<()> {
        let url = format!("{}/work", self.join_result.api_endpoints.workload_api);

        match self.http_client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                let work_assignments: Vec<WorkAssignment> = response.json().await?;

                for assignment in work_assignments {
                    debug!("Received work assignment: {}", assignment.id);

                    if let Err(e) = self.workload_manager.start_workload(assignment).await {
                        error!("Failed to start workload: {}", e);
                    }
                }
            }
            Ok(response) if response.status().as_u16() == 204 => {
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

    /// Get disk usage for data directory (internal implementation)
    async fn get_disk_usage(&self) -> Result<f32> {
        use std::fs;

        match fs::metadata(&self.config.data_dir) {
            Ok(metadata) => Ok(metadata.len() as f32 / (1024.0 * 1024.0 * 1024.0)),
            Err(_) => Ok(0.0),
        }
    }
}

/// Handle for interacting with the swarmlet agent actor
///
/// This handle is cheap to clone and can be used from multiple tasks.
/// All operations are cancel-safe - if the caller's future is dropped,
/// the actor continues processing.
#[derive(Clone)]
pub struct SwarmletAgentHandle {
    sender: mpsc::Sender<SwarmletRequest>,
}

impl SwarmletAgentHandle {
    /// Create a new handle from a sender
    pub fn new(sender: mpsc::Sender<SwarmletRequest>) -> Self {
        Self { sender }
    }

    /// Get current health status
    ///
    /// This operation is cancel-safe. If the caller's future is dropped,
    /// the actor continues processing.
    pub async fn get_health(&self) -> Result<HealthStatus> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmletRequest::GetHealth { reply: tx })
            .await
            .map_err(|_| SwarmletError::AgentRuntime("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| SwarmletError::AgentRuntime("Actor dropped".to_string()))
    }

    /// Get node ID
    pub async fn get_node_id(&self) -> Result<Uuid> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmletRequest::GetNodeId { reply: tx })
            .await
            .map_err(|_| SwarmletError::AgentRuntime("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| SwarmletError::AgentRuntime("Actor dropped".to_string()))
    }

    /// Execute a command
    pub async fn execute_command(
        &self,
        request: crate::command::CommandRequest,
    ) -> Result<crate::command::CommandResult> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmletRequest::ExecuteCommand { request, reply: tx })
            .await
            .map_err(|_| SwarmletError::AgentRuntime("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| SwarmletError::AgentRuntime("Actor dropped".to_string()))?
    }

    /// Execute a shell script
    pub async fn execute_shell(&self, script: &str) -> Result<crate::command::CommandResult> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmletRequest::ExecuteShell {
                script: script.to_string(),
                reply: tx,
            })
            .await
            .map_err(|_| SwarmletError::AgentRuntime("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| SwarmletError::AgentRuntime("Actor dropped".to_string()))?
    }

    /// Start a workload
    pub async fn start_workload(&self, assignment: WorkAssignment) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmletRequest::StartWorkload { assignment, reply: tx })
            .await
            .map_err(|_| SwarmletError::AgentRuntime("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| SwarmletError::AgentRuntime("Actor dropped".to_string()))?
    }

    /// Stop all workloads
    pub async fn stop_all_workloads(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(SwarmletRequest::StopAllWorkloads { reply: tx })
            .await
            .map_err(|_| SwarmletError::AgentRuntime("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| SwarmletError::AgentRuntime("Actor dropped".to_string()))?
    }

    /// Request graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        self.sender
            .send(SwarmletRequest::Shutdown)
            .await
            .map_err(|_| SwarmletError::AgentRuntime("Actor already stopped".to_string()))
    }
}

/// Create an actor and its handle
///
/// # Returns
/// A tuple of (actor, handle). The actor should be spawned as a task,
/// and the handle used to communicate with it.
///
/// # Example
/// ```ignore
/// let (actor, handle) = create_swarmlet_actor(join_result, data_dir).await?;
///
/// // Spawn the actor
/// tokio::spawn(actor.run());
///
/// // Use the handle
/// let health = handle.get_health().await?;
///
/// // Shutdown
/// handle.shutdown().await?;
/// ```
pub async fn create_swarmlet_actor(
    join_result: JoinResult,
    data_dir: String,
) -> Result<(SwarmletAgentActor, SwarmletAgentHandle)> {
    let (tx, rx) = mpsc::channel(64);

    let actor = SwarmletAgentActor::new(join_result, data_dir, rx).await?;
    let handle = SwarmletAgentHandle::new(tx);

    Ok((actor, handle))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::join::ClusterApiEndpoints;
    use tempfile::TempDir;

    fn create_test_join_result() -> JoinResult {
        JoinResult {
            node_id: Uuid::new_v4(),
            cluster_name: "test-cluster".to_string(),
            node_certificate: generate_test_certificate(),
            cluster_endpoints: vec!["http://localhost:7946".to_string()],
            assigned_capabilities: vec!["compute".to_string()],
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

    fn generate_test_certificate() -> String {
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
    async fn test_actor_creation() {
        let join_result = create_test_join_result();
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        let result = create_swarmlet_actor(join_result, data_dir).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_actor_get_health() {
        let join_result = create_test_join_result();
        let node_id = join_result.node_id;
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        let (actor, handle) = create_swarmlet_actor(join_result, data_dir).await.unwrap();

        // Spawn actor
        let actor_task = tokio::spawn(actor.run());

        // Get health
        let health = handle.get_health().await.unwrap();
        assert_eq!(health.node_id, node_id);
        assert_eq!(health.status, NodeStatus::Healthy);

        // Shutdown
        handle.shutdown().await.unwrap();
        actor_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_actor_cancel_safety() {
        let join_result = create_test_join_result();
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        let (actor, handle) = create_swarmlet_actor(join_result, data_dir).await.unwrap();
        tokio::spawn(actor.run());

        // Start a request but cancel it
        let handle_clone = handle.clone();
        let future = handle_clone.get_health();
        drop(future); // Cancel the request

        // Give actor time to process
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Actor should still be responsive
        let health = handle.get_health().await.unwrap();
        assert_eq!(health.status, NodeStatus::Healthy);

        handle.shutdown().await.unwrap();
    }
}
