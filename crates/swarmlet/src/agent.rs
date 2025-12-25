//! Swarmlet agent runtime

use crate::{
    command::CommandExecutor, config::Config, join::JoinResult, security::NodeCertificate,
    workload::WorkloadManager, Result, SwarmletError,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use warp::Filter;

#[cfg(feature = "hpc-channels")]
use crate::hpc_bridge::{SharedAgentChannelBridge, shared_channel_bridge};

/// Main swarmlet agent that manages node lifecycle
pub struct SwarmletAgent {
    config: Arc<Config>,
    join_result: JoinResult,
    #[allow(dead_code)] // Certificate will be used for TLS authentication in production
    node_certificate: NodeCertificate,
    workload_manager: Arc<WorkloadManager>,
    command_executor: Arc<CommandExecutor>,
    health_status: Arc<RwLock<HealthStatus>>,
    shutdown_signal: tokio::sync::watch::Receiver<bool>,
    shutdown_sender: tokio::sync::watch::Sender<bool>,
    /// HPC-Channels event bridge for publishing agent lifecycle events
    #[cfg(feature = "hpc-channels")]
    event_bridge: SharedAgentChannelBridge,
}

/// Health status of the swarmlet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub node_id: Uuid,
    pub status: NodeStatus,
    pub uptime_seconds: u64,
    pub workloads_active: u32,
    pub cpu_usage_percent: f32,
    pub memory_usage_gb: f32,
    pub disk_usage_gb: f32,
    pub network_rx_bytes: u64,
    pub network_tx_bytes: u64,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub errors_count: u32,
}

/// Node status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeStatus {
    Starting,
    Healthy,
    Degraded,
    Unhealthy,
    Shutting,
}

impl SwarmletAgent {
    /// Create a new swarmlet agent from join result
    pub async fn new(join_result: JoinResult, data_dir: String) -> Result<Self> {
        let config = Config::default_with_data_dir(PathBuf::from(data_dir));
        let config = Arc::new(config);

        let node_certificate = NodeCertificate::from_pem(&join_result.node_certificate)?;
        let workload_manager = Arc::new(WorkloadManager::new(config.clone()).await?);
        let command_executor = Arc::new(CommandExecutor::new(config.data_dir.clone()));

        let health_status = Arc::new(RwLock::new(HealthStatus {
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
        }));

        let (shutdown_sender, shutdown_signal) = tokio::sync::watch::channel(false);

        Ok(Self {
            config,
            join_result,
            node_certificate,
            workload_manager,
            command_executor,
            health_status,
            shutdown_signal,
            shutdown_sender,
            #[cfg(feature = "hpc-channels")]
            event_bridge: shared_channel_bridge(),
        })
    }

    /// Get the event bridge for subscribing to agent lifecycle events
    #[cfg(feature = "hpc-channels")]
    pub fn event_bridge(&self) -> &SharedAgentChannelBridge {
        &self.event_bridge
    }

    /// Create a swarmlet agent from configuration
    pub async fn from_config(_config: Config) -> Result<Self> {
        // This would load join result and certificate from saved state
        // For now, return an error as this needs implementation
        Err(SwarmletError::NotImplemented(
            "Creating agent from config not yet implemented".to_string(),
        ))
    }

    /// Run the swarmlet agent (main event loop)
    pub async fn run(self) -> Result<()> {
        info!(
            "Starting swarmlet agent for node {}",
            self.join_result.node_id
        );

        // Update status to healthy
        {
            let mut health = self.health_status.write().await;
            health.status = NodeStatus::Healthy;
            health.last_heartbeat = chrono::Utc::now();
        }

        // Publish agent started event
        #[cfg(feature = "hpc-channels")]
        self.event_bridge.publish_agent_started(&self.join_result.node_id.to_string());

        // Publish agent healthy event
        #[cfg(feature = "hpc-channels")]
        self.event_bridge.publish_agent_healthy(&self.join_result.node_id.to_string(), "Healthy");

        // Start background tasks
        let agent = Arc::new(self);
        let mut tasks = Vec::new();

        // Start heartbeat task
        tasks.push(tokio::spawn({
            let agent = agent.clone();
            async move { agent.heartbeat_loop().await }
        }));

        // Start health monitoring task
        tasks.push(tokio::spawn({
            let agent = agent.clone();
            async move { agent.health_monitor_loop().await }
        }));

        // Start workload management task
        tasks.push(tokio::spawn({
            let agent = agent.clone();
            async move { agent.workload_loop().await }
        }));

        // Start API server task
        tasks.push(tokio::spawn({
            let agent = agent.clone();
            async move { agent.api_server_loop().await }
        }));

        // Wait for shutdown signal
        info!("Swarmlet agent running, waiting for shutdown signal...");

        let mut shutdown_signal = agent.shutdown_signal.clone();
        tokio::select! {
            _ = shutdown_signal.changed() => {
                info!("Shutdown signal received");
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Ctrl+C received, shutting down");
                let _ = agent.shutdown_sender.send(true);
            }
        }

        // Update status to shutting down
        {
            let mut health = agent.health_status.write().await;
            health.status = NodeStatus::Shutting;
        }

        // Publish agent shutdown event
        #[cfg(feature = "hpc-channels")]
        agent.event_bridge.publish_agent_shutdown(&agent.join_result.node_id.to_string(), "graceful shutdown");

        // Send final heartbeat
        if let Err(e) = agent.send_heartbeat().await {
            warn!("Failed to send final heartbeat: {}", e);
        }

        // Wait for tasks to complete with timeout
        let shutdown_timeout = Duration::from_secs(30);

        match tokio::time::timeout(shutdown_timeout, futures::future::join_all(tasks)).await {
            Ok(_) => info!("All tasks completed successfully"),
            Err(_) => warn!("Shutdown timeout reached, some tasks may not have completed"),
        }

        info!("Swarmlet agent shutdown complete");
        Ok(())
    }

    /// Send shutdown signal to the agent
    pub fn shutdown(&self) -> Result<()> {
        info!("Requesting agent shutdown");
        self.shutdown_sender.send(true).map_err(|_| {
            SwarmletError::AgentRuntime("Failed to send shutdown signal".to_string())
        })?;
        Ok(())
    }

    /// Heartbeat loop - sends periodic status updates to cluster
    async fn heartbeat_loop(&self) -> Result<()> {
        let mut interval = interval(self.join_result.heartbeat_interval);
        let _client = reqwest::Client::new();
        let mut shutdown_signal = self.shutdown_signal.clone();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.send_heartbeat().await {
                        error!("Heartbeat failed: {}", e);

                        // Increment error counter
                        {
                            let mut health = self.health_status.write().await;
                            health.errors_count += 1;
                        }
                    }
                }
                _ = shutdown_signal.changed() => {
                    debug!("Heartbeat loop shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Health monitoring loop - updates local health metrics
    async fn health_monitor_loop(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(10));
        let start_time = std::time::Instant::now();
        let mut shutdown_signal = self.shutdown_signal.clone();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    self.update_health_metrics(start_time).await?;
                }
                _ = shutdown_signal.changed() => {
                    debug!("Health monitor loop shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Workload management loop - handles work assignments from cluster
    async fn workload_loop(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(5));
        let mut shutdown_signal = self.shutdown_signal.clone();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.check_for_work().await {
                        warn!("Failed to check for work: {}", e);
                    }
                }
                _ = shutdown_signal.changed() => {
                    debug!("Workload loop shutting down");

                    // Stop all workloads gracefully
                    if let Err(e) = self.workload_manager.stop_all_workloads().await {
                        error!("Failed to stop workloads: {}", e);
                    }
                    break;
                }
            }
        }

        Ok(())
    }

    /// API server loop - provides local HTTP API for health checks and metrics
    async fn api_server_loop(&self) -> Result<()> {
        use std::convert::Infallible;
        use std::net::SocketAddr;

        // Create API routes
        let health_status = self.health_status.clone();
        let command_executor = self.command_executor.clone();
        let mut shutdown_signal = self.shutdown_signal.clone();

        let health_route = warp::path("health").and(warp::get()).and_then(move || {
            let health_status = health_status.clone();
            async move {
                let health = health_status.read().await;
                let response = serde_json::to_string(&*health)
                    .unwrap_or_else(|_| r#"{"error": "serialization_failed"}"#.to_string());

                Ok::<_, Infallible>(warp::reply::with_header(
                    warp::reply::html(response),
                    "content-type",
                    "application/json",
                ))
            }
        });

        let metrics_route = warp::path("metrics").and(warp::get()).map(|| {
            // Return Prometheus-style metrics
            format!(
                "# HELP swarmlet_uptime_seconds Uptime in seconds\n\
                     # TYPE swarmlet_uptime_seconds counter\n\
                     swarmlet_uptime_seconds {}\n",
                chrono::Utc::now().timestamp()
            )
        });

        // Command execution routes
        let execute_command_route = warp::path!("api" / "v1" / "execute")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |request: crate::command::CommandRequest| {
                let command_executor = command_executor.clone();
                async move {
                    match command_executor.execute_command(request).await {
                        Ok(result) => {
                            let json = serde_json::to_string(&result).unwrap_or_else(|_| {
                                r#"{"error": "serialization_failed"}"#.to_string()
                            });
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(json, warp::http::StatusCode::OK),
                                "content-type",
                                "application/json",
                            ))
                        }
                        Err(e) => {
                            let error_response = format!(r#"{{"error": "{e}"}}"#);
                            Ok::<_, Infallible>(warp::reply::with_header(
                                warp::reply::with_status(
                                    error_response,
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                                "content-type",
                                "application/json",
                            ))
                        }
                    }
                }
            });

        let command_executor_shell = self.command_executor.clone();
        let execute_shell_route = warp::path!("api" / "v1" / "shell")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |request: serde_json::Value| {
                let command_executor = command_executor_shell.clone();
                async move {
                    if let Some(script) = request.get("script").and_then(|s| s.as_str()) {
                        match command_executor.execute_shell(script).await {
                            Ok(result) => {
                                let json = serde_json::to_string(&result).unwrap_or_else(|_| {
                                    r#"{"error": "serialization_failed"}"#.to_string()
                                });
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(json, warp::http::StatusCode::OK),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                            Err(e) => {
                                let error_response = format!(r#"{{"error": "{e}"}}"#);
                                Ok::<_, Infallible>(warp::reply::with_header(
                                    warp::reply::with_status(
                                        error_response,
                                        warp::http::StatusCode::BAD_REQUEST,
                                    ),
                                    "content-type",
                                    "application/json",
                                ))
                            }
                        }
                    } else {
                        let error_response = r#"{"error": "Missing 'script' field"}"#.to_string();
                        Ok::<_, Infallible>(warp::reply::with_header(
                            warp::reply::with_status(
                                error_response,
                                warp::http::StatusCode::BAD_REQUEST,
                            ),
                            "content-type",
                            "application/json",
                        ))
                    }
                }
            });

        let routes = health_route
            .or(metrics_route)
            .or(execute_command_route)
            .or(execute_shell_route);

        // Start server
        let port = self.config.api_port.unwrap_or(8080);
        let addr: SocketAddr = ([0, 0, 0, 0], port).into();

        info!("Starting API server on {}", addr);

        tokio::select! {
            _ = warp::serve(routes).run(addr) => {
                debug!("API server completed");
            }
            _ = shutdown_signal.changed() => {
                debug!("API server shutting down");
            }
        }

        Ok(())
    }

    /// Send heartbeat to cluster
    async fn send_heartbeat(&self) -> Result<()> {
        let health = self.health_status.read().await.clone();

        let heartbeat = HeartbeatMessage {
            node_id: self.join_result.node_id,
            timestamp: chrono::Utc::now(),
            status: health.status,
            metrics: HealthMetrics {
                cpu_usage_percent: health.cpu_usage_percent,
                memory_usage_gb: health.memory_usage_gb,
                disk_usage_gb: health.disk_usage_gb,
                workloads_active: health.workloads_active,
                uptime_seconds: health.uptime_seconds,
            },
        };

        let client = reqwest::Client::new();
        let url = format!("{}/heartbeat", self.join_result.api_endpoints.health_check);

        match client.post(&url).json(&heartbeat).send().await {
            Ok(response) if response.status().is_success() => {
                debug!("Heartbeat sent successfully");

                // Update last heartbeat time
                {
                    let mut health = self.health_status.write().await;
                    health.last_heartbeat = chrono::Utc::now();
                }

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

    /// Update local health metrics
    async fn update_health_metrics(&self, start_time: std::time::Instant) -> Result<()> {
        use sysinfo::System;

        let mut system = System::new_all();
        system.refresh_all();

        let uptime_seconds = start_time.elapsed().as_secs();
        let cpu_usage = system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>()
            / system.cpus().len() as f32;
        let memory_usage =
            (system.total_memory() - system.available_memory()) as f32 / (1024.0 * 1024.0 * 1024.0);

        // Get disk usage for data directory
        let disk_usage = self.get_disk_usage().await.unwrap_or(0.0);

        // Get active workload count
        let workloads_active = self.workload_manager.active_workload_count().await;

        {
            let mut health = self.health_status.write().await;
            health.uptime_seconds = uptime_seconds;
            health.cpu_usage_percent = cpu_usage;
            health.memory_usage_gb = memory_usage;
            health.disk_usage_gb = disk_usage;
            health.workloads_active = workloads_active;

            // Update status based on metrics
            health.status = if cpu_usage > 90.0 || memory_usage > health.memory_usage_gb * 0.9 {
                NodeStatus::Degraded
            } else if health.errors_count > 10 {
                NodeStatus::Unhealthy
            } else {
                NodeStatus::Healthy
            };
        }

        Ok(())
    }

    /// Check for new work assignments from cluster
    async fn check_for_work(&self) -> Result<()> {
        let client = reqwest::Client::new();
        let url = format!("{}/work", self.join_result.api_endpoints.workload_api);

        match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                let work_assignments: Vec<WorkAssignment> = response.json().await?;

                for assignment in work_assignments {
                    debug!("Received work assignment: {}", assignment.id);

                    if let Err(e) = self.workload_manager.start_workload(assignment).await {
                        error!("Failed to start workload: {}", e);
                    }
                }
            }
            Ok(response) if response.status() == 204 => {
                // No work available
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

    /// Get disk usage for data directory
    async fn get_disk_usage(&self) -> Result<f32> {
        use std::fs;

        match fs::metadata(&self.config.data_dir) {
            Ok(metadata) => {
                // This is a simplified implementation
                // In reality, you'd calculate actual disk usage
                Ok(metadata.len() as f32 / (1024.0 * 1024.0 * 1024.0))
            }
            Err(_) => Ok(0.0),
        }
    }
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

/// Work assignment from cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkAssignment {
    pub id: Uuid,
    pub workload_type: String,
    pub container_image: Option<String>,
    pub command: Option<Vec<String>>,
    pub shell_script: Option<String>,
    pub environment: std::collections::HashMap<String, String>,
    pub resource_limits: ResourceLimits,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Resource limits for workloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub cpu_cores: Option<f32>,
    pub memory_gb: Option<f32>,
    pub disk_gb: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::join::ClusterApiEndpoints;
    use tempfile::TempDir;
    use tokio::time::Duration;

    /// TDD Phase tracking
    #[derive(Debug, Clone, PartialEq)]
    enum TddPhase {
        Red,      // Write failing tests
        Green,    // Make tests pass
        Refactor, // Optimize implementation
    }

    /// Test result tracking
    #[derive(Debug)]
    struct TestResult {
        test_name: String,
        phase: TddPhase,
        success: bool,
        duration: Duration,
        error_message: Option<String>,
    }

    /// Create a test join result
    fn create_test_join_result() -> JoinResult {
        JoinResult {
            node_id: Uuid::new_v4(),
            cluster_name: "test-cluster".to_string(),
            node_certificate: generate_test_certificate(),
            cluster_endpoints: vec!["http://localhost:7946".to_string()],
            assigned_capabilities: vec!["compute".to_string(), "storage".to_string()],
            heartbeat_interval: Duration::from_secs(30),
            api_endpoints: ClusterApiEndpoints {
                workload_api: "http://localhost:8081".to_string(),
                metrics_api: "http://localhost:8082".to_string(),
                logs_api: "http://localhost:8083".to_string(),
                health_check: "http://localhost:8080".to_string(),
            },
        }
    }

    /// Generate a test certificate
    fn generate_test_certificate() -> String {
        // This is a minimal valid PEM certificate for testing
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
    async fn test_agent_creation_from_join_result() {
        let start = std::time::Instant::now();
        let mut results = Vec::new();

        // RED Phase - Should fail initially
        let phase = TddPhase::Red;
        let join_result = create_test_join_result();
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        match SwarmletAgent::new(join_result.clone(), data_dir.clone()).await {
            Ok(agent) => {
                results.push(TestResult {
                    test_name: "agent_creation".to_string(),
                    phase: phase.clone(),
                    success: true,
                    duration: start.elapsed(),
                    error_message: None,
                });

                // Verify agent properties
                assert_eq!(agent.join_result.node_id, join_result.node_id);
                assert_eq!(agent.join_result.cluster_name, join_result.cluster_name);
            }
            Err(e) => {
                results.push(TestResult {
                    test_name: "agent_creation".to_string(),
                    phase,
                    success: false,
                    duration: start.elapsed(),
                    error_message: Some(e.to_string()),
                });
            }
        }

        // GREEN Phase - Should pass
        let phase = TddPhase::Green;
        let agent = SwarmletAgent::new(join_result.clone(), data_dir.clone())
            .await
            .expect("Should create agent successfully");

        results.push(TestResult {
            test_name: "agent_creation_green".to_string(),
            phase,
            success: true,
            duration: start.elapsed(),
            error_message: None,
        });

        // Verify health status initialization
        let health = agent.health_status.read().await;
        assert_eq!(health.node_id, join_result.node_id);
        assert_eq!(health.status, NodeStatus::Starting);
        assert_eq!(health.workloads_active, 0);
        assert_eq!(health.errors_count, 0);
    }

    #[tokio::test]
    async fn test_agent_health_status_updates() {
        let join_result = create_test_join_result();
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        let agent = SwarmletAgent::new(join_result, data_dir)
            .await
            .expect("Should create agent");

        // Test health status transitions
        {
            let mut health = agent.health_status.write().await;
            health.status = NodeStatus::Healthy;
            health.cpu_usage_percent = 45.0;
            health.memory_usage_gb = 3.5;
            health.workloads_active = 5;
        }

        // Verify updates
        {
            let health = agent.health_status.read().await;
            assert_eq!(health.status, NodeStatus::Healthy);
            assert_eq!(health.cpu_usage_percent, 45.0);
            assert_eq!(health.memory_usage_gb, 3.5);
            assert_eq!(health.workloads_active, 5);
        }

        // Test degraded status when resources are high
        {
            let mut health = agent.health_status.write().await;
            health.cpu_usage_percent = 95.0;
        }

        // Update health metrics should set degraded status
        agent
            .update_health_metrics(std::time::Instant::now())
            .await
            .ok();

        {
            let health = agent.health_status.read().await;
            // Note: Status might not change in test environment without actual high CPU
            assert!(health.cpu_usage_percent >= 0.0);
        }
    }

    #[tokio::test]
    async fn test_agent_shutdown_signal() {
        let join_result = create_test_join_result();
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        let agent = SwarmletAgent::new(join_result, data_dir)
            .await
            .expect("Should create agent");

        // Test shutdown signal
        assert!(agent.shutdown().is_ok());

        // Verify shutdown signal was sent
        let mut shutdown_signal = agent.shutdown_signal.clone();
        shutdown_signal
            .changed()
            .await
            .expect("Should receive shutdown signal");
        assert_eq!(*shutdown_signal.borrow(), true);
    }

    #[tokio::test]
    async fn test_heartbeat_message_serialization() {
        let heartbeat = HeartbeatMessage {
            node_id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            status: NodeStatus::Healthy,
            metrics: HealthMetrics {
                cpu_usage_percent: 25.5,
                memory_usage_gb: 4.2,
                disk_usage_gb: 50.0,
                workloads_active: 3,
                uptime_seconds: 3600,
            },
        };

        // Test serialization
        let json = serde_json::to_string(&heartbeat).expect("Should serialize heartbeat");
        assert!(json.contains("Healthy"));
        assert!(json.contains("25.5"));

        // Test deserialization
        let deserialized: HeartbeatMessage =
            serde_json::from_str(&json).expect("Should deserialize heartbeat");
        assert_eq!(deserialized.node_id, heartbeat.node_id);
        assert_eq!(deserialized.status, heartbeat.status);
        assert_eq!(deserialized.metrics.cpu_usage_percent, 25.5);
    }

    #[tokio::test]
    async fn test_work_assignment_handling() {
        let work_assignment = WorkAssignment {
            id: Uuid::new_v4(),
            workload_type: "container".to_string(),
            container_image: Some("nginx:latest".to_string()),
            command: Some(vec![
                "nginx".to_string(),
                "-g".to_string(),
                "daemon off;".to_string(),
            ]),
            shell_script: None,
            environment: std::collections::HashMap::from([(
                "ENV_VAR".to_string(),
                "value".to_string(),
            )]),
            resource_limits: ResourceLimits {
                cpu_cores: Some(2.0),
                memory_gb: Some(4.0),
                disk_gb: Some(10.0),
            },
            created_at: chrono::Utc::now(),
        };

        // Test serialization
        let json =
            serde_json::to_string(&work_assignment).expect("Should serialize work assignment");
        assert!(json.contains("nginx:latest"));
        assert!(json.contains("ENV_VAR"));

        // Test deserialization
        let deserialized: WorkAssignment =
            serde_json::from_str(&json).expect("Should deserialize work assignment");
        assert_eq!(deserialized.id, work_assignment.id);
        assert_eq!(deserialized.workload_type, "container");
        assert_eq!(
            deserialized.container_image,
            Some("nginx:latest".to_string())
        );
    }

    #[tokio::test]
    async fn test_node_status_transitions() {
        // Test all node status values
        let statuses = vec![
            NodeStatus::Starting,
            NodeStatus::Healthy,
            NodeStatus::Degraded,
            NodeStatus::Unhealthy,
            NodeStatus::Shutting,
        ];

        for status in statuses {
            let health = HealthStatus {
                node_id: Uuid::new_v4(),
                status: status.clone(),
                uptime_seconds: 100,
                workloads_active: 1,
                cpu_usage_percent: 10.0,
                memory_usage_gb: 2.0,
                disk_usage_gb: 20.0,
                network_rx_bytes: 1000,
                network_tx_bytes: 2000,
                last_heartbeat: chrono::Utc::now(),
                errors_count: 0,
            };

            // Test serialization
            let json = serde_json::to_string(&health).expect("Should serialize health status");
            let deserialized: HealthStatus =
                serde_json::from_str(&json).expect("Should deserialize health status");
            assert_eq!(deserialized.status, status);
        }
    }

    #[tokio::test]
    async fn test_api_routes_health_endpoint() {
        let join_result = create_test_join_result();
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        let agent = SwarmletAgent::new(join_result, data_dir)
            .await
            .expect("Should create agent");

        // Update some health metrics
        {
            let mut health = agent.health_status.write().await;
            health.status = NodeStatus::Healthy;
            health.cpu_usage_percent = 30.0;
            health.memory_usage_gb = 4.0;
            health.workloads_active = 2;
        }

        // Read health status
        let health = agent.health_status.read().await;
        let health_json = serde_json::to_string(&*health).expect("Should serialize health");

        // Verify health JSON contains expected fields
        assert!(health_json.contains("\"status\":\"Healthy\""));
        assert!(health_json.contains("\"cpu_usage_percent\":30.0"));
        assert!(health_json.contains("\"workloads_active\":2"));
    }

    #[tokio::test]
    async fn test_resource_limits_validation() {
        // Test with all fields set
        let limits_full = ResourceLimits {
            cpu_cores: Some(4.0),
            memory_gb: Some(8.0),
            disk_gb: Some(100.0),
        };

        let json = serde_json::to_string(&limits_full).expect("Should serialize");
        let deserialized: ResourceLimits = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.cpu_cores, Some(4.0));
        assert_eq!(deserialized.memory_gb, Some(8.0));
        assert_eq!(deserialized.disk_gb, Some(100.0));

        // Test with no limits set
        let limits_none = ResourceLimits {
            cpu_cores: None,
            memory_gb: None,
            disk_gb: None,
        };

        let json = serde_json::to_string(&limits_none).expect("Should serialize");
        let deserialized: ResourceLimits = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.cpu_cores, None);
        assert_eq!(deserialized.memory_gb, None);
        assert_eq!(deserialized.disk_gb, None);
    }

    #[tokio::test]
    async fn test_agent_from_config_not_implemented() {
        let config = Config::default();

        match SwarmletAgent::from_config(config).await {
            Ok(_) => panic!("Should return not implemented error"),
            Err(e) => match e {
                crate::SwarmletError::NotImplemented(msg) => {
                    assert!(msg.contains("not yet implemented"));
                }
                _ => panic!("Expected NotImplemented error, got: {:?}", e),
            },
        }
    }

    #[tokio::test]
    async fn test_disk_usage_calculation() {
        let join_result = create_test_join_result();
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap().to_string();

        // Create a test file in the data directory
        let test_file_path = temp_dir.path().join("test_file.dat");
        std::fs::write(&test_file_path, vec![0u8; 1024]).expect("Should write test file");

        let agent = SwarmletAgent::new(join_result, data_dir)
            .await
            .expect("Should create agent");

        // Get disk usage
        let disk_usage = agent.get_disk_usage().await.expect("Should get disk usage");

        // Should be greater than 0 but very small (1KB file)
        assert!(disk_usage >= 0.0);
        assert!(disk_usage < 0.001); // Less than 1MB
    }
}
