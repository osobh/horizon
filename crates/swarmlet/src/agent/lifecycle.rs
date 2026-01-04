//! Agent lifecycle management - heartbeat, health monitoring, workload loops

use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, error, warn};
use uuid::Uuid;

use crate::{Result, SwarmletError};

use super::{NodeStatus, SwarmletAgent, WorkAssignment};

impl SwarmletAgent {
    /// Heartbeat loop - sends periodic status updates to cluster
    pub(super) async fn heartbeat_loop(&self) -> Result<()> {
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
    pub(super) async fn health_monitor_loop(&self) -> Result<()> {
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
    pub(super) async fn workload_loop(&self) -> Result<()> {
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

    /// Build job management loop - monitors and cleans up build jobs
    pub(super) async fn build_job_loop(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(60)); // Check every minute
        let mut shutdown_signal = self.shutdown_signal.clone();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Clean up old completed/failed build jobs (older than 1 hour)
                    self.build_job_manager.cleanup_old_jobs(3600).await;

                    // Log active build count
                    let active_count = self.build_job_manager.active_job_count().await;
                    if active_count > 0 {
                        debug!("Active build jobs: {}", active_count);
                    }
                }
                _ = shutdown_signal.changed() => {
                    debug!("Build job loop shutting down");

                    // Cancel all active builds gracefully
                    if let Err(e) = self.build_job_manager.cancel_all().await {
                        error!("Failed to cancel build jobs: {}", e);
                    }
                    break;
                }
            }
        }

        Ok(())
    }

    /// Send heartbeat to cluster
    pub(super) async fn send_heartbeat(&self) -> Result<()> {
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

    /// Get disk usage for data directory's mount point
    async fn get_disk_usage(&self) -> Result<f32> {
        use std::path::Path;
        use sysinfo::Disks;

        let data_path = Path::new(&self.config.data_dir)
            .canonicalize()
            .unwrap_or_else(|_| Path::new(&self.config.data_dir).to_path_buf());

        let disks = Disks::new_with_refreshed_list();

        // Find the disk with the longest mount point that is a prefix of data_path
        let mut best_disk: Option<&sysinfo::Disk> = None;
        let mut best_mount_len = 0;

        for disk in disks.list() {
            let mount_point = disk.mount_point();
            if data_path.starts_with(mount_point) {
                let mount_len = mount_point.as_os_str().len();
                if mount_len > best_mount_len {
                    best_mount_len = mount_len;
                    best_disk = Some(disk);
                }
            }
        }

        match best_disk {
            Some(disk) => {
                let total = disk.total_space();
                let available = disk.available_space();
                let used = total.saturating_sub(available);
                Ok(used as f32 / (1024.0 * 1024.0 * 1024.0))
            }
            None => {
                // Fallback: if no disk found, try to get the first disk's usage
                if let Some(disk) = disks.list().first() {
                    let total = disk.total_space();
                    let available = disk.available_space();
                    let used = total.saturating_sub(available);
                    Ok(used as f32 / (1024.0 * 1024.0 * 1024.0))
                } else {
                    Ok(0.0)
                }
            }
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
