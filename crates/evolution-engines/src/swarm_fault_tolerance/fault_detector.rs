//! Node failure detection for distributed SwarmAgentic systems

use super::types::{AlertThresholds, FailureDetectionAlgorithm, HealthStatus, NodeHealth};
use crate::error::EvolutionEngineResult;
use crate::swarm_distributed::SwarmNode;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for fault tolerance
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Failure detection timeout in milliseconds
    pub failure_timeout_ms: u64,
    /// Checkpoint interval in generations
    pub checkpoint_interval: u32,
    /// Recovery strategy to use
    pub recovery_strategy: crate::swarm_distributed::RecoveryStrategy,
    /// Enable adaptive failure detection
    pub adaptive_detection: bool,
    /// Maximum recovery attempts
    pub max_recovery_attempts: u32,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            failure_timeout_ms: 30000,
            checkpoint_interval: 10,
            recovery_strategy: crate::swarm_distributed::RecoveryStrategy::Redistribute,
            adaptive_detection: true,
            max_recovery_attempts: 3,
        }
    }
}

/// Fault detector for monitoring node health
pub struct FaultDetector {
    /// Configuration
    pub(crate) config: FaultToleranceConfig,
    /// Node health tracking
    pub(crate) node_health: Arc<RwLock<HashMap<String, NodeHealth>>>,
    /// Detection algorithms
    pub(crate) detection_algorithms: Vec<FailureDetectionAlgorithm>,
    /// Alert thresholds
    pub(crate) alert_thresholds: AlertThresholds,
}

impl FaultDetector {
    /// Create new fault detector
    pub async fn new(config: FaultToleranceConfig) -> EvolutionEngineResult<Self> {
        Ok(Self {
            config: config.clone(),
            node_health: Arc::new(RwLock::new(HashMap::new())),
            detection_algorithms: vec![
                FailureDetectionAlgorithm::Heartbeat,
                FailureDetectionAlgorithm::Adaptive,
            ],
            alert_thresholds: AlertThresholds::default(),
        })
    }

    /// Get monitored nodes
    pub async fn get_monitored_nodes(&self) -> Vec<String> {
        self.node_health.read().await.keys().cloned().collect()
    }

    /// Start monitoring a node
    pub async fn start_monitoring(&self, node: SwarmNode) -> EvolutionEngineResult<()> {
        let health = NodeHealth {
            node_id: node.node_id.clone(),
            status: HealthStatus::Healthy,
            last_heartbeat: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                ?
                .as_millis() as u64,
            response_times: Vec::new(),
            error_count: 0,
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            network_score: 1.0,
        };

        self.node_health.write().await.insert(node.node_id, health);
        Ok(())
    }

    /// Stop monitoring a node
    pub async fn stop_monitoring(&self, node_id: &str) -> EvolutionEngineResult<()> {
        self.node_health.write().await.remove(node_id);
        Ok(())
    }

    /// Update node health information
    pub async fn update_node_health(
        &self,
        node_id: &str,
        response_time: f64,
        cpu_util: f64,
        memory_util: f64,
    ) -> EvolutionEngineResult<()> {
        let mut health_map = self.node_health.write().await;
        if let Some(health) = health_map.get_mut(node_id) {
            health.last_heartbeat = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            health.response_times.push(response_time);
            health.cpu_utilization = cpu_util;
            health.memory_utilization = memory_util;

            // Keep only recent response times
            if health.response_times.len() > 100 {
                health.response_times.drain(0..50);
            }

            // Update health status based on thresholds
            health.status = self.assess_health_status(health).await;
        }
        Ok(())
    }

    /// Check for failed nodes
    pub async fn check_for_failures(&self) -> EvolutionEngineResult<Vec<String>> {
        let mut failed_nodes = Vec::new();
        let health_map = self.node_health.read().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            ?
            .as_millis() as u64;

        for (node_id, health) in health_map.iter() {
            // Check heartbeat timeout
            if now - health.last_heartbeat > self.config.failure_timeout_ms {
                failed_nodes.push(node_id.clone());
                continue;
            }

            // Check other failure conditions
            if health.cpu_utilization > self.alert_thresholds.cpu_threshold
                || health.memory_utilization > self.alert_thresholds.memory_threshold
                || health.error_count > 10
            {
                failed_nodes.push(node_id.clone());
            }
        }

        Ok(failed_nodes)
    }

    /// Get health status of all nodes
    pub async fn get_cluster_health(&self) -> EvolutionEngineResult<HashMap<String, NodeHealth>> {
        Ok(self.node_health.read().await.clone())
    }

    /// Assess health status based on metrics
    async fn assess_health_status(&self, health: &NodeHealth) -> HealthStatus {
        // Simple assessment logic
        if health.cpu_utilization > 0.9 || health.memory_utilization > 0.9 {
            HealthStatus::Degraded
        } else if health.error_count > 5 {
            HealthStatus::Suspect
        } else {
            HealthStatus::Healthy
        }
    }
}
