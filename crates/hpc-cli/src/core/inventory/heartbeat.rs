//! Heartbeat monitoring for inventory nodes
//!
//! Provides background health monitoring for registered nodes,
//! updating status based on agent responses.

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::store::InventoryStore;
use hpc_inventory::{NodeInfo, NodeStatus};

/// Default heartbeat interval in seconds
pub const DEFAULT_HEARTBEAT_INTERVAL_SECS: u64 = 30;

/// Default timeout for considering a node unreachable
pub const DEFAULT_UNREACHABLE_TIMEOUT_SECS: i64 = 90;

/// Heartbeat state for tracking node health
#[derive(Debug, Clone)]
pub struct HeartbeatState {
    /// Last heartbeat time per node
    pub last_heartbeat: HashMap<String, DateTime<Utc>>,
    /// Node health scores (0-100)
    pub health_scores: HashMap<String, u8>,
}

impl Default for HeartbeatState {
    fn default() -> Self {
        Self::new()
    }
}

impl HeartbeatState {
    /// Create new heartbeat state
    pub fn new() -> Self {
        Self {
            last_heartbeat: HashMap::new(),
            health_scores: HashMap::new(),
        }
    }

    /// Record a heartbeat for a node
    pub fn record_heartbeat(&mut self, node_id: &str) {
        self.last_heartbeat
            .insert(node_id.to_string(), Utc::now());
        // Increase health score on successful heartbeat
        let score = self.health_scores.get(node_id).copied().unwrap_or(50);
        self.health_scores
            .insert(node_id.to_string(), (score + 10).min(100));
    }

    /// Record a failed heartbeat
    pub fn record_failure(&mut self, node_id: &str) {
        // Decrease health score on failure
        let score = self.health_scores.get(node_id).copied().unwrap_or(50);
        self.health_scores
            .insert(node_id.to_string(), score.saturating_sub(20));
    }

    /// Get time since last heartbeat
    pub fn time_since_heartbeat(&self, node_id: &str) -> Option<Duration> {
        self.last_heartbeat
            .get(node_id)
            .map(|time| Utc::now().signed_duration_since(*time))
    }

    /// Check if node is responsive
    pub fn is_responsive(&self, node_id: &str, timeout_secs: i64) -> bool {
        match self.time_since_heartbeat(node_id) {
            Some(duration) => duration.num_seconds() < timeout_secs,
            None => false,
        }
    }

    /// Get health score for a node
    pub fn health_score(&self, node_id: &str) -> u8 {
        self.health_scores.get(node_id).copied().unwrap_or(0)
    }
}

/// Heartbeat monitor for background health checking
pub struct HeartbeatMonitor {
    /// Shared heartbeat state
    state: Arc<RwLock<HeartbeatState>>,
    /// Heartbeat check interval
    interval_secs: u64,
    /// Unreachable timeout
    unreachable_timeout_secs: i64,
    /// Shutdown signal
    shutdown: Arc<RwLock<bool>>,
}

impl HeartbeatMonitor {
    /// Create a new heartbeat monitor
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(HeartbeatState::new())),
            interval_secs: DEFAULT_HEARTBEAT_INTERVAL_SECS,
            unreachable_timeout_secs: DEFAULT_UNREACHABLE_TIMEOUT_SECS,
            shutdown: Arc::new(RwLock::new(false)),
        }
    }

    /// Set check interval
    pub fn with_interval(mut self, secs: u64) -> Self {
        self.interval_secs = secs;
        self
    }

    /// Set unreachable timeout
    pub fn with_timeout(mut self, secs: i64) -> Self {
        self.unreachable_timeout_secs = secs;
        self
    }

    /// Get shared state reference
    pub fn state(&self) -> Arc<RwLock<HeartbeatState>> {
        self.state.clone()
    }

    /// Record a heartbeat for a node
    pub async fn record_heartbeat(&self, node_id: &str) {
        self.state.write().await.record_heartbeat(node_id);
    }

    /// Record a failed heartbeat
    pub async fn record_failure(&self, node_id: &str) {
        self.state.write().await.record_failure(node_id);
    }

    /// Check if a node is responsive
    pub async fn is_responsive(&self, node_id: &str) -> bool {
        self.state
            .read()
            .await
            .is_responsive(node_id, self.unreachable_timeout_secs)
    }

    /// Get health score for a node
    pub async fn health_score(&self, node_id: &str) -> u8 {
        self.state.read().await.health_score(node_id)
    }

    /// Signal shutdown
    pub async fn shutdown(&self) {
        *self.shutdown.write().await = true;
    }

    /// Check if shutdown was requested
    async fn should_shutdown(&self) -> bool {
        *self.shutdown.read().await
    }

    /// Start the background monitoring task
    pub async fn start_monitoring(&self) -> Result<()> {
        let state = self.state.clone();
        let interval = self.interval_secs;
        let timeout = self.unreachable_timeout_secs;
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            loop {
                // Check shutdown flag
                if *shutdown.read().await {
                    break;
                }

                // Check all nodes
                if let Ok(store) = InventoryStore::new() {
                    let nodes: Vec<NodeInfo> = store.list_nodes().into_iter().cloned().collect();
                    let mut state = state.write().await;

                    for node in nodes {
                        // Only check connected/unreachable nodes
                        if matches!(
                            node.status,
                            NodeStatus::Connected | NodeStatus::Unreachable
                        ) {
                            // Check if we've received a heartbeat recently
                            if !state.is_responsive(&node.id, timeout) {
                                // Node is not responsive
                                if let Ok(mut store) = InventoryStore::new() {
                                    if let Some(n) = store.find_node_mut(&node.id) {
                                        if n.status == NodeStatus::Connected {
                                            n.set_status(NodeStatus::Unreachable);
                                            let _ = store.save();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Wait for next check
                tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
            }
        });

        Ok(())
    }
}

impl Default for HeartbeatMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Ping a node to check if it's responsive
pub async fn ping_node(node: &NodeInfo) -> Result<bool> {
    use super::ssh::{SshAuth, SshClient};

    // Build auth from credential
    let auth = SshAuth::from(&node.credential_ref);

    // Try to connect and run a simple command
    let client = SshClient::new(
        node.address.clone(),
        node.port,
        node.username.clone(),
        auth,
    )
    .with_timeout(10);

    match client.connect().await {
        Ok(session) => {
            // Try a simple command
            let result = session.exec("echo ping").await;
            let _ = session.disconnect().await;
            Ok(result.is_ok())
        }
        Err(_) => Ok(false),
    }
}

/// Quick health check for a node
pub async fn check_node_health(node: &NodeInfo) -> NodeHealthResult {
    use super::ssh::{SshAuth, SshClient};

    let auth = SshAuth::from(&node.credential_ref);

    let client = SshClient::new(
        node.address.clone(),
        node.port,
        node.username.clone(),
        auth,
    )
    .with_timeout(15);

    match client.connect().await {
        Ok(session) => {
            // Check agent status
            let agent_check = session.exec("pgrep -x swarmlet || docker ps | grep swarmlet").await;
            let agent_running = agent_check.map(|o| o.success()).unwrap_or(false);

            // Get load average
            let load = session
                .exec("cat /proc/loadavg 2>/dev/null | awk '{print $1}' || sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}'")
                .await
                .map(|o| o.stdout.trim().parse::<f32>().unwrap_or(0.0))
                .unwrap_or(0.0);

            // Get memory usage
            let mem_used = session
                .exec("free -m 2>/dev/null | awk '/Mem:/ {printf \"%.1f\", $3/$2*100}' || vm_stat 2>/dev/null | awk '/Pages active/ {print $3}' | tr -d '.'")
                .await
                .map(|o| o.stdout.trim().parse::<f32>().unwrap_or(0.0))
                .unwrap_or(0.0);

            let _ = session.disconnect().await;

            NodeHealthResult {
                reachable: true,
                agent_running,
                load_average: load,
                memory_percent: mem_used,
                error: None,
            }
        }
        Err(e) => NodeHealthResult {
            reachable: false,
            agent_running: false,
            load_average: 0.0,
            memory_percent: 0.0,
            error: Some(e.to_string()),
        },
    }
}

/// Result of a node health check
#[derive(Debug, Clone)]
pub struct NodeHealthResult {
    /// Node is reachable via SSH
    pub reachable: bool,
    /// Agent process is running
    pub agent_running: bool,
    /// System load average
    pub load_average: f32,
    /// Memory usage percentage
    pub memory_percent: f32,
    /// Error message if check failed
    pub error: Option<String>,
}

impl NodeHealthResult {
    /// Check if node is healthy
    pub fn is_healthy(&self) -> bool {
        self.reachable && self.agent_running
    }

    /// Get health status as string
    pub fn status_str(&self) -> &str {
        if !self.reachable {
            "Unreachable"
        } else if !self.agent_running {
            "Agent Down"
        } else if self.load_average > 10.0 {
            "High Load"
        } else if self.memory_percent > 95.0 {
            "Low Memory"
        } else {
            "Healthy"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heartbeat_state_new() {
        let state = HeartbeatState::new();
        assert!(state.last_heartbeat.is_empty());
        assert!(state.health_scores.is_empty());
    }

    #[test]
    fn test_record_heartbeat() {
        let mut state = HeartbeatState::new();
        state.record_heartbeat("node-1");

        assert!(state.last_heartbeat.contains_key("node-1"));
        assert_eq!(state.health_score("node-1"), 60); // 50 + 10
    }

    #[test]
    fn test_record_failure() {
        let mut state = HeartbeatState::new();
        state.record_heartbeat("node-1"); // Set initial score to 60
        state.record_failure("node-1");

        assert_eq!(state.health_score("node-1"), 40); // 60 - 20
    }

    #[test]
    fn test_is_responsive() {
        let mut state = HeartbeatState::new();
        state.record_heartbeat("node-1");

        assert!(state.is_responsive("node-1", 60));
        assert!(!state.is_responsive("unknown-node", 60));
    }

    #[tokio::test]
    async fn test_heartbeat_monitor() {
        let monitor = HeartbeatMonitor::new().with_interval(1).with_timeout(5);

        monitor.record_heartbeat("node-1").await;
        assert!(monitor.is_responsive("node-1").await);

        let score = monitor.health_score("node-1").await;
        assert_eq!(score, 60);

        monitor.record_failure("node-1").await;
        let score = monitor.health_score("node-1").await;
        assert_eq!(score, 40);
    }

    #[test]
    fn test_node_health_result() {
        let healthy = NodeHealthResult {
            reachable: true,
            agent_running: true,
            load_average: 1.5,
            memory_percent: 60.0,
            error: None,
        };
        assert!(healthy.is_healthy());
        assert_eq!(healthy.status_str(), "Healthy");

        let unreachable = NodeHealthResult {
            reachable: false,
            agent_running: false,
            load_average: 0.0,
            memory_percent: 0.0,
            error: Some("Connection refused".to_string()),
        };
        assert!(!unreachable.is_healthy());
        assert_eq!(unreachable.status_str(), "Unreachable");

        let agent_down = NodeHealthResult {
            reachable: true,
            agent_running: false,
            load_average: 1.0,
            memory_percent: 50.0,
            error: None,
        };
        assert!(!agent_down.is_healthy());
        assert_eq!(agent_down.status_str(), "Agent Down");
    }
}
