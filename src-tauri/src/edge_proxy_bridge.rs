//! Edge Proxy Bridge (Synergy 5)
//!
//! Visualizes Vortex edge proxy with SLAI brain-driven intelligent routing.
//!
//! Vortex capabilities:
//! - Protocol transmutation: HTTP -> RMPI, HTTP -> WARP, HTTP -> TCP
//! - Radix-tree routing with hot-swap config
//! - Data affinity routing to nodes with data in RAM
//! - SLAI brain-link for real-time routing decisions
//! - WireGuard encryption for backend traffic
//!
//! SLAI Brain capabilities:
//! - GPU health monitoring (13-dimension feature vector)
//! - ML failure prediction (ensemble: threshold + statistical + random forest)
//! - Multi-factor routing (availability, utilization, proximity, tier, labels, carbon)
//! - Preemptive migration before failures
//!
//! Currently uses mock data until Vortex and SLAI crates are fully integrated.

use std::sync::Arc;
use tokio::sync::RwLock;

/// Edge proxy status.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EdgeProxyStatus {
    /// Active connections
    pub active_connections: u32,
    /// Requests per second
    pub requests_per_second: f64,
    /// Protocol statistics
    pub protocols: Vec<ProtocolStats>,
    /// Recent routing decisions
    pub routing_decisions: Vec<RoutingDecision>,
    /// Backend health status
    pub backend_health: Vec<BackendHealth>,
    /// Proxy uptime in seconds
    pub uptime_seconds: u64,
    /// Total requests handled
    pub total_requests: u64,
}

/// SLAI brain status.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BrainStatus {
    /// Total registered nodes
    pub registered_nodes: u32,
    /// Healthy GPUs
    pub healthy_gpus: u32,
    /// At-risk GPUs (predicted to fail soon)
    pub at_risk_gpus: u32,
    /// Failed GPUs
    pub failed_gpus: u32,
    /// Total predictions made
    pub predictions_made: u64,
    /// Migrations triggered proactively
    pub migrations_triggered: u64,
    /// Jobs saved from failure
    pub jobs_saved: u64,
    /// Model accuracy percentage
    pub model_accuracy_pct: f32,
    /// Active monitors
    pub active_monitors: u32,
}

/// Protocol statistics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProtocolStats {
    /// Source protocol
    pub source_protocol: String,
    /// Target protocol after transmutation
    pub target_protocol: String,
    /// Requests using this path
    pub request_count: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Success rate percentage
    pub success_rate_pct: f32,
}

/// Individual routing decision.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoutingDecision {
    /// Request ID
    pub request_id: String,
    /// Request path/endpoint
    pub path: String,
    /// Source protocol
    pub source_protocol: String,
    /// Target protocol after transmutation
    pub target_protocol: String,
    /// Target node selected
    pub target_node: String,
    /// Routing reason
    pub reason: RoutingReason,
    /// Decision latency in milliseconds
    pub decision_latency_ms: f64,
    /// Timestamp (unix epoch)
    pub timestamp: u64,
}

/// Routing reason from SLAI brain.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingReason {
    /// Best availability score
    BestAvailability,
    /// Data affinity (data already in node's RAM)
    DataAffinity,
    /// Lowest utilization
    LowestUtilization,
    /// Closest proximity (network topology)
    ClosestProximity,
    /// Label/tier matching
    LabelMatch,
    /// Lowest carbon footprint
    LowCarbon,
    /// Load balancing
    LoadBalance,
}

/// Backend node health status.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackendHealth {
    /// Node ID
    pub node_id: String,
    /// Node hostname
    pub hostname: String,
    /// Health status
    pub status: HealthStatus,
    /// GPU utilization percentage
    pub gpu_utilization_pct: f32,
    /// Memory utilization percentage
    pub memory_utilization_pct: f32,
    /// CPU utilization percentage
    pub cpu_utilization_pct: f32,
    /// Active jobs on this node
    pub active_jobs: u32,
    /// Failure probability (from SLAI prediction)
    pub failure_probability: f32,
    /// Last heartbeat (seconds ago)
    pub last_heartbeat_secs: u32,
}

/// Node health status.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    AtRisk,
    Failed,
    Unknown,
}

/// GPU failure prediction from SLAI.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FailurePrediction {
    /// GPU identifier
    pub gpu_id: String,
    /// Node hosting this GPU
    pub node_id: String,
    /// Probability of failure (0.0 - 1.0)
    pub probability: f32,
    /// Time to failure estimate (seconds)
    pub estimated_ttf_secs: Option<u64>,
    /// Primary contributing factor
    pub primary_factor: String,
    /// Recommended action
    pub recommended_action: String,
    /// Current jobs at risk
    pub jobs_at_risk: u32,
}

/// Combined edge proxy and brain status.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EdgeProxyBrainStatus {
    /// Edge proxy status
    pub proxy: EdgeProxyStatus,
    /// SLAI brain status
    pub brain: BrainStatus,
    /// Recent failure predictions
    pub predictions: Vec<FailurePrediction>,
}

/// Bridge to Vortex edge proxy and SLAI brain.
pub struct EdgeProxyBridge {
    state: Arc<RwLock<MockEdgeProxyState>>,
}

struct MockEdgeProxyState {
    proxy: EdgeProxyStatus,
    brain: BrainStatus,
    predictions: Vec<FailurePrediction>,
}

impl MockEdgeProxyState {
    fn new() -> Self {
        let protocols = vec![
            ProtocolStats {
                source_protocol: "HTTP/2".to_string(),
                target_protocol: "RMPI".to_string(),
                request_count: 1_234_567,
                avg_latency_ms: 0.45,
                success_rate_pct: 99.95,
            },
            ProtocolStats {
                source_protocol: "HTTP/2".to_string(),
                target_protocol: "WARP".to_string(),
                request_count: 456_789,
                avg_latency_ms: 0.38,
                success_rate_pct: 99.98,
            },
            ProtocolStats {
                source_protocol: "HTTP/2".to_string(),
                target_protocol: "TCP".to_string(),
                request_count: 234_567,
                avg_latency_ms: 0.62,
                success_rate_pct: 99.90,
            },
            ProtocolStats {
                source_protocol: "gRPC".to_string(),
                target_protocol: "RMPI".to_string(),
                request_count: 789_012,
                avg_latency_ms: 0.35,
                success_rate_pct: 99.97,
            },
        ];

        let routing_decisions = vec![
            RoutingDecision {
                request_id: "req-001".to_string(),
                path: "/api/v1/inference".to_string(),
                source_protocol: "HTTP/2".to_string(),
                target_protocol: "RMPI".to_string(),
                target_node: "gpu-server-1".to_string(),
                reason: RoutingReason::DataAffinity,
                decision_latency_ms: 0.12,
                timestamp: current_timestamp(),
            },
            RoutingDecision {
                request_id: "req-002".to_string(),
                path: "/api/v1/train".to_string(),
                source_protocol: "HTTP/2".to_string(),
                target_protocol: "RMPI".to_string(),
                target_node: "gpu-server-2".to_string(),
                reason: RoutingReason::LowestUtilization,
                decision_latency_ms: 0.08,
                timestamp: current_timestamp() - 1,
            },
            RoutingDecision {
                request_id: "req-003".to_string(),
                path: "/storage/upload".to_string(),
                source_protocol: "HTTP/2".to_string(),
                target_protocol: "WARP".to_string(),
                target_node: "storage-1".to_string(),
                reason: RoutingReason::BestAvailability,
                decision_latency_ms: 0.15,
                timestamp: current_timestamp() - 2,
            },
            RoutingDecision {
                request_id: "req-004".to_string(),
                path: "/api/v1/embed".to_string(),
                source_protocol: "gRPC".to_string(),
                target_protocol: "RMPI".to_string(),
                target_node: "gpu-server-3".to_string(),
                reason: RoutingReason::LowCarbon,
                decision_latency_ms: 0.10,
                timestamp: current_timestamp() - 3,
            },
        ];

        let backend_health = vec![
            BackendHealth {
                node_id: "node-001".to_string(),
                hostname: "gpu-server-1".to_string(),
                status: HealthStatus::Healthy,
                gpu_utilization_pct: 78.5,
                memory_utilization_pct: 65.2,
                cpu_utilization_pct: 45.0,
                active_jobs: 3,
                failure_probability: 0.02,
                last_heartbeat_secs: 1,
            },
            BackendHealth {
                node_id: "node-002".to_string(),
                hostname: "gpu-server-2".to_string(),
                status: HealthStatus::Healthy,
                gpu_utilization_pct: 45.2,
                memory_utilization_pct: 52.8,
                cpu_utilization_pct: 32.5,
                active_jobs: 2,
                failure_probability: 0.01,
                last_heartbeat_secs: 2,
            },
            BackendHealth {
                node_id: "node-003".to_string(),
                hostname: "gpu-server-3".to_string(),
                status: HealthStatus::AtRisk,
                gpu_utilization_pct: 92.5,
                memory_utilization_pct: 88.0,
                cpu_utilization_pct: 78.5,
                active_jobs: 5,
                failure_probability: 0.35,
                last_heartbeat_secs: 1,
            },
            BackendHealth {
                node_id: "node-004".to_string(),
                hostname: "gpu-server-4".to_string(),
                status: HealthStatus::Degraded,
                gpu_utilization_pct: 65.0,
                memory_utilization_pct: 70.2,
                cpu_utilization_pct: 55.0,
                active_jobs: 2,
                failure_probability: 0.15,
                last_heartbeat_secs: 5,
            },
        ];

        let proxy = EdgeProxyStatus {
            active_connections: 1_245,
            requests_per_second: 12_456.7,
            protocols,
            routing_decisions,
            backend_health,
            uptime_seconds: 86_400 * 7, // 7 days
            total_requests: 987_654_321,
        };

        let brain = BrainStatus {
            registered_nodes: 12,
            healthy_gpus: 42,
            at_risk_gpus: 3,
            failed_gpus: 1,
            predictions_made: 1_234_567,
            migrations_triggered: 156,
            jobs_saved: 142,
            model_accuracy_pct: 94.5,
            active_monitors: 46,
        };

        let predictions = vec![
            FailurePrediction {
                gpu_id: "gpu-server-3:0".to_string(),
                node_id: "node-003".to_string(),
                probability: 0.35,
                estimated_ttf_secs: Some(3600), // 1 hour
                primary_factor: "Memory temperature elevated".to_string(),
                recommended_action: "Migrate jobs to gpu-server-1".to_string(),
                jobs_at_risk: 2,
            },
            FailurePrediction {
                gpu_id: "gpu-server-4:1".to_string(),
                node_id: "node-004".to_string(),
                probability: 0.15,
                estimated_ttf_secs: Some(86400), // 1 day
                primary_factor: "ECC errors trending up".to_string(),
                recommended_action: "Schedule maintenance window".to_string(),
                jobs_at_risk: 1,
            },
        ];

        Self {
            proxy,
            brain,
            predictions,
        }
    }

    fn simulate_activity(&mut self) {
        // Update proxy metrics
        let conn_variance = (rand_float() * 100.0) as i32 - 50;
        self.proxy.active_connections = (self.proxy.active_connections as i32 + conn_variance).max(100) as u32;

        let rps_variance = rand_float() as f64 * 1000.0 - 500.0;
        self.proxy.requests_per_second = (self.proxy.requests_per_second + rps_variance).max(1000.0);

        self.proxy.total_requests += (rand_float() as u64 * 1000) + 100;

        // Update protocol stats
        for protocol in &mut self.proxy.protocols {
            protocol.request_count += (rand_float() as u64 * 100) + 10;
            let latency_variance = rand_float() as f64 * 0.1 - 0.05;
            protocol.avg_latency_ms = (protocol.avg_latency_ms + latency_variance).max(0.1).min(2.0);
        }

        // Rotate routing decisions
        let new_decision = RoutingDecision {
            request_id: format!("req-{}", rand_float() as u64 * 1000),
            path: ["/api/v1/inference", "/api/v1/train", "/storage/upload", "/api/v1/embed"]
                [(rand_float() * 4.0) as usize % 4].to_string(),
            source_protocol: ["HTTP/2", "gRPC"][(rand_float() * 2.0) as usize % 2].to_string(),
            target_protocol: ["RMPI", "WARP", "TCP"][(rand_float() * 3.0) as usize % 3].to_string(),
            target_node: format!("gpu-server-{}", (rand_float() * 4.0) as usize + 1),
            reason: match (rand_float() * 7.0) as usize {
                0 => RoutingReason::BestAvailability,
                1 => RoutingReason::DataAffinity,
                2 => RoutingReason::LowestUtilization,
                3 => RoutingReason::ClosestProximity,
                4 => RoutingReason::LabelMatch,
                5 => RoutingReason::LowCarbon,
                _ => RoutingReason::LoadBalance,
            },
            decision_latency_ms: rand_float() as f64 * 0.2,
            timestamp: current_timestamp(),
        };
        self.proxy.routing_decisions.insert(0, new_decision);
        if self.proxy.routing_decisions.len() > 10 {
            self.proxy.routing_decisions.pop();
        }

        // Update backend health
        for backend in &mut self.proxy.backend_health {
            let gpu_variance = rand_float() * 10.0 - 5.0;
            backend.gpu_utilization_pct = (backend.gpu_utilization_pct + gpu_variance).max(10.0).min(100.0);

            let mem_variance = rand_float() * 5.0 - 2.5;
            backend.memory_utilization_pct = (backend.memory_utilization_pct + mem_variance).max(10.0).min(100.0);

            // Update failure probability based on utilization
            if backend.gpu_utilization_pct > 90.0 {
                backend.failure_probability = (backend.failure_probability + 0.02).min(0.5);
                backend.status = HealthStatus::AtRisk;
            } else if backend.failure_probability > 0.1 {
                backend.status = HealthStatus::Degraded;
            } else {
                backend.failure_probability = (backend.failure_probability - 0.01).max(0.01);
                backend.status = HealthStatus::Healthy;
            }
        }

        // Update brain stats
        self.brain.predictions_made += (rand_float() as u64 * 10) + 1;
        if rand_float() > 0.95 {
            self.brain.migrations_triggered += 1;
            self.brain.jobs_saved += 1;
        }
    }
}

fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn rand_float() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as f32 % 100.0) / 100.0
}

impl EdgeProxyBridge {
    /// Create a new edge proxy bridge.
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MockEdgeProxyState::new())),
        }
    }

    /// Initialize the edge proxy bridge.
    pub async fn initialize(&self) -> Result<(), String> {
        tracing::info!("Edge proxy bridge initialized (mock mode)");
        Ok(())
    }

    /// Get complete edge proxy and brain status.
    pub async fn get_status(&self) -> EdgeProxyBrainStatus {
        let state = self.state.read().await;
        EdgeProxyBrainStatus {
            proxy: state.proxy.clone(),
            brain: state.brain.clone(),
            predictions: state.predictions.clone(),
        }
    }

    /// Get edge proxy status only.
    pub async fn get_proxy_status(&self) -> EdgeProxyStatus {
        let state = self.state.read().await;
        state.proxy.clone()
    }

    /// Get SLAI brain status.
    pub async fn get_brain_status(&self) -> BrainStatus {
        let state = self.state.read().await;
        state.brain.clone()
    }

    /// Get failure predictions.
    pub async fn get_predictions(&self) -> Vec<FailurePrediction> {
        let state = self.state.read().await;
        state.predictions.clone()
    }

    /// Get backend health.
    pub async fn get_backend_health(&self) -> Vec<BackendHealth> {
        let state = self.state.read().await;
        state.proxy.backend_health.clone()
    }

    /// Get recent routing decisions.
    pub async fn get_routing_decisions(&self) -> Vec<RoutingDecision> {
        let state = self.state.read().await;
        state.proxy.routing_decisions.clone()
    }

    /// Simulate activity (for demo purposes).
    pub async fn simulate_activity(&self) {
        let mut state = self.state.write().await;
        state.simulate_activity();
    }
}

impl Default for EdgeProxyBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_creation() {
        let bridge = EdgeProxyBridge::new();
        let status = bridge.get_status().await;
        assert!(!status.proxy.protocols.is_empty());
        assert!(!status.proxy.backend_health.is_empty());
    }

    #[tokio::test]
    async fn test_simulate_activity() {
        let bridge = EdgeProxyBridge::new();
        let before = bridge.get_proxy_status().await;
        bridge.simulate_activity().await;
        let after = bridge.get_proxy_status().await;
        assert!(after.total_requests > before.total_requests);
    }
}
