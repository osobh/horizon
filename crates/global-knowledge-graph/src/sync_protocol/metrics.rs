//! Metrics collection and monitoring for synchronization

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

/// Real-time synchronization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMetrics {
    pub propagation_latency: Duration,
    pub consensus_duration: Duration,
    pub operations_per_second: f64,
    pub conflicts_resolved: u64,
    pub gpu_utilization: f32,
    pub network_bandwidth_used: u64,
    pub active_clusters: usize,
    pub sync_success_rate: f32,
}

/// Consensus algorithm metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusMetrics {
    pub round_count: usize,
    pub vote_count: usize,
    pub proposal_count: usize,
    pub consensus_time: Duration,
}

/// Per-cluster metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMetrics {
    pub cluster_id: String,
    pub operations_processed: u64,
    pub sync_latency: Duration,
    pub cpu_usage: f32,
    pub memory_usage: f32,
}

/// Network performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub average_latency: Duration,
}

/// Aggregated metrics across all clusters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub total_operations: u64,
    pub average_latency: Duration,
    pub success_rate: f32,
    pub cluster_metrics: HashMap<String, ClusterMetrics>,
}

/// Performance anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnomaly {
    pub anomaly_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub affected_clusters: Vec<String>,
    pub description: String,
}

/// Types of performance anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    HighLatency,
    LowThroughput,
    ConsensusFailure,
    NetworkPartition,
    ResourceExhaustion,
    Other(String),
}

/// Severity levels for anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance report for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub report_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub sync_metrics: SyncMetrics,
    pub consensus_metrics: ConsensusMetrics,
    pub network_metrics: NetworkMetrics,
    pub anomalies: Vec<PerformanceAnomaly>,
}

/// Trait for metrics collection
#[async_trait::async_trait]
pub trait MetricsCollector: Send + Sync {
    async fn collect(&self) -> AggregatedMetrics;
    async fn report_anomaly(&self, anomaly: PerformanceAnomaly);
    async fn get_historical_metrics(&self, duration: Duration) -> Vec<AggregatedMetrics>;
}

/// Default metrics collector implementation
pub struct DefaultMetricsCollector {
    metrics_history: tokio::sync::RwLock<Vec<AggregatedMetrics>>,
    anomalies: tokio::sync::RwLock<Vec<PerformanceAnomaly>>,
}

impl DefaultMetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics_history: tokio::sync::RwLock::new(Vec::new()),
            anomalies: tokio::sync::RwLock::new(Vec::new()),
        }
    }
}

#[async_trait::async_trait]
impl MetricsCollector for DefaultMetricsCollector {
    async fn collect(&self) -> AggregatedMetrics {
        // Implementation would collect real metrics
        AggregatedMetrics {
            total_operations: 0,
            average_latency: Duration::from_millis(50),
            success_rate: 0.99,
            cluster_metrics: HashMap::new(),
        }
    }

    async fn report_anomaly(&self, anomaly: PerformanceAnomaly) {
        let mut anomalies = self.anomalies.write().await;
        anomalies.push(anomaly);
        
        // Keep only last 1000 anomalies
        if anomalies.len() > 1000 {
            anomalies.drain(0..anomalies.len() - 1000);
        }
    }

    async fn get_historical_metrics(&self, _duration: Duration) -> Vec<AggregatedMetrics> {
        let history = self.metrics_history.read().await;
        history.clone()
    }
}