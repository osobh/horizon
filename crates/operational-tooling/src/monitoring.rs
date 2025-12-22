//! Production monitoring and analytics for operational tooling

use crate::error::OperationalResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Operational metrics collection and analysis
#[derive(Debug, Clone)]
pub struct OperationalMetrics {
    /// Resource usage metrics
    pub resource_usage: ResourceMetrics,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Error metrics
    pub errors: ErrorMetrics,
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
}

impl Default for OperationalMetrics {
    fn default() -> Self {
        Self {
            resource_usage: ResourceMetrics::default(),
            performance: PerformanceMetrics::default(),
            errors: ErrorMetrics::default(),
            timestamp: SystemTime::UNIX_EPOCH,
        }
    }
}

/// Resource usage metrics
#[derive(Debug, Clone, Default)]
pub struct ResourceMetrics {
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Memory usage in MB
    pub memory_usage_mb: u32,
    /// GPU utilization per device
    pub gpu_utilization: HashMap<u32, GpuUtilization>,
    /// Network bandwidth usage
    pub network_bandwidth_mbps: f32,
    /// Storage usage in MB
    pub storage_usage_mb: u32,
}

/// GPU utilization metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuUtilization {
    /// GPU compute utilization percentage
    pub compute_utilization: f32,
    /// GPU memory utilization percentage
    pub memory_utilization: f32,
    /// GPU temperature in Celsius
    pub temperature: f32,
    /// Power consumption in watts
    pub power_usage_watts: f32,
    /// Active kernels count
    pub active_kernels: u32,
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Request latency percentiles
    pub latency_p50_ms: f32,
    pub latency_p95_ms: f32,
    pub latency_p99_ms: f32,
    /// Throughput in operations per second
    pub throughput_ops_per_sec: f32,
    /// Success rate percentage
    pub success_rate: f32,
}

/// Error metrics
#[derive(Debug, Clone, Default)]
pub struct ErrorMetrics {
    /// Total error count
    pub total_errors: u32,
    /// Error rate per minute
    pub error_rate_per_min: f32,
    /// Error types breakdown
    pub error_types: HashMap<String, u32>,
}

/// Resource monitor for tracking system resources
pub struct ResourceMonitor {
    /// Monitoring interval
    interval: Duration,
    /// Resource thresholds
    thresholds: ResourceThresholds,
}

/// Resource usage thresholds
#[derive(Debug, Clone)]
pub struct ResourceThresholds {
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f32,
    /// Maximum memory usage in MB
    pub max_memory_mb: u32,
    /// Maximum GPU utilization percentage
    pub max_gpu_utilization: f32,
    /// Maximum network bandwidth in Mbps
    pub max_network_mbps: f32,
}

/// Agent behavior analytics
pub struct AgentAnalytics {
    /// Agent performance metrics
    agent_metrics: HashMap<String, AgentMetrics>,
}

/// Metrics for individual agents
#[derive(Debug, Clone, Default)]
pub struct AgentMetrics {
    /// Agent execution time
    pub execution_time_ms: f32,
    /// Memory consumption
    pub memory_usage_mb: u32,
    /// Success/failure rate
    pub success_rate: f32,
    /// Resource efficiency score
    pub efficiency_score: f32,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            thresholds: ResourceThresholds::default(),
        }
    }

    /// Set resource thresholds
    pub fn set_thresholds(&mut self, thresholds: ResourceThresholds) {
        self.thresholds = thresholds;
    }

    /// Start monitoring resources
    pub async fn start_monitoring(&self) -> OperationalResult<()> {
        // Mock implementation
        Ok(())
    }

    /// Collect current resource metrics
    pub async fn collect_metrics(&self) -> OperationalResult<OperationalMetrics> {
        // Mock implementation
        Ok(OperationalMetrics::default())
    }

    /// Check if resources exceed thresholds
    pub fn check_thresholds(&self, metrics: &OperationalMetrics) -> Vec<String> {
        let mut violations = Vec::new();

        if metrics.resource_usage.cpu_usage > self.thresholds.max_cpu_usage {
            violations.push(format!(
                "CPU usage {} exceeds threshold {}",
                metrics.resource_usage.cpu_usage, self.thresholds.max_cpu_usage
            ));
        }

        if metrics.resource_usage.memory_usage_mb > self.thresholds.max_memory_mb {
            violations.push(format!(
                "Memory usage {} MB exceeds threshold {} MB",
                metrics.resource_usage.memory_usage_mb, self.thresholds.max_memory_mb
            ));
        }

        violations
    }
}

impl AgentAnalytics {
    /// Create new agent analytics
    pub fn new() -> Self {
        Self {
            agent_metrics: HashMap::new(),
        }
    }

    /// Record agent metrics
    pub fn record_agent_metrics(&mut self, agent_id: String, metrics: AgentMetrics) {
        self.agent_metrics.insert(agent_id, metrics);
    }

    /// Get agent metrics
    pub fn get_agent_metrics(&self, agent_id: &str) -> Option<&AgentMetrics> {
        self.agent_metrics.get(agent_id)
    }

    /// Get top performing agents
    pub fn get_top_performers(&self, limit: usize) -> Vec<(String, AgentMetrics)> {
        let mut agents: Vec<_> = self
            .agent_metrics
            .iter()
            .map(|(id, metrics)| (id.clone(), metrics.clone()))
            .collect();

        agents.sort_by(|a, b| {
            b.1.efficiency_score
                .partial_cmp(&a.1.efficiency_score)
                .unwrap()
        });
        agents.truncate(limit);
        agents
    }
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            max_cpu_usage: 80.0,
            max_memory_mb: 8192,
            max_gpu_utilization: 90.0,
            max_network_mbps: 1000.0,
        }
    }
}

impl Default for AgentAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operational_metrics_default() {
        let metrics = OperationalMetrics::default();
        assert_eq!(metrics.resource_usage.cpu_usage, 0.0);
        assert_eq!(metrics.performance.throughput_ops_per_sec, 0.0);
        assert_eq!(metrics.errors.total_errors, 0);
    }

    #[test]
    fn test_gpu_utilization_serialization() {
        let gpu_util = GpuUtilization {
            compute_utilization: 75.5,
            memory_utilization: 80.0,
            temperature: 65.0,
            power_usage_watts: 250.0,
            active_kernels: 4,
        };

        let serialized = serde_json::to_string(&gpu_util).unwrap();
        let deserialized: GpuUtilization = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            gpu_util.compute_utilization,
            deserialized.compute_utilization
        );
        assert_eq!(gpu_util.memory_utilization, deserialized.memory_utilization);
        assert_eq!(gpu_util.temperature, deserialized.temperature);
        assert_eq!(gpu_util.power_usage_watts, deserialized.power_usage_watts);
        assert_eq!(gpu_util.active_kernels, deserialized.active_kernels);
    }

    #[test]
    fn test_resource_monitor_creation() {
        let monitor = ResourceMonitor::new(Duration::from_secs(10));
        assert_eq!(monitor.interval, Duration::from_secs(10));
    }

    #[test]
    fn test_resource_thresholds_default() {
        let thresholds = ResourceThresholds::default();
        assert_eq!(thresholds.max_cpu_usage, 80.0);
        assert_eq!(thresholds.max_memory_mb, 8192);
        assert_eq!(thresholds.max_gpu_utilization, 90.0);
        assert_eq!(thresholds.max_network_mbps, 1000.0);
    }

    #[test]
    fn test_threshold_violations() {
        let monitor = ResourceMonitor::new(Duration::from_secs(1));
        let mut metrics = OperationalMetrics::default();

        // Set metrics that exceed thresholds
        metrics.resource_usage.cpu_usage = 85.0; // Above default 80%
        metrics.resource_usage.memory_usage_mb = 9000; // Above default 8192 MB

        let violations = monitor.check_thresholds(&metrics);
        assert_eq!(violations.len(), 2);
        assert!(violations[0].contains("CPU usage"));
        assert!(violations[1].contains("Memory usage"));
    }

    #[test]
    fn test_agent_analytics() {
        let mut analytics = AgentAnalytics::new();

        let metrics = AgentMetrics {
            execution_time_ms: 100.0,
            memory_usage_mb: 256,
            success_rate: 95.0,
            efficiency_score: 85.0,
        };

        analytics.record_agent_metrics("agent-1".to_string(), metrics.clone());

        let retrieved = analytics.get_agent_metrics("agent-1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().execution_time_ms, 100.0);
    }

    #[test]
    fn test_top_performers() {
        let mut analytics = AgentAnalytics::new();

        // Add agents with different efficiency scores
        analytics.record_agent_metrics(
            "agent-1".to_string(),
            AgentMetrics {
                efficiency_score: 90.0,
                ..Default::default()
            },
        );
        analytics.record_agent_metrics(
            "agent-2".to_string(),
            AgentMetrics {
                efficiency_score: 85.0,
                ..Default::default()
            },
        );
        analytics.record_agent_metrics(
            "agent-3".to_string(),
            AgentMetrics {
                efficiency_score: 95.0,
                ..Default::default()
            },
        );

        let top_performers = analytics.get_top_performers(2);
        assert_eq!(top_performers.len(), 2);
        assert_eq!(top_performers[0].0, "agent-3"); // Highest score
        assert_eq!(top_performers[1].0, "agent-1"); // Second highest
    }

    #[tokio::test]
    async fn test_resource_monitor_operations() {
        let monitor = ResourceMonitor::new(Duration::from_secs(1));

        // Test monitoring start
        let result = monitor.start_monitoring().await;
        assert!(result.is_ok());

        // Test metrics collection
        let metrics = monitor.collect_metrics().await;
        assert!(metrics.is_ok());
    }

    #[test]
    fn test_custom_thresholds() {
        let mut monitor = ResourceMonitor::new(Duration::from_secs(1));
        let custom_thresholds = ResourceThresholds {
            max_cpu_usage: 90.0,
            max_memory_mb: 16384,
            max_gpu_utilization: 95.0,
            max_network_mbps: 2000.0,
        };

        monitor.set_thresholds(custom_thresholds);
        assert_eq!(monitor.thresholds.max_cpu_usage, 90.0);
    }

    #[test]
    fn test_no_threshold_violations() {
        let monitor = ResourceMonitor::new(Duration::from_secs(1));
        let metrics = OperationalMetrics::default(); // All zeros, should not violate

        let violations = monitor.check_thresholds(&metrics);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_agent_analytics_default() {
        let analytics = AgentAnalytics::default();
        assert!(analytics.agent_metrics.is_empty());
    }

    #[test]
    fn test_agent_metrics_default() {
        let metrics = AgentMetrics::default();
        assert_eq!(metrics.execution_time_ms, 0.0);
        assert_eq!(metrics.memory_usage_mb, 0);
        assert_eq!(metrics.success_rate, 0.0);
        assert_eq!(metrics.efficiency_score, 0.0);
    }

    #[test]
    fn test_gpu_utilization_default() {
        let gpu = GpuUtilization::default();
        assert_eq!(gpu.compute_utilization, 0.0);
        assert_eq!(gpu.memory_utilization, 0.0);
        assert_eq!(gpu.temperature, 0.0);
        assert_eq!(gpu.power_usage_watts, 0.0);
        assert_eq!(gpu.active_kernels, 0);
    }
}
