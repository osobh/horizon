//! Metrics collection traits and implementations

use super::types::Metric;
use crate::{MonitoringError, SystemMetrics};
use async_trait::async_trait;
use std::time::{SystemTime, UNIX_EPOCH};

/// Trait for metrics collectors
#[async_trait]
pub trait MetricsCollector: Send + Sync {
    /// Collect current metrics
    async fn collect(&self) -> Result<Vec<Metric>, MonitoringError>;

    /// Get collector name
    fn name(&self) -> &str;

    /// Check if collector is healthy
    async fn is_healthy(&self) -> bool {
        true
    }
}

/// System metrics collector for basic OS/hardware metrics
pub struct SystemMetricsCollector {
    name: String,
}

impl SystemMetricsCollector {
    pub fn new() -> Self {
        Self {
            name: "system".to_string(),
        }
    }

    pub fn with_name(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Collect system metrics
    fn collect_system_metrics(&self) -> SystemMetrics {
        // In a real implementation, this would gather actual system metrics
        // For now, return mock data
        SystemMetrics {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            cpu_usage_percent: 45.5,
            memory_usage_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            gpu_usage_percent: 75.0,
            gpu_memory_usage_bytes: 4 * 1024 * 1024 * 1024, // 4GB
            network_io_bytes: 1024 * 1024,                  // 1MB
            disk_io_bytes: 10 * 1024 * 1024,                // 10MB
        }
    }
}

impl Default for SystemMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MetricsCollector for SystemMetricsCollector {
    async fn collect(&self) -> Result<Vec<Metric>, MonitoringError> {
        let sys_metrics = self.collect_system_metrics();

        let metrics = vec![
            Metric::gauge("cpu_usage_percent", sys_metrics.cpu_usage_percent)
                .with_label("collector", &self.name),
            Metric::gauge("memory_usage_bytes", sys_metrics.memory_usage_bytes as f64)
                .with_label("collector", &self.name),
            Metric::gauge("gpu_usage_percent", sys_metrics.gpu_usage_percent)
                .with_label("collector", &self.name),
            Metric::gauge(
                "gpu_memory_usage_bytes",
                sys_metrics.gpu_memory_usage_bytes as f64,
            )
            .with_label("collector", &self.name),
            Metric::counter(
                "network_io_bytes_total",
                sys_metrics.network_io_bytes as f64,
            )
            .with_label("collector", &self.name),
            Metric::counter("disk_io_bytes_total", sys_metrics.disk_io_bytes as f64)
                .with_label("collector", &self.name),
        ];

        Ok(metrics)
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn is_healthy(&self) -> bool {
        // Check if we can collect metrics
        self.collect().await.is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::MetricType;

    #[tokio::test]
    async fn test_system_metrics_collector() {
        let collector = SystemMetricsCollector::new();

        assert_eq!(collector.name(), "system");
        assert!(collector.is_healthy().await);

        let metrics = collector.collect().await.unwrap();
        assert!(!metrics.is_empty());

        // Check we have all expected metrics
        let metric_names: Vec<_> = metrics.iter().map(|m| &m.name).collect();
        assert!(metric_names.contains(&&"cpu_usage_percent".to_string()));
        assert!(metric_names.contains(&&"memory_usage_bytes".to_string()));
        assert!(metric_names.contains(&&"gpu_usage_percent".to_string()));
    }

    #[tokio::test]
    async fn test_system_metrics_collector_with_name() {
        let collector = SystemMetricsCollector::with_name("custom_system");
        assert_eq!(collector.name(), "custom_system");

        let metrics = collector.collect().await.unwrap();
        // All metrics should have the custom collector label
        for metric in metrics {
            let has_label = metric
                .labels
                .iter()
                .any(|(k, v)| k == "collector" && v == "custom_system");
            assert!(has_label);
        }
    }

    #[tokio::test]
    async fn test_metric_types() {
        let collector = SystemMetricsCollector::new();
        let metrics = collector.collect().await.unwrap();

        // CPU, memory, GPU metrics should be gauges
        let cpu_metric = metrics
            .iter()
            .find(|m| m.name == "cpu_usage_percent")
            .unwrap();
        assert_eq!(cpu_metric.metric_type, MetricType::Gauge);

        // IO metrics should be counters
        let io_metric = metrics
            .iter()
            .find(|m| m.name == "network_io_bytes_total")
            .unwrap();
        assert_eq!(io_metric.metric_type, MetricType::Counter);
    }

    // Mock collector for testing
    struct MockCollector {
        name: String,
        healthy: bool,
    }

    #[async_trait]
    impl MetricsCollector for MockCollector {
        async fn collect(&self) -> Result<Vec<Metric>, MonitoringError> {
            if self.healthy {
                Ok(vec![Metric::gauge("mock_metric", 42.0)])
            } else {
                Err(MonitoringError::MetricsFailed {
                    reason: "Mock failure".to_string(),
                })
            }
        }

        fn name(&self) -> &str {
            &self.name
        }

        async fn is_healthy(&self) -> bool {
            self.healthy
        }
    }

    #[tokio::test]
    async fn test_collector_trait() {
        let collector = MockCollector {
            name: "mock".to_string(),
            healthy: true,
        };

        assert_eq!(collector.name(), "mock");
        assert!(collector.is_healthy().await);

        let metrics = collector.collect().await.unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].value, 42.0);
    }

    #[tokio::test]
    async fn test_unhealthy_collector() {
        let collector = MockCollector {
            name: "unhealthy".to_string(),
            healthy: false,
        };

        assert!(!collector.is_healthy().await);
        assert!(collector.collect().await.is_err());
    }

    // TDD: Test default implementation and trait default method
    #[tokio::test]
    async fn test_system_collector_default() {
        let collector = SystemMetricsCollector::default();
        assert_eq!(collector.name(), "system");
        assert!(collector.is_healthy().await);
    }
}
