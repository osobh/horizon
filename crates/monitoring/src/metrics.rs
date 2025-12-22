//! Metrics collection and aggregation for GPU container systems

pub mod aggregator;
pub mod collector;
pub mod exporter;
pub mod types;

pub use aggregator::MetricsAggregator;
pub use collector::{MetricsCollector, SystemMetricsCollector};
pub use exporter::PrometheusExporter;
pub use types::{EvolutionMetrics, KernelMetrics, Metric, MetricType};

// Re-export common types
pub use crate::MonitoringError;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn test_metrics_integration() {
        // Create components
        let collector = Arc::new(SystemMetricsCollector::new());
        let exporter = PrometheusExporter::new(9090);
        let aggregator = MetricsAggregator::new();

        // Add collector
        aggregator.add_collector(collector);

        // Wait for async add to complete
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Collect metrics
        let metrics = aggregator.collect_all().await.unwrap();
        assert!(!metrics.is_empty());

        // Export metrics
        let output = exporter.format_metrics(&metrics).unwrap();
        assert!(output.contains("TYPE"));
        assert!(output.contains("HELP"));
    }

    #[tokio::test]
    async fn test_metric_types() {
        let gauge = Metric {
            name: "test_gauge".to_string(),
            value: 42.0,
            metric_type: MetricType::Gauge,
            labels: vec![("env".to_string(), "test".to_string())],
            timestamp: None,
        };

        let counter = Metric {
            name: "test_counter".to_string(),
            value: 100.0,
            metric_type: MetricType::Counter,
            labels: vec![],
            timestamp: None,
        };

        let histogram = Metric {
            name: "test_histogram".to_string(),
            value: 250.0,
            metric_type: MetricType::Histogram,
            labels: vec![("bucket".to_string(), "le_500".to_string())],
            timestamp: None,
        };

        assert_eq!(gauge.name, "test_gauge");
        assert_eq!(counter.value, 100.0);
        assert_eq!(histogram.labels.len(), 1);
    }
}
