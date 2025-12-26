//! Prometheus metrics exporter

use super::types::{Metric, MetricType};
use crate::MonitoringError;
use dashmap::DashMap;
use prometheus::{Encoder, Registry, TextEncoder};
use std::collections::HashMap;
use std::sync::Arc;

/// Prometheus metrics exporter
pub struct PrometheusExporter {
    registry: Registry,
    port: u16,
    metrics_cache:
        Arc<DashMap<String, prometheus::core::GenericGauge<prometheus::core::AtomicF64>>>,
    counters_cache:
        Arc<DashMap<String, prometheus::core::GenericCounter<prometheus::core::AtomicF64>>>,
}

impl PrometheusExporter {
    /// Create new Prometheus exporter
    pub fn new(port: u16) -> Self {
        Self {
            registry: Registry::new(),
            port,
            metrics_cache: Arc::new(DashMap::new()),
            counters_cache: Arc::new(DashMap::new()),
        }
    }

    /// Get the configured port
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Update metrics in Prometheus registry
    pub async fn update_metrics(&self, metrics: &[Metric]) -> Result<(), MonitoringError> {
        for metric in metrics {
            match metric.metric_type {
                MetricType::Gauge => self.update_gauge(metric).await?,
                MetricType::Counter => self.update_counter(metric).await?,
                MetricType::Histogram => self.update_histogram(metric).await?,
                MetricType::Summary => self.update_summary(metric).await?,
            }
        }
        Ok(())
    }

    /// Format metrics for Prometheus text exposition format
    pub fn format_metrics(&self, metrics: &[Metric]) -> Result<String, MonitoringError> {
        let mut output = String::new();

        // Group metrics by name
        let mut grouped: HashMap<String, Vec<&Metric>> = HashMap::new();
        for metric in metrics {
            grouped.entry(metric.name.clone()).or_default().push(metric);
        }

        // Format each metric group
        for (name, metric_group) in grouped {
            if let Some(first) = metric_group.first() {
                // Write HELP and TYPE lines
                output.push_str(&format!("# HELP {} {}\n", name, self.get_help_text(&name)));
                output.push_str(&format!(
                    "# TYPE {} {}\n",
                    name,
                    self.metric_type_string(first.metric_type)
                ));

                // Write metric values
                for metric in metric_group {
                    output.push_str(&self.format_metric_line(metric));
                }
            }
        }

        Ok(output)
    }

    /// Get metrics endpoint path
    pub fn metrics_path(&self) -> &str {
        "/metrics"
    }

    /// Export metrics using Prometheus encoder
    pub async fn export(&self) -> Result<Vec<u8>, MonitoringError> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();

        encoder.encode(&metric_families, &mut buffer).map_err(|e| {
            MonitoringError::MetricsFailed {
                reason: format!("Failed to encode metrics: {e}"),
            }
        })?;

        Ok(buffer)
    }

    // Private helper methods

    async fn update_gauge(&self, metric: &Metric) -> Result<(), MonitoringError> {
        let gauge = self
            .metrics_cache
            .entry(metric.name.clone())
            .or_insert_with(|| {
                let gauge =
                    prometheus::Gauge::new(metric.name.clone(), self.get_help_text(&metric.name))
                        .expect("Failed to create gauge");
                self.registry.register(Box::new(gauge.clone())).ok();
                gauge
            });

        gauge.set(metric.value);
        Ok(())
    }

    async fn update_counter(&self, metric: &Metric) -> Result<(), MonitoringError> {
        let counter = self
            .counters_cache
            .entry(metric.name.clone())
            .or_insert_with(|| {
                let counter =
                    prometheus::Counter::new(metric.name.clone(), self.get_help_text(&metric.name))
                        .expect("Failed to create counter");
                self.registry.register(Box::new(counter.clone())).ok();
                counter
            });

        counter.inc_by(metric.value);
        Ok(())
    }

    async fn update_histogram(&self, _metric: &Metric) -> Result<(), MonitoringError> {
        // Histogram implementation would go here
        Ok(())
    }

    async fn update_summary(&self, _metric: &Metric) -> Result<(), MonitoringError> {
        // Summary implementation would go here
        Ok(())
    }

    fn get_help_text(&self, metric_name: &str) -> &str {
        match metric_name {
            "cpu_usage_percent" => "Current CPU usage percentage",
            "memory_usage_bytes" => "Current memory usage in bytes",
            "gpu_usage_percent" => "Current GPU usage percentage",
            "gpu_memory_usage_bytes" => "Current GPU memory usage in bytes",
            "network_io_bytes_total" => "Total network I/O in bytes",
            "disk_io_bytes_total" => "Total disk I/O in bytes",
            _ => "Metric description",
        }
    }

    fn metric_type_string(&self, metric_type: MetricType) -> &str {
        match metric_type {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram => "histogram",
            MetricType::Summary => "summary",
        }
    }

    fn format_metric_line(&self, metric: &Metric) -> String {
        let labels = if metric.labels.is_empty() {
            String::new()
        } else {
            let label_pairs: Vec<String> = metric
                .labels
                .iter()
                .map(|(k, v)| format!("{k}=\"{v}\""))
                .collect();
            format!("{{{}}}", label_pairs.join(","))
        };

        if let Some(timestamp) = metric.timestamp {
            format!("{}{} {} {}\n", metric.name, labels, metric.value, timestamp)
        } else {
            format!("{}{} {}\n", metric.name, labels, metric.value)
        }
    }
}

impl Default for PrometheusExporter {
    fn default() -> Self {
        Self::new(9090)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prometheus_exporter_creation() {
        let exporter = PrometheusExporter::new(8080);
        assert_eq!(exporter.port(), 8080);
        assert_eq!(exporter.metrics_path(), "/metrics");
    }

    #[test]
    fn test_format_single_metric() {
        let exporter = PrometheusExporter::default();
        let metric = Metric::gauge("test_gauge", 42.0);
        let metrics = vec![metric];

        let output = exporter.format_metrics(&metrics).unwrap();
        assert!(output.contains("# HELP test_gauge"));
        assert!(output.contains("# TYPE test_gauge gauge"));
        assert!(output.contains("test_gauge 42"));
    }

    #[test]
    fn test_format_metric_with_labels() {
        let exporter = PrometheusExporter::default();
        let metric = Metric::counter("requests_total", 100.0)
            .with_label("method", "GET")
            .with_label("status", "200");
        let metrics = vec![metric];

        let output = exporter.format_metrics(&metrics).unwrap();
        assert!(output.contains("requests_total{method=\"GET\",status=\"200\"} 100"));
    }

    #[test]
    fn test_format_metric_with_timestamp() {
        let exporter = PrometheusExporter::default();
        let metric = Metric::gauge("temperature", 25.5).with_timestamp(1234567890);
        let metrics = vec![metric];

        let output = exporter.format_metrics(&metrics).unwrap();
        assert!(output.contains("temperature 25.5 1234567890"));
    }

    #[test]
    fn test_format_multiple_metrics() {
        let exporter = PrometheusExporter::default();
        let metrics = vec![
            Metric::gauge("cpu_usage_percent", 45.5),
            Metric::gauge("memory_usage_bytes", 1024.0),
            Metric::counter("requests_total", 1000.0),
        ];

        let output = exporter.format_metrics(&metrics).unwrap();
        assert!(output.contains("# TYPE cpu_usage_percent gauge"));
        assert!(output.contains("# TYPE memory_usage_bytes gauge"));
        assert!(output.contains("# TYPE requests_total counter"));
    }

    #[tokio::test]
    async fn test_update_metrics() {
        let exporter = PrometheusExporter::default();
        let metrics = vec![
            Metric::gauge("test_gauge", 42.0),
            Metric::counter("test_counter", 10.0),
        ];

        let result = exporter.update_metrics(&metrics).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_metric_type_string() {
        let exporter = PrometheusExporter::default();
        assert_eq!(exporter.metric_type_string(MetricType::Counter), "counter");
        assert_eq!(exporter.metric_type_string(MetricType::Gauge), "gauge");
        assert_eq!(
            exporter.metric_type_string(MetricType::Histogram),
            "histogram"
        );
        assert_eq!(exporter.metric_type_string(MetricType::Summary), "summary");
    }

    #[test]
    fn test_empty_metrics() {
        let exporter = PrometheusExporter::default();
        let metrics: Vec<Metric> = vec![];

        let output = exporter.format_metrics(&metrics).unwrap();
        assert_eq!(output, "");
    }
}
