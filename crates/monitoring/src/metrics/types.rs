//! Metric types and data structures

use serde::{Deserialize, Serialize};

/// Core metric structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    pub metric_type: MetricType,
    pub labels: Vec<(String, String)>,
    pub timestamp: Option<u64>,
}

/// Metric types following Prometheus conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// GPU kernel execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMetrics {
    pub kernel_id: String,
    pub execution_time_ms: f64,
    pub memory_usage_bytes: u64,
    pub gpu_utilization_percent: f64,
    pub power_usage_watts: f64,
    pub temperature_celsius: f64,
}

/// Evolution process metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetrics {
    pub generation: u64,
    pub population_size: usize,
    pub best_fitness: f64,
    pub average_fitness: f64,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub diversity_score: f64,
}

impl Metric {
    /// Create a new counter metric
    pub fn counter(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            metric_type: MetricType::Counter,
            labels: Vec::new(),
            timestamp: None,
        }
    }

    /// Create a new gauge metric
    pub fn gauge(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            metric_type: MetricType::Gauge,
            labels: Vec::new(),
            timestamp: None,
        }
    }

    /// Create a new histogram metric
    pub fn histogram(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            metric_type: MetricType::Histogram,
            labels: Vec::new(),
            timestamp: None,
        }
    }

    /// Add a label to the metric
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.push((key.into(), value.into()));
        self
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = Some(timestamp);
        self
    }
}

impl Default for KernelMetrics {
    fn default() -> Self {
        Self {
            kernel_id: String::new(),
            execution_time_ms: 0.0,
            memory_usage_bytes: 0,
            gpu_utilization_percent: 0.0,
            power_usage_watts: 0.0,
            temperature_celsius: 0.0,
        }
    }
}

impl Default for EvolutionMetrics {
    fn default() -> Self {
        Self {
            generation: 0,
            population_size: 0,
            best_fitness: 0.0,
            average_fitness: 0.0,
            mutation_rate: 0.0,
            crossover_rate: 0.0,
            diversity_score: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_creation() {
        let counter = Metric::counter("requests_total", 100.0);
        assert_eq!(counter.name, "requests_total");
        assert_eq!(counter.value, 100.0);
        assert_eq!(counter.metric_type, MetricType::Counter);
    }

    #[test]
    fn test_metric_with_labels() {
        let metric = Metric::gauge("temperature", 25.5)
            .with_label("location", "gpu0")
            .with_label("unit", "celsius");

        assert_eq!(metric.labels.len(), 2);
        assert_eq!(
            metric.labels[0],
            ("location".to_string(), "gpu0".to_string())
        );
        assert_eq!(
            metric.labels[1],
            ("unit".to_string(), "celsius".to_string())
        );
    }

    #[test]
    fn test_metric_with_timestamp() {
        let timestamp = 1234567890;
        let metric = Metric::histogram("latency", 150.0).with_timestamp(timestamp);

        assert_eq!(metric.timestamp, Some(timestamp));
    }

    #[test]
    fn test_kernel_metrics_default() {
        let metrics = KernelMetrics::default();
        assert_eq!(metrics.execution_time_ms, 0.0);
        assert_eq!(metrics.memory_usage_bytes, 0);
        assert_eq!(metrics.gpu_utilization_percent, 0.0);
    }

    #[test]
    fn test_evolution_metrics_default() {
        let metrics = EvolutionMetrics::default();
        assert_eq!(metrics.generation, 0);
        assert_eq!(metrics.population_size, 0);
        assert_eq!(metrics.best_fitness, 0.0);
    }

    #[test]
    fn test_metric_type_equality() {
        assert_eq!(MetricType::Counter, MetricType::Counter);
        assert_ne!(MetricType::Counter, MetricType::Gauge);
        assert_ne!(MetricType::Histogram, MetricType::Summary);
    }
}
