//! Metrics aggregation and management

use super::collector::MetricsCollector;
use super::types::Metric;
use crate::MonitoringError;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Aggregates metrics from multiple collectors
pub struct MetricsAggregator {
    collectors: Arc<RwLock<Vec<Arc<dyn MetricsCollector + Send + Sync>>>>,
    cache: Arc<RwLock<MetricsCache>>,
    cache_duration: Duration,
}

struct MetricsCache {
    metrics: Vec<Metric>,
    last_update: Instant,
}

impl MetricsAggregator {
    /// Create new metrics aggregator
    pub fn new() -> Self {
        Self {
            collectors: Arc::new(RwLock::new(Vec::new())),
            cache: Arc::new(RwLock::new(MetricsCache {
                metrics: Vec::new(),
                last_update: Instant::now() - Duration::from_secs(3600), // Force initial update
            })),
            cache_duration: Duration::from_secs(10),
        }
    }

    /// Create with custom cache duration
    pub fn with_cache_duration(cache_duration: Duration) -> Self {
        Self {
            cache_duration,
            ..Self::new()
        }
    }

    /// Add a metrics collector
    pub fn add_collector(&self, collector: Arc<dyn MetricsCollector + Send + Sync>) {
        tokio::spawn({
            let collectors = self.collectors.clone();
            async move {
                collectors.write().await.push(collector);
            }
        });
    }

    /// Remove all collectors
    pub async fn clear_collectors(&self) {
        self.collectors.write().await.clear();
    }

    /// Get collector count
    pub async fn collector_count(&self) -> usize {
        self.collectors.read().await.len()
    }

    /// Collect metrics from all collectors
    pub async fn collect_all(&self) -> Result<Vec<Metric>, MonitoringError> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if cache.last_update.elapsed() < self.cache_duration {
                return Ok(cache.metrics.clone());
            }
        }

        // Collect from all collectors
        let collectors = self.collectors.read().await;
        let mut all_metrics = Vec::new();
        let mut errors = Vec::new();

        for collector in collectors.iter() {
            match collector.collect().await {
                Ok(metrics) => all_metrics.extend(metrics),
                Err(e) => errors.push(format!("{}: {}", collector.name(), e)),
            }
        }

        if all_metrics.is_empty() && !errors.is_empty() {
            return Err(MonitoringError::MetricsFailed {
                reason: format!("All collectors failed: {}", errors.join(", ")),
            });
        }

        // Update cache
        {
            let mut cache = self.cache.write().await;
            cache.metrics = all_metrics.clone();
            cache.last_update = Instant::now();
        }

        Ok(all_metrics)
    }

    /// Get latest metrics without re-collecting
    pub async fn get_cached_metrics(&self) -> Vec<Metric> {
        self.cache.read().await.metrics.clone()
    }

    /// Force refresh metrics cache
    pub async fn refresh_cache(&self) -> Result<(), MonitoringError> {
        self.collect_all().await.map(|_| ())
    }

    /// Get metrics by name
    pub async fn get_metrics_by_name(&self, name: &str) -> Result<Vec<Metric>, MonitoringError> {
        let metrics = self.collect_all().await?;
        Ok(metrics.into_iter().filter(|m| m.name == name).collect())
    }

    /// Get metrics with specific labels
    pub async fn get_metrics_with_label(
        &self,
        key: &str,
        value: &str,
    ) -> Result<Vec<Metric>, MonitoringError> {
        let metrics = self.collect_all().await?;
        Ok(metrics
            .into_iter()
            .filter(|m| m.labels.iter().any(|(k, v)| k == key && v == value))
            .collect())
    }

    /// Aggregate metrics by computing statistics
    pub async fn compute_statistics(
        &self,
        metric_name: &str,
    ) -> Result<MetricStats, MonitoringError> {
        let metrics = self.get_metrics_by_name(metric_name).await?;

        if metrics.is_empty() {
            return Err(MonitoringError::MetricsFailed {
                reason: format!("No metrics found with name: {metric_name}"),
            });
        }

        let values: Vec<f64> = metrics.iter().map(|m| m.value).collect();
        let sum: f64 = values.iter().sum();
        let count = values.len() as f64;
        let mean = sum / count;

        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count;
        let std_dev = variance.sqrt();

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Ok(MetricStats {
            count: metrics.len(),
            sum,
            mean,
            std_dev,
            min,
            max,
        })
    }
}

impl Default for MetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a metric
#[derive(Debug, Clone)]
pub struct MetricStats {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::collector::SystemMetricsCollector;

    #[tokio::test]
    async fn test_metrics_aggregator_basic() {
        let aggregator = MetricsAggregator::new();

        // Add collector
        let collector = Arc::new(SystemMetricsCollector::new());
        aggregator.add_collector(collector);

        // Wait for async add
        tokio::time::sleep(Duration::from_millis(10)).await;

        assert_eq!(aggregator.collector_count().await, 1);

        // Collect metrics
        let metrics = aggregator.collect_all().await.unwrap();
        assert!(!metrics.is_empty());
    }

    #[tokio::test]
    async fn test_metrics_cache() {
        let aggregator = MetricsAggregator::with_cache_duration(Duration::from_millis(100));

        let collector = Arc::new(SystemMetricsCollector::new());
        aggregator.add_collector(collector);

        tokio::time::sleep(Duration::from_millis(10)).await;

        // First collection
        let metrics1 = aggregator.collect_all().await.unwrap();

        // Second collection should return cached results
        let metrics2 = aggregator.collect_all().await.unwrap();
        assert_eq!(metrics1.len(), metrics2.len());

        // Wait for cache to expire
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should collect fresh metrics
        let metrics3 = aggregator.collect_all().await.unwrap();
        assert_eq!(metrics1.len(), metrics3.len());
    }

    #[tokio::test]
    async fn test_get_metrics_by_name() {
        let aggregator = MetricsAggregator::new();
        let collector = Arc::new(SystemMetricsCollector::new());
        aggregator.add_collector(collector);

        tokio::time::sleep(Duration::from_millis(10)).await;

        let cpu_metrics = aggregator
            .get_metrics_by_name("cpu_usage_percent")
            .await
            .unwrap();
        assert!(!cpu_metrics.is_empty());
        assert!(cpu_metrics.iter().all(|m| m.name == "cpu_usage_percent"));
    }

    #[tokio::test]
    async fn test_get_metrics_with_label() {
        let aggregator = MetricsAggregator::new();
        let collector = Arc::new(SystemMetricsCollector::new());
        aggregator.add_collector(collector);

        tokio::time::sleep(Duration::from_millis(10)).await;

        let system_metrics = aggregator
            .get_metrics_with_label("collector", "system")
            .await
            .unwrap();
        assert!(!system_metrics.is_empty());
    }

    #[tokio::test]
    async fn test_compute_statistics() {
        let aggregator = MetricsAggregator::new();

        // Add multiple collectors to get multiple values
        for i in 0..3 {
            let collector = Arc::new(SystemMetricsCollector::with_name(format!("system_{i}")));
            aggregator.add_collector(collector);
        }

        tokio::time::sleep(Duration::from_millis(20)).await;

        let stats = aggregator
            .compute_statistics("cpu_usage_percent")
            .await
            .unwrap();
        assert_eq!(stats.count, 3);
        assert!(stats.mean > 0.0);
        assert!(stats.min <= stats.max);
    }

    #[tokio::test]
    async fn test_clear_collectors() {
        let aggregator = MetricsAggregator::new();

        let collector1 = Arc::new(SystemMetricsCollector::new());
        let collector2 = Arc::new(SystemMetricsCollector::new());

        aggregator.add_collector(collector1);
        aggregator.add_collector(collector2);

        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(aggregator.collector_count().await, 2);

        aggregator.clear_collectors().await;
        assert_eq!(aggregator.collector_count().await, 0);
    }

    #[tokio::test]
    async fn test_refresh_cache() {
        let aggregator = MetricsAggregator::new();
        let collector = Arc::new(SystemMetricsCollector::new());
        aggregator.add_collector(collector);

        tokio::time::sleep(Duration::from_millis(10)).await;

        // Force refresh
        let result = aggregator.refresh_cache().await;
        assert!(result.is_ok());

        // Should have fresh metrics
        let cached = aggregator.get_cached_metrics().await;
        assert!(!cached.is_empty());
    }

    #[tokio::test]
    async fn test_no_collectors() {
        let aggregator = MetricsAggregator::new();

        let metrics = aggregator.collect_all().await.unwrap();
        assert!(metrics.is_empty());
    }

    #[tokio::test]
    async fn test_statistics_error() {
        let aggregator = MetricsAggregator::new();

        let result = aggregator.compute_statistics("nonexistent_metric").await;
        assert!(result.is_err());
    }
}
