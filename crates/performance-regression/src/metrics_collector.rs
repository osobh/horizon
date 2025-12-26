//! Real-time performance metrics collection and aggregation

use crate::error::{PerformanceRegressionError, PerformanceRegressionResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::info;

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Collection interval in seconds
    pub collection_interval: u64,
    /// Maximum samples to retain per metric
    pub max_samples: usize,
    /// Aggregation window size in seconds
    pub aggregation_window: u64,
    /// Enable high-resolution metrics
    pub high_resolution: bool,
    /// Metric retention period in hours
    pub retention_hours: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            collection_interval: 10,
            max_samples: 10000,
            aggregation_window: 60,
            high_resolution: false,
            retention_hours: 24,
        }
    }
}

/// Performance metric types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// Response time in milliseconds
    ResponseTime,
    /// Throughput in requests per second
    Throughput,
    /// CPU utilization percentage
    CpuUsage,
    /// Memory usage in bytes
    MemoryUsage,
    /// Disk I/O operations per second
    DiskIops,
    /// Network bandwidth in bytes per second
    NetworkBandwidth,
    /// Error rate percentage
    ErrorRate,
    /// Custom application metric
    Custom(String),
}

/// Performance metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    /// Metric type
    pub metric_type: MetricType,
    /// Metric value
    pub value: OrderedFloat<f64>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Additional tags
    pub tags: HashMap<String, String>,
    /// Source identifier
    pub source: String,
}

/// Aggregated metric statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStatistics {
    /// Metric type
    pub metric_type: MetricType,
    /// Average value
    pub average: OrderedFloat<f64>,
    /// Minimum value
    pub minimum: OrderedFloat<f64>,
    /// Maximum value
    pub maximum: OrderedFloat<f64>,
    /// Standard deviation
    pub std_deviation: OrderedFloat<f64>,
    /// 95th percentile
    pub p95: OrderedFloat<f64>,
    /// 99th percentile
    pub p99: OrderedFloat<f64>,
    /// Sample count
    pub sample_count: usize,
    /// Time window
    pub window_start: DateTime<Utc>,
    /// Window end
    pub window_end: DateTime<Utc>,
}

/// Metrics collector
pub struct MetricsCollector {
    config: MetricsConfig,
    metrics: Arc<DashMap<MetricType, VecDeque<MetricDataPoint>>>,
    aggregated_stats: Arc<RwLock<HashMap<MetricType, MetricStatistics>>>,
    collection_channel: mpsc::UnboundedSender<MetricDataPoint>,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new(config: MetricsConfig) -> PerformanceRegressionResult<Self> {
        let metrics = Arc::new(DashMap::new());
        let aggregated_stats = Arc::new(RwLock::new(HashMap::new()));
        let (tx, mut rx) = mpsc::unbounded_channel();

        let collector_metrics = metrics.clone();
        let collector_stats = aggregated_stats.clone();
        let collector_config = config.clone();

        // Background task for metric processing
        tokio::spawn(async move {
            while let Some(metric) = rx.recv().await {
                Self::process_metric(
                    metric,
                    &collector_metrics,
                    &collector_stats,
                    &collector_config,
                )
                .await;
            }
        });

        Ok(Self {
            config,
            metrics,
            aggregated_stats,
            collection_channel: tx,
        })
    }

    /// Record a metric data point
    pub async fn record_metric(&self, metric: MetricDataPoint) -> PerformanceRegressionResult<()> {
        self.collection_channel.send(metric).map_err(|e| {
            PerformanceRegressionError::MetricsCollectionFailed {
                source_name: "channel".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(())
    }

    /// Get current statistics for a metric type
    pub fn get_statistics(&self, metric_type: MetricType) -> Option<MetricStatistics> {
        self.aggregated_stats.read().get(&metric_type).cloned()
    }

    /// Get raw metric data points
    pub fn get_raw_metrics(
        &self,
        metric_type: MetricType,
        limit: Option<usize>,
    ) -> Vec<MetricDataPoint> {
        if let Some(data) = self.metrics.get(&metric_type) {
            let limit = limit.unwrap_or(self.config.max_samples);
            data.iter().rev().take(limit).cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Process incoming metric
    async fn process_metric(
        metric: MetricDataPoint,
        metrics: &DashMap<MetricType, VecDeque<MetricDataPoint>>,
        stats: &RwLock<HashMap<MetricType, MetricStatistics>>,
        config: &MetricsConfig,
    ) {
        // Store raw metric
        let metric_type = metric.metric_type.clone();
        let mut data = metrics.entry(metric_type).or_insert_with(VecDeque::new);
        data.push_back(metric.clone());

        // Maintain size limit
        while data.len() > config.max_samples {
            data.pop_front();
        }

        // Update aggregated statistics
        Self::update_statistics(&metric, stats, config);
    }

    /// Update aggregated statistics
    fn update_statistics(
        metric: &MetricDataPoint,
        stats: &RwLock<HashMap<MetricType, MetricStatistics>>,
        _config: &MetricsConfig,
    ) {
        // This is a simplified version - real implementation would include
        // proper windowed aggregation and percentile calculation
        let mut stats_guard = stats.write();
        let current_stats = stats_guard
            .entry(metric.metric_type.clone())
            .or_insert_with(|| MetricStatistics {
                metric_type: metric.metric_type.clone(),
                average: OrderedFloat(0.0),
                minimum: metric.value,
                maximum: metric.value,
                std_deviation: OrderedFloat(0.0),
                p95: metric.value,
                p99: metric.value,
                sample_count: 0,
                window_start: metric.timestamp,
                window_end: metric.timestamp,
            });

        // Update statistics (simplified)
        current_stats.sample_count += 1;
        current_stats.window_end = metric.timestamp;
        current_stats.minimum = current_stats.minimum.min(metric.value);
        current_stats.maximum = current_stats.maximum.max(metric.value);

        // Update running average
        let count = current_stats.sample_count as f64;
        let old_avg = current_stats.average.0;
        current_stats.average = OrderedFloat((old_avg * (count - 1.0) + metric.value.0) / count);
    }

    /// Start automatic metric collection
    pub async fn start_collection(&self) -> PerformanceRegressionResult<()> {
        info!(
            "Starting metrics collection with interval: {}s",
            self.config.collection_interval
        );

        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(
            self.config.collection_interval,
        ));

        loop {
            interval.tick().await;
            self.collect_system_metrics().await?;
        }
    }

    /// Collect system metrics
    async fn collect_system_metrics(&self) -> PerformanceRegressionResult<()> {
        let timestamp = Utc::now();

        // Simulate system metric collection
        let metrics = vec![
            MetricDataPoint {
                metric_type: MetricType::CpuUsage,
                value: OrderedFloat(self.get_cpu_usage()),
                timestamp,
                tags: HashMap::new(),
                source: "system".to_string(),
            },
            MetricDataPoint {
                metric_type: MetricType::MemoryUsage,
                value: OrderedFloat(self.get_memory_usage()),
                timestamp,
                tags: HashMap::new(),
                source: "system".to_string(),
            },
        ];

        for metric in metrics {
            self.record_metric(metric).await?;
        }

        Ok(())
    }

    /// Get CPU usage (simulated)
    fn get_cpu_usage(&self) -> f64 {
        // In real implementation, this would query the system
        50.0 + (rand::random::<f64>() - 0.5) * 20.0
    }

    /// Get memory usage (simulated)
    fn get_memory_usage(&self) -> f64 {
        // In real implementation, this would query the system
        1024.0 * 1024.0 * 1024.0 * (0.6 + rand::random::<f64>() * 0.3)
    }
}

use rand;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chrono::Duration;

    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config);
        assert!(collector.is_ok());
    }

    #[tokio::test]
    async fn test_metric_recording() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();

        let metric = MetricDataPoint {
            metric_type: MetricType::ResponseTime,
            value: OrderedFloat(150.0),
            timestamp: Utc::now(),
            tags: HashMap::new(),
            source: "test".to_string(),
        };

        let result = collector.record_metric(metric).await;
        assert!(result.is_ok());

        // Give some time for background processing
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let stats = collector.get_statistics(MetricType::ResponseTime);
        assert!(stats.is_some());
    }

    #[tokio::test]
    async fn test_metric_statistics_calculation() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();

        let values = vec![100.0, 150.0, 200.0, 175.0, 125.0];
        for (i, value) in values.iter().enumerate() {
            let metric = MetricDataPoint {
                metric_type: MetricType::Throughput,
                value: OrderedFloat(*value),
                timestamp: Utc::now() + Duration::seconds(i as i64),
                tags: HashMap::new(),
                source: "test".to_string(),
            };
            collector.record_metric(metric).await.unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let stats = collector.get_statistics(MetricType::Throughput).unwrap();
        assert_eq!(stats.sample_count, 5);
        assert_relative_eq!(stats.average.0, 150.0, epsilon = 1e-10);
        assert_eq!(stats.minimum.0, 100.0);
        assert_eq!(stats.maximum.0, 200.0);
    }

    #[test]
    fn test_metric_type_serialization() {
        let metric_type = MetricType::Custom("test_metric".to_string());
        let json = serde_json::to_string(&metric_type).unwrap();
        let deserialized: MetricType = serde_json::from_str(&json).unwrap();
        assert_eq!(metric_type, deserialized);
    }

    #[tokio::test]
    async fn test_raw_metrics_retrieval() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();

        // Record multiple metrics
        for i in 0..5 {
            let metric = MetricDataPoint {
                metric_type: MetricType::ErrorRate,
                value: OrderedFloat(i as f64),
                timestamp: Utc::now() + Duration::seconds(i),
                tags: HashMap::new(),
                source: "test".to_string(),
            };
            collector.record_metric(metric).await.unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let raw_metrics = collector.get_raw_metrics(MetricType::ErrorRate, Some(3));
        assert_eq!(raw_metrics.len(), 3);

        // Should be in reverse order (most recent first)
        assert_eq!(raw_metrics[0].value.0, 4.0);
        assert_eq!(raw_metrics[1].value.0, 3.0);
        assert_eq!(raw_metrics[2].value.0, 2.0);
    }

    #[tokio::test]
    async fn test_metric_data_point_creation() {
        let metric = MetricDataPoint {
            metric_type: MetricType::NetworkBandwidth,
            value: OrderedFloat(1024.0 * 1024.0),
            timestamp: Utc::now(),
            tags: HashMap::from([
                ("interface".to_string(), "eth0".to_string()),
                ("direction".to_string(), "inbound".to_string()),
            ]),
            source: "network_monitor".to_string(),
        };

        assert_eq!(metric.metric_type, MetricType::NetworkBandwidth);
        assert_eq!(metric.value.0, 1024.0 * 1024.0);
        assert_eq!(metric.tags.len(), 2);
        assert_eq!(metric.source, "network_monitor");
    }

    #[tokio::test]
    async fn test_multiple_metric_types() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();

        let metric_types = vec![
            MetricType::ResponseTime,
            MetricType::Throughput,
            MetricType::CpuUsage,
            MetricType::MemoryUsage,
            MetricType::DiskIops,
        ];

        for metric_type in metric_types {
            let metric = MetricDataPoint {
                metric_type,
                value: OrderedFloat(100.0),
                timestamp: Utc::now(),
                tags: HashMap::new(),
                source: "test".to_string(),
            };
            collector.record_metric(metric).await.unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Verify all metric types have statistics
        assert!(collector.get_statistics(MetricType::ResponseTime).is_some());
        assert!(collector.get_statistics(MetricType::Throughput).is_some());
        assert!(collector.get_statistics(MetricType::CpuUsage).is_some());
        assert!(collector.get_statistics(MetricType::MemoryUsage).is_some());
        assert!(collector.get_statistics(MetricType::DiskIops).is_some());
    }

    #[tokio::test]
    async fn test_max_samples_limit() {
        let config = MetricsConfig {
            max_samples: 3,
            ..Default::default()
        };
        let collector = MetricsCollector::new(config).unwrap();

        // Record more metrics than max_samples
        for i in 0..5 {
            let metric = MetricDataPoint {
                metric_type: MetricType::DiskIops,
                value: OrderedFloat(i as f64),
                timestamp: Utc::now() + Duration::seconds(i),
                tags: HashMap::new(),
                source: "test".to_string(),
            };
            collector.record_metric(metric).await.unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let raw_metrics = collector.get_raw_metrics(MetricType::DiskIops, None);
        assert_eq!(raw_metrics.len(), 3); // Should be limited to max_samples

        // Should contain the most recent metrics
        assert_eq!(raw_metrics[0].value.0, 4.0);
        assert_eq!(raw_metrics[1].value.0, 3.0);
        assert_eq!(raw_metrics[2].value.0, 2.0);
    }

    #[tokio::test]
    async fn test_custom_metric_type() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();

        let custom_metric = MetricDataPoint {
            metric_type: MetricType::Custom("database_connections".to_string()),
            value: OrderedFloat(42.0),
            timestamp: Utc::now(),
            tags: HashMap::from([("pool".to_string(), "primary".to_string())]),
            source: "db_monitor".to_string(),
        };

        collector.record_metric(custom_metric).await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let stats =
            collector.get_statistics(MetricType::Custom("database_connections".to_string()));
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().average.0, 42.0);
    }

    #[tokio::test]
    async fn test_concurrent_metric_recording() {
        let config = MetricsConfig::default();
        let collector = Arc::new(MetricsCollector::new(config).unwrap());

        let mut handles = vec![];

        for i in 0..10 {
            let collector_clone = collector.clone();
            let handle = tokio::spawn(async move {
                let metric = MetricDataPoint {
                    metric_type: MetricType::ResponseTime,
                    value: OrderedFloat(i as f64 * 10.0),
                    timestamp: Utc::now(),
                    tags: HashMap::new(),
                    source: format!("source_{}", i),
                };
                collector_clone.record_metric(metric).await.unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let stats = collector.get_statistics(MetricType::ResponseTime).unwrap();
        assert_eq!(stats.sample_count, 10);
    }

    #[tokio::test]
    async fn test_metric_statistics_fields() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();

        let metric = MetricDataPoint {
            metric_type: MetricType::CpuUsage,
            value: OrderedFloat(75.5),
            timestamp: Utc::now(),
            tags: HashMap::new(),
            source: "test".to_string(),
        };

        collector.record_metric(metric.clone()).await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let stats = collector.get_statistics(MetricType::CpuUsage).unwrap();
        assert_eq!(stats.metric_type, MetricType::CpuUsage);
        assert_eq!(stats.average.0, 75.5);
        assert_eq!(stats.minimum.0, 75.5);
        assert_eq!(stats.maximum.0, 75.5);
        assert_eq!(stats.sample_count, 1);
        assert!(stats.window_start <= stats.window_end);
    }
}
