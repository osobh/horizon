//! Bottleneck detection for performance regression analysis
//!
//! This module provides comprehensive resource bottleneck identification capabilities
//! to help identify performance degradation causes and remediation strategies.

use crate::error::PerformanceRegressionResult;
use crate::metrics_collector::{MetricDataPoint, MetricType};
use chrono::{DateTime, Duration, Utc};
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Bottleneck detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckConfig {
    /// CPU utilization threshold for bottleneck detection (percentage)
    pub cpu_threshold: f64,
    /// Memory utilization threshold for bottleneck detection (percentage)
    pub memory_threshold: f64,
    /// Disk I/O threshold for bottleneck detection (IOPS)
    pub disk_io_threshold: f64,
    /// Network bandwidth threshold for bottleneck detection (percentage)
    pub network_threshold: f64,
    /// Response time threshold for bottleneck detection (milliseconds)
    pub response_time_threshold: f64,
    /// Minimum duration for bottleneck confirmation (seconds)
    pub min_bottleneck_duration: u64,
    /// Correlation threshold for related bottlenecks
    pub correlation_threshold: f64,
    /// Enable dependency analysis
    pub enable_dependency_analysis: bool,
    /// Enable transaction flow analysis
    pub enable_transaction_flow_analysis: bool,
    /// Historical window for analysis (hours)
    pub analysis_window_hours: u64,
}

impl Default for BottleneckConfig {
    fn default() -> Self {
        Self {
            cpu_threshold: 80.0,
            memory_threshold: 85.0,
            disk_io_threshold: 1000.0,
            network_threshold: 90.0,
            response_time_threshold: 1000.0,
            min_bottleneck_duration: 60,
            correlation_threshold: 0.7,
            enable_dependency_analysis: true,
            enable_transaction_flow_analysis: true,
            analysis_window_hours: 24,
        }
    }
}

/// Bottleneck type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BottleneckType {
    /// CPU resource bottleneck
    CpuBottleneck,
    /// Memory resource bottleneck
    MemoryBottleneck,
    /// Disk I/O bottleneck
    DiskIoBottleneck,
    /// Network bandwidth bottleneck
    NetworkBottleneck,
    /// Application response time bottleneck
    ResponseTimeBottleneck,
    /// Database connection pool bottleneck
    DatabaseConnectionBottleneck,
    /// Thread pool exhaustion
    ThreadPoolBottleneck,
    /// Dependency service bottleneck
    DependencyBottleneck(String),
    /// Custom bottleneck type
    Custom(String),
}

/// Bottleneck severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    /// Low severity - minor impact
    Low,
    /// Medium severity - noticeable impact
    Medium,
    /// High severity - significant impact
    High,
    /// Critical severity - severe impact
    Critical,
}

/// Detected bottleneck information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedBottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity level
    pub severity: BottleneckSeverity,
    /// Bottleneck score (0.0 to 1.0)
    pub score: OrderedFloat<f64>,
    /// Resource utilization percentage
    pub utilization: OrderedFloat<f64>,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Duration of bottleneck
    pub duration: Duration,
    /// Affected components
    pub affected_components: Vec<String>,
    /// Related metrics
    pub related_metrics: HashMap<MetricType, OrderedFloat<f64>>,
    /// Correlated bottlenecks
    pub correlated_bottlenecks: Vec<BottleneckType>,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Bottleneck insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckInsights {
    /// Primary bottleneck if any
    pub primary_bottleneck: Option<DetectedBottleneck>,
    /// All detected bottlenecks
    pub all_bottlenecks: Vec<DetectedBottleneck>,
    /// Resource correlation matrix
    pub correlation_matrix: HashMap<(BottleneckType, BottleneckType), OrderedFloat<f64>>,
    /// Transaction flow impact analysis
    pub transaction_flow_impact: HashMap<String, OrderedFloat<f64>>,
    /// Overall system health score (0.0 to 1.0)
    pub system_health_score: OrderedFloat<f64>,
    /// Trend analysis
    pub bottleneck_trends: HashMap<BottleneckType, TrendIndicator>,
}

/// Trend indicator for bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendIndicator {
    /// Improving trend
    Improving,
    /// Stable trend
    Stable,
    /// Degrading trend
    Degrading,
    /// Rapidly degrading
    RapidlyDegrading,
}

/// Bottleneck detector for performance analysis
pub struct BottleneckDetector {
    config: BottleneckConfig,
    metric_history: Arc<RwLock<HashMap<MetricType, VecDeque<MetricDataPoint>>>>,
    detected_bottlenecks: Arc<RwLock<Vec<DetectedBottleneck>>>,
    resource_correlations: Arc<RwLock<HashMap<(MetricType, MetricType), OrderedFloat<f64>>>>,
    #[allow(dead_code)]
    dependency_graph: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl BottleneckDetector {
    /// Create a new bottleneck detector
    pub fn new(config: BottleneckConfig) -> Self {
        Self {
            config,
            metric_history: Arc::new(RwLock::new(HashMap::new())),
            detected_bottlenecks: Arc::new(RwLock::new(Vec::new())),
            resource_correlations: Arc::new(RwLock::new(HashMap::new())),
            dependency_graph: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add metric to history for analysis
    pub fn add_metric(&self, metric: MetricDataPoint) {
        let mut history = self.metric_history.write();
        let deque = history
            .entry(metric.metric_type.clone())
            .or_insert_with(VecDeque::new);
        deque.push_back(metric);

        // Maintain window size
        let window_duration = Duration::hours(self.config.analysis_window_hours as i64);
        let cutoff_time = Utc::now() - window_duration;
        while let Some(front) = deque.front() {
            if front.timestamp < cutoff_time {
                deque.pop_front();
            } else {
                break;
            }
        }
    }

    /// Detect bottlenecks from current metrics
    pub async fn detect_bottlenecks(&self) -> PerformanceRegressionResult<Vec<DetectedBottleneck>> {
        let mut bottlenecks = Vec::new();

        // Analyze resource utilization
        let resource_bottlenecks = self.analyze_resource_utilization()?;
        bottlenecks.extend(resource_bottlenecks);

        // Analyze dependencies if enabled
        if self.config.enable_dependency_analysis {
            let dependency_bottlenecks = self.analyze_dependencies().await?;
            bottlenecks.extend(dependency_bottlenecks);
        }

        // Analyze transaction flow if enabled
        if self.config.enable_transaction_flow_analysis {
            let flow_bottlenecks = self.analyze_transaction_flow().await?;
            bottlenecks.extend(flow_bottlenecks);
        }

        // Update detected bottlenecks
        {
            let mut detected = self.detected_bottlenecks.write();
            *detected = bottlenecks.clone();
        }

        // Calculate correlations
        self.track_resource_correlations();

        Ok(bottlenecks)
    }

    /// Analyze resource utilization for bottlenecks
    pub fn analyze_resource_utilization(
        &self,
    ) -> PerformanceRegressionResult<Vec<DetectedBottleneck>> {
        let history = self.metric_history.read();
        let mut bottlenecks = Vec::new();

        // Check CPU utilization
        if let Some(cpu_metrics) = history.get(&MetricType::CpuUsage) {
            if let Some(bottleneck) = self.check_cpu_bottleneck(cpu_metrics) {
                bottlenecks.push(bottleneck);
            }
        }

        // Check memory utilization
        if let Some(memory_metrics) = history.get(&MetricType::MemoryUsage) {
            if let Some(bottleneck) = self.check_memory_bottleneck(memory_metrics) {
                bottlenecks.push(bottleneck);
            }
        }

        // Check disk I/O
        if let Some(disk_metrics) = history.get(&MetricType::DiskIops) {
            if let Some(bottleneck) = self.check_disk_bottleneck(disk_metrics) {
                bottlenecks.push(bottleneck);
            }
        }

        // Check network bandwidth
        if let Some(network_metrics) = history.get(&MetricType::NetworkBandwidth) {
            if let Some(bottleneck) = self.check_network_bottleneck(network_metrics) {
                bottlenecks.push(bottleneck);
            }
        }

        // Check response time
        if let Some(response_metrics) = history.get(&MetricType::ResponseTime) {
            if let Some(bottleneck) = self.check_response_time_bottleneck(response_metrics) {
                bottlenecks.push(bottleneck);
            }
        }

        Ok(bottlenecks)
    }

    /// Analyze dependencies for bottlenecks
    pub async fn analyze_dependencies(
        &self,
    ) -> PerformanceRegressionResult<Vec<DetectedBottleneck>> {
        // Placeholder for dependency analysis
        // In a real implementation, this would analyze service dependencies
        Ok(Vec::new())
    }

    /// Analyze transaction flow for bottlenecks
    pub async fn analyze_transaction_flow(
        &self,
    ) -> PerformanceRegressionResult<Vec<DetectedBottleneck>> {
        // Placeholder for transaction flow analysis
        // In a real implementation, this would trace transaction paths
        Ok(Vec::new())
    }

    /// Calculate bottleneck score
    pub fn calculate_bottleneck_score(
        &self,
        utilization: f64,
        threshold: f64,
        duration: Duration,
    ) -> OrderedFloat<f64> {
        // Score based on how much utilization exceeds threshold
        let excess_ratio = (utilization - threshold) / threshold;
        let base_score = excess_ratio.min(1.0).max(0.0);

        // Factor in duration (ensure minimum factor of 1.0 for valid durations)
        let duration_seconds = duration.num_seconds().max(1) as f64;
        let min_duration_seconds = self.config.min_bottleneck_duration.max(1) as f64;
        let duration_factor = (duration_seconds / min_duration_seconds).min(2.0).max(1.0);

        let score = base_score * duration_factor;

        // Ensure minimum score if utilization exceeds threshold
        if utilization > threshold && score < 0.1 {
            OrderedFloat(0.1)
        } else {
            OrderedFloat(score)
        }
    }

    /// Get bottleneck insights and recommendations
    pub fn get_bottleneck_insights(&self) -> BottleneckInsights {
        let bottlenecks = self.detected_bottlenecks.read().clone();
        let correlations = self.resource_correlations.read().clone();

        // Find primary bottleneck (highest score)
        let primary_bottleneck = bottlenecks.iter().max_by_key(|b| b.score).cloned();

        // Calculate system health score
        let system_health_score = self.calculate_system_health_score(&bottlenecks);

        // Build correlation matrix
        let mut correlation_matrix = HashMap::new();
        for (key, value) in correlations {
            let bottleneck_key = (
                self.metric_type_to_bottleneck_type(&key.0),
                self.metric_type_to_bottleneck_type(&key.1),
            );
            correlation_matrix.insert(bottleneck_key, value);
        }

        // Analyze trends
        let bottleneck_trends = self.analyze_bottleneck_trends(&bottlenecks);

        BottleneckInsights {
            primary_bottleneck,
            all_bottlenecks: bottlenecks,
            correlation_matrix,
            transaction_flow_impact: HashMap::new(), // Placeholder
            system_health_score,
            bottleneck_trends,
        }
    }

    /// Track correlations between resources
    pub fn track_resource_correlations(&self) {
        let history = self.metric_history.read();
        let mut correlations = self.resource_correlations.write();

        // Calculate correlations between different metric types
        let metric_types: Vec<_> = history.keys().cloned().collect();

        for i in 0..metric_types.len() {
            for j in (i + 1)..metric_types.len() {
                let metric1 = &metric_types[i];
                let metric2 = &metric_types[j];

                if let (Some(data1), Some(data2)) = (history.get(metric1), history.get(metric2)) {
                    if let Some(correlation) = self.calculate_correlation(data1, data2) {
                        correlations.insert((metric1.clone(), metric2.clone()), correlation);
                        correlations.insert((metric2.clone(), metric1.clone()), correlation);
                    }
                }
            }
        }
    }

    // Helper methods for specific bottleneck checks
    fn check_cpu_bottleneck(
        &self,
        metrics: &VecDeque<MetricDataPoint>,
    ) -> Option<DetectedBottleneck> {
        self.check_threshold_bottleneck(
            metrics,
            self.config.cpu_threshold,
            BottleneckType::CpuBottleneck,
            vec![
                "Increase CPU resources",
                "Optimize CPU-intensive operations",
                "Enable horizontal scaling",
            ],
        )
    }

    fn check_memory_bottleneck(
        &self,
        metrics: &VecDeque<MetricDataPoint>,
    ) -> Option<DetectedBottleneck> {
        self.check_threshold_bottleneck(
            metrics,
            self.config.memory_threshold,
            BottleneckType::MemoryBottleneck,
            vec![
                "Increase memory allocation",
                "Optimize memory usage",
                "Fix memory leaks",
            ],
        )
    }

    fn check_disk_bottleneck(
        &self,
        metrics: &VecDeque<MetricDataPoint>,
    ) -> Option<DetectedBottleneck> {
        self.check_threshold_bottleneck(
            metrics,
            self.config.disk_io_threshold,
            BottleneckType::DiskIoBottleneck,
            vec![
                "Use faster storage (SSD)",
                "Optimize I/O operations",
                "Implement caching",
            ],
        )
    }

    fn check_network_bottleneck(
        &self,
        metrics: &VecDeque<MetricDataPoint>,
    ) -> Option<DetectedBottleneck> {
        self.check_threshold_bottleneck(
            metrics,
            self.config.network_threshold,
            BottleneckType::NetworkBottleneck,
            vec![
                "Increase network bandwidth",
                "Optimize network protocols",
                "Use CDN",
            ],
        )
    }

    fn check_response_time_bottleneck(
        &self,
        metrics: &VecDeque<MetricDataPoint>,
    ) -> Option<DetectedBottleneck> {
        self.check_threshold_bottleneck(
            metrics,
            self.config.response_time_threshold,
            BottleneckType::ResponseTimeBottleneck,
            vec![
                "Optimize slow queries",
                "Add caching layers",
                "Review application logic",
            ],
        )
    }

    fn check_threshold_bottleneck(
        &self,
        metrics: &VecDeque<MetricDataPoint>,
        threshold: f64,
        bottleneck_type: BottleneckType,
        recommendations: Vec<&str>,
    ) -> Option<DetectedBottleneck> {
        if metrics.is_empty() {
            return None;
        }

        // Get all metrics sorted by timestamp
        let mut all_metrics: Vec<_> = metrics.iter().collect();
        all_metrics.sort_by_key(|m| m.timestamp);

        if all_metrics.is_empty() {
            return None;
        }

        // Find the earliest timestamp where we exceed threshold
        let mut bottleneck_start = None;
        let mut bottleneck_metrics = Vec::new();

        for metric in all_metrics.iter().rev() {
            if metric.value.0 > threshold {
                if bottleneck_start.is_none() {
                    bottleneck_start = Some(metric.timestamp);
                }
                bottleneck_metrics.push(*metric);
            } else if !bottleneck_metrics.is_empty() {
                // Break if we encounter a metric below threshold after finding bottleneck metrics
                break;
            }
        }

        if bottleneck_metrics.is_empty() {
            return None;
        }

        // Calculate duration from oldest to newest bottleneck metric
        let oldest_timestamp = bottleneck_metrics.last().map(|m| m.timestamp).unwrap();
        let newest_timestamp = bottleneck_metrics.first().map(|m| m.timestamp).unwrap();
        let duration = newest_timestamp - oldest_timestamp;

        // Check if duration meets minimum requirement
        if duration.num_seconds() < self.config.min_bottleneck_duration as i64 {
            return None;
        }

        // Calculate average utilization
        let avg_utilization = bottleneck_metrics.iter().map(|m| m.value.0).sum::<f64>()
            / bottleneck_metrics.len() as f64;

        let score = self.calculate_bottleneck_score(avg_utilization, threshold, duration);

        // Determine severity
        let severity = self.calculate_severity(score.0);

        Some(DetectedBottleneck {
            bottleneck_type,
            severity,
            score,
            utilization: OrderedFloat(avg_utilization),
            detected_at: newest_timestamp,
            duration,
            affected_components: vec![],
            related_metrics: HashMap::new(),
            correlated_bottlenecks: vec![],
            recommendations: recommendations.into_iter().map(String::from).collect(),
        })
    }

    fn calculate_severity(&self, score: f64) -> BottleneckSeverity {
        match score {
            s if s >= 0.8 => BottleneckSeverity::Critical,
            s if s >= 0.6 => BottleneckSeverity::High,
            s if s >= 0.4 => BottleneckSeverity::Medium,
            _ => BottleneckSeverity::Low,
        }
    }

    fn calculate_system_health_score(
        &self,
        bottlenecks: &[DetectedBottleneck],
    ) -> OrderedFloat<f64> {
        if bottlenecks.is_empty() {
            return OrderedFloat(1.0);
        }

        // Calculate weighted score based on severity
        let total_weight: f64 = bottlenecks
            .iter()
            .map(|b| match b.severity {
                BottleneckSeverity::Critical => 1.0,
                BottleneckSeverity::High => 0.7,
                BottleneckSeverity::Medium => 0.4,
                BottleneckSeverity::Low => 0.2,
            })
            .sum();

        OrderedFloat((1.0 - (total_weight / bottlenecks.len() as f64)).max(0.0))
    }

    fn metric_type_to_bottleneck_type(&self, metric_type: &MetricType) -> BottleneckType {
        match metric_type {
            MetricType::CpuUsage => BottleneckType::CpuBottleneck,
            MetricType::MemoryUsage => BottleneckType::MemoryBottleneck,
            MetricType::DiskIops => BottleneckType::DiskIoBottleneck,
            MetricType::NetworkBandwidth => BottleneckType::NetworkBottleneck,
            MetricType::ResponseTime => BottleneckType::ResponseTimeBottleneck,
            MetricType::Custom(name) => BottleneckType::Custom(name.clone()),
            _ => BottleneckType::Custom("Unknown".to_string()),
        }
    }

    fn calculate_correlation(
        &self,
        data1: &VecDeque<MetricDataPoint>,
        data2: &VecDeque<MetricDataPoint>,
    ) -> Option<OrderedFloat<f64>> {
        // Simple correlation calculation
        if data1.len() < 2 || data2.len() < 2 {
            return None;
        }

        // Find overlapping time windows
        let mut values1 = Vec::new();
        let mut values2 = Vec::new();

        for metric1 in data1 {
            // Find closest metric in data2
            if let Some(metric2) = data2
                .iter()
                .min_by_key(|m| (m.timestamp - metric1.timestamp).num_seconds().abs())
            {
                if (metric2.timestamp - metric1.timestamp).num_seconds().abs() < 60 {
                    values1.push(metric1.value.0);
                    values2.push(metric2.value.0);
                }
            }
        }

        if values1.len() < 2 {
            return None;
        }

        // Calculate Pearson correlation
        let n = values1.len() as f64;
        let sum1: f64 = values1.iter().sum();
        let sum2: f64 = values2.iter().sum();
        let sum1_sq: f64 = values1.iter().map(|x| x * x).sum();
        let sum2_sq: f64 = values2.iter().map(|x| x * x).sum();
        let sum_prod: f64 = values1.iter().zip(values2.iter()).map(|(x, y)| x * y).sum();

        let num = n * sum_prod - sum1 * sum2;
        let den = ((n * sum1_sq - sum1 * sum1) * (n * sum2_sq - sum2 * sum2)).sqrt();

        if den == 0.0 {
            None
        } else {
            Some(OrderedFloat(num / den))
        }
    }

    fn analyze_bottleneck_trends(
        &self,
        bottlenecks: &[DetectedBottleneck],
    ) -> HashMap<BottleneckType, TrendIndicator> {
        let mut trends = HashMap::new();

        // Group bottlenecks by type
        let mut grouped: HashMap<BottleneckType, Vec<&DetectedBottleneck>> = HashMap::new();
        for bottleneck in bottlenecks {
            grouped
                .entry(bottleneck.bottleneck_type.clone())
                .or_insert_with(Vec::new)
                .push(bottleneck);
        }

        // Analyze trend for each type
        for (bottleneck_type, group) in grouped {
            if group.len() < 2 {
                trends.insert(bottleneck_type, TrendIndicator::Stable);
                continue;
            }

            // Sort by time
            let mut sorted: Vec<_> = group.into_iter().collect();
            sorted.sort_by_key(|b| b.detected_at);

            // Calculate trend based on score changes
            let recent_avg = sorted.iter().rev().take(3).map(|b| b.score.0).sum::<f64>()
                / 3.0_f64.min(sorted.len() as f64);
            let older_avg = sorted.iter().take(3).map(|b| b.score.0).sum::<f64>()
                / 3.0_f64.min(sorted.len() as f64);

            let trend = if recent_avg > older_avg * 1.2 {
                if recent_avg > older_avg * 1.5 {
                    TrendIndicator::RapidlyDegrading
                } else {
                    TrendIndicator::Degrading
                }
            } else if recent_avg < older_avg * 0.8 {
                TrendIndicator::Improving
            } else {
                TrendIndicator::Stable
            };

            trends.insert(bottleneck_type, trend);
        }

        trends
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    // Helper function to create test metrics
    fn create_test_metric(
        metric_type: MetricType,
        value: f64,
        offset_seconds: i64,
    ) -> MetricDataPoint {
        MetricDataPoint {
            metric_type,
            value: OrderedFloat(value),
            timestamp: Utc::now() - Duration::seconds(offset_seconds),
            tags: HashMap::new(),
            source: "test".to_string(),
        }
    }

    #[test]
    fn test_bottleneck_config_default() {
        let config = BottleneckConfig::default();
        assert_eq!(config.cpu_threshold, 80.0);
        assert_eq!(config.memory_threshold, 85.0);
        assert_eq!(config.disk_io_threshold, 1000.0);
        assert_eq!(config.network_threshold, 90.0);
        assert_eq!(config.response_time_threshold, 1000.0);
        assert_eq!(config.min_bottleneck_duration, 60);
        assert_eq!(config.correlation_threshold, 0.7);
        assert!(config.enable_dependency_analysis);
        assert!(config.enable_transaction_flow_analysis);
        assert_eq!(config.analysis_window_hours, 24);
    }

    #[test]
    fn test_bottleneck_detector_creation() {
        let config = BottleneckConfig::default();
        let detector = BottleneckDetector::new(config);
        assert!(detector.detected_bottlenecks.read().is_empty());
    }

    #[test]
    fn test_add_metric_to_history() {
        let detector = BottleneckDetector::new(BottleneckConfig::default());
        let metric = create_test_metric(MetricType::CpuUsage, 75.0, 0);

        detector.add_metric(metric.clone());

        let history = detector.metric_history.read();
        assert!(history.contains_key(&MetricType::CpuUsage));
        assert_eq!(history.get(&MetricType::CpuUsage).unwrap().len(), 1);
    }

    #[test]
    fn test_metric_history_window() {
        let mut config = BottleneckConfig::default();
        config.analysis_window_hours = 1; // 1 hour window
        let detector = BottleneckDetector::new(config);

        // Add old metric (outside window)
        let old_metric = create_test_metric(MetricType::CpuUsage, 75.0, 7200); // 2 hours ago
        detector.add_metric(old_metric);

        // Add recent metric
        let recent_metric = create_test_metric(MetricType::CpuUsage, 85.0, 1800); // 30 minutes ago
        detector.add_metric(recent_metric);

        let history = detector.metric_history.read();
        let cpu_history = history.get(&MetricType::CpuUsage).unwrap();
        assert_eq!(cpu_history.len(), 1); // Only recent metric should remain
        assert_eq!(cpu_history.back().unwrap().value.0, 85.0);
    }

    #[tokio::test]
    async fn test_detect_cpu_bottleneck() {
        let mut config = BottleneckConfig::default();
        config.cpu_threshold = 80.0;
        config.min_bottleneck_duration = 30;
        let detector = BottleneckDetector::new(config);

        // Add CPU metrics above threshold
        for i in 0..10 {
            let metric = create_test_metric(MetricType::CpuUsage, 85.0 + i as f64, i * 5);
            detector.add_metric(metric);
        }

        let bottlenecks = detector.detect_bottlenecks().await.unwrap();
        assert!(!bottlenecks.is_empty());

        let cpu_bottleneck = bottlenecks
            .iter()
            .find(|b| matches!(b.bottleneck_type, BottleneckType::CpuBottleneck));
        assert!(cpu_bottleneck.is_some());

        let bottleneck = cpu_bottleneck.unwrap();
        assert!(bottleneck.utilization.0 > 80.0);
        assert!(bottleneck.score.0 > 0.0);
        assert!(!bottleneck.recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_detect_memory_bottleneck() {
        let mut config = BottleneckConfig::default();
        config.memory_threshold = 85.0;
        config.min_bottleneck_duration = 30;
        let detector = BottleneckDetector::new(config);

        // Add memory metrics above threshold
        for i in 0..10 {
            let metric = create_test_metric(MetricType::MemoryUsage, 90.0 + i as f64, i * 5);
            detector.add_metric(metric);
        }

        let bottlenecks = detector.detect_bottlenecks().await.unwrap();
        let memory_bottleneck = bottlenecks
            .iter()
            .find(|b| matches!(b.bottleneck_type, BottleneckType::MemoryBottleneck));
        assert!(memory_bottleneck.is_some());
    }

    #[tokio::test]
    async fn test_detect_disk_io_bottleneck() {
        let mut config = BottleneckConfig::default();
        config.disk_io_threshold = 1000.0;
        config.min_bottleneck_duration = 30;
        let detector = BottleneckDetector::new(config);

        // Add disk I/O metrics above threshold
        for i in 0..10 {
            let metric = create_test_metric(MetricType::DiskIops, 1200.0 + i as f64 * 10.0, i * 5);
            detector.add_metric(metric);
        }

        let bottlenecks = detector.detect_bottlenecks().await.unwrap();
        let disk_bottleneck = bottlenecks
            .iter()
            .find(|b| matches!(b.bottleneck_type, BottleneckType::DiskIoBottleneck));
        assert!(disk_bottleneck.is_some());
    }

    #[tokio::test]
    async fn test_detect_network_bottleneck() {
        let mut config = BottleneckConfig::default();
        config.network_threshold = 90.0;
        config.min_bottleneck_duration = 30;
        let detector = BottleneckDetector::new(config);

        // Add network metrics above threshold
        for i in 0..10 {
            let metric = create_test_metric(MetricType::NetworkBandwidth, 95.0, i * 5);
            detector.add_metric(metric);
        }

        let bottlenecks = detector.detect_bottlenecks().await.unwrap();
        let network_bottleneck = bottlenecks
            .iter()
            .find(|b| matches!(b.bottleneck_type, BottleneckType::NetworkBottleneck));
        assert!(network_bottleneck.is_some());
    }

    #[tokio::test]
    async fn test_detect_response_time_bottleneck() {
        let mut config = BottleneckConfig::default();
        config.response_time_threshold = 1000.0;
        config.min_bottleneck_duration = 30;
        let detector = BottleneckDetector::new(config);

        // Add response time metrics above threshold
        for i in 0..10 {
            let metric =
                create_test_metric(MetricType::ResponseTime, 1500.0 + i as f64 * 50.0, i * 5);
            detector.add_metric(metric);
        }

        let bottlenecks = detector.detect_bottlenecks().await.unwrap();
        let response_bottleneck = bottlenecks
            .iter()
            .find(|b| matches!(b.bottleneck_type, BottleneckType::ResponseTimeBottleneck));
        assert!(response_bottleneck.is_some());
    }

    #[tokio::test]
    async fn test_no_bottleneck_below_threshold() {
        let mut config = BottleneckConfig::default();
        config.cpu_threshold = 80.0;
        let detector = BottleneckDetector::new(config);

        // Add CPU metrics below threshold
        for i in 0..10 {
            let metric = create_test_metric(MetricType::CpuUsage, 50.0 + i as f64, i * 5);
            detector.add_metric(metric);
        }

        let bottlenecks = detector.detect_bottlenecks().await.unwrap();
        let cpu_bottleneck = bottlenecks
            .iter()
            .find(|b| matches!(b.bottleneck_type, BottleneckType::CpuBottleneck));
        assert!(cpu_bottleneck.is_none());
    }

    #[test]
    fn test_calculate_bottleneck_score() {
        let config = BottleneckConfig::default();
        let detector = BottleneckDetector::new(config);

        // Test score calculation
        let score1 = detector.calculate_bottleneck_score(90.0, 80.0, Duration::seconds(60));
        assert!(score1.0 > 0.0 && score1.0 <= 1.0);

        // Higher utilization should give higher score
        let score2 = detector.calculate_bottleneck_score(95.0, 80.0, Duration::seconds(60));
        assert!(score2.0 > score1.0);

        // Longer duration should give higher score
        let score3 = detector.calculate_bottleneck_score(90.0, 80.0, Duration::seconds(120));
        assert!(score3.0 > score1.0);
    }

    #[test]
    fn test_bottleneck_severity_calculation() {
        let config = BottleneckConfig::default();
        let detector = BottleneckDetector::new(config);

        assert_eq!(
            detector.calculate_severity(0.9),
            BottleneckSeverity::Critical
        );
        assert_eq!(detector.calculate_severity(0.7), BottleneckSeverity::High);
        assert_eq!(detector.calculate_severity(0.5), BottleneckSeverity::Medium);
        assert_eq!(detector.calculate_severity(0.3), BottleneckSeverity::Low);
    }

    #[tokio::test]
    async fn test_multiple_bottlenecks_detection() {
        let mut config = BottleneckConfig::default();
        config.cpu_threshold = 80.0;
        config.memory_threshold = 85.0;
        config.min_bottleneck_duration = 30;
        let detector = BottleneckDetector::new(config);

        // Add both CPU and memory metrics above thresholds
        for i in 0..10 {
            detector.add_metric(create_test_metric(MetricType::CpuUsage, 85.0, i * 5));
            detector.add_metric(create_test_metric(MetricType::MemoryUsage, 90.0, i * 5));
        }

        let bottlenecks = detector.detect_bottlenecks().await.unwrap();
        assert!(bottlenecks.len() >= 2);

        let cpu_found = bottlenecks
            .iter()
            .any(|b| matches!(b.bottleneck_type, BottleneckType::CpuBottleneck));
        let memory_found = bottlenecks
            .iter()
            .any(|b| matches!(b.bottleneck_type, BottleneckType::MemoryBottleneck));

        assert!(cpu_found);
        assert!(memory_found);
    }

    #[test]
    fn test_get_bottleneck_insights() {
        let config = BottleneckConfig::default();
        let detector = BottleneckDetector::new(config);

        // Add some test bottlenecks
        let mut bottlenecks = detector.detected_bottlenecks.write();
        bottlenecks.push(DetectedBottleneck {
            bottleneck_type: BottleneckType::CpuBottleneck,
            severity: BottleneckSeverity::High,
            score: OrderedFloat(0.8),
            utilization: OrderedFloat(90.0),
            detected_at: Utc::now(),
            duration: Duration::seconds(120),
            affected_components: vec!["api-server".to_string()],
            related_metrics: HashMap::new(),
            correlated_bottlenecks: vec![],
            recommendations: vec!["Increase CPU".to_string()],
        });
        drop(bottlenecks);

        let insights = detector.get_bottleneck_insights();
        assert!(insights.primary_bottleneck.is_some());
        assert_eq!(insights.all_bottlenecks.len(), 1);
        assert!(insights.system_health_score.0 < 1.0);
    }

    #[test]
    fn test_resource_correlation_tracking() {
        let detector = BottleneckDetector::new(BottleneckConfig::default());

        // Add correlated metrics
        for i in 0..20 {
            let cpu_value = 50.0 + i as f64 * 2.0;
            let memory_value = 60.0 + i as f64 * 1.8; // Correlated with CPU

            detector.add_metric(create_test_metric(MetricType::CpuUsage, cpu_value, i * 10));
            detector.add_metric(create_test_metric(
                MetricType::MemoryUsage,
                memory_value,
                i * 10,
            ));
        }

        detector.track_resource_correlations();

        let correlations = detector.resource_correlations.read();
        let correlation = correlations.get(&(MetricType::CpuUsage, MetricType::MemoryUsage));
        assert!(correlation.is_some());
        assert!(correlation.unwrap().0 > 0.7); // Should be highly correlated
    }

    #[test]
    fn test_system_health_score_calculation() {
        let config = BottleneckConfig::default();
        let detector = BottleneckDetector::new(config);

        // No bottlenecks = perfect health
        let score1 = detector.calculate_system_health_score(&[]);
        assert_eq!(score1.0, 1.0);

        // Single high-severity bottleneck
        let bottlenecks = vec![DetectedBottleneck {
            bottleneck_type: BottleneckType::CpuBottleneck,
            severity: BottleneckSeverity::High,
            score: OrderedFloat(0.8),
            utilization: OrderedFloat(90.0),
            detected_at: Utc::now(),
            duration: Duration::seconds(120),
            affected_components: vec![],
            related_metrics: HashMap::new(),
            correlated_bottlenecks: vec![],
            recommendations: vec![],
        }];
        let score2 = detector.calculate_system_health_score(&bottlenecks);
        assert!(score2.0 < 1.0);
        assert!(score2.0 > 0.0);
    }

    #[test]
    fn test_bottleneck_trend_analysis() {
        let config = BottleneckConfig::default();
        let detector = BottleneckDetector::new(config);

        // Create bottlenecks with increasing severity
        let mut bottlenecks = Vec::new();
        for i in 0..5 {
            bottlenecks.push(DetectedBottleneck {
                bottleneck_type: BottleneckType::CpuBottleneck,
                severity: BottleneckSeverity::Medium,
                score: OrderedFloat(0.4 + i as f64 * 0.1),
                utilization: OrderedFloat(80.0 + i as f64 * 2.0),
                detected_at: Utc::now() - Duration::hours(5 - i),
                duration: Duration::seconds(120),
                affected_components: vec![],
                related_metrics: HashMap::new(),
                correlated_bottlenecks: vec![],
                recommendations: vec![],
            });
        }

        let trends = detector.analyze_bottleneck_trends(&bottlenecks);
        assert_eq!(
            trends.get(&BottleneckType::CpuBottleneck),
            Some(&TrendIndicator::Degrading)
        );
    }

    #[test]
    fn test_custom_metric_bottleneck() {
        let config = BottleneckConfig::default();
        let detector = BottleneckDetector::new(config);

        let custom_type = MetricType::Custom("database_connections".to_string());
        let bottleneck_type = detector.metric_type_to_bottleneck_type(&custom_type);

        match bottleneck_type {
            BottleneckType::Custom(name) => assert_eq!(name, "database_connections"),
            _ => panic!("Expected custom bottleneck type"),
        }
    }

    #[tokio::test]
    async fn test_bottleneck_duration_calculation() {
        let mut config = BottleneckConfig::default();
        config.cpu_threshold = 80.0;
        config.min_bottleneck_duration = 30;
        let detector = BottleneckDetector::new(config);

        // Add metrics over time
        for i in 0..20 {
            let metric = create_test_metric(MetricType::CpuUsage, 85.0, i * 5);
            detector.add_metric(metric);
        }

        let bottlenecks = detector.detect_bottlenecks().await.unwrap();
        let cpu_bottleneck = bottlenecks
            .iter()
            .find(|b| matches!(b.bottleneck_type, BottleneckType::CpuBottleneck))
            .unwrap();

        assert!(cpu_bottleneck.duration.num_seconds() >= 30);
    }

    #[test]
    fn test_insufficient_data_for_bottleneck() {
        let mut config = BottleneckConfig::default();
        config.min_bottleneck_duration = 60;
        let detector = BottleneckDetector::new(config);

        // Add only one metric (insufficient for duration check)
        detector.add_metric(create_test_metric(MetricType::CpuUsage, 90.0, 0));

        let result = detector.analyze_resource_utilization().unwrap();
        assert!(result.is_empty()); // No bottleneck should be detected
    }

    #[test]
    fn test_bottleneck_recommendations() {
        let mut config = BottleneckConfig::default();
        config.min_bottleneck_duration = 30;
        let detector = BottleneckDetector::new(config);

        // Add CPU bottleneck metrics with sufficient duration
        for i in 0..15 {
            detector.add_metric(create_test_metric(MetricType::CpuUsage, 90.0, i * 5));
        }

        let bottlenecks = detector.analyze_resource_utilization().unwrap();
        let cpu_bottleneck = bottlenecks
            .iter()
            .find(|b| matches!(b.bottleneck_type, BottleneckType::CpuBottleneck));

        assert!(cpu_bottleneck.is_some());
        let bottleneck = cpu_bottleneck.unwrap();
        assert!(!bottleneck.recommendations.is_empty());
        assert!(bottleneck.recommendations.iter().any(|r| r.contains("CPU")));
    }

    #[tokio::test]
    async fn test_edge_case_empty_metrics() {
        let config = BottleneckConfig::default();
        let detector = BottleneckDetector::new(config);

        // Detect bottlenecks with no metrics
        let result = detector.detect_bottlenecks().await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_correlation_with_insufficient_data() {
        let detector = BottleneckDetector::new(BottleneckConfig::default());

        // Add only one metric for each type
        detector.add_metric(create_test_metric(MetricType::CpuUsage, 80.0, 0));
        detector.add_metric(create_test_metric(MetricType::MemoryUsage, 85.0, 0));

        detector.track_resource_correlations();

        let correlations = detector.resource_correlations.read();
        assert!(correlations.is_empty()); // No correlation should be calculated
    }

    #[test]
    fn test_bottleneck_type_equality() {
        assert_eq!(BottleneckType::CpuBottleneck, BottleneckType::CpuBottleneck);
        assert_ne!(
            BottleneckType::CpuBottleneck,
            BottleneckType::MemoryBottleneck
        );

        let custom1 = BottleneckType::Custom("test".to_string());
        let custom2 = BottleneckType::Custom("test".to_string());
        let custom3 = BottleneckType::Custom("other".to_string());

        assert_eq!(custom1, custom2);
        assert_ne!(custom1, custom3);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(BottleneckSeverity::Critical > BottleneckSeverity::High);
        assert!(BottleneckSeverity::High > BottleneckSeverity::Medium);
        assert!(BottleneckSeverity::Medium > BottleneckSeverity::Low);
    }
}
