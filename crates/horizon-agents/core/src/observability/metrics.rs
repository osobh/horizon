use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub name: String,
    pub value: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub tags: HashMap<String, String>,
}

impl MetricPoint {
    pub fn new(name: String, value: f64) -> Self {
        Self {
            name,
            value,
            timestamp: chrono::Utc::now(),
            tags: HashMap::new(),
        }
    }

    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }
}

pub struct MetricsCollector {
    metrics: Vec<MetricPoint>,
    max_metrics: usize,
}

impl MetricsCollector {
    pub fn new(max_metrics: usize) -> Self {
        Self {
            metrics: Vec::new(),
            max_metrics,
        }
    }

    pub fn record(&mut self, metric: MetricPoint) {
        if self.metrics.len() >= self.max_metrics {
            self.metrics.remove(0);
        }
        self.metrics.push(metric);
    }

    pub fn record_counter(&mut self, name: String, value: f64) {
        self.record(MetricPoint::new(name, value));
    }

    pub fn record_gauge(&mut self, name: String, value: f64) {
        self.record(MetricPoint::new(name, value));
    }

    pub fn record_histogram(&mut self, name: String, value: f64) {
        self.record(MetricPoint::new(name, value));
    }

    pub fn get_metrics(&self, name: &str) -> Vec<&MetricPoint> {
        self.metrics.iter().filter(|m| m.name == name).collect()
    }

    pub fn get_recent_metrics(&self, limit: usize) -> Vec<&MetricPoint> {
        let start = if self.metrics.len() > limit {
            self.metrics.len() - limit
        } else {
            0
        };

        self.metrics[start..].iter().rev().collect()
    }

    pub fn clear(&mut self) {
        self.metrics.clear();
    }

    pub fn size(&self) -> usize {
        self.metrics.len()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_actions: u64,
    pub total_cost: f64,
    pub average_response_time_ms: f64,
}

impl AgentMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_actions: 0,
            total_cost: 0.0,
            average_response_time_ms: 0.0,
        }
    }

    pub fn record_request(&mut self, success: bool, response_time_ms: f64) {
        self.total_requests += 1;
        if success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }

        // Update rolling average
        let total = self.total_requests as f64;
        self.average_response_time_ms =
            (self.average_response_time_ms * (total - 1.0) + response_time_ms) / total;
    }

    pub fn record_action(&mut self, cost: f64) {
        self.total_actions += 1;
        self.total_cost += cost;
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        (self.successful_requests as f64 / self.total_requests as f64) * 100.0
    }

    pub fn failure_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        (self.failed_requests as f64 / self.total_requests as f64) * 100.0
    }
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_point_creation() {
        let metric = MetricPoint::new("test_metric".to_string(), 42.0);
        assert_eq!(metric.name, "test_metric");
        assert_eq!(metric.value, 42.0);
        assert!(metric.tags.is_empty());
    }

    #[test]
    fn test_metric_point_with_tag() {
        let metric = MetricPoint::new("test_metric".to_string(), 42.0)
            .with_tag("env".to_string(), "prod".to_string());

        assert_eq!(metric.tags.len(), 1);
        assert_eq!(metric.tags.get("env").unwrap(), "prod");
    }

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new(100);
        assert_eq!(collector.size(), 0);
    }

    #[test]
    fn test_metrics_collector_record() {
        let mut collector = MetricsCollector::new(100);

        let metric = MetricPoint::new("test".to_string(), 1.0);
        collector.record(metric);

        assert_eq!(collector.size(), 1);
    }

    #[test]
    fn test_metrics_collector_eviction() {
        let mut collector = MetricsCollector::new(2);

        collector.record(MetricPoint::new("metric1".to_string(), 1.0));
        collector.record(MetricPoint::new("metric2".to_string(), 2.0));
        collector.record(MetricPoint::new("metric3".to_string(), 3.0));

        assert_eq!(collector.size(), 2);
        let metrics = collector.get_recent_metrics(10);
        assert_eq!(metrics[0].name, "metric3");
        assert_eq!(metrics[1].name, "metric2");
    }

    #[test]
    fn test_metrics_collector_record_counter() {
        let mut collector = MetricsCollector::new(100);
        collector.record_counter("requests".to_string(), 1.0);

        let metrics = collector.get_metrics("requests");
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].value, 1.0);
    }

    #[test]
    fn test_metrics_collector_record_gauge() {
        let mut collector = MetricsCollector::new(100);
        collector.record_gauge("cpu_usage".to_string(), 75.5);

        let metrics = collector.get_metrics("cpu_usage");
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].value, 75.5);
    }

    #[test]
    fn test_metrics_collector_get_metrics() {
        let mut collector = MetricsCollector::new(100);

        collector.record(MetricPoint::new("metric1".to_string(), 1.0));
        collector.record(MetricPoint::new("metric2".to_string(), 2.0));
        collector.record(MetricPoint::new("metric1".to_string(), 3.0));

        let metrics = collector.get_metrics("metric1");
        assert_eq!(metrics.len(), 2);
    }

    #[test]
    fn test_metrics_collector_get_recent_metrics() {
        let mut collector = MetricsCollector::new(100);

        for i in 0..5 {
            collector.record(MetricPoint::new(format!("metric{}", i), i as f64));
        }

        let recent = collector.get_recent_metrics(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].name, "metric4");
    }

    #[test]
    fn test_metrics_collector_clear() {
        let mut collector = MetricsCollector::new(100);
        collector.record(MetricPoint::new("test".to_string(), 1.0));

        assert_eq!(collector.size(), 1);
        collector.clear();
        assert_eq!(collector.size(), 0);
    }

    #[test]
    fn test_agent_metrics_creation() {
        let metrics = AgentMetrics::new();
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.successful_requests, 0);
        assert_eq!(metrics.failed_requests, 0);
        assert_eq!(metrics.total_actions, 0);
        assert_eq!(metrics.total_cost, 0.0);
    }

    #[test]
    fn test_agent_metrics_record_request() {
        let mut metrics = AgentMetrics::new();

        metrics.record_request(true, 100.0);
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.successful_requests, 1);
        assert_eq!(metrics.average_response_time_ms, 100.0);

        metrics.record_request(false, 200.0);
        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.average_response_time_ms, 150.0);
    }

    #[test]
    fn test_agent_metrics_record_action() {
        let mut metrics = AgentMetrics::new();

        metrics.record_action(10.0);
        assert_eq!(metrics.total_actions, 1);
        assert_eq!(metrics.total_cost, 10.0);

        metrics.record_action(20.0);
        assert_eq!(metrics.total_actions, 2);
        assert_eq!(metrics.total_cost, 30.0);
    }

    #[test]
    fn test_agent_metrics_success_rate() {
        let mut metrics = AgentMetrics::new();

        assert_eq!(metrics.success_rate(), 0.0);

        metrics.record_request(true, 100.0);
        metrics.record_request(true, 100.0);
        metrics.record_request(false, 100.0);
        metrics.record_request(true, 100.0);

        assert!((metrics.success_rate() - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_agent_metrics_failure_rate() {
        let mut metrics = AgentMetrics::new();

        assert_eq!(metrics.failure_rate(), 0.0);

        metrics.record_request(true, 100.0);
        metrics.record_request(false, 100.0);
        metrics.record_request(false, 100.0);
        metrics.record_request(true, 100.0);

        assert!((metrics.failure_rate() - 50.0).abs() < 0.01);
    }
}
