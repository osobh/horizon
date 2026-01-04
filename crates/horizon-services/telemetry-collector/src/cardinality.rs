use anyhow::{anyhow, Result};
use hpc_types::{CpuMetric, GpuMetric, MetricBatch, NicMetric};
use std::collections::HashSet;

/// Tracks unique metric series and enforces cardinality limits
pub struct CardinalityTracker {
    /// Maximum allowed number of unique series
    max_cardinality: u64,
    /// Set of unique series IDs
    series: HashSet<String>,
}

impl CardinalityTracker {
    /// Create a new cardinality tracker with the given limit
    pub fn new(max_cardinality: u64) -> Self {
        Self {
            max_cardinality,
            series: HashSet::new(),
        }
    }

    /// Get current number of unique series
    pub fn current_cardinality(&self) -> usize {
        self.series.len()
    }

    /// Get maximum cardinality limit
    pub fn max_cardinality(&self) -> u64 {
        self.max_cardinality
    }

    /// Track a GPU metric and return error if limit is exceeded
    pub fn track_gpu_metric(&mut self, metric: &GpuMetric) -> Result<()> {
        let series_id = self.generate_gpu_series_id(metric);
        self.track_series(series_id)
    }

    /// Track a CPU metric and return error if limit is exceeded
    pub fn track_cpu_metric(&mut self, metric: &CpuMetric) -> Result<()> {
        let series_id = self.generate_cpu_series_id(metric);
        self.track_series(series_id)
    }

    /// Track a NIC metric and return error if limit is exceeded
    pub fn track_nic_metric(&mut self, metric: &NicMetric) -> Result<()> {
        let series_id = self.generate_nic_series_id(metric);
        self.track_series(series_id)
    }

    /// Track all metrics in a batch
    pub fn track_batch(&mut self, batch: &MetricBatch) -> Result<()> {
        // First, check if adding all new series would exceed the limit
        let mut new_series = Vec::new();

        for metric in &batch.gpu_metrics {
            let series_id = self.generate_gpu_series_id(metric);
            if !self.series.contains(&series_id) {
                new_series.push(series_id);
            }
        }

        for metric in &batch.cpu_metrics {
            let series_id = self.generate_cpu_series_id(metric);
            if !self.series.contains(&series_id) {
                new_series.push(series_id);
            }
        }

        for metric in &batch.nic_metrics {
            let series_id = self.generate_nic_series_id(metric);
            if !self.series.contains(&series_id) {
                new_series.push(series_id);
            }
        }

        // Check if adding these would exceed limit
        let new_total = self.series.len() + new_series.len();
        if new_total > self.max_cardinality as usize {
            return Err(anyhow!(
                "Batch would exceed cardinality limit: {} + {} > {}",
                self.series.len(),
                new_series.len(),
                self.max_cardinality
            ));
        }

        // Add all new series
        for series_id in new_series {
            self.series.insert(series_id);
        }

        Ok(())
    }

    /// Reset all tracked series
    pub fn reset(&mut self) {
        self.series.clear();
    }

    /// Get all tracked series IDs
    pub fn get_all_series(&self) -> Vec<String> {
        self.series.iter().cloned().collect()
    }

    /// Track a series ID, returning error if limit exceeded
    fn track_series(&mut self, series_id: String) -> Result<()> {
        if self.series.contains(&series_id) {
            // Already tracked, no need to check limit
            return Ok(());
        }

        if self.series.len() >= self.max_cardinality as usize {
            return Err(anyhow!(
                "Cardinality limit exceeded: {} >= {}",
                self.series.len(),
                self.max_cardinality
            ));
        }

        self.series.insert(series_id);
        Ok(())
    }

    /// Generate a unique series ID for a GPU metric
    /// Format: gpu:host_id:gpu_id
    fn generate_gpu_series_id(&self, metric: &GpuMetric) -> String {
        format!("gpu:{}:{}", metric.host_id, metric.gpu_id)
    }

    /// Generate a unique series ID for a CPU metric
    /// Format: cpu:host_id:socket
    fn generate_cpu_series_id(&self, metric: &CpuMetric) -> String {
        format!("cpu:{}:{}", metric.host_id, metric.socket)
    }

    /// Generate a unique series ID for a NIC metric
    /// Format: nic:host_id:interface
    fn generate_nic_series_id(&self, metric: &NicMetric) -> String {
        format!("nic:{}:{}", metric.host_id, metric.interface)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_series_id_format() {
        let tracker = CardinalityTracker::new(100);

        let gpu_metric = GpuMetric {
            host_id: "host1".to_string(),
            gpu_id: "gpu0".to_string(),
            ..Default::default()
        };

        let cpu_metric = CpuMetric {
            host_id: "host1".to_string(),
            socket: 0,
            ..Default::default()
        };

        let nic_metric = NicMetric {
            host_id: "host1".to_string(),
            interface: "eth0".to_string(),
            ..Default::default()
        };

        assert_eq!(
            tracker.generate_gpu_series_id(&gpu_metric),
            "gpu:host1:gpu0"
        );
        assert_eq!(tracker.generate_cpu_series_id(&cpu_metric), "cpu:host1:0");
        assert_eq!(
            tracker.generate_nic_series_id(&nic_metric),
            "nic:host1:eth0"
        );
    }
}
