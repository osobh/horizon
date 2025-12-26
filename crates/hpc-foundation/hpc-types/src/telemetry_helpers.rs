//! Telemetry message helpers and builders
//!
//! This module provides convenient builder patterns and helper functions
//! for creating and validating telemetry messages.

use crate::telemetry::v1::{CpuMetric, GpuMetric, MetricBatch, NicMetric};
use crate::{create_timestamp, current_timestamp, Timestamp};

/// Builder for GpuMetric
///
/// # Examples
///
/// ```
/// use hpc_types::telemetry_helpers::GpuMetricBuilder;
///
/// let metric = GpuMetricBuilder::new("host-001", "gpu-0")
///     .utilization(87.5)
///     .memory(64.5, 80.0)
///     .temperature(68.5)
///     .power(350.0)
///     .build();
///
/// assert_eq!(metric.host_id, "host-001");
/// assert_eq!(metric.utilization, 87.5);
/// ```
pub struct GpuMetricBuilder {
    metric: GpuMetric,
}

impl GpuMetricBuilder {
    /// Creates a new GPU metric builder
    pub fn new(host_id: impl Into<String>, gpu_id: impl Into<String>) -> Self {
        Self {
            metric: GpuMetric {
                host_id: host_id.into(),
                gpu_id: gpu_id.into(),
                timestamp: Some(current_timestamp()),
                ..Default::default()
            },
        }
    }

    /// Sets the timestamp (defaults to current time)
    pub fn timestamp(mut self, ts: Timestamp) -> Self {
        self.metric.timestamp = Some(ts);
        self
    }

    /// Sets the timestamp from seconds and nanos
    pub fn timestamp_from(mut self, seconds: i64, nanos: i32) -> Self {
        self.metric.timestamp = Some(create_timestamp(seconds, nanos));
        self
    }

    /// Sets GPU utilization percentage
    pub fn utilization(mut self, utilization: f32) -> Self {
        self.metric.utilization = utilization;
        self
    }

    /// Sets SM occupancy percentage
    pub fn sm_occupancy(mut self, occupancy: f32) -> Self {
        self.metric.sm_occupancy = occupancy;
        self
    }

    /// Sets memory usage in GB
    pub fn memory(mut self, used_gb: f32, total_gb: f32) -> Self {
        self.metric.memory_used_gb = used_gb;
        self.metric.memory_total_gb = total_gb;
        self
    }

    /// Sets PCIe bandwidth in Gbps
    pub fn pcie_bandwidth(mut self, tx_gbps: f32, rx_gbps: f32) -> Self {
        self.metric.pcie_tx_gbps = tx_gbps;
        self.metric.pcie_rx_gbps = rx_gbps;
        self
    }

    /// Sets NVLink bandwidth in Gbps
    pub fn nvlink_bandwidth(mut self, bandwidth_gbps: f32) -> Self {
        self.metric.nvlink_bandwidth_gbps = bandwidth_gbps;
        self
    }

    /// Sets temperature in Celsius
    pub fn temperature(mut self, celsius: f32) -> Self {
        self.metric.temperature_celsius = celsius;
        self
    }

    /// Sets power consumption in Watts
    pub fn power(mut self, watts: f32) -> Self {
        self.metric.power_watts = watts;
        self
    }

    /// Sets ECC error flag
    pub fn ecc_errors(mut self, has_errors: bool) -> Self {
        self.metric.ecc_errors = has_errors;
        self
    }

    /// Sets MIG profile
    pub fn mig_profile(mut self, profile: impl Into<String>) -> Self {
        self.metric.mig_profile = profile.into();
        self
    }

    /// Builds the GpuMetric
    pub fn build(self) -> GpuMetric {
        self.metric
    }
}

/// Builder for CpuMetric
pub struct CpuMetricBuilder {
    metric: CpuMetric,
}

impl CpuMetricBuilder {
    /// Creates a new CPU metric builder
    pub fn new(host_id: impl Into<String>, socket: u32) -> Self {
        Self {
            metric: CpuMetric {
                host_id: host_id.into(),
                timestamp: Some(current_timestamp()),
                socket,
                ..Default::default()
            },
        }
    }

    /// Sets the timestamp
    pub fn timestamp(mut self, ts: Timestamp) -> Self {
        self.metric.timestamp = Some(ts);
        self
    }

    /// Sets CPU utilization percentage
    pub fn utilization(mut self, utilization: f32) -> Self {
        self.metric.utilization = utilization;
        self
    }

    /// Sets instructions per cycle
    pub fn ipc(mut self, ipc: f32) -> Self {
        self.metric.ipc = ipc;
        self
    }

    /// Sets cache misses count
    pub fn cache_misses(mut self, misses: u64) -> Self {
        self.metric.cache_misses = misses;
        self
    }

    /// Builds the CpuMetric
    pub fn build(self) -> CpuMetric {
        self.metric
    }
}

/// Builder for NicMetric
pub struct NicMetricBuilder {
    metric: NicMetric,
}

impl NicMetricBuilder {
    /// Creates a new NIC metric builder
    pub fn new(host_id: impl Into<String>, interface: impl Into<String>) -> Self {
        Self {
            metric: NicMetric {
                host_id: host_id.into(),
                timestamp: Some(current_timestamp()),
                interface: interface.into(),
                ..Default::default()
            },
        }
    }

    /// Sets the timestamp
    pub fn timestamp(mut self, ts: Timestamp) -> Self {
        self.metric.timestamp = Some(ts);
        self
    }

    /// Sets network bandwidth in Gbps
    pub fn bandwidth(mut self, rx_gbps: f32, tx_gbps: f32) -> Self {
        self.metric.rx_gbps = rx_gbps;
        self.metric.tx_gbps = tx_gbps;
        self
    }

    /// Sets error and drop counts
    pub fn errors_and_drops(mut self, errors: u64, drops: u64) -> Self {
        self.metric.errors = errors;
        self.metric.drops = drops;
        self
    }

    /// Builds the NicMetric
    pub fn build(self) -> NicMetric {
        self.metric
    }
}

/// Builder for MetricBatch
pub struct MetricBatchBuilder {
    batch: MetricBatch,
}

impl MetricBatchBuilder {
    /// Creates a new metric batch builder
    pub fn new() -> Self {
        Self {
            batch: MetricBatch {
                gpu_metrics: vec![],
                cpu_metrics: vec![],
                nic_metrics: vec![],
            },
        }
    }

    /// Adds a GPU metric to the batch
    pub fn add_gpu_metric(mut self, metric: GpuMetric) -> Self {
        self.batch.gpu_metrics.push(metric);
        self
    }

    /// Adds multiple GPU metrics to the batch
    pub fn add_gpu_metrics(mut self, metrics: Vec<GpuMetric>) -> Self {
        self.batch.gpu_metrics.extend(metrics);
        self
    }

    /// Adds a CPU metric to the batch
    pub fn add_cpu_metric(mut self, metric: CpuMetric) -> Self {
        self.batch.cpu_metrics.push(metric);
        self
    }

    /// Adds multiple CPU metrics to the batch
    pub fn add_cpu_metrics(mut self, metrics: Vec<CpuMetric>) -> Self {
        self.batch.cpu_metrics.extend(metrics);
        self
    }

    /// Adds a NIC metric to the batch
    pub fn add_nic_metric(mut self, metric: NicMetric) -> Self {
        self.batch.nic_metrics.push(metric);
        self
    }

    /// Adds multiple NIC metrics to the batch
    pub fn add_nic_metrics(mut self, metrics: Vec<NicMetric>) -> Self {
        self.batch.nic_metrics.extend(metrics);
        self
    }

    /// Builds the MetricBatch
    pub fn build(self) -> MetricBatch {
        self.batch
    }
}

impl Default for MetricBatchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation helpers
pub mod validation {
    use super::*;

    /// Validates a GpuMetric
    pub fn validate_gpu_metric(metric: &GpuMetric) -> Result<(), String> {
        if metric.host_id.is_empty() {
            return Err("host_id cannot be empty".to_string());
        }
        if metric.gpu_id.is_empty() {
            return Err("gpu_id cannot be empty".to_string());
        }
        if metric.timestamp.is_none() {
            return Err("timestamp is required".to_string());
        }
        if metric.utilization < 0.0 || metric.utilization > 100.0 {
            return Err(format!(
                "utilization must be 0-100, got {}",
                metric.utilization
            ));
        }
        if metric.memory_total_gb > 0.0 && metric.memory_used_gb > metric.memory_total_gb {
            return Err("memory_used_gb cannot exceed memory_total_gb".to_string());
        }
        Ok(())
    }

    /// Validates a CpuMetric
    pub fn validate_cpu_metric(metric: &CpuMetric) -> Result<(), String> {
        if metric.host_id.is_empty() {
            return Err("host_id cannot be empty".to_string());
        }
        if metric.timestamp.is_none() {
            return Err("timestamp is required".to_string());
        }
        if metric.utilization < 0.0 || metric.utilization > 100.0 {
            return Err(format!(
                "utilization must be 0-100, got {}",
                metric.utilization
            ));
        }
        Ok(())
    }

    /// Validates a NicMetric
    pub fn validate_nic_metric(metric: &NicMetric) -> Result<(), String> {
        if metric.host_id.is_empty() {
            return Err("host_id cannot be empty".to_string());
        }
        if metric.interface.is_empty() {
            return Err("interface cannot be empty".to_string());
        }
        if metric.timestamp.is_none() {
            return Err("timestamp is required".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_metric_builder() {
        let metric = GpuMetricBuilder::new("host-001", "gpu-0")
            .utilization(87.5)
            .memory(64.5, 80.0)
            .temperature(68.5)
            .power(350.0)
            .build();

        assert_eq!(metric.host_id, "host-001");
        assert_eq!(metric.gpu_id, "gpu-0");
        assert_eq!(metric.utilization, 87.5);
        assert_eq!(metric.memory_used_gb, 64.5);
        assert_eq!(metric.memory_total_gb, 80.0);
    }

    #[test]
    fn test_cpu_metric_builder() {
        let metric = CpuMetricBuilder::new("host-001", 0)
            .utilization(45.6)
            .ipc(2.3)
            .cache_misses(1_234_567)
            .build();

        assert_eq!(metric.host_id, "host-001");
        assert_eq!(metric.socket, 0);
        assert_eq!(metric.utilization, 45.6);
    }

    #[test]
    fn test_nic_metric_builder() {
        let metric = NicMetricBuilder::new("host-001", "eth0")
            .bandwidth(8.5, 7.2)
            .errors_and_drops(0, 0)
            .build();

        assert_eq!(metric.host_id, "host-001");
        assert_eq!(metric.interface, "eth0");
        assert_eq!(metric.rx_gbps, 8.5);
    }

    #[test]
    fn test_batch_builder() {
        let gpu = GpuMetricBuilder::new("host-001", "gpu-0").build();
        let cpu = CpuMetricBuilder::new("host-001", 0).build();
        let nic = NicMetricBuilder::new("host-001", "eth0").build();

        let batch = MetricBatchBuilder::new()
            .add_gpu_metric(gpu)
            .add_cpu_metric(cpu)
            .add_nic_metric(nic)
            .build();

        assert_eq!(batch.gpu_metrics.len(), 1);
        assert_eq!(batch.cpu_metrics.len(), 1);
        assert_eq!(batch.nic_metrics.len(), 1);
    }

    #[test]
    fn test_validation_valid_metric() {
        let metric = GpuMetricBuilder::new("host-001", "gpu-0")
            .utilization(50.0)
            .memory(40.0, 80.0)
            .build();

        assert!(validation::validate_gpu_metric(&metric).is_ok());
    }

    #[test]
    fn test_validation_invalid_utilization() {
        let metric = GpuMetricBuilder::new("host-001", "gpu-0")
            .utilization(150.0)
            .build();

        let result = validation::validate_gpu_metric(&metric);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("utilization"));
    }

    #[test]
    fn test_validation_empty_host_id() {
        let metric = GpuMetric {
            host_id: "".to_string(),
            gpu_id: "gpu-0".to_string(),
            timestamp: Some(current_timestamp()),
            ..Default::default()
        };

        let result = validation::validate_gpu_metric(&metric);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("host_id"));
    }
}
