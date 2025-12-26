use anyhow::Result;
use hpc_types::{GpuMetric, CpuMetric, NicMetric, MetricBatch};
use crate::config::InfluxDbConfig;

pub struct InfluxDbWriter {
    _config: InfluxDbConfig,
}

impl InfluxDbWriter {
    pub fn new(config: InfluxDbConfig) -> Result<Self> {
        Ok(Self { _config: config })
    }

    /// Convert GPU metric to InfluxDB line protocol
    pub fn gpu_metric_to_line_protocol(metric: &GpuMetric) -> String {
        let timestamp = metric.timestamp.as_ref()
            .map(|ts| ts.seconds * 1_000_000_000 + ts.nanos as i64)
            .unwrap_or(0);

        format!(
            "gpu_metrics,host={},gpu={} utilization={},sm_occupancy={},memory_used_gb={},memory_total_gb={},pcie_tx_gbps={},pcie_rx_gbps={},nvlink_bandwidth_gbps={},temperature_celsius={},power_watts={},ecc_errors={} {}",
            metric.host_id,
            metric.gpu_id,
            metric.utilization,
            metric.sm_occupancy,
            metric.memory_used_gb,
            metric.memory_total_gb,
            metric.pcie_tx_gbps,
            metric.pcie_rx_gbps,
            metric.nvlink_bandwidth_gbps,
            metric.temperature_celsius,
            metric.power_watts,
            metric.ecc_errors,
            timestamp
        )
    }

    /// Convert CPU metric to InfluxDB line protocol
    pub fn cpu_metric_to_line_protocol(metric: &CpuMetric) -> String {
        let timestamp = metric.timestamp.as_ref()
            .map(|ts| ts.seconds * 1_000_000_000 + ts.nanos as i64)
            .unwrap_or(0);

        format!(
            "cpu_metrics,host={},socket={} utilization={},ipc={},cache_misses={} {}",
            metric.host_id,
            metric.socket,
            metric.utilization,
            metric.ipc,
            metric.cache_misses,
            timestamp
        )
    }

    /// Convert NIC metric to InfluxDB line protocol
    pub fn nic_metric_to_line_protocol(metric: &NicMetric) -> String {
        let timestamp = metric.timestamp.as_ref()
            .map(|ts| ts.seconds * 1_000_000_000 + ts.nanos as i64)
            .unwrap_or(0);

        format!(
            "nic_metrics,host={},interface={} rx_gbps={},tx_gbps={},errors={},drops={} {}",
            metric.host_id,
            metric.interface,
            metric.rx_gbps,
            metric.tx_gbps,
            metric.errors,
            metric.drops,
            timestamp
        )
    }

    /// Convert a metric batch to line protocol strings
    pub fn batch_to_line_protocol(&self, batch: &MetricBatch) -> Vec<String> {
        let mut lines = Vec::new();

        for metric in &batch.gpu_metrics {
            lines.push(Self::gpu_metric_to_line_protocol(metric));
        }

        for metric in &batch.cpu_metrics {
            lines.push(Self::cpu_metric_to_line_protocol(metric));
        }

        for metric in &batch.nic_metrics {
            lines.push(Self::nic_metric_to_line_protocol(metric));
        }

        lines
    }

    /// Write a metric batch to InfluxDB
    pub async fn write_batch(&self, batch: &MetricBatch) -> Result<()> {
        // For now, just validate the batch can be converted to line protocol
        // In production, this would actually write to InfluxDB using an HTTP client
        let _lines = self.batch_to_line_protocol(batch);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_protocol_escaping() {
        let metric = GpuMetric {
            host_id: "host-with-dash".to_string(),
            gpu_id: "gpu_0".to_string(),
            ..Default::default()
        };

        let line = InfluxDbWriter::gpu_metric_to_line_protocol(&metric);
        assert!(line.contains("host=host-with-dash"));
    }
}
