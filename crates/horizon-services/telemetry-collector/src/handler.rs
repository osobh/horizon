use anyhow::{Context, Result};
use prost::Message;
use horizon_hpc_types::MetricBatch;
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::cardinality::CardinalityTracker;
use crate::writers::{InfluxDbWriter, ParquetWriter};

pub struct StreamHandler {
    cardinality_tracker: Arc<Mutex<CardinalityTracker>>,
    influxdb_writer: Arc<InfluxDbWriter>,
    parquet_writer: Arc<Mutex<ParquetWriter>>,
}

impl StreamHandler {
    pub fn new(
        cardinality_tracker: Arc<Mutex<CardinalityTracker>>,
        influxdb_writer: Arc<InfluxDbWriter>,
        parquet_writer: Arc<Mutex<ParquetWriter>>,
    ) -> Self {
        Self {
            cardinality_tracker,
            influxdb_writer,
            parquet_writer,
        }
    }

    /// Decode a length-prefixed MetricBatch from bytes
    pub fn decode_length_prefixed(data: &[u8]) -> Result<MetricBatch> {
        if data.len() < 4 {
            anyhow::bail!("Data too short for length prefix");
        }

        let length = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;

        if data.len() < 4 + length {
            anyhow::bail!("Data shorter than indicated length");
        }

        MetricBatch::decode(&data[4..4 + length])
            .context("Failed to decode MetricBatch")
    }

    /// Handle a received metric batch
    pub async fn handle_batch(&self, batch: MetricBatch) -> Result<()> {
        // Check cardinality
        {
            let mut tracker = self.cardinality_tracker.lock().await;
            tracker.track_batch(&batch)
                .context("Cardinality limit exceeded")?;
        }

        // Write to InfluxDB (async)
        self.influxdb_writer.write_batch(&batch).await
            .context("Failed to write to InfluxDB")?;

        // Write to Parquet (async)
        {
            let mut writer = self.parquet_writer.lock().await;
            writer.write_batch(&batch).await
                .context("Failed to write to Parquet")?;
        }

        Ok(())
    }

    /// Encode a length-prefixed MetricBatch
    pub fn encode_length_prefixed(batch: &MetricBatch) -> Vec<u8> {
        let encoded = batch.encode_to_vec();
        let length = (encoded.len() as u32).to_be_bytes();

        let mut result = Vec::with_capacity(4 + encoded.len());
        result.extend_from_slice(&length);
        result.extend_from_slice(&encoded);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_hpc_types::{GpuMetric, Timestamp};

    #[test]
    fn test_length_prefixed_encoding() {
        let batch = MetricBatch {
            gpu_metrics: vec![GpuMetric {
                host_id: "test".to_string(),
                gpu_id: "0".to_string(),
                timestamp: Some(Timestamp { seconds: 123, nanos: 0 }),
                ..Default::default()
            }],
            cpu_metrics: vec![],
            nic_metrics: vec![],
        };

        let encoded = StreamHandler::encode_length_prefixed(&batch);
        let decoded = StreamHandler::decode_length_prefixed(&encoded).unwrap();

        assert_eq!(batch.gpu_metrics.len(), decoded.gpu_metrics.len());
    }
}
