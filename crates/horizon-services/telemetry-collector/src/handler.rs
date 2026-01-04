use crate::cardinality::CardinalityTracker;
use crate::channels::TelemetryChannels;
use crate::writers::{InfluxDbWriter, ParquetWriter};
use anyhow::{Context, Result};
use hpc_channels::TelemetryMessage;
use hpc_types::MetricBatch;
use prost::Message;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

pub struct StreamHandler {
    cardinality_tracker: Arc<Mutex<CardinalityTracker>>,
    influxdb_writer: Arc<InfluxDbWriter>,
    parquet_writer: Arc<Mutex<ParquetWriter>>,
    channels: TelemetryChannels,
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
            channels: TelemetryChannels::new(),
        }
    }

    pub fn with_channels(
        cardinality_tracker: Arc<Mutex<CardinalityTracker>>,
        influxdb_writer: Arc<InfluxDbWriter>,
        parquet_writer: Arc<Mutex<ParquetWriter>>,
        channels: TelemetryChannels,
    ) -> Self {
        Self {
            cardinality_tracker,
            influxdb_writer,
            parquet_writer,
            channels,
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

        MetricBatch::decode(&data[4..4 + length]).context("Failed to decode MetricBatch")
    }

    /// Handle a received metric batch
    pub async fn handle_batch(&self, batch: MetricBatch) -> Result<()> {
        let metric_count =
            (batch.gpu_metrics.len() + batch.cpu_metrics.len() + batch.nic_metrics.len()) as u32;

        // Extract node_id from first metric if available
        let node_id = batch
            .gpu_metrics
            .first()
            .map(|m| m.host_id.clone())
            .or_else(|| batch.cpu_metrics.first().map(|m| m.host_id.clone()))
            .unwrap_or_else(|| "unknown".to_string());

        // Check cardinality
        {
            let mut tracker = self.cardinality_tracker.lock().await;
            tracker
                .track_batch(&batch)
                .context("Cardinality limit exceeded")?;
        }

        // Write to InfluxDB (async)
        if let Err(e) = self.influxdb_writer.write_batch(&batch).await {
            // Publish writer error event
            self.channels.publish_alert(TelemetryMessage::WriterError {
                writer_type: "influxdb".to_string(),
                error: e.to_string(),
            });
            return Err(e).context("Failed to write to InfluxDB");
        }

        // Write to Parquet (async)
        {
            let mut writer = self.parquet_writer.lock().await;
            if let Err(e) = writer.write_batch(&batch).await {
                // Publish writer error event
                self.channels.publish_alert(TelemetryMessage::WriterError {
                    writer_type: "parquet".to_string(),
                    error: e.to_string(),
                });
                return Err(e).context("Failed to write to Parquet");
            }
        }

        // Publish metrics received event via hpc-channels
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.channels
            .publish_metrics(TelemetryMessage::MetricsBatchReceived {
                node_id,
                metric_count,
                timestamp_ms,
            });

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
    use hpc_types::{GpuMetric, Timestamp};

    #[test]
    fn test_length_prefixed_encoding() {
        let batch = MetricBatch {
            gpu_metrics: vec![GpuMetric {
                host_id: "test".to_string(),
                gpu_id: "0".to_string(),
                timestamp: Some(Timestamp {
                    seconds: 123,
                    nanos: 0,
                }),
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
