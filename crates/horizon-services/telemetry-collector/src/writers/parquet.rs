use crate::config::ParquetConfig;
use anyhow::{anyhow, Context, Result};
use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hpc_types::MetricBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::{create_dir_all, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct ParquetWriter {
    config: ParquetConfig,
    current_file: Option<PathBuf>,
    rotation_time: SystemTime,
}

impl ParquetWriter {
    pub fn new(config: ParquetConfig) -> Result<Self> {
        // Create output directory if it doesn't exist
        create_dir_all(&config.output_dir).context("Failed to create output directory")?;

        Ok(Self {
            config,
            current_file: None,
            rotation_time: SystemTime::now(),
        })
    }

    /// Check if file rotation is needed
    fn needs_rotation(&self) -> bool {
        let elapsed = SystemTime::now()
            .duration_since(self.rotation_time)
            .unwrap_or_default();

        elapsed.as_secs() >= self.config.rotation_interval_secs
    }

    /// Generate a new file path for the current rotation
    fn generate_file_path(&self) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let filename = format!("metrics_{}.parquet", timestamp);
        Path::new(&self.config.output_dir).join(filename)
    }

    /// Get current file path
    pub fn current_file_path(&self) -> Option<PathBuf> {
        self.current_file.clone()
    }

    /// Convert MetricBatch to Arrow RecordBatch
    pub fn metrics_to_record_batch(&self, batch: &MetricBatch) -> Result<RecordBatch> {
        let mut timestamps = Vec::new();
        let mut host_ids = Vec::new();
        let mut metric_types = Vec::new();
        let mut labels = Vec::new();
        let mut values = Vec::new();

        // Process GPU metrics
        for metric in &batch.gpu_metrics {
            let ts = metric.timestamp.as_ref().map(|t| t.seconds).unwrap_or(0);

            // Add multiple fields as separate rows
            let fields = vec![
                ("utilization", metric.utilization as f64),
                ("sm_occupancy", metric.sm_occupancy as f64),
                ("memory_used_gb", metric.memory_used_gb as f64),
                ("temperature_celsius", metric.temperature_celsius as f64),
                ("power_watts", metric.power_watts as f64),
            ];

            for (field_name, value) in fields {
                timestamps.push(ts);
                host_ids.push(metric.host_id.clone());
                metric_types.push(format!("gpu_{}", field_name));
                labels.push(format!("gpu={}", metric.gpu_id));
                values.push(value);
            }
        }

        // Process CPU metrics
        for metric in &batch.cpu_metrics {
            let ts = metric.timestamp.as_ref().map(|t| t.seconds).unwrap_or(0);

            let fields = vec![
                ("utilization", metric.utilization as f64),
                ("ipc", metric.ipc as f64),
                ("cache_misses", metric.cache_misses as f64),
            ];

            for (field_name, value) in fields {
                timestamps.push(ts);
                host_ids.push(metric.host_id.clone());
                metric_types.push(format!("cpu_{}", field_name));
                labels.push(format!("socket={}", metric.socket));
                values.push(value);
            }
        }

        // Process NIC metrics
        for metric in &batch.nic_metrics {
            let ts = metric.timestamp.as_ref().map(|t| t.seconds).unwrap_or(0);

            let fields = vec![
                ("rx_gbps", metric.rx_gbps as f64),
                ("tx_gbps", metric.tx_gbps as f64),
                ("errors", metric.errors as f64),
                ("drops", metric.drops as f64),
            ];

            for (field_name, value) in fields {
                timestamps.push(ts);
                host_ids.push(metric.host_id.clone());
                metric_types.push(format!("nic_{}", field_name));
                labels.push(format!("interface={}", metric.interface));
                values.push(value);
            }
        }

        // Create schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("host_id", DataType::Utf8, false),
            Field::new("metric_type", DataType::Utf8, false),
            Field::new("labels", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
        ]));

        // Create arrays
        let timestamp_array = Arc::new(Int64Array::from(timestamps));
        let host_id_array = Arc::new(StringArray::from(host_ids));
        let metric_type_array = Arc::new(StringArray::from(metric_types));
        let labels_array = Arc::new(StringArray::from(labels));
        let values_array = Arc::new(Float64Array::from(values));

        RecordBatch::try_new(
            schema,
            vec![
                timestamp_array,
                host_id_array,
                metric_type_array,
                labels_array,
                values_array,
            ],
        )
        .context("Failed to create RecordBatch")
    }

    /// Get compression from config
    fn get_compression(&self) -> Compression {
        match self.config.compression.as_str() {
            "gzip" => Compression::GZIP(Default::default()),
            "snappy" => Compression::SNAPPY,
            "lz4" => Compression::LZ4,
            "zstd" => Compression::ZSTD(Default::default()),
            _ => Compression::UNCOMPRESSED,
        }
    }

    /// Write a metric batch to Parquet file
    pub async fn write_batch(&mut self, batch: &MetricBatch) -> Result<()> {
        // Check if we need to rotate the file
        if self.needs_rotation() || self.current_file.is_none() {
            self.current_file = Some(self.generate_file_path());
            self.rotation_time = SystemTime::now();
        }

        let file_path = self
            .current_file
            .as_ref()
            .ok_or_else(|| anyhow!("No current file"))?;

        // Convert batch to RecordBatch
        let record_batch = self.metrics_to_record_batch(batch)?;

        // Create writer properties with compression
        let props = WriterProperties::builder()
            .set_compression(self.get_compression())
            .build();

        // Write to file
        let file = File::create(file_path).context("Failed to create parquet file")?;

        let mut writer = ArrowWriter::try_new(file, record_batch.schema(), Some(props))
            .context("Failed to create ArrowWriter")?;

        writer
            .write(&record_batch)
            .context("Failed to write record batch")?;

        writer.close().context("Failed to close writer")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_compression_parsing() {
        let temp_dir = TempDir::new().unwrap();

        let config = ParquetConfig {
            output_dir: temp_dir.path().to_str().unwrap().to_string(),
            rotation_interval_secs: 3600,
            compression: "snappy".to_string(),
        };

        let writer = ParquetWriter::new(config).unwrap();
        matches!(writer.get_compression(), Compression::SNAPPY);
    }
}
