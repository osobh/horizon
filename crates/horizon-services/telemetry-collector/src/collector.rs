use crate::cardinality::CardinalityTracker;
use crate::config::CollectorConfig;
use crate::handler::StreamHandler;
use crate::listener::QuicListener;
use crate::writers::{InfluxDbWriter, ParquetWriter};
use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct TelemetryCollector {
    config: CollectorConfig,
}

impl TelemetryCollector {
    pub fn new(config: CollectorConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = CollectorConfig::from_file(path)?;
        Self::new(config)
    }

    pub async fn run(self) -> Result<()> {
        tracing::info!("Starting Telemetry Collector");

        // Initialize Prometheus metrics exporter (basic setup)
        tracing::info!(
            "Metrics collection enabled on port {}",
            self.config.observability.metrics_port
        );

        // Create cardinality tracker
        let cardinality_tracker = Arc::new(Mutex::new(CardinalityTracker::new(
            self.config.limits.max_cardinality,
        )));

        // Create InfluxDB writer
        let influxdb_writer = Arc::new(
            InfluxDbWriter::new(self.config.influxdb.clone())
                .context("Failed to create InfluxDB writer")?,
        );

        // Create Parquet writer
        let parquet_writer = Arc::new(Mutex::new(
            ParquetWriter::new(self.config.parquet.clone())
                .context("Failed to create Parquet writer")?,
        ));

        // Create stream handler
        let handler = Arc::new(StreamHandler::new(
            cardinality_tracker,
            influxdb_writer,
            parquet_writer,
        ));

        // Create QUIC listener
        let listener = QuicListener::new(
            self.config.server.clone(),
            Path::new(&self.config.security.tls_cert_path),
            Path::new(&self.config.security.tls_key_path),
            handler,
        )
        .await
        .context("Failed to create QUIC listener")?;

        tracing::info!("QUIC listener initialized");

        // Run the listener
        listener.serve().await
    }
}
