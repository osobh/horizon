use telemetry_collector::config::CollectorConfig;
use telemetry_collector::cardinality::CardinalityTracker;
use telemetry_collector::writers::{InfluxDbWriter, ParquetWriter};
use telemetry_collector::handler::StreamHandler;
use hpc_types::{GpuMetric, MetricBatch, Timestamp};
use std::sync::Arc;
use tokio::sync::Mutex;
use tempfile::TempDir;

#[tokio::test]
async fn test_end_to_end_metric_flow() {
    // Create temporary directory for Parquet files
    let temp_dir = TempDir::new().unwrap();

    // Create configuration
    let mut config = CollectorConfig::default();
    config.influxdb.token = "test-token".to_string();
    config.parquet.output_dir = temp_dir.path().to_str().unwrap().to_string();

    // Setup components
    let cardinality_tracker = Arc::new(Mutex::new(
        CardinalityTracker::new(config.limits.max_cardinality)
    ));

    let influxdb_writer = Arc::new(
        InfluxDbWriter::new(config.influxdb.clone()).unwrap()
    );

    let parquet_writer = Arc::new(Mutex::new(
        ParquetWriter::new(config.parquet.clone()).unwrap()
    ));

    let handler = StreamHandler::new(
        cardinality_tracker.clone(),
        influxdb_writer,
        parquet_writer,
    );

    // Create test batch
    let batch = MetricBatch {
        gpu_metrics: vec![GpuMetric {
            host_id: "test-host".to_string(),
            gpu_id: "gpu0".to_string(),
            timestamp: Some(Timestamp { seconds: 1234567890, nanos: 0 }),
            utilization: 0.85,
            sm_occupancy: 0.90,
            memory_used_gb: 12.0,
            memory_total_gb: 16.0,
            pcie_tx_gbps: 5.0,
            pcie_rx_gbps: 6.0,
            nvlink_bandwidth_gbps: 100.0,
            temperature_celsius: 65.0,
            power_watts: 250.0,
            ecc_errors: false,
            mig_profile: "".to_string(),
        }],
        cpu_metrics: vec![],
        nic_metrics: vec![],
    };

    // Test encoding/decoding
    let encoded = StreamHandler::encode_length_prefixed(&batch);
    let decoded = StreamHandler::decode_length_prefixed(&encoded).unwrap();
    assert_eq!(batch.gpu_metrics.len(), decoded.gpu_metrics.len());

    // Test batch handling
    let result = handler.handle_batch(batch).await;
    assert!(result.is_ok());

    // Verify cardinality was tracked
    let tracker = cardinality_tracker.lock().await;
    assert_eq!(tracker.current_cardinality(), 1);
}

#[tokio::test]
async fn test_cardinality_limit_enforcement() {
    let temp_dir = TempDir::new().unwrap();

    let mut config = CollectorConfig::default();
    config.influxdb.token = "test-token".to_string();
    config.parquet.output_dir = temp_dir.path().to_str().unwrap().to_string();
    config.limits.max_cardinality = 2; // Very low limit for testing

    let cardinality_tracker = Arc::new(Mutex::new(
        CardinalityTracker::new(config.limits.max_cardinality)
    ));

    let influxdb_writer = Arc::new(
        InfluxDbWriter::new(config.influxdb.clone()).unwrap()
    );

    let parquet_writer = Arc::new(Mutex::new(
        ParquetWriter::new(config.parquet.clone()).unwrap()
    ));

    let handler = StreamHandler::new(
        cardinality_tracker,
        influxdb_writer,
        parquet_writer,
    );

    // First batch should succeed
    let batch1 = MetricBatch {
        gpu_metrics: vec![
            GpuMetric {
                host_id: "host1".to_string(),
                gpu_id: "gpu0".to_string(),
                timestamp: Some(Timestamp { seconds: 1234567890, nanos: 0 }),
                ..Default::default()
            },
            GpuMetric {
                host_id: "host1".to_string(),
                gpu_id: "gpu1".to_string(),
                timestamp: Some(Timestamp { seconds: 1234567890, nanos: 0 }),
                ..Default::default()
            },
        ],
        cpu_metrics: vec![],
        nic_metrics: vec![],
    };

    assert!(handler.handle_batch(batch1).await.is_ok());

    // Second batch with new series should fail (exceeds cardinality)
    let batch2 = MetricBatch {
        gpu_metrics: vec![GpuMetric {
            host_id: "host2".to_string(),
            gpu_id: "gpu0".to_string(),
            timestamp: Some(Timestamp { seconds: 1234567890, nanos: 0 }),
            ..Default::default()
        }],
        cpu_metrics: vec![],
        nic_metrics: vec![],
    };

    assert!(handler.handle_batch(batch2).await.is_err());
}

#[tokio::test]
async fn test_concurrent_batch_handling() {
    let temp_dir = TempDir::new().unwrap();

    let mut config = CollectorConfig::default();
    config.influxdb.token = "test-token".to_string();
    config.parquet.output_dir = temp_dir.path().to_str().unwrap().to_string();

    let cardinality_tracker = Arc::new(Mutex::new(
        CardinalityTracker::new(config.limits.max_cardinality)
    ));

    let influxdb_writer = Arc::new(
        InfluxDbWriter::new(config.influxdb.clone()).unwrap()
    );

    let parquet_writer = Arc::new(Mutex::new(
        ParquetWriter::new(config.parquet.clone()).unwrap()
    ));

    let handler = Arc::new(StreamHandler::new(
        cardinality_tracker.clone(),
        influxdb_writer,
        parquet_writer,
    ));

    // Spawn multiple concurrent handlers
    let mut handles = vec![];
    for i in 0..10 {
        let handler_clone = handler.clone();
        let handle = tokio::spawn(async move {
            let batch = MetricBatch {
                gpu_metrics: vec![GpuMetric {
                    host_id: format!("host{}", i),
                    gpu_id: "gpu0".to_string(),
                    timestamp: Some(Timestamp { seconds: 1234567890, nanos: 0 }),
                    ..Default::default()
                }],
                cpu_metrics: vec![],
                nic_metrics: vec![],
            };
            handler_clone.handle_batch(batch).await
        });
        handles.push(handle);
    }

    // Wait for all to complete
    for handle in handles {
        assert!(handle.await.unwrap().is_ok());
    }

    // Verify all series were tracked
    let tracker = cardinality_tracker.lock().await;
    assert_eq!(tracker.current_cardinality(), 10);
}
