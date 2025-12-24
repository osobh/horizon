use telemetry_collector::writers::{InfluxDbWriter, ParquetWriter};
use telemetry_collector::config::{InfluxDbConfig, ParquetConfig};
use hpc_types::{GpuMetric, CpuMetric, NicMetric, MetricBatch, Timestamp};
use tempfile::TempDir;

fn create_test_batch() -> MetricBatch {
    MetricBatch {
        gpu_metrics: vec![GpuMetric {
            host_id: "host1".to_string(),
            gpu_id: "gpu0".to_string(),
            timestamp: Some(Timestamp { seconds: 1234567890, nanos: 0 }),
            utilization: 0.75,
            sm_occupancy: 0.80,
            memory_used_gb: 10.0,
            memory_total_gb: 16.0,
            pcie_tx_gbps: 5.0,
            pcie_rx_gbps: 6.0,
            nvlink_bandwidth_gbps: 100.0,
            temperature_celsius: 65.0,
            power_watts: 250.0,
            ecc_errors: false,
            mig_profile: "".to_string(),
        }],
        cpu_metrics: vec![CpuMetric {
            host_id: "host1".to_string(),
            timestamp: Some(Timestamp { seconds: 1234567890, nanos: 0 }),
            socket: 0,
            utilization: 0.6,
            ipc: 1.8,
            cache_misses: 1000,
        }],
        nic_metrics: vec![NicMetric {
            host_id: "host1".to_string(),
            timestamp: Some(Timestamp { seconds: 1234567890, nanos: 0 }),
            interface: "eth0".to_string(),
            rx_gbps: 10.0,
            tx_gbps: 8.0,
            errors: 0,
            drops: 0,
        }],
    }
}

// InfluxDB Writer Tests

#[test]
fn test_influxdb_writer_creation() {
    let config = InfluxDbConfig {
        url: "http://localhost:8086".to_string(),
        org: "horizon".to_string(),
        bucket: "telemetry".to_string(),
        token: "test-token".to_string(),
    };

    let writer = InfluxDbWriter::new(config);
    assert!(writer.is_ok());
}

#[test]
fn test_influxdb_line_protocol_gpu() {
    let metric = GpuMetric {
        host_id: "host1".to_string(),
        gpu_id: "gpu0".to_string(),
        timestamp: Some(Timestamp { seconds: 1234567890, nanos: 0 }),
        utilization: 0.75,
        sm_occupancy: 0.80,
        memory_used_gb: 10.0,
        memory_total_gb: 16.0,
        pcie_tx_gbps: 5.0,
        pcie_rx_gbps: 6.0,
        nvlink_bandwidth_gbps: 100.0,
        temperature_celsius: 65.0,
        power_watts: 250.0,
        ecc_errors: false,
        mig_profile: "".to_string(),
    };

    let line = InfluxDbWriter::gpu_metric_to_line_protocol(&metric);
    assert!(line.contains("gpu_metrics"));
    assert!(line.contains("host=host1"));
    assert!(line.contains("gpu=gpu0"));
    assert!(line.contains("utilization=0.75"));
}

#[test]
fn test_influxdb_line_protocol_cpu() {
    let metric = CpuMetric {
        host_id: "host1".to_string(),
        timestamp: Some(Timestamp { seconds: 1234567890, nanos: 0 }),
        socket: 0,
        utilization: 0.6,
        ipc: 1.8,
        cache_misses: 1000,
    };

    let line = InfluxDbWriter::cpu_metric_to_line_protocol(&metric);
    assert!(line.contains("cpu_metrics"));
    assert!(line.contains("host=host1"));
    assert!(line.contains("socket=0"));
    assert!(line.contains("utilization=0.6"));
}

#[test]
fn test_influxdb_line_protocol_nic() {
    let metric = NicMetric {
        host_id: "host1".to_string(),
        timestamp: Some(Timestamp { seconds: 1234567890, nanos: 0 }),
        interface: "eth0".to_string(),
        rx_gbps: 10.0,
        tx_gbps: 8.0,
        errors: 0,
        drops: 0,
    };

    let line = InfluxDbWriter::nic_metric_to_line_protocol(&metric);
    assert!(line.contains("nic_metrics"));
    assert!(line.contains("host=host1"));
    assert!(line.contains("interface=eth0"));
    assert!(line.contains("rx_gbps=10"));
}

#[tokio::test]
async fn test_influxdb_batch_conversion() {
    let config = InfluxDbConfig {
        url: "http://localhost:8086".to_string(),
        org: "horizon".to_string(),
        bucket: "telemetry".to_string(),
        token: "test-token".to_string(),
    };

    let writer = InfluxDbWriter::new(config).unwrap();
    let batch = create_test_batch();
    let lines = writer.batch_to_line_protocol(&batch);

    assert_eq!(lines.len(), 3); // 1 GPU + 1 CPU + 1 NIC
    assert!(lines[0].contains("gpu_metrics"));
    assert!(lines[1].contains("cpu_metrics"));
    assert!(lines[2].contains("nic_metrics"));
}

// Parquet Writer Tests

#[test]
fn test_parquet_writer_creation() {
    let temp_dir = TempDir::new().unwrap();
    let config = ParquetConfig {
        output_dir: temp_dir.path().to_str().unwrap().to_string(),
        rotation_interval_secs: 3600,
        compression: "snappy".to_string(),
    };

    let writer = ParquetWriter::new(config);
    assert!(writer.is_ok());
}

#[tokio::test]
async fn test_parquet_write_batch() {
    let temp_dir = TempDir::new().unwrap();
    let config = ParquetConfig {
        output_dir: temp_dir.path().to_str().unwrap().to_string(),
        rotation_interval_secs: 3600,
        compression: "snappy".to_string(),
    };

    let mut writer = ParquetWriter::new(config).unwrap();
    let batch = create_test_batch();

    let result = writer.write_batch(&batch).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_parquet_file_rotation() {
    let temp_dir = TempDir::new().unwrap();
    let config = ParquetConfig {
        output_dir: temp_dir.path().to_str().unwrap().to_string(),
        rotation_interval_secs: 1, // 1 second for testing
        compression: "snappy".to_string(),
    };

    let mut writer = ParquetWriter::new(config).unwrap();
    let batch = create_test_batch();

    // Write first batch
    writer.write_batch(&batch).await.unwrap();
    let first_file = writer.current_file_path();

    // Wait for rotation interval
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Write second batch - should trigger rotation
    writer.write_batch(&batch).await.unwrap();
    let second_file = writer.current_file_path();

    assert_ne!(first_file, second_file);
}

#[tokio::test]
async fn test_parquet_compression_types() {
    let temp_dir = TempDir::new().unwrap();

    for compression in &["snappy", "gzip", "lz4", "zstd", "none"] {
        let config = ParquetConfig {
            output_dir: temp_dir.path().to_str().unwrap().to_string(),
            rotation_interval_secs: 3600,
            compression: compression.to_string(),
        };

        let writer = ParquetWriter::new(config);
        assert!(writer.is_ok(), "Failed for compression: {}", compression);
    }
}

#[tokio::test]
async fn test_parquet_schema_conversion() {
    let temp_dir = TempDir::new().unwrap();
    let config = ParquetConfig {
        output_dir: temp_dir.path().to_str().unwrap().to_string(),
        rotation_interval_secs: 3600,
        compression: "snappy".to_string(),
    };

    let writer = ParquetWriter::new(config).unwrap();
    let batch = create_test_batch();

    let record_batch = writer.metrics_to_record_batch(&batch);
    assert!(record_batch.is_ok());

    let rb = record_batch.unwrap();
    assert!(rb.num_rows() > 0);
    assert!(rb.num_columns() > 0);
}

#[tokio::test]
async fn test_parquet_large_batch() {
    let temp_dir = TempDir::new().unwrap();
    let config = ParquetConfig {
        output_dir: temp_dir.path().to_str().unwrap().to_string(),
        rotation_interval_secs: 3600,
        compression: "snappy".to_string(),
    };

    let mut writer = ParquetWriter::new(config).unwrap();

    // Create a large batch
    let mut gpu_metrics = Vec::new();
    for i in 0..1000 {
        gpu_metrics.push(GpuMetric {
            host_id: format!("host{}", i % 10),
            gpu_id: format!("gpu{}", i % 8),
            timestamp: Some(Timestamp { seconds: 1234567890 + i, nanos: 0 }),
            utilization: (i % 100) as f32 / 100.0,
            sm_occupancy: 0.80,
            memory_used_gb: 10.0,
            memory_total_gb: 16.0,
            pcie_tx_gbps: 5.0,
            pcie_rx_gbps: 6.0,
            nvlink_bandwidth_gbps: 100.0,
            temperature_celsius: 65.0,
            power_watts: 250.0,
            ecc_errors: false,
            mig_profile: "".to_string(),
        });
    }

    let batch = MetricBatch {
        gpu_metrics,
        cpu_metrics: vec![],
        nic_metrics: vec![],
    };

    let result = writer.write_batch(&batch).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_parquet_concurrent_writes() {
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let temp_dir = TempDir::new().unwrap();
    let config = ParquetConfig {
        output_dir: temp_dir.path().to_str().unwrap().to_string(),
        rotation_interval_secs: 3600,
        compression: "snappy".to_string(),
    };

    let writer = Arc::new(Mutex::new(ParquetWriter::new(config).unwrap()));
    let mut handles = vec![];

    for _ in 0..10 {
        let writer_clone = writer.clone();
        let handle = tokio::spawn(async move {
            let batch = create_test_batch();
            let mut w = writer_clone.lock().await;
            w.write_batch(&batch).await
        });
        handles.push(handle);
    }

    for handle in handles {
        assert!(handle.await.unwrap().is_ok());
    }
}
