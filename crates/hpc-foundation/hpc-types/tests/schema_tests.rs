use prost::Message;
use hpc_types::common::v1::{HealthCheckRequest, HealthCheckResponse, Timestamp};
use hpc_types::telemetry::v1::{CpuMetric, GpuMetric, MetricBatch, NicMetric};
use hpc_types::{create_timestamp, timestamp_from_system_time, timestamp_to_system_time};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[test]
fn test_health_check_message_creation() {
    let request = HealthCheckRequest {};
    assert!(!request.encode_length_delimited_to_vec().is_empty());

    let response = HealthCheckResponse {
        healthy: true,
        version: "0.1.0".to_string(),
        uptime_seconds: 3600,
    };
    assert!(response.healthy);
    assert_eq!(response.version, "0.1.0");
    assert_eq!(response.uptime_seconds, 3600);
}

#[test]
fn test_timestamp_creation() {
    let ts = Timestamp {
        seconds: 1234567890,
        nanos: 123456789,
    };
    assert_eq!(ts.seconds, 1234567890);
    assert_eq!(ts.nanos, 123456789);
}

#[test]
fn test_timestamp_helper_creation() {
    let ts = create_timestamp(1234567890, 123456789);
    assert_eq!(ts.seconds, 1234567890);
    assert_eq!(ts.nanos, 123456789);
}

#[test]
fn test_timestamp_from_system_time() {
    let system_time = UNIX_EPOCH + Duration::from_secs(1234567890);
    let ts = timestamp_from_system_time(system_time);
    assert_eq!(ts.seconds, 1234567890);
    assert!(ts.nanos >= 0 && ts.nanos < 1_000_000_000);
}

#[test]
fn test_timestamp_to_system_time() {
    let ts = Timestamp {
        seconds: 1234567890,
        nanos: 500_000_000,
    };
    let system_time = timestamp_to_system_time(&ts);
    let duration = system_time.duration_since(UNIX_EPOCH).unwrap();
    assert_eq!(duration.as_secs(), 1234567890);
    assert_eq!(duration.subsec_nanos(), 500_000_000);
}

#[test]
fn test_timestamp_roundtrip() {
    let original = SystemTime::now();
    let ts = timestamp_from_system_time(original);
    let recovered = timestamp_to_system_time(&ts);

    let original_duration = original.duration_since(UNIX_EPOCH).unwrap();
    let recovered_duration = recovered.duration_since(UNIX_EPOCH).unwrap();

    // Should be within 1 nanosecond (accounting for truncation)
    let diff = if original_duration > recovered_duration {
        original_duration - recovered_duration
    } else {
        recovered_duration - original_duration
    };
    assert!(diff.as_nanos() < 2);
}

#[test]
fn test_gpu_metric_creation() {
    let ts = create_timestamp(1234567890, 0);
    let metric = GpuMetric {
        host_id: "host-001".to_string(),
        gpu_id: "gpu-0".to_string(),
        timestamp: Some(ts),
        utilization: 87.5,
        sm_occupancy: 92.3,
        memory_used_gb: 64.5,
        memory_total_gb: 80.0,
        pcie_tx_gbps: 12.5,
        pcie_rx_gbps: 11.8,
        nvlink_bandwidth_gbps: 300.0,
        temperature_celsius: 68.5,
        power_watts: 350.0,
        ecc_errors: false,
        mig_profile: "1g.10gb".to_string(),
    };

    assert_eq!(metric.host_id, "host-001");
    assert_eq!(metric.gpu_id, "gpu-0");
    assert_eq!(metric.utilization, 87.5);
    assert_eq!(metric.memory_used_gb, 64.5);
    assert!(!metric.ecc_errors);
}

#[test]
fn test_cpu_metric_creation() {
    let ts = create_timestamp(1234567890, 0);
    let metric = CpuMetric {
        host_id: "host-001".to_string(),
        timestamp: Some(ts),
        socket: 0,
        utilization: 45.6,
        ipc: 2.3,
        cache_misses: 1_234_567,
    };

    assert_eq!(metric.host_id, "host-001");
    assert_eq!(metric.socket, 0);
    assert_eq!(metric.utilization, 45.6);
    assert_eq!(metric.cache_misses, 1_234_567);
}

#[test]
fn test_nic_metric_creation() {
    let ts = create_timestamp(1234567890, 0);
    let metric = NicMetric {
        host_id: "host-001".to_string(),
        timestamp: Some(ts),
        interface: "eth0".to_string(),
        rx_gbps: 8.5,
        tx_gbps: 7.2,
        errors: 0,
        drops: 0,
    };

    assert_eq!(metric.host_id, "host-001");
    assert_eq!(metric.interface, "eth0");
    assert_eq!(metric.rx_gbps, 8.5);
    assert_eq!(metric.errors, 0);
}

#[test]
fn test_metric_batch_creation() {
    let ts = create_timestamp(1234567890, 0);

    let gpu_metric = GpuMetric {
        host_id: "host-001".to_string(),
        gpu_id: "gpu-0".to_string(),
        timestamp: Some(ts.clone()),
        utilization: 87.5,
        sm_occupancy: 92.3,
        memory_used_gb: 64.5,
        memory_total_gb: 80.0,
        pcie_tx_gbps: 12.5,
        pcie_rx_gbps: 11.8,
        nvlink_bandwidth_gbps: 300.0,
        temperature_celsius: 68.5,
        power_watts: 350.0,
        ecc_errors: false,
        mig_profile: "".to_string(),
    };

    let cpu_metric = CpuMetric {
        host_id: "host-001".to_string(),
        timestamp: Some(ts.clone()),
        socket: 0,
        utilization: 45.6,
        ipc: 2.3,
        cache_misses: 1_234_567,
    };

    let nic_metric = NicMetric {
        host_id: "host-001".to_string(),
        timestamp: Some(ts),
        interface: "eth0".to_string(),
        rx_gbps: 8.5,
        tx_gbps: 7.2,
        errors: 0,
        drops: 0,
    };

    let batch = MetricBatch {
        gpu_metrics: vec![gpu_metric],
        cpu_metrics: vec![cpu_metric],
        nic_metrics: vec![nic_metric],
    };

    assert_eq!(batch.gpu_metrics.len(), 1);
    assert_eq!(batch.cpu_metrics.len(), 1);
    assert_eq!(batch.nic_metrics.len(), 1);
}

#[test]
fn test_protobuf_serialization() {
    let ts = create_timestamp(1234567890, 0);
    let metric = GpuMetric {
        host_id: "host-001".to_string(),
        gpu_id: "gpu-0".to_string(),
        timestamp: Some(ts),
        utilization: 87.5,
        sm_occupancy: 92.3,
        memory_used_gb: 64.5,
        memory_total_gb: 80.0,
        pcie_tx_gbps: 12.5,
        pcie_rx_gbps: 11.8,
        nvlink_bandwidth_gbps: 300.0,
        temperature_celsius: 68.5,
        power_watts: 350.0,
        ecc_errors: false,
        mig_profile: "".to_string(),
    };

    // Serialize to bytes
    let mut buf = Vec::new();
    metric.encode(&mut buf).expect("Failed to encode");

    assert!(!buf.is_empty());

    // Deserialize back
    let decoded = GpuMetric::decode(&buf[..]).expect("Failed to decode");

    assert_eq!(decoded.host_id, "host-001");
    assert_eq!(decoded.gpu_id, "gpu-0");
    assert_eq!(decoded.utilization, 87.5);
    assert_eq!(decoded.memory_used_gb, 64.5);
}

#[test]
fn test_protobuf_serialization_batch() {
    let ts = create_timestamp(1234567890, 0);

    let mut gpu_metrics = Vec::new();
    for i in 0..10 {
        gpu_metrics.push(GpuMetric {
            host_id: format!("host-{:03}", i / 8),
            gpu_id: format!("gpu-{}", i % 8),
            timestamp: Some(ts.clone()),
            utilization: 50.0 + i as f32,
            sm_occupancy: 90.0,
            memory_used_gb: 40.0,
            memory_total_gb: 80.0,
            pcie_tx_gbps: 10.0,
            pcie_rx_gbps: 10.0,
            nvlink_bandwidth_gbps: 300.0,
            temperature_celsius: 65.0,
            power_watts: 300.0,
            ecc_errors: false,
            mig_profile: "".to_string(),
        });
    }

    let batch = MetricBatch {
        gpu_metrics,
        cpu_metrics: vec![],
        nic_metrics: vec![],
    };

    let mut buf = Vec::new();
    batch.encode(&mut buf).expect("Failed to encode batch");

    let decoded = MetricBatch::decode(&buf[..]).expect("Failed to decode batch");
    assert_eq!(decoded.gpu_metrics.len(), 10);
    assert_eq!(decoded.gpu_metrics[5].utilization, 55.0);
}

#[test]
fn test_json_serialization() {
    let ts = create_timestamp(1234567890, 123456789);
    let metric = GpuMetric {
        host_id: "host-001".to_string(),
        gpu_id: "gpu-0".to_string(),
        timestamp: Some(ts),
        utilization: 87.5,
        sm_occupancy: 92.3,
        memory_used_gb: 64.5,
        memory_total_gb: 80.0,
        pcie_tx_gbps: 12.5,
        pcie_rx_gbps: 11.8,
        nvlink_bandwidth_gbps: 300.0,
        temperature_celsius: 68.5,
        power_watts: 350.0,
        ecc_errors: false,
        mig_profile: "1g.10gb".to_string(),
    };

    let json = serde_json::to_string(&metric).expect("Failed to serialize to JSON");
    assert!(json.contains("host-001"));
    assert!(json.contains("gpu-0"));

    let decoded: GpuMetric = serde_json::from_str(&json).expect("Failed to deserialize from JSON");
    assert_eq!(decoded.host_id, "host-001");
    assert_eq!(decoded.utilization, 87.5);
}

#[test]
fn test_default_values() {
    let metric = GpuMetric::default();
    assert_eq!(metric.host_id, "");
    assert_eq!(metric.gpu_id, "");
    assert_eq!(metric.utilization, 0.0);
    assert!(!metric.ecc_errors);
    assert!(metric.timestamp.is_none());
}

#[test]
fn test_nested_message_timestamp() {
    let ts = create_timestamp(1234567890, 999999999);
    let metric = GpuMetric {
        timestamp: Some(ts.clone()),
        ..Default::default()
    };

    assert!(metric.timestamp.is_some());
    let extracted = metric.timestamp.unwrap();
    assert_eq!(extracted.seconds, 1234567890);
    assert_eq!(extracted.nanos, 999999999);
}

#[test]
fn test_large_batch_serialization() {
    let ts = create_timestamp(1234567890, 0);
    let mut gpu_metrics = Vec::new();

    for i in 0..1000 {
        gpu_metrics.push(GpuMetric {
            host_id: format!("host-{:03}", i / 8),
            gpu_id: format!("gpu-{}", i % 8),
            timestamp: Some(ts.clone()),
            utilization: (i % 100) as f32,
            sm_occupancy: 90.0,
            memory_used_gb: 40.0,
            memory_total_gb: 80.0,
            pcie_tx_gbps: 10.0,
            pcie_rx_gbps: 10.0,
            nvlink_bandwidth_gbps: 300.0,
            temperature_celsius: 65.0,
            power_watts: 300.0,
            ecc_errors: false,
            mig_profile: "".to_string(),
        });
    }

    let batch = MetricBatch {
        gpu_metrics,
        cpu_metrics: vec![],
        nic_metrics: vec![],
    };

    let mut buf = Vec::new();
    batch
        .encode(&mut buf)
        .expect("Failed to encode large batch");

    assert!(!buf.is_empty());

    let decoded = MetricBatch::decode(&buf[..]).expect("Failed to decode large batch");
    assert_eq!(decoded.gpu_metrics.len(), 1000);
}

#[test]
fn test_protobuf_roundtrip_all_types() {
    let ts = create_timestamp(1234567890, 0);

    let gpu = GpuMetric {
        host_id: "host-001".to_string(),
        gpu_id: "gpu-0".to_string(),
        timestamp: Some(ts.clone()),
        utilization: 87.5,
        sm_occupancy: 92.3,
        memory_used_gb: 64.5,
        memory_total_gb: 80.0,
        pcie_tx_gbps: 12.5,
        pcie_rx_gbps: 11.8,
        nvlink_bandwidth_gbps: 300.0,
        temperature_celsius: 68.5,
        power_watts: 350.0,
        ecc_errors: false,
        mig_profile: "".to_string(),
    };

    let cpu = CpuMetric {
        host_id: "host-001".to_string(),
        timestamp: Some(ts.clone()),
        socket: 0,
        utilization: 45.6,
        ipc: 2.3,
        cache_misses: 1_234_567,
    };

    let nic = NicMetric {
        host_id: "host-001".to_string(),
        timestamp: Some(ts),
        interface: "eth0".to_string(),
        rx_gbps: 8.5,
        tx_gbps: 7.2,
        errors: 123,
        drops: 456,
    };

    let batch = MetricBatch {
        gpu_metrics: vec![gpu],
        cpu_metrics: vec![cpu],
        nic_metrics: vec![nic],
    };

    let mut buf = Vec::new();
    batch.encode(&mut buf).unwrap();

    let decoded = MetricBatch::decode(&buf[..]).unwrap();

    assert_eq!(decoded.gpu_metrics[0].host_id, "host-001");
    assert_eq!(decoded.cpu_metrics[0].socket, 0);
    assert_eq!(decoded.nic_metrics[0].interface, "eth0");
    assert_eq!(decoded.nic_metrics[0].errors, 123);
    assert_eq!(decoded.nic_metrics[0].drops, 456);
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_timestamp_random_roundtrip(seconds in 0i64..2_000_000_000i64, nanos in 0i32..1_000_000_000i32) {
            let ts = create_timestamp(seconds, nanos);
            let system_time = timestamp_to_system_time(&ts);
            let ts2 = timestamp_from_system_time(system_time);

            assert_eq!(ts.seconds, ts2.seconds);
            // Nanos might differ slightly due to SystemTime precision
            let nanos_diff = (ts.nanos - ts2.nanos).abs();
            assert!(nanos_diff < 1000); // Within 1 microsecond
        }

        #[test]
        fn test_gpu_metric_serialization_random(
            utilization in 0.0f32..100.0f32,
            memory_used in 0.0f32..1000.0f32,
            temperature in 0.0f32..100.0f32
        ) {
            let ts = create_timestamp(1234567890, 0);
            let metric = GpuMetric {
                host_id: "test".to_string(),
                gpu_id: "gpu-0".to_string(),
                timestamp: Some(ts),
                utilization,
                sm_occupancy: 90.0,
                memory_used_gb: memory_used,
                memory_total_gb: 1000.0,
                pcie_tx_gbps: 10.0,
                pcie_rx_gbps: 10.0,
                nvlink_bandwidth_gbps: 300.0,
                temperature_celsius: temperature,
                power_watts: 300.0,
                ecc_errors: false,
                mig_profile: "".to_string(),
            };

            let mut buf = Vec::new();
            metric.encode(&mut buf).unwrap();
            let decoded = GpuMetric::decode(&buf[..]).unwrap();

            assert_eq!(decoded.utilization, utilization);
            assert_eq!(decoded.memory_used_gb, memory_used);
            assert_eq!(decoded.temperature_celsius, temperature);
        }
    }
}

#[test]
fn test_message_size_estimation() {
    let ts = create_timestamp(1234567890, 0);
    let metric = GpuMetric {
        host_id: "host-001".to_string(),
        gpu_id: "gpu-0".to_string(),
        timestamp: Some(ts),
        utilization: 87.5,
        sm_occupancy: 92.3,
        memory_used_gb: 64.5,
        memory_total_gb: 80.0,
        pcie_tx_gbps: 12.5,
        pcie_rx_gbps: 11.8,
        nvlink_bandwidth_gbps: 300.0,
        temperature_celsius: 68.5,
        power_watts: 350.0,
        ecc_errors: false,
        mig_profile: "".to_string(),
    };

    let encoded_size = metric.encoded_len();
    let mut buf = Vec::new();
    metric.encode(&mut buf).unwrap();

    assert_eq!(encoded_size, buf.len());
}

#[test]
fn test_empty_batch() {
    let batch = MetricBatch {
        gpu_metrics: vec![],
        cpu_metrics: vec![],
        nic_metrics: vec![],
    };

    let mut buf = Vec::new();
    batch.encode(&mut buf).unwrap();

    let decoded = MetricBatch::decode(&buf[..]).unwrap();
    assert_eq!(decoded.gpu_metrics.len(), 0);
    assert_eq!(decoded.cpu_metrics.len(), 0);
    assert_eq!(decoded.nic_metrics.len(), 0);
}

#[test]
fn test_health_check_roundtrip() {
    let response = HealthCheckResponse {
        healthy: true,
        version: "0.1.0".to_string(),
        uptime_seconds: 86400,
    };

    let mut buf = Vec::new();
    response.encode(&mut buf).unwrap();

    let decoded = HealthCheckResponse::decode(&buf[..]).unwrap();
    assert!(decoded.healthy);
    assert_eq!(decoded.version, "0.1.0");
    assert_eq!(decoded.uptime_seconds, 86400);
}
