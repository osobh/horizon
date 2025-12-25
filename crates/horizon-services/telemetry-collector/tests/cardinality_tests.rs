use telemetry_collector::cardinality::CardinalityTracker;
use hpc_types::{GpuMetric, CpuMetric, NicMetric, MetricBatch, Timestamp};

fn create_test_gpu_metric(host_id: &str, gpu_id: &str) -> GpuMetric {
    GpuMetric {
        host_id: host_id.to_string(),
        gpu_id: gpu_id.to_string(),
        timestamp: Some(Timestamp {
            seconds: 1234567890,
            nanos: 0,
        }),
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
    }
}

fn create_test_cpu_metric(host_id: &str, socket: u32) -> CpuMetric {
    CpuMetric {
        host_id: host_id.to_string(),
        timestamp: Some(Timestamp {
            seconds: 1234567890,
            nanos: 0,
        }),
        socket,
        utilization: 0.6,
        ipc: 1.8,
        cache_misses: 1000,
    }
}

fn create_test_nic_metric(host_id: &str, interface: &str) -> NicMetric {
    NicMetric {
        host_id: host_id.to_string(),
        timestamp: Some(Timestamp {
            seconds: 1234567890,
            nanos: 0,
        }),
        interface: interface.to_string(),
        rx_gbps: 10.0,
        tx_gbps: 8.0,
        errors: 0,
        drops: 0,
    }
}

#[test]
fn test_cardinality_tracker_creation() {
    let tracker = CardinalityTracker::new(1000);
    assert_eq!(tracker.current_cardinality(), 0);
    assert_eq!(tracker.max_cardinality(), 1000);
}

#[test]
fn test_track_single_metric_series() {
    let mut tracker = CardinalityTracker::new(1000);
    let metric = create_test_gpu_metric("host1", "gpu0");

    let result = tracker.track_gpu_metric(&metric);
    assert!(result.is_ok());
    assert_eq!(tracker.current_cardinality(), 1);
}

#[test]
fn test_track_multiple_unique_series() {
    let mut tracker = CardinalityTracker::new(1000);

    let metric1 = create_test_gpu_metric("host1", "gpu0");
    let metric2 = create_test_gpu_metric("host1", "gpu1");
    let metric3 = create_test_gpu_metric("host2", "gpu0");

    tracker.track_gpu_metric(&metric1).unwrap();
    tracker.track_gpu_metric(&metric2).unwrap();
    tracker.track_gpu_metric(&metric3).unwrap();

    assert_eq!(tracker.current_cardinality(), 3);
}

#[test]
fn test_track_duplicate_series() {
    let mut tracker = CardinalityTracker::new(1000);

    let metric1 = create_test_gpu_metric("host1", "gpu0");
    let metric2 = create_test_gpu_metric("host1", "gpu0");

    tracker.track_gpu_metric(&metric1).unwrap();
    tracker.track_gpu_metric(&metric2).unwrap();

    // Should only count as 1 series
    assert_eq!(tracker.current_cardinality(), 1);
}

#[test]
fn test_cardinality_limit_enforcement() {
    let mut tracker = CardinalityTracker::new(2);

    let metric1 = create_test_gpu_metric("host1", "gpu0");
    let metric2 = create_test_gpu_metric("host1", "gpu1");
    let metric3 = create_test_gpu_metric("host2", "gpu0");

    tracker.track_gpu_metric(&metric1).unwrap();
    tracker.track_gpu_metric(&metric2).unwrap();

    // Third unique series should be rejected
    let result = tracker.track_gpu_metric(&metric3);
    assert!(result.is_err());
    assert_eq!(tracker.current_cardinality(), 2);
}

#[test]
fn test_track_cpu_metrics() {
    let mut tracker = CardinalityTracker::new(1000);

    let metric1 = create_test_cpu_metric("host1", 0);
    let metric2 = create_test_cpu_metric("host1", 1);
    let metric3 = create_test_cpu_metric("host2", 0);

    tracker.track_cpu_metric(&metric1).unwrap();
    tracker.track_cpu_metric(&metric2).unwrap();
    tracker.track_cpu_metric(&metric3).unwrap();

    assert_eq!(tracker.current_cardinality(), 3);
}

#[test]
fn test_track_nic_metrics() {
    let mut tracker = CardinalityTracker::new(1000);

    let metric1 = create_test_nic_metric("host1", "eth0");
    let metric2 = create_test_nic_metric("host1", "eth1");
    let metric3 = create_test_nic_metric("host2", "eth0");

    tracker.track_nic_metric(&metric1).unwrap();
    tracker.track_nic_metric(&metric2).unwrap();
    tracker.track_nic_metric(&metric3).unwrap();

    assert_eq!(tracker.current_cardinality(), 3);
}

#[test]
fn test_track_metric_batch() {
    let mut tracker = CardinalityTracker::new(1000);

    let batch = MetricBatch {
        gpu_metrics: vec![
            create_test_gpu_metric("host1", "gpu0"),
            create_test_gpu_metric("host1", "gpu1"),
        ],
        cpu_metrics: vec![
            create_test_cpu_metric("host1", 0),
        ],
        nic_metrics: vec![
            create_test_nic_metric("host1", "eth0"),
        ],
    };

    let result = tracker.track_batch(&batch);
    assert!(result.is_ok());
    assert_eq!(tracker.current_cardinality(), 4);
}

#[test]
fn test_track_batch_exceeds_limit() {
    let mut tracker = CardinalityTracker::new(2);

    let batch = MetricBatch {
        gpu_metrics: vec![
            create_test_gpu_metric("host1", "gpu0"),
            create_test_gpu_metric("host1", "gpu1"),
            create_test_gpu_metric("host1", "gpu2"),
        ],
        cpu_metrics: vec![],
        nic_metrics: vec![],
    };

    let result = tracker.track_batch(&batch);
    assert!(result.is_err());
}

#[test]
fn test_series_id_generation() {
    let mut tracker = CardinalityTracker::new(1000);

    let gpu_metric = create_test_gpu_metric("host1", "gpu0");
    let cpu_metric = create_test_cpu_metric("host1", 0);
    let nic_metric = create_test_nic_metric("host1", "eth0");

    tracker.track_gpu_metric(&gpu_metric).unwrap();
    tracker.track_cpu_metric(&cpu_metric).unwrap();
    tracker.track_nic_metric(&nic_metric).unwrap();

    // Different metric types should create different series
    assert_eq!(tracker.current_cardinality(), 3);
}

#[test]
fn test_reset_cardinality() {
    let mut tracker = CardinalityTracker::new(1000);

    let metric = create_test_gpu_metric("host1", "gpu0");
    tracker.track_gpu_metric(&metric).unwrap();

    assert_eq!(tracker.current_cardinality(), 1);

    tracker.reset();
    assert_eq!(tracker.current_cardinality(), 0);
}

#[test]
fn test_get_series_info() {
    let mut tracker = CardinalityTracker::new(1000);

    let metric = create_test_gpu_metric("host1", "gpu0");
    tracker.track_gpu_metric(&metric).unwrap();

    let series = tracker.get_all_series();
    assert_eq!(series.len(), 1);
    assert!(series.iter().any(|s| s.contains("host1") && s.contains("gpu0")));
}

#[test]
fn test_concurrent_tracking() {
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let tracker = Arc::new(Mutex::new(CardinalityTracker::new(1000)));

        let mut handles = vec![];

        for i in 0..10 {
            let tracker_clone = tracker.clone();
            let handle = tokio::spawn(async move {
                let metric = create_test_gpu_metric(&format!("host{}", i), "gpu0");
                let mut t = tracker_clone.lock().await;
                t.track_gpu_metric(&metric)
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        let t = tracker.lock().await;
        assert_eq!(t.current_cardinality(), 10);
    });
}
