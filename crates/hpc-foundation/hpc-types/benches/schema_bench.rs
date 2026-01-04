use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use hpc_types::telemetry::v1::{GpuMetric, MetricBatch};
use hpc_types::{create_timestamp, timestamp_from_system_time};
use prost::Message;
use std::time::SystemTime;

fn create_gpu_metric(id: usize) -> GpuMetric {
    let ts = create_timestamp(1234567890, 0);
    GpuMetric {
        host_id: format!("host-{:03}", id / 8),
        gpu_id: format!("gpu-{}", id % 8),
        timestamp: Some(ts),
        utilization: 50.0 + (id % 50) as f32,
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
    }
}

fn bench_single_message_serialization(c: &mut Criterion) {
    let metric = create_gpu_metric(0);

    c.bench_function("serialize_single_gpu_metric", |b| {
        b.iter(|| {
            let mut buf = Vec::new();
            black_box(&metric).encode(&mut buf).unwrap();
            black_box(buf);
        });
    });
}

fn bench_single_message_deserialization(c: &mut Criterion) {
    let metric = create_gpu_metric(0);
    let mut buf = Vec::new();
    metric.encode(&mut buf).unwrap();

    c.bench_function("deserialize_single_gpu_metric", |b| {
        b.iter(|| {
            let decoded = GpuMetric::decode(black_box(&buf[..])).unwrap();
            black_box(decoded);
        });
    });
}

fn bench_batch_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_serialization");

    for size in [10, 100, 1000].iter() {
        let gpu_metrics: Vec<GpuMetric> = (0..*size).map(create_gpu_metric).collect();
        let batch = MetricBatch {
            gpu_metrics,
            cpu_metrics: vec![],
            nic_metrics: vec![],
        };

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(format!("size_{}", size), size, |b, _| {
            b.iter(|| {
                let mut buf = Vec::new();
                black_box(&batch).encode(&mut buf).unwrap();
                black_box(buf);
            });
        });
    }
    group.finish();
}

fn bench_batch_deserialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_deserialization");

    for size in [10, 100, 1000].iter() {
        let gpu_metrics: Vec<GpuMetric> = (0..*size).map(create_gpu_metric).collect();
        let batch = MetricBatch {
            gpu_metrics,
            cpu_metrics: vec![],
            nic_metrics: vec![],
        };

        let mut buf = Vec::new();
        batch.encode(&mut buf).unwrap();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(format!("size_{}", size), size, |b, _| {
            b.iter(|| {
                let decoded = MetricBatch::decode(black_box(&buf[..])).unwrap();
                black_box(decoded);
            });
        });
    }
    group.finish();
}

fn bench_json_serialization(c: &mut Criterion) {
    let metric = create_gpu_metric(0);

    c.bench_function("serialize_to_json", |b| {
        b.iter(|| {
            let json = serde_json::to_string(black_box(&metric)).unwrap();
            black_box(json);
        });
    });
}

fn bench_json_deserialization(c: &mut Criterion) {
    let metric = create_gpu_metric(0);
    let json = serde_json::to_string(&metric).unwrap();

    c.bench_function("deserialize_from_json", |b| {
        b.iter(|| {
            let decoded: GpuMetric = serde_json::from_str(black_box(&json)).unwrap();
            black_box(decoded);
        });
    });
}

fn bench_timestamp_creation(c: &mut Criterion) {
    c.bench_function("timestamp_from_system_time", |b| {
        b.iter(|| {
            let ts = timestamp_from_system_time(black_box(SystemTime::now()));
            black_box(ts);
        });
    });
}

fn bench_message_size(c: &mut Criterion) {
    let metric = create_gpu_metric(0);

    c.bench_function("message_encoded_len", |b| {
        b.iter(|| {
            let size = black_box(&metric).encoded_len();
            black_box(size);
        });
    });
}

criterion_group!(
    benches,
    bench_single_message_serialization,
    bench_single_message_deserialization,
    bench_batch_serialization,
    bench_batch_deserialization,
    bench_json_serialization,
    bench_json_deserialization,
    bench_timestamp_creation,
    bench_message_size
);
criterion_main!(benches);
