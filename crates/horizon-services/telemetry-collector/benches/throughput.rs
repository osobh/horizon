use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use telemetry_collector::cardinality::CardinalityTracker;
use telemetry_collector::handler::StreamHandler;
use horizon_hpc_types::{GpuMetric, MetricBatch, Timestamp};

fn create_batch(num_metrics: usize) -> MetricBatch {
    let mut gpu_metrics = Vec::new();
    for i in 0..num_metrics {
        gpu_metrics.push(GpuMetric {
            host_id: format!("host{}", i % 10),
            gpu_id: format!("gpu{}", i % 8),
            timestamp: Some(Timestamp { seconds: 1234567890 + i as i64, nanos: 0 }),
            utilization: (i % 100) as f32 / 100.0,
            ..Default::default()
        });
    }
    MetricBatch {
        gpu_metrics,
        cpu_metrics: vec![],
        nic_metrics: vec![],
    }
}

fn bench_cardinality_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("cardinality");

    for size in [100, 500, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut tracker = CardinalityTracker::new(100000);
                let batch = create_batch(size);
                tracker.track_batch(&batch).ok();
            });
        });
    }
    group.finish();
}

fn bench_encoding_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoding");

    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let batch = create_batch(size);
            b.iter(|| {
                let encoded = StreamHandler::encode_length_prefixed(&batch);
                StreamHandler::decode_length_prefixed(&encoded).ok();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cardinality_tracking, bench_encoding_decoding);
criterion_main!(benches);
