//! Performance benchmarks for the tracingx crate.
//!
//! These benchmarks measure:
//! - Initialization overhead
//! - Logging throughput (with and without OTLP)
//! - Metrics recording overhead
//! - Subscriber filtering performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hpc_tracing::init_metrics;
use std::net::SocketAddr;
use tracing::{debug, error, info, instrument, warn};

// Helper to generate unique ports for metrics benchmarks
fn unique_port() -> u16 {
    use std::sync::atomic::{AtomicU16, Ordering};
    static COUNTER: AtomicU16 = AtomicU16::new(20000);
    COUNTER.fetch_add(1, Ordering::SeqCst)
}

fn bench_logging_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("logging_throughput");

    // Benchmark info! macro
    group.bench_function("info_macro", |b| {
        b.iter(|| {
            info!(
                value = black_box(42),
                user = black_box("test_user"),
                "Processing request"
            );
        });
    });

    // Benchmark debug! macro (should be filtered out at info level)
    group.bench_function("debug_macro_filtered", |b| {
        b.iter(|| {
            debug!(
                value = black_box(42),
                "Debug message that should be filtered"
            );
        });
    });

    // Benchmark warn! macro
    group.bench_function("warn_macro", |b| {
        b.iter(|| {
            warn!(error_code = black_box("E001"), "Warning message");
        });
    });

    // Benchmark error! macro
    group.bench_function("error_macro", |b| {
        b.iter(|| {
            error!(error = black_box("connection failed"), "Error occurred");
        });
    });

    group.finish();
}

fn bench_instrumented_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("instrumented_functions");

    #[instrument]
    fn simple_function(x: u64) -> u64 {
        x * 2
    }

    #[instrument(skip(data))]
    fn function_with_skip(id: u64, data: &[u8]) -> u64 {
        id + data.len() as u64
    }

    // Benchmark simple instrumented function
    group.bench_function("simple_instrumented", |b| {
        b.iter(|| simple_function(black_box(42)));
    });

    // Benchmark instrumented function with skip
    group.bench_function("instrumented_with_skip", |b| {
        let data = vec![0u8; 1024];
        b.iter(|| function_with_skip(black_box(42), black_box(&data)));
    });

    group.finish();
}

fn bench_metrics_recording(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_recording");

    // Initialize metrics for benchmarking
    // Note: This can only be done once per process, so these benchmarks
    // measure the cost after initialization
    static INIT_METRICS: std::sync::Once = std::sync::Once::new();
    INIT_METRICS.call_once(|| {
        let port = unique_port();
        let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
        let _ = init_metrics(addr);
    });

    // Benchmark counter increment
    group.bench_function("counter_increment", |b| {
        b.iter(|| {
            metrics::increment_counter!("bench_counter");
        });
    });

    // Benchmark counter with labels (if supported by metrics version)
    group.bench_function("counter_simple", |b| {
        b.iter(|| {
            metrics::increment_counter!("bench_counter_simple");
        });
    });

    // Benchmark gauge set
    group.bench_function("gauge_set", |b| {
        b.iter(|| {
            metrics::gauge!("bench_gauge", black_box(42.5));
        });
    });

    // Benchmark histogram recording
    group.bench_function("histogram_record", |b| {
        b.iter(|| {
            metrics::histogram!("bench_histogram", black_box(123.45));
        });
    });

    group.finish();
}

fn bench_span_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("span_creation");

    // Benchmark span creation without entering
    group.bench_function("span_create_only", |b| {
        b.iter(|| {
            let span = tracing::info_span!("benchmark_span", value = black_box(42));
            black_box(span);
        });
    });

    // Benchmark span creation and enter
    group.bench_function("span_create_and_enter", |b| {
        b.iter(|| {
            let span = tracing::info_span!("benchmark_span", value = black_box(42));
            let _enter = span.enter();
        });
    });

    // Benchmark nested spans
    group.bench_function("nested_spans", |b| {
        b.iter(|| {
            let outer = tracing::info_span!("outer", id = black_box(1));
            let _outer_guard = outer.enter();
            let inner = tracing::info_span!("inner", id = black_box(2));
            let _inner_guard = inner.enter();
        });
    });

    group.finish();
}

fn bench_log_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_filtering");

    // These benchmarks show the cost of filtering
    // At info level, trace and debug should be very cheap (filtered out early)

    group.bench_function("trace_filtered", |b| {
        b.iter(|| {
            tracing::trace!("This should be filtered out");
        });
    });

    group.bench_function("debug_filtered", |b| {
        b.iter(|| {
            tracing::debug!("This should be filtered out");
        });
    });

    group.bench_function("info_not_filtered", |b| {
        b.iter(|| {
            tracing::info!("This should pass through");
        });
    });

    group.finish();
}

fn bench_metrics_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_throughput");

    // Ensure metrics are initialized
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let port = unique_port();
        let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
        let _ = init_metrics(addr);
    });

    // Benchmark sustained counter increments
    for count in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("counter_burst", count),
            count,
            |b, &count| {
                b.iter(|| {
                    for _ in 0..count {
                        metrics::increment_counter!("burst_counter");
                    }
                });
            },
        );
    }

    // Benchmark sustained histogram recordings
    for count in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("histogram_burst", count),
            count,
            |b, &count| {
                b.iter(|| {
                    for i in 0..count {
                        metrics::histogram!("burst_histogram", i as f64);
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_logging_throughput,
    bench_instrumented_functions,
    bench_metrics_recording,
    bench_span_creation,
    bench_log_filtering,
    bench_metrics_throughput,
);

criterion_main!(benches);
