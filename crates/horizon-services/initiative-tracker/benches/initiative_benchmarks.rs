use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_tracking(c: &mut Criterion) {
    c.bench_function("track", |b| {
        b.iter(|| black_box(1 + 1))
    });
}

criterion_group!(benches, benchmark_tracking);
criterion_main!(benches);
