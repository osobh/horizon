use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_reporting(c: &mut Criterion) {
    c.bench_function("report", |b| {
        b.iter(|| black_box(1 + 1))
    });
}

criterion_group!(benches, benchmark_reporting);
criterion_main!(benches);
