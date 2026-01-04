use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_vendor(c: &mut Criterion) {
    c.bench_function("vendor", |b| b.iter(|| black_box(1 + 1)));
}

criterion_group!(benches, benchmark_vendor);
criterion_main!(benches);
