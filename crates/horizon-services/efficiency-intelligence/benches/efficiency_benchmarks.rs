use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_analysis(c: &mut Criterion) {
    c.bench_function("analyze", |b| {
        b.iter(|| black_box(1 + 1))
    });
}

criterion_group!(benches, benchmark_analysis);
criterion_main!(benches);
