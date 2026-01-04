use criterion::{black_box, criterion_group, criterion_main, Criterion};
use margin_intelligence::calculator::MarginCalculator;
use rust_decimal_macros::dec;

fn benchmark_calculations(c: &mut Criterion) {
    c.bench_function("gross_margin", |b| {
        b.iter(|| MarginCalculator::gross_margin(black_box(dec!(10000)), black_box(dec!(7000))))
    });

    c.bench_function("contribution_margin", |b| {
        b.iter(|| {
            MarginCalculator::contribution_margin(black_box(dec!(10000)), black_box(dec!(7000)))
        })
    });
}

criterion_group!(benches, benchmark_calculations);
criterion_main!(benches);
