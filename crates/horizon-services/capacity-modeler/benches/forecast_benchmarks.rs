use capacity_modeler::{forecaster::EtsForecaster, ForecastService};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn generate_seasonal_data(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 50.0;
            let trend = i as f64 * 0.1;
            let seasonal = ((i as f64 * 2.0 * std::f64::consts::PI) / 7.0).sin() * 10.0;
            base + trend + seasonal
        })
        .collect()
}

fn benchmark_ets_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("ets_training");

    for size in [100, 500, 1000, 2000].iter() {
        let data = generate_seasonal_data(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut forecaster = EtsForecaster::new();
                forecaster.train(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

fn benchmark_forecast_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("forecast_generation");

    let data = generate_seasonal_data(1000);
    let mut forecaster = EtsForecaster::new();
    forecaster.train(&data).unwrap();

    for horizon in [30, 91, 182, 365].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(horizon), horizon, |b, &h| {
            b.iter(|| {
                forecaster.forecast(black_box(h)).unwrap();
            });
        });
    }

    group.finish();
}

fn benchmark_forecast_with_intervals(c: &mut Criterion) {
    let data = generate_seasonal_data(1000);
    let mut forecaster = EtsForecaster::new();
    forecaster.train(&data).unwrap();

    c.bench_function("forecast_with_intervals_91_days", |b| {
        b.iter(|| {
            forecaster
                .forecast_with_intervals(black_box(91), black_box(0.95))
                .unwrap();
        });
    });
}

fn benchmark_end_to_end_forecast(c: &mut Criterion) {
    let service = ForecastService::new(100);
    let data = generate_seasonal_data(200);

    c.bench_function("end_to_end_forecast_13_weeks", |b| {
        b.iter(|| {
            service
                .forecast_gpu_demand(black_box(&data), black_box(13), black_box(true))
                .unwrap();
        });
    });
}

fn benchmark_backtest(c: &mut Criterion) {
    let service = ForecastService::new(50);
    let data = generate_seasonal_data(300);

    c.bench_function("backtest_200_train_30_test", |b| {
        b.iter(|| {
            service
                .backtest(black_box(&data), black_box(200), black_box(30))
                .unwrap();
        });
    });
}

fn benchmark_data_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_preprocessing");
    let service = ForecastService::new(100);

    for size in [100, 500, 1000, 5000].iter() {
        let mut data = generate_seasonal_data(*size);
        // Add some outliers
        if *size > 50 {
            data[*size / 4] = 1000.0;
            data[*size / 2] = -500.0;
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                // Access private method via service (testing indirectly)
                service.forecast_gpu_demand(black_box(&data), 1, false).ok();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_ets_training,
    benchmark_forecast_generation,
    benchmark_forecast_with_intervals,
    benchmark_end_to_end_forecast,
    benchmark_backtest,
    benchmark_data_preprocessing,
);
criterion_main!(benches);
