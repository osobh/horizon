use chrono::Utc;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cost_reporter::{
    export::{CsvExporter, JsonExporter, MarkdownExporter},
    models::{CostAttribution, DailyCostSummary, HasCostBreakdown},
    reports::{CostForecaster, TrendAnalyzer},
};
use rust_decimal_macros::dec;
use uuid::Uuid;

fn create_test_attribution() -> CostAttribution {
    let now = Utc::now();
    CostAttribution {
        id: Uuid::new_v4(),
        job_id: Some(Uuid::new_v4()),
        user_id: "user123".to_string(),
        team_id: Some("team456".to_string()),
        customer_id: Some("customer789".to_string()),
        gpu_cost: dec!(100.50),
        cpu_cost: dec!(20.25),
        network_cost: dec!(5.75),
        storage_cost: dec!(3.50),
        total_cost: dec!(130.00),
        period_start: now,
        period_end: now,
        created_at: now,
        updated_at: now,
    }
}

fn create_test_daily_summaries(count: usize) -> Vec<DailyCostSummary> {
    let now = Utc::now();
    (0..count)
        .map(|i| DailyCostSummary {
            day: now - chrono::Duration::days(i as i64),
            team_id: Some("team1".to_string()),
            user_id: Some("user1".to_string()),
            total_cost: dec!(100.00) + rust_decimal::Decimal::from(i),
            gpu_cost: dec!(80.00),
            cpu_cost: dec!(10.00),
            network_cost: dec!(5.00),
            storage_cost: dec!(5.00),
            job_count: 10,
        })
        .collect()
}

fn benchmark_csv_export(c: &mut Criterion) {
    let attributions: Vec<_> = (0..1000).map(|_| create_test_attribution()).collect();
    let exporter = CsvExporter::new();

    c.bench_function("csv_export_1000_records", |b| {
        b.iter(|| exporter.export(black_box(&attributions)).unwrap())
    });
}

fn benchmark_json_export(c: &mut Criterion) {
    let attributions: Vec<_> = (0..1000).map(|_| create_test_attribution()).collect();
    let exporter = JsonExporter::new();

    c.bench_function("json_export_1000_records", |b| {
        b.iter(|| exporter.export(black_box(&attributions)).unwrap())
    });
}

fn benchmark_markdown_export(c: &mut Criterion) {
    let attributions: Vec<_> = (0..1000).map(|_| create_test_attribution()).collect();
    let exporter = MarkdownExporter::new();

    c.bench_function("markdown_export_1000_records", |b| {
        b.iter(|| exporter.export(black_box(&attributions)).unwrap())
    });
}

fn benchmark_trend_analysis(c: &mut Criterion) {
    let summaries = create_test_daily_summaries(30);
    let analyzer = TrendAnalyzer::new();

    c.bench_function("trend_analysis_30_days", |b| {
        b.iter(|| analyzer.calculate_trend(black_box(&summaries)).unwrap())
    });
}

fn benchmark_cost_forecast(c: &mut Criterion) {
    let summaries = create_test_daily_summaries(30);
    let forecaster = CostForecaster::new();

    c.bench_function("cost_forecast_30_days_ahead", |b| {
        b.iter(|| forecaster.forecast(black_box(&summaries), 30).unwrap())
    });
}

criterion_group!(
    benches,
    benchmark_csv_export,
    benchmark_json_export,
    benchmark_markdown_export,
    benchmark_trend_analysis,
    benchmark_cost_forecast,
);
criterion_main!(benches);
