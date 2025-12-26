//! Benchmarks for natural language parsing performance

use ai_assistant::parser::NaturalLanguageParser;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_intent_parsing(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let parser = rt.block_on(async { NaturalLanguageParser::new().unwrap() });

    c.bench_function("parse_deploy_intent", |b| {
        b.to_async(&rt).iter(|| async {
            parser
                .parse(black_box("deploy my application from github.com/user/repo"))
                .await
                .unwrap()
        })
    });

    c.bench_function("parse_scale_intent", |b| {
        b.to_async(&rt).iter(|| async {
            parser
                .parse(black_box("scale web-service to 5 replicas with 2 CPU"))
                .await
                .unwrap()
        })
    });

    c.bench_function("parse_query_intent", |b| {
        b.to_async(&rt).iter(|| async {
            parser
                .parse(black_box("show me all running agents with high CPU usage"))
                .await
                .unwrap()
        })
    });

    c.bench_function("parse_complex_intent", |b| {
        b.to_async(&rt).iter(|| async {
            parser.parse(black_box("debug my machine learning training service that's running slowly and consuming too much GPU memory")).await.unwrap()
        })
    });

    c.bench_function("parse_unknown_intent", |b| {
        b.to_async(&rt).iter(|| async {
            parser
                .parse(black_box(
                    "this is complete gibberish that should not match any pattern",
                ))
                .await
                .unwrap()
        })
    });
}

fn benchmark_entity_extraction(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let parser = rt.block_on(async { NaturalLanguageParser::new().unwrap() });

    c.bench_function("extract_entities_simple", |b| {
        b.to_async(&rt).iter(|| async {
            parser
                .parse(black_box("deploy app with 2 CPU"))
                .await
                .unwrap()
        })
    });

    c.bench_function("extract_entities_complex", |b| {
        b.to_async(&rt).iter(|| async {
            parser.parse(black_box("scale service to 10 replicas with 4 CPU, 8GB memory, 1 GPU from https://github.com/user/repo")).await.unwrap()
        })
    });

    c.bench_function("extract_entities_urls", |b| {
        b.to_async(&rt).iter(|| async {
            parser.parse(black_box("deploy from https://github.com/user/repo or https://gitlab.com/another/project")).await.unwrap()
        })
    });
}

fn benchmark_batch_parsing(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let parser = rt.block_on(async { NaturalLanguageParser::new().unwrap() });

    let test_inputs = vec![
        "deploy my app",
        "scale to 5",
        "show agents",
        "debug service",
        "help me",
        "optimize for speed",
        "rollback deployment",
        "evolve model",
        "status check",
        "tail logs",
    ];

    c.bench_function("parse_batch_10", |b| {
        b.to_async(&rt).iter(|| async {
            for input in &test_inputs {
                parser.parse(black_box(input)).await.unwrap();
            }
        })
    });
}

criterion_group!(
    benches,
    benchmark_intent_parsing,
    benchmark_entity_extraction,
    benchmark_batch_parsing
);
criterion_main!(benches);
