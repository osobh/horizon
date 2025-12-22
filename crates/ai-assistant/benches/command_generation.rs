//! Benchmarks for command generation performance

use ai_assistant::command_generator::CommandGenerator;
use ai_assistant::parser::{Intent, ParsedQuery};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

fn benchmark_command_generation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let generator = rt.block_on(async { CommandGenerator::new().unwrap() });

    let deploy_query = ParsedQuery {
        intent: Intent::Deploy {
            target: "test-app".to_string(),
            source: Some("https://github.com/user/repo".to_string()),
            config: HashMap::new(),
        },
        confidence: 0.9,
        entities: HashMap::new(),
        context: Default::default(),
        raw_input: "deploy test-app".to_string(),
    };

    let scale_query = ParsedQuery {
        intent: Intent::Scale {
            target: "web-service".to_string(),
            replicas: Some(5),
            resources: None,
        },
        confidence: 0.85,
        entities: HashMap::new(),
        context: Default::default(),
        raw_input: "scale web-service to 5".to_string(),
    };

    c.bench_function("generate_deploy_command", |b| {
        b.to_async(&rt)
            .iter(|| async { generator.generate(black_box(&deploy_query)).await.unwrap() })
    });

    c.bench_function("generate_scale_command", |b| {
        b.to_async(&rt)
            .iter(|| async { generator.generate(black_box(&scale_query)).await.unwrap() })
    });

    let mut filters = HashMap::new();
    filters.insert("status".to_string(), "running".to_string());

    let query_query = ParsedQuery {
        intent: Intent::Query {
            resource_type: "agents".to_string(),
            filters,
            projection: Some(vec!["name".to_string(), "status".to_string()]),
        },
        confidence: 0.8,
        entities: HashMap::new(),
        context: Default::default(),
        raw_input: "show running agents".to_string(),
    };

    c.bench_function("generate_query_command", |b| {
        b.to_async(&rt)
            .iter(|| async { generator.generate(black_box(&query_query)).await.unwrap() })
    });
}

fn benchmark_batch_generation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let generator = rt.block_on(async { CommandGenerator::new().unwrap() });

    let queries = vec![
        ParsedQuery {
            intent: Intent::Deploy {
                target: "app1".to_string(),
                source: None,
                config: HashMap::new(),
            },
            confidence: 0.9,
            entities: HashMap::new(),
            context: Default::default(),
            raw_input: "deploy app1".to_string(),
        },
        ParsedQuery {
            intent: Intent::Scale {
                target: "app2".to_string(),
                replicas: Some(3),
                resources: None,
            },
            confidence: 0.85,
            entities: HashMap::new(),
            context: Default::default(),
            raw_input: "scale app2".to_string(),
        },
        ParsedQuery {
            intent: Intent::Status { target: None },
            confidence: 0.95,
            entities: HashMap::new(),
            context: Default::default(),
            raw_input: "status".to_string(),
        },
    ];

    c.bench_function("generate_batch_commands", |b| {
        b.to_async(&rt).iter(|| async {
            for query in &queries {
                generator.generate(black_box(query)).await.unwrap();
            }
        })
    });
}

criterion_group!(
    benches,
    benchmark_command_generation,
    benchmark_batch_generation
);
criterion_main!(benches);
