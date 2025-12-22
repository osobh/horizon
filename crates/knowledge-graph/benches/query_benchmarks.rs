use criterion::{black_box, criterion_group, criterion_main, Criterion};
use exorust_knowledge_graph::*;
use std::collections::HashMap;
use tokio::runtime::Runtime;

fn benchmark_node_operations(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("create_knowledge_graph", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = graph::KnowledgeGraphConfig {
                    gpu_enabled: false,
                    ..Default::default()
                };
                black_box(graph::KnowledgeGraph::new(config).await?)
            })
        })
    });

    c.bench_function("add_1000_nodes", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = graph::KnowledgeGraphConfig {
                    gpu_enabled: false,
                    ..Default::default()
                };
                let mut graph = graph::KnowledgeGraph::new(config).await.unwrap();

                for i in 0..1000 {
                    let mut properties = HashMap::new();
                    properties.insert("id".to_string(), serde_json::Value::Number(i.into()));
                    let node = graph::Node::new(graph::NodeType::Agent, properties);
                    graph.add_node(node).unwrap();
                }

                black_box(graph)
            })
        })
    });
}

fn benchmark_query_operations(c: &mut Criterion) {
    let rt = Runtime::new()?;

    // Setup test graph
    let (mut graph, node_ids) = rt.block_on(async {
        let config = graph::KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = graph::KnowledgeGraph::new(config).await?;
        let mut node_ids = Vec::new();

        // Add nodes
        for i in 0..1000 {
            let mut properties = HashMap::new();
            properties.insert("value".to_string(), serde_json::Value::Number(i.into()));
            let node = graph::Node::new(graph::NodeType::Agent, properties);
            let node_id = graph.add_node(node).unwrap();
            node_ids.push(node_id);
        }

        // Add edges
        for i in 0..999 {
            let edge = graph::Edge::new(
                node_ids[i].clone(),
                node_ids[i + 1].clone(),
                graph::EdgeType::Has,
                1.0,
            );
            graph.add_edge(edge).unwrap();
        }

        (graph, node_ids)
    });

    c.bench_function("find_nodes_by_type", |b| {
        b.iter(|| {
            let nodes = graph.get_nodes_by_type(&graph::NodeType::Agent);
            black_box(nodes)
        })
    });

    c.bench_function("path_finding_bfs", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut engine = query::QueryEngine::new(false).await.unwrap();
                let query = query::Query {
                    query_type: query::QueryType::FindPath {
                        source_id: node_ids[0].clone(),
                        target_id: node_ids[100].clone(),
                        max_length: 150,
                    },
                    timeout_ms: None,
                    limit: None,
                    offset: None,
                    use_gpu: false,
                };

                let result = engine.execute(&graph, query).await.unwrap();
                black_box(result)
            })
        })
    });

    c.bench_function("neighborhood_query", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut engine = query::QueryEngine::new(false).await.unwrap();
                let query = query::Query {
                    query_type: query::QueryType::Neighborhood {
                        node_id: node_ids[500].clone(),
                        radius: 3,
                        edge_types: None,
                    },
                    timeout_ms: None,
                    limit: None,
                    offset: None,
                    use_gpu: false,
                };

                let result = engine.execute(&graph, query).await.unwrap();
                black_box(result)
            })
        })
    });
}

criterion_group!(
    benches,
    benchmark_node_operations,
    benchmark_query_operations
);
criterion_main!(benches);
