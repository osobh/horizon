use criterion::{black_box, criterion_group, criterion_main, Criterion};
use exorust_knowledge_graph::*;
use std::collections::HashMap;
use tokio::runtime::Runtime;

fn benchmark_semantic_operations(c: &mut Criterion) {
    let rt = Runtime::new()?;

    c.bench_function("create_semantic_engine", |b| {
        b.iter(|| {
            let config = semantic::EmbeddingConfig {
                gpu_enabled: false,
                dimension: 128,
                ..Default::default()
            };
            black_box(semantic::SemanticSearchEngine::new(config))
        })
    });

    c.bench_function("text_to_embedding", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = semantic::EmbeddingConfig {
                    gpu_enabled: false,
                    dimension: 256,
                    ..Default::default()
                };
                let mut engine = semantic::SemanticSearchEngine::new(config);

                let embedding = engine
                    .text_to_embedding("This is a test sentence for embedding benchmark")
                    .await
                    .unwrap();
                black_box(embedding)
            })
        })
    });

    c.bench_function("cosine_similarity_256d", |b| {
        b.iter(|| {
            let config = semantic::EmbeddingConfig {
                gpu_enabled: false,
                dimension: 256,
                ..Default::default()
            };
            let engine = semantic::SemanticSearchEngine::new(config);

            let vec1: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
            let vec2: Vec<f32> = (0..256).map(|i| ((i + 128) % 256) as f32 / 256.0).collect();

            let similarity = engine.compute_similarity(&vec1, &vec2).unwrap();
            black_box(similarity)
        })
    });
}

fn benchmark_semantic_search(c: &mut Criterion) {
    let rt = Runtime::new()?;

    // Setup test data
    let (graph, mut engine) = rt.block_on(async {
        let graph_config = graph::KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = graph::KnowledgeGraph::new(graph_config).await?;

        let config = semantic::EmbeddingConfig {
            gpu_enabled: false,
            dimension: 128,
            ..Default::default()
        };
        let mut engine = semantic::SemanticSearchEngine::new(config);

        // Add test nodes with different content
        let test_content = [
            "artificial intelligence and machine learning",
            "deep learning neural networks",
            "natural language processing",
            "computer vision and image recognition",
            "robotics and automation systems",
            "data science and analytics",
            "cloud computing infrastructure",
            "distributed systems architecture",
            "cybersecurity and threat detection",
            "blockchain and cryptocurrency technology",
        ];

        for (i, content) in test_content.iter().enumerate() {
            let mut properties = HashMap::new();
            properties.insert(
                "description".to_string(),
                serde_json::Value::String(content.to_string()),
            );
            properties.insert("id".to_string(), serde_json::Value::Number(i.into()));

            let node = graph::Node::new(graph::NodeType::Concept, properties);
            graph.add_node(node).unwrap();
        }

        // Pre-compute embeddings for all nodes
        let nodes = graph.get_nodes_by_type(&graph::NodeType::Concept);
        for node in nodes {
            engine.update_node_embedding(&node.id, node).await.unwrap();
        }

        (graph, engine)
    });

    c.bench_function("semantic_search_10_nodes", |b| {
        b.iter(|| {
            rt.block_on(async {
                let query = semantic::SemanticQuery {
                    query: semantic::QueryInput::Text("machine learning algorithms".to_string()),
                    node_types: Some(vec![graph::NodeType::Concept]),
                    top_k: 5,
                    threshold: 0.0,
                    use_gpu: false,
                };

                let mut engine_clone =
                    semantic::SemanticSearchEngine::new(semantic::EmbeddingConfig {
                        gpu_enabled: false,
                        dimension: 128,
                        ..Default::default()
                    });

                // Copy cache
                for node in graph.get_nodes_by_type(&graph::NodeType::Concept) {
                    let _ = engine_clone.update_node_embedding(&node.id, node).await;
                }

                let results = engine_clone.search(&graph, query).await.unwrap();
                black_box(results)
            })
        })
    });

    c.bench_function("find_similar_nodes", |b| {
        b.iter(|| {
            rt.block_on(async {
                let nodes = graph.get_nodes_by_type(&graph::NodeType::Concept);
                if let Some(target_node) = nodes.first() {
                    let mut engine_clone =
                        semantic::SemanticSearchEngine::new(semantic::EmbeddingConfig {
                            gpu_enabled: false,
                            dimension: 128,
                            ..Default::default()
                        });

                    // Copy cache
                    for node in &nodes {
                        let _ = engine_clone.update_node_embedding(&node.id, node).await;
                    }

                    let similar = engine_clone
                        .find_similar_nodes(&graph, &target_node.id, 3, 0.0)
                        .await
                        .unwrap();
                    black_box(similar)
                } else {
                    black_box(Vec::new())
                }
            })
        })
    });

    c.bench_function("batch_embedding_update", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut engine_clone =
                    semantic::SemanticSearchEngine::new(semantic::EmbeddingConfig {
                        gpu_enabled: false,
                        dimension: 64,
                        ..Default::default()
                    });

                let nodes: Vec<&graph::Node> = graph.get_nodes_by_type(&graph::NodeType::Concept);
                let result = engine_clone.batch_update_embeddings(nodes).await.unwrap();
                black_box(result)
            })
        })
    });
}

criterion_group!(
    benches,
    benchmark_semantic_operations,
    benchmark_semantic_search
);
criterion_main!(benches);
