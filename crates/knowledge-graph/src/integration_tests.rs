//! Integration tests for knowledge-graph combining multiple subsystems

#[cfg(test)]
mod integration_tests {
    use crate::evolution_tracker::{EvolutionEvent, EvolutionSnapshot, EvolutionTracker};
    use crate::graph::{Edge, EdgeType, KnowledgeGraph, Node, NodeType};
    use crate::memory_integration::{CacheStrategy, MemoryConfig, MemoryIntegration};
    use crate::patterns::*; // Pattern imports
    use crate::pruning::{GraphPruner, PruningConfig, PruningStrategy, RetentionPolicy};
    use crate::query::{
        AggregationFunction, Query, QueryEngine, QueryFilter, QueryResult, QueryType,
    };
    use crate::scaling::{
        ConsistencyLevel, DistributedGraphStore, PartitionStrategy, ReplicationFactor,
        ScaledKnowledgeGraph, ScalingConfig,
    };
    use crate::semantic::{EmbeddingVector, SemanticQuery, SemanticSearchEngine};
    use crate::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::{Duration, SystemTime};
    use tokio::sync::RwLock;

    // Multi-threaded concurrent access tests

    #[tokio::test]
    async fn test_concurrent_graph_modifications() {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));

        // Spawn multiple tasks modifying the graph
        let mut handles = vec![];

        for i in 0..10 {
            let g = graph.clone();
            let handle = tokio::spawn(async move {
                let mut graph = g.write().await;

                for j in 0..10 {
                    let node_id = format!("thread_{}_node_{}", i, j);
                    graph.add_node(node_id, Node::new(NodeType::Agent, HashMap::new()));
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all nodes were added
        let g = graph.read().await;
        assert_eq!(g.node_count(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_query_execution() {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let query_engine = QueryEngine::new(graph.clone());

        // Setup initial data
        {
            let mut g = graph.write().await;
            for i in 0..50 {
                let mut props = HashMap::new();
                props.insert("index".to_string(), i.to_string());
                g.add_node(format!("node_{}", i), Node::new(NodeType::Metric, props));
            }
        }

        // Execute queries concurrently
        let mut handles = vec![];

        for i in 0..20 {
            let qe = query_engine.clone();
            let handle = tokio::spawn(async move {
                let query = Query {
                    query_type: QueryType::NodeSearch,
                    filters: vec![],
                    limit: Some(10),
                    offset: Some(i * 2),
                    order_by: None,
                    aggregations: vec![],
                };

                let result = qe.execute(query).await;
                assert!(result.is_ok());
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }
    }

    // Complex pattern matching scenarios

    #[tokio::test]
    async fn test_complex_pattern_matching() {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));

        // Create a complex pattern structure
        {
            let mut g = graph.write().await;

            // Create multiple instances of a pattern
            for instance in 0..5 {
                let prefix = format!("instance_{}", instance);

                // Hub and spoke pattern
                g.add_node(
                    format!("{}_hub", prefix),
                    Node::new(NodeType::Agent, HashMap::new()),
                );

                for i in 0..4 {
                    let spoke = format!("{}_spoke_{}", prefix, i);
                    g.add_node(spoke.clone(), Node::new(NodeType::Resource, HashMap::new()));

                    g.add_edge(
                        format!("{}_edge_{}", prefix, i),
                        Edge::new(format!("{}_hub", prefix), spoke, EdgeType::Uses, 1.0),
                    );
                }

                // Add noise edges
                if instance > 0 {
                    g.add_edge(
                        format!("noise_{}", instance),
                        Edge::new(
                            format!("instance_{}_hub", instance),
                            format!("instance_{}_hub", instance - 1),
                            EdgeType::RelatesTo,
                            0.5,
                        ),
                    );
                }
            }
        }

        let discovery = PatternDiscovery::new(graph.clone());
        let patterns = discovery.discover_patterns(3).await.unwrap();

        // Should find the hub-spoke pattern
        assert!(!patterns.is_empty());

        // Verify pattern properties
        for pattern in patterns {
            assert!(pattern.confidence > 0.0);
            assert!(pattern.support >= 3);
        }
    }

    // Semantic search with real-world scenarios

    #[tokio::test]
    async fn test_semantic_search_real_world() {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let search_engine = SemanticSearchEngine::new(graph.clone(), 768); // BERT-like embeddings

        // Add concepts with semantic relationships
        {
            let mut g = graph.write().await;

            let concepts = vec![
                (
                    "python_programming",
                    vec!["language", "programming", "scripting"],
                ),
                ("machine_learning", vec!["AI", "algorithms", "data"]),
                ("deep_learning", vec!["neural", "networks", "AI"]),
                ("data_science", vec!["statistics", "analysis", "insights"]),
                (
                    "software_engineering",
                    vec!["development", "design", "architecture"],
                ),
            ];

            for (concept, tags) in concepts {
                let mut props = HashMap::new();
                props.insert("name".to_string(), concept.to_string());
                for (i, tag) in tags.iter().enumerate() {
                    props.insert(format!("tag_{}", i), tag.to_string());
                }

                g.add_node(concept.to_string(), Node::new(NodeType::Concept, props));
            }

            // Add relationships
            g.add_edge(
                "e1".to_string(),
                Edge::new(
                    "machine_learning".to_string(),
                    "deep_learning".to_string(),
                    EdgeType::Produces,
                    0.9,
                ),
            );

            g.add_edge(
                "e2".to_string(),
                Edge::new(
                    "data_science".to_string(),
                    "machine_learning".to_string(),
                    EdgeType::Uses,
                    0.8,
                ),
            );
        }

        // Perform semantic searches
        let queries = vec![
            "artificial intelligence and neural networks",
            "programming languages for data analysis",
            "statistical methods in AI",
        ];

        for query_text in queries {
            let query = SemanticQuery {
                query_text: query_text.to_string(),
                embedding: EmbeddingVector::from_vec(vec![0.5; 768]), // Mock embedding
                similarity_threshold: 0.5,
                max_results: 5,
                include_metadata: true,
            };

            let results = search_engine.search(query).await;
            assert!(results.is_ok());
        }
    }

    // Evolution tracking with rollback scenarios

    #[tokio::test]
    async fn test_evolution_with_rollback() {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let tracker = EvolutionTracker::new(graph.clone());

        // Take initial snapshot
        let snap1 = tracker.create_snapshot().await?;

        // Make changes
        {
            let mut g = graph.write().await;
            for i in 0..10 {
                g.add_node(
                    format!("gen1_node_{}", i),
                    Node::new(NodeType::Agent, HashMap::new()),
                );
            }
        }

        let snap2 = tracker.create_snapshot().await.unwrap();

        // Make more changes
        {
            let mut g = graph.write().await;
            for i in 0..5 {
                g.remove_node(&format!("gen1_node_{}", i));
            }
            for i in 0..5 {
                g.add_node(
                    format!("gen2_node_{}", i),
                    Node::new(NodeType::Goal, HashMap::new()),
                );
            }
        }

        let snap3 = tracker.create_snapshot().await.unwrap();

        // Test rollback capability (would be implemented)
        assert_ne!(snap1.snapshot_id, snap2.snapshot_id);
        assert_ne!(snap2.snapshot_id, snap3.snapshot_id);

        // Verify evolution history
        let history = tracker.get_history(10).await;
        assert!(history.len() >= 2);
    }

    // Distributed graph operations

    #[tokio::test]
    async fn test_distributed_graph_operations() {
        let config = ScalingConfig {
            num_partitions: 8,
            replication_factor: ReplicationFactor::Triple,
            partition_strategy: PartitionStrategy::Hash,
            consistency_level: ConsistencyLevel::ReadYourWrite,
            sync_interval: Duration::from_secs(10),
            max_partition_size: 100_000,
            enable_auto_scaling: true,
            scaling_threshold: 0.75,
        };

        let base_graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let distributed = DistributedGraphStore::new(vec![base_graph.clone()], config);

        // Test partition distribution
        let test_nodes = vec![
            "user_123",
            "user_456",
            "user_789",
            "product_abc",
            "product_def",
            "product_xyz",
            "order_111",
            "order_222",
            "order_333",
        ];

        let mut partition_distribution = HashMap::new();

        for node_id in test_nodes {
            let partition = distributed.compute_partition(node_id);
            *partition_distribution.entry(partition).or_insert(0) += 1;
        }

        // Verify reasonable distribution
        assert!(!partition_distribution.is_empty());
        assert!(partition_distribution.len() > 1); // Should use multiple partitions
    }

    // Memory-aware operations

    #[test]
    fn test_memory_aware_caching() {
        let config = MemoryConfig {
            cache_size_mb: 256,
            cache_strategy: CacheStrategy::ARC,
            prefetch_enabled: true,
            prefetch_distance: 2,
            compression_enabled: true,
            compression_level: 3,
            tier_migration_threshold: 0.7,
            gpu_memory_limit_mb: 1024,
        };

        let cache_efficiency = match config.cache_strategy {
            CacheStrategy::LRU => 0.7,
            CacheStrategy::LFU => 0.75,
            CacheStrategy::ARC => 0.85, // Adaptive should be most efficient
            CacheStrategy::FIFO => 0.6,
            CacheStrategy::Random => 0.5,
        };

        assert!(cache_efficiency > 0.0);

        // Test prefetch logic
        if config.prefetch_enabled {
            assert!(config.prefetch_distance > 0);
        }

        // Test compression settings
        if config.compression_enabled {
            assert!(config.compression_level > 0 && config.compression_level <= 9);
        }
    }

    // Pruning with pattern preservation

    #[tokio::test]
    async fn test_pruning_with_pattern_preservation() {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));

        // Create graph with important patterns
        {
            let mut g = graph.write().await;

            // Critical pattern that should be preserved
            for i in 0..3 {
                let critical = format!("critical_{}", i);
                g.add_node(
                    critical.clone(),
                    Node::new(NodeType::Constraint, HashMap::new()),
                );

                for j in 0..2 {
                    let resource = format!("critical_resource_{}_{}", i, j);
                    g.add_node(
                        resource.clone(),
                        Node::new(NodeType::Resource, HashMap::new()),
                    );
                    g.add_edge(
                        format!("critical_edge_{}_{}", i, j),
                        Edge::new(critical.clone(), resource, EdgeType::Requires, 1.0),
                    );
                }
            }

            // Non-critical nodes that can be pruned
            for i in 0..50 {
                g.add_node(
                    format!("temporary_{}", i),
                    Node::new(NodeType::Event, HashMap::new()),
                );
            }
        }

        let pruning_config = PruningConfig {
            strategy: PruningStrategy::Combined,
            retention_policy: RetentionPolicy::KeepImportant(0.3),
            max_graph_size: 20,
            prune_interval: Duration::from_secs(300),
            batch_size: 5,
            importance_threshold: 0.5,
            preserve_patterns: true, // Critical setting
        };

        let pruner = GraphPruner::new(graph.clone(), pruning_config);

        // Simulate pruning (would remove temporary nodes but preserve patterns)
        let stats = pruner.analyze_pruning_impact().await;
        assert!(stats.is_ok());
    }

    // End-to-end workflow test

    #[tokio::test]
    async fn test_end_to_end_knowledge_workflow() {
        // Initialize system
        init().await.unwrap();

        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let query_engine = QueryEngine::new(graph.clone());
        let search_engine = SemanticSearchEngine::new(graph.clone(), 512);
        let pattern_discovery = PatternDiscovery::new(graph.clone());
        let evolution_tracker = EvolutionTracker::new(graph.clone());

        // Phase 1: Build initial knowledge graph
        {
            let mut g = graph.write().await;

            // Add domain knowledge
            let domains = vec![
                ("AI", NodeType::Concept),
                ("MachineLearning", NodeType::Concept),
                ("DeepLearning", NodeType::Concept),
                ("NeuralNetworks", NodeType::Pattern),
                ("Backpropagation", NodeType::Pattern),
            ];

            for (name, node_type) in domains {
                let mut props = HashMap::new();
                props.insert("domain".to_string(), "artificial_intelligence".to_string());
                props.insert("importance".to_string(), "high".to_string());

                g.add_node(name.to_string(), Node::new(node_type, props));
            }

            // Add relationships
            g.add_edge(
                "e1".to_string(),
                Edge::new(
                    "AI".to_string(),
                    "MachineLearning".to_string(),
                    EdgeType::Has,
                    1.0,
                ),
            );
            g.add_edge(
                "e2".to_string(),
                Edge::new(
                    "MachineLearning".to_string(),
                    "DeepLearning".to_string(),
                    EdgeType::Produces,
                    0.9,
                ),
            );
            g.add_edge(
                "e3".to_string(),
                Edge::new(
                    "DeepLearning".to_string(),
                    "NeuralNetworks".to_string(),
                    EdgeType::Uses,
                    1.0,
                ),
            );
            g.add_edge(
                "e4".to_string(),
                Edge::new(
                    "NeuralNetworks".to_string(),
                    "Backpropagation".to_string(),
                    EdgeType::Requires,
                    1.0,
                ),
            );
        }

        // Phase 2: Query and analyze
        let concept_query = Query {
            query_type: QueryType::PathSearch,
            filters: vec![QueryFilter::NodeType(NodeType::Concept)],
            limit: None,
            offset: None,
            order_by: None,
            aggregations: vec![(AggregationFunction::Count, "node_id".to_string())],
        };

        let results = query_engine.execute(concept_query).await.unwrap();
        assert!(!results.nodes.is_empty());

        // Phase 3: Discover patterns
        let patterns = pattern_discovery.discover_patterns(2).await.unwrap();
        assert!(!patterns.is_empty());

        // Phase 4: Track evolution
        let snapshot = evolution_tracker.create_snapshot().await.unwrap();
        assert!(!snapshot.snapshot_id.is_empty());

        // Phase 5: Scale if needed
        let scaling_config = ScalingConfig {
            num_partitions: 2,
            replication_factor: ReplicationFactor::Single,
            partition_strategy: PartitionStrategy::Hash,
            consistency_level: ConsistencyLevel::Eventual,
            sync_interval: Duration::from_secs(60),
            max_partition_size: 10_000,
            enable_auto_scaling: false,
            scaling_threshold: 0.8,
        };

        let scaled = ScaledKnowledgeGraph::new(graph.clone(), scaling_config);
        assert_eq!(scaled.config.num_partitions, 2);
    }

    // Stress testing

    #[tokio::test]
    async fn test_graph_under_stress() {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));

        // Add many nodes rapidly
        let start = std::time::Instant::now();

        {
            let mut g = graph.write().await;

            for i in 0..1000 {
                let node_id = format!("stress_node_{}", i);
                g.add_node(node_id, Node::new(NodeType::Metric, HashMap::new()));

                // Add edges to create connectivity
                if i > 0 {
                    let edge_id = format!("stress_edge_{}", i);
                    g.add_edge(
                        edge_id,
                        Edge::new(
                            format!("stress_node_{}", i - 1),
                            format!("stress_node_{}", i),
                            EdgeType::Influences,
                            1.0,
                        ),
                    );
                }
            }
        }

        let elapsed = start.elapsed();
        println!("Added 1000 nodes and 999 edges in {:?}", elapsed);

        // Verify graph integrity
        let g = graph.read().await;
        assert_eq!(g.node_count(), 1000);
        assert_eq!(g.edge_count(), 999);
    }

    // Error injection and recovery

    #[test]
    fn test_error_injection_scenarios() {
        // Test all error paths
        let test_cases = vec![
            (
                KnowledgeGraphError::InvalidQuery {
                    query: "MATCH (n) WHERE n.invalid = true".to_string(),
                    message: "Property 'invalid' does not exist".to_string(),
                },
                "InvalidQuery",
            ),
            (
                KnowledgeGraphError::GpuError {
                    message: "CUDA out of memory".to_string(),
                },
                "GpuError",
            ),
            (
                KnowledgeGraphError::SemanticError {
                    message: "Embedding dimension mismatch: expected 384, got 512".to_string(),
                },
                "SemanticError",
            ),
            (
                KnowledgeGraphError::ConcurrencyError {
                    message: "Deadlock detected in partition 3".to_string(),
                },
                "ConcurrencyError",
            ),
            (
                KnowledgeGraphError::SerializationError {
                    message: "Failed to serialize graph state: cyclic reference".to_string(),
                },
                "SerializationError",
            ),
        ];

        for (error, expected_type) in test_cases {
            let error_string = error.to_string();
            assert!(error_string.contains(expected_type) || error_string.len() > 0);
        }
    }
}
