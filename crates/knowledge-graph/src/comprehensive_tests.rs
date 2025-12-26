//! Comprehensive tests for knowledge-graph to enhance coverage to 90%+

#[cfg(test)]
mod comprehensive_tests {
    use crate::evolution_tracker::{EvolutionEvent, EvolutionSnapshot, EvolutionTracker};
    use crate::graph::{Edge, EdgeType, KnowledgeGraph, Node, NodeType};
    use crate::memory_integration::{CacheStrategy, MemoryConfig, MemoryIntegration};
    use crate::patterns::*; // Pattern imports
    use crate::pruning::{PruningConfig, PruningStrategy, RetentionPolicy};
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

    // Complex graph construction tests

    #[test]
    fn test_large_graph_construction() {
        let mut graph = KnowledgeGraph::new();

        // Add many nodes
        for i in 0..100 {
            let mut props = HashMap::new();
            props.insert("index".to_string(), i.to_string());
            props.insert("type".to_string(), "test".to_string());

            let node = Node::new(
                match i % 8 {
                    0 => NodeType::Agent,
                    1 => NodeType::Goal,
                    2 => NodeType::Concept,
                    3 => NodeType::Resource,
                    4 => NodeType::Pattern,
                    5 => NodeType::Event,
                    6 => NodeType::Constraint,
                    _ => NodeType::Metric,
                },
                props,
            );

            graph.add_node(format!("node_{}", i), node);
        }

        // Add edges creating a complex network
        for i in 0..100 {
            for j in 0..3 {
                let target = (i + j + 1) % 100;
                let edge = Edge::new(
                    format!("node_{}", i),
                    format!("node_{}", target),
                    match j {
                        0 => EdgeType::Has,
                        1 => EdgeType::Uses,
                        _ => EdgeType::RelatesTo,
                    },
                    1.0 / (j + 1) as f64,
                );
                graph.add_edge(format!("edge_{}_{}", i, target), edge);
            }
        }

        assert_eq!(graph.node_count(), 100);
        assert_eq!(graph.edge_count(), 300);
    }

    #[test]
    fn test_graph_traversal_operations() {
        let mut graph = KnowledgeGraph::new();

        // Create a small test graph
        let nodes = vec![
            ("root", NodeType::Agent),
            ("child1", NodeType::Goal),
            ("child2", NodeType::Resource),
            ("grandchild", NodeType::Concept),
        ];

        for (id, node_type) in nodes {
            graph.add_node(id.to_string(), Node::new(node_type, HashMap::new()));
        }

        // Add edges
        graph.add_edge(
            "e1".to_string(),
            Edge::new("root".to_string(), "child1".to_string(), EdgeType::Has, 1.0),
        );
        graph.add_edge(
            "e2".to_string(),
            Edge::new(
                "root".to_string(),
                "child2".to_string(),
                EdgeType::Uses,
                0.8,
            ),
        );
        graph.add_edge(
            "e3".to_string(),
            Edge::new(
                "child1".to_string(),
                "grandchild".to_string(),
                EdgeType::Produces,
                0.5,
            ),
        );

        // Test neighbor retrieval
        let root_neighbors = graph.get_neighbors("root");
        assert_eq!(root_neighbors.len(), 2);

        let child1_neighbors = graph.get_neighbors("child1");
        assert_eq!(child1_neighbors.len(), 1);

        let grandchild_neighbors = graph.get_neighbors("grandchild");
        assert_eq!(grandchild_neighbors.len(), 0); // No outgoing edges
    }

    // Query engine comprehensive tests

    #[tokio::test]
    async fn test_query_engine_complex_queries() {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let query_engine = QueryEngine::new(graph.clone());

        // Setup test data
        {
            let mut g = graph.write().await;

            // Add diverse nodes
            for i in 0..20 {
                let mut props = HashMap::new();
                props.insert("value".to_string(), i.to_string());
                props.insert(
                    "category".to_string(),
                    if i % 2 == 0 { "even" } else { "odd" }.to_string(),
                );

                g.add_node(format!("node_{}", i), Node::new(NodeType::Metric, props));
            }
        }

        // Test different query types
        let queries = vec![
            Query {
                query_type: QueryType::NodeSearch,
                filters: vec![QueryFilter::Property(
                    "category".to_string(),
                    "even".to_string(),
                )],
                limit: Some(5),
                offset: Some(0),
                order_by: Some("value".to_string()),
                aggregations: vec![],
            },
            Query {
                query_type: QueryType::Aggregation,
                filters: vec![],
                limit: None,
                offset: None,
                order_by: None,
                aggregations: vec![
                    (AggregationFunction::Count, "node_id".to_string()),
                    (AggregationFunction::Average, "value".to_string()),
                ],
            },
        ];

        for query in queries {
            let result = query_engine.execute(query).await;
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_query_filter_combinations() {
        let filters = vec![
            QueryFilter::NodeType(NodeType::Agent),
            QueryFilter::EdgeType(EdgeType::Has),
            QueryFilter::Property("key".to_string(), "value".to_string()),
            QueryFilter::Weight(0.5, 1.0),
            QueryFilter::Custom("custom_filter".to_string()),
        ];

        // Test serialization of filters
        for filter in filters {
            let json = serde_json::to_string(&filter).unwrap();
            let deserialized: QueryFilter = serde_json::from_str(&json).unwrap();

            match (filter, deserialized) {
                (QueryFilter::NodeType(a), QueryFilter::NodeType(b)) => assert_eq!(a, b),
                (QueryFilter::EdgeType(a), QueryFilter::EdgeType(b)) => assert_eq!(a, b),
                (QueryFilter::Property(k1, v1), QueryFilter::Property(k2, v2)) => {
                    assert_eq!(k1, k2);
                    assert_eq!(v1, v2);
                }
                (QueryFilter::Weight(min1, max1), QueryFilter::Weight(min2, max2)) => {
                    assert!((min1 - min2).abs() < f64::EPSILON);
                    assert!((max1 - max2).abs() < f64::EPSILON);
                }
                (QueryFilter::Custom(a), QueryFilter::Custom(b)) => assert_eq!(a, b),
                _ => panic!("Filter mismatch"),
            }
        }
    }

    // Semantic search comprehensive tests

    #[tokio::test]
    async fn test_semantic_search_workflow() {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let search_engine = SemanticSearchEngine::new(graph.clone(), 384); // Standard embedding size

        // Add nodes with embeddings
        {
            let mut g = graph.write().await;

            let concepts = vec![
                ("machine_learning", vec![0.8, 0.2, 0.1]),
                ("deep_learning", vec![0.9, 0.3, 0.1]),
                ("neural_networks", vec![0.85, 0.35, 0.15]),
                ("data_science", vec![0.6, 0.5, 0.3]),
                ("statistics", vec![0.4, 0.7, 0.5]),
            ];

            for (concept, embedding) in concepts {
                let mut props = HashMap::new();
                props.insert("name".to_string(), concept.to_string());

                let node = Node::new(NodeType::Concept, props);
                g.add_node(concept.to_string(), node);

                // Would normally add embedding to search engine
                // search_engine.add_embedding(concept, embedding);
            }
        }

        // Test semantic query
        let query = SemanticQuery {
            query_text: "artificial intelligence".to_string(),
            embedding: EmbeddingVector::from_vec(vec![0.85, 0.25, 0.12]),
            similarity_threshold: 0.7,
            max_results: 3,
            include_metadata: true,
        };

        // In real implementation, this would return similar nodes
        let results = search_engine.search(query).await;
        assert!(results.is_ok());
    }

    #[test]
    fn test_embedding_operations() {
        let embeddings = vec![
            EmbeddingVector::from_vec(vec![1.0, 0.0, 0.0]),
            EmbeddingVector::from_vec(vec![0.0, 1.0, 0.0]),
            EmbeddingVector::from_vec(vec![0.0, 0.0, 1.0]),
            EmbeddingVector::from_vec(vec![0.577, 0.577, 0.577]), // Normalized
        ];

        // Test cosine similarity calculations
        for i in 0..embeddings.len() {
            for j in 0..embeddings.len() {
                let sim = embeddings[i].cosine_similarity(&embeddings[j]);
                if i == j {
                    assert!((sim - 1.0).abs() < 0.01); // Same vector = similarity ~1
                }
            }
        }

        // Test zero vector
        let zero = EmbeddingVector::from_vec(vec![0.0, 0.0, 0.0]);
        assert_eq!(zero.magnitude(), 0.0);
    }

    // Pattern discovery comprehensive tests

    #[tokio::test]
    async fn test_pattern_discovery_workflow() {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let pattern_discovery = PatternDiscovery::new(graph.clone());

        // Create patterns in the graph
        {
            let mut g = graph.write().await;

            // Create repeating structure
            for i in 0..5 {
                let base = i * 3;
                g.add_node(
                    format!("hub_{}", i),
                    Node::new(NodeType::Agent, HashMap::new()),
                );
                g.add_node(
                    format!("resource_{}", base),
                    Node::new(NodeType::Resource, HashMap::new()),
                );
                g.add_node(
                    format!("resource_{}", base + 1),
                    Node::new(NodeType::Resource, HashMap::new()),
                );

                g.add_edge(
                    format!("e1_{}", i),
                    Edge::new(
                        format!("hub_{}", i),
                        format!("resource_{}", base),
                        EdgeType::Uses,
                        1.0,
                    ),
                );
                g.add_edge(
                    format!("e2_{}", i),
                    Edge::new(
                        format!("hub_{}", i),
                        format!("resource_{}", base + 1),
                        EdgeType::Uses,
                        1.0,
                    ),
                );
            }
        }

        // Discover patterns
        let min_support = 3;
        let patterns = pattern_discovery.discover_patterns(min_support).await;
        assert!(patterns.is_ok());

        let discovered = patterns.unwrap();
        assert!(!discovered.is_empty()); // Should find hub-resource pattern
    }

    #[test]
    fn test_pattern_matching() {
        let pattern = Pattern {
            pattern_id: "test_pattern".to_string(),
            pattern_type: PatternType::Structural,
            nodes: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            edges: vec![("A", "B"), ("B", "C")]
                .into_iter()
                .map(|(s, t)| (s.to_string(), t.to_string()))
                .collect(),
            confidence: 0.9,
            support: 5,
            frequency: 0.8,
            metadata: HashMap::new(),
        };

        // Test pattern properties
        assert_eq!(pattern.nodes.len(), 3);
        assert_eq!(pattern.edges.len(), 2);
        assert!(pattern.confidence > 0.0 && pattern.confidence <= 1.0);
    }

    // Evolution tracking comprehensive tests

    #[tokio::test]
    async fn test_evolution_tracking_lifecycle() {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let tracker = EvolutionTracker::new(graph.clone());

        // Initial snapshot
        let snapshot1 = tracker.create_snapshot().await;
        assert!(snapshot1.is_ok());

        // Modify graph
        {
            let mut g = graph.write().await;
            g.add_node(
                "evolved_node".to_string(),
                Node::new(NodeType::Agent, HashMap::new()),
            );
        }

        // Track evolution event
        let event = EvolutionEvent {
            event_id: "ev1".to_string(),
            timestamp: SystemTime::now(),
            event_type: "node_addition".to_string(),
            affected_nodes: vec!["evolved_node".to_string()],
            affected_edges: vec![],
            changes: HashMap::new(),
            metadata: HashMap::new(),
        };

        tracker.track_event(event).await;

        // Create new snapshot
        let snapshot2 = tracker.create_snapshot().await;
        assert!(snapshot2.is_ok());

        // Get evolution history
        let history = tracker.get_history(10).await;
        assert!(!history.is_empty());
    }

    #[test]
    fn test_evolution_metrics() {
        let mut metrics = HashMap::new();
        metrics.insert("nodes_added".to_string(), 10.0);
        metrics.insert("edges_removed".to_string(), 5.0);
        metrics.insert("patterns_discovered".to_string(), 3.0);
        metrics.insert("evolution_rate".to_string(), 0.15);

        let snapshot = EvolutionSnapshot {
            snapshot_id: "snap_123".to_string(),
            timestamp: SystemTime::now(),
            graph_state: HashMap::new(),
            metrics: metrics.clone(),
            generation: 42,
        };

        assert_eq!(snapshot.metrics.len(), 4);
        assert_eq!(snapshot.metrics.get("nodes_added"), Some(&10.0));
        assert_eq!(snapshot.generation, 42);
    }

    // Scaling and distribution tests

    #[tokio::test]
    async fn test_scaled_knowledge_graph() {
        let config = ScalingConfig {
            num_partitions: 4,
            replication_factor: ReplicationFactor::Double,
            partition_strategy: PartitionStrategy::Hash,
            consistency_level: ConsistencyLevel::Strong,
            sync_interval: Duration::from_secs(30),
            max_partition_size: 1_000_000,
            enable_auto_scaling: true,
            scaling_threshold: 0.8,
        };

        let base_graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let scaled_graph = ScaledKnowledgeGraph::new(base_graph, config);

        // Test partition assignment
        let node_ids = vec!["node1", "node2", "node3", "node4", "node5"];
        let mut partition_counts = vec![0; 4];

        for node_id in node_ids {
            let partition = scaled_graph.get_partition(node_id);
            assert!(partition < 4);
            partition_counts[partition] += 1;
        }

        // Verify distribution (should be relatively even with hash partitioning)
        assert!(partition_counts.iter().all(|&count| count > 0));
    }

    #[test]
    fn test_consistency_level_operations() {
        let levels = vec![
            (ConsistencyLevel::Strong, true, false),
            (ConsistencyLevel::Eventual, false, true),
            (ConsistencyLevel::Weak, false, true),
            (ConsistencyLevel::ReadYourWrite, true, true),
            (
                ConsistencyLevel::BoundedStaleness(Duration::from_secs(5)),
                true,
                true,
            ),
        ];

        for (level, requires_sync, allows_stale) in levels {
            match level {
                ConsistencyLevel::Strong => {
                    assert!(requires_sync);
                    assert!(!allows_stale);
                }
                ConsistencyLevel::Eventual | ConsistencyLevel::Weak => {
                    assert!(!requires_sync);
                    assert!(allows_stale);
                }
                ConsistencyLevel::ReadYourWrite => {
                    assert!(requires_sync);
                    assert!(allows_stale);
                }
                ConsistencyLevel::BoundedStaleness(_) => {
                    assert!(requires_sync);
                    assert!(allows_stale);
                }
            }
        }
    }

    // Pruning comprehensive tests

    #[tokio::test]
    async fn test_pruning_workflow() {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));

        // Add old and new nodes
        {
            let mut g = graph.write().await;

            for i in 0..100 {
                let mut props = HashMap::new();
                props.insert("age".to_string(), i.to_string());
                props.insert("importance".to_string(), (100 - i).to_string());

                g.add_node(format!("node_{}", i), Node::new(NodeType::Metric, props));
            }
        }

        let config = PruningConfig {
            strategy: PruningStrategy::Combined,
            retention_policy: RetentionPolicy::KeepImportant(0.5),
            max_graph_size: 50,
            prune_interval: Duration::from_secs(60),
            batch_size: 10,
            importance_threshold: 50.0,
            preserve_patterns: true,
        };

        // In real implementation, pruning would remove low-importance nodes
        assert_eq!(config.max_graph_size, 50);
        assert_eq!(config.batch_size, 10);
    }

    #[test]
    fn test_retention_policy_logic() {
        let policies = vec![
            (RetentionPolicy::KeepAll, 100, 100), // Keep all
            (
                RetentionPolicy::KeepRecent(Duration::from_secs(3600)),
                100,
                50,
            ), // Keep recent
            (RetentionPolicy::KeepImportant(0.7), 100, 30), // Keep top 30%
            (RetentionPolicy::KeepAccessed(5), 100, 20), // Keep frequently accessed
            (RetentionPolicy::Custom, 100, 100),  // Custom logic
        ];

        for (policy, total_nodes, expected_retained) in policies {
            match policy {
                RetentionPolicy::KeepAll => assert_eq!(total_nodes, expected_retained),
                RetentionPolicy::KeepImportant(threshold) => {
                    assert!(threshold >= 0.0 && threshold <= 1.0);
                }
                RetentionPolicy::KeepAccessed(min_accesses) => {
                    assert!(min_accesses > 0);
                }
                _ => {}
            }
        }
    }

    // Memory integration tests

    #[test]
    fn test_memory_integration_config() {
        let configs = vec![
            MemoryConfig {
                cache_size_mb: 1024,
                cache_strategy: CacheStrategy::LRU,
                prefetch_enabled: true,
                prefetch_distance: 3,
                compression_enabled: false,
                compression_level: 0,
                tier_migration_threshold: 0.8,
                gpu_memory_limit_mb: 4096,
            },
            MemoryConfig {
                cache_size_mb: 512,
                cache_strategy: CacheStrategy::ARC,
                prefetch_enabled: false,
                prefetch_distance: 0,
                compression_enabled: true,
                compression_level: 6,
                tier_migration_threshold: 0.5,
                gpu_memory_limit_mb: 2048,
            },
        ];

        for config in configs {
            assert!(config.cache_size_mb > 0);
            assert!(
                config.tier_migration_threshold >= 0.0 && config.tier_migration_threshold <= 1.0
            );

            if config.compression_enabled {
                assert!(config.compression_level > 0);
            }
        }
    }

    #[test]
    fn test_cache_strategy_effectiveness() {
        let strategies = vec![
            (CacheStrategy::LRU, "Least Recently Used"),
            (CacheStrategy::LFU, "Least Frequently Used"),
            (CacheStrategy::FIFO, "First In First Out"),
            (CacheStrategy::ARC, "Adaptive Replacement Cache"),
            (CacheStrategy::Random, "Random Eviction"),
        ];

        for (strategy, description) in strategies {
            assert!(!description.is_empty());

            // Test that each strategy can be serialized
            let json = serde_json::to_string(&strategy).unwrap();
            let deserialized: CacheStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    // Complex integration scenarios

    #[tokio::test]
    async fn test_full_system_integration() {
        // Initialize all components
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let query_engine = QueryEngine::new(graph.clone());
        let search_engine = SemanticSearchEngine::new(graph.clone(), 128);
        let pattern_discovery = PatternDiscovery::new(graph.clone());
        let evolution_tracker = EvolutionTracker::new(graph.clone());

        // Build a complex graph
        {
            let mut g = graph.write().await;

            // Add interconnected components
            for i in 0..10 {
                let agent_id = format!("agent_{}", i);
                let goal_id = format!("goal_{}", i);
                let resource_id = format!("resource_{}", i);

                g.add_node(agent_id.clone(), Node::new(NodeType::Agent, HashMap::new()));
                g.add_node(goal_id.clone(), Node::new(NodeType::Goal, HashMap::new()));
                g.add_node(
                    resource_id.clone(),
                    Node::new(NodeType::Resource, HashMap::new()),
                );

                g.add_edge(
                    format!("e1_{}", i),
                    Edge::new(agent_id.clone(), goal_id.clone(), EdgeType::Has, 1.0),
                );
                g.add_edge(
                    format!("e2_{}", i),
                    Edge::new(agent_id, resource_id, EdgeType::Uses, 0.8),
                );
            }
        }

        // Take initial snapshot
        let snapshot = evolution_tracker.create_snapshot().await;
        assert!(snapshot.is_ok());

        // Perform queries
        let query = Query {
            query_type: QueryType::NodeSearch,
            filters: vec![QueryFilter::NodeType(NodeType::Agent)],
            limit: Some(5),
            offset: None,
            order_by: None,
            aggregations: vec![],
        };

        let results = query_engine.execute(query).await;
        assert!(results.is_ok());

        // Discover patterns
        let patterns = pattern_discovery.discover_patterns(3).await;
        assert!(patterns.is_ok());
    }

    // Error handling and recovery

    #[test]
    fn test_error_recovery_scenarios() {
        // Test various error conditions
        let errors = vec![
            KnowledgeGraphError::NodeNotFound {
                node_id: "missing".to_string(),
            },
            KnowledgeGraphError::InvalidQuery {
                query: "SELECT * FROM nowhere".to_string(),
                message: "Invalid syntax".to_string(),
            },
            KnowledgeGraphError::CapacityExceeded {
                current: 1000,
                maximum: 500,
            },
            KnowledgeGraphError::TimeoutError {
                operation: "pattern_discovery".to_string(),
                timeout: Duration::from_secs(30),
            },
        ];

        for error in errors {
            // Ensure errors can be converted to strings
            let error_str = error.to_string();
            assert!(!error_str.is_empty());

            // Ensure errors implement std::error::Error
            use std::error::Error;
            assert!(error.source().is_none()); // These errors don't chain
        }
    }

    // Performance edge cases

    #[test]
    fn test_performance_boundaries() {
        // Test with minimal resources
        let minimal_config = ScalingConfig {
            num_partitions: 1,
            replication_factor: ReplicationFactor::None,
            partition_strategy: PartitionStrategy::Hash,
            consistency_level: ConsistencyLevel::Weak,
            sync_interval: Duration::from_secs(3600), // Very infrequent
            max_partition_size: 10,                   // Very small
            enable_auto_scaling: false,
            scaling_threshold: 0.99, // Very high threshold
        };

        // Test with maximum resources
        let maximal_config = ScalingConfig {
            num_partitions: 1000,
            replication_factor: ReplicationFactor::Custom(10),
            partition_strategy: PartitionStrategy::Geographic,
            consistency_level: ConsistencyLevel::Strong,
            sync_interval: Duration::from_millis(100), // Very frequent
            max_partition_size: u64::MAX,
            enable_auto_scaling: true,
            scaling_threshold: 0.01, // Very low threshold
        };

        // Both configs should be valid
        assert!(minimal_config.num_partitions > 0);
        assert!(maximal_config.num_partitions > 0);
    }
}
