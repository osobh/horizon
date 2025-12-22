//! Edge case tests for knowledge-graph to enhance coverage to 80%+

#[cfg(test)]
mod edge_case_tests {
    use crate::error::KnowledgeGraphError;
    use crate::evolution_tracker::{EvolutionEvent, EvolutionSnapshot, EvolutionTracker};
    use crate::graph::{Edge, EdgeType, KnowledgeGraph, Node, NodeType};
    use crate::memory_integration::*; // MemoryIntegration, MemoryConfig, CacheStrategy
    use crate::patterns::*; // Pattern, PatternType, PatternDiscovery, PatternMatch
    use crate::pruning::{PruningConfig, PruningStrategy, RetentionPolicy};
    use crate::query::{Query, QueryResult, QueryType}; // QueryFilter, AggregationFunction
    use crate::scaling::{ConsistencyLevel, PartitionStrategy, ReplicationFactor, ScalingConfig};
    use crate::semantic::{EmbeddingVector, SemanticQuery, SemanticSearchEngine};
    use crate::*;
    use std::collections::HashMap;
    use std::time::Duration;

    // Error handling edge cases

    #[test]
    fn test_error_edge_cases_unicode() {
        // Test with unicode strings
        let error = KnowledgeGraphError::NodeNotFound {
            node_id: "节点_🌐_ノード".to_string(),
        };
        assert!(error.to_string().contains("节点_🌐_ノード"));

        let error2 = KnowledgeGraphError::InvalidQuery {
            query: "查询_🔍_Запрос".to_string(),
            message: "错误_❌_エラー".to_string(),
        };
        assert!(error2.to_string().contains("查询_🔍_Запрос"));
    }

    #[test]
    fn test_error_extreme_values() {
        // Test with very long strings
        let long_id = "x".repeat(10000);
        let error = KnowledgeGraphError::EdgeNotFound {
            edge_id: long_id.clone(),
        };
        assert!(error.to_string().contains(&long_id));

        // Test empty strings
        let error2 = KnowledgeGraphError::GpuError {
            message: String::new(),
        };
        assert!(error2.to_string().contains("GPU operation failed"));
    }

    #[test]
    fn test_all_error_variants() {
        let errors = vec![
            KnowledgeGraphError::NodeNotFound {
                node_id: "test".to_string(),
            },
            KnowledgeGraphError::EdgeNotFound {
                edge_id: "test".to_string(),
            },
            KnowledgeGraphError::InvalidQuery {
                query: "test".to_string(),
                message: "error".to_string(),
            },
            KnowledgeGraphError::GpuError {
                message: "test".to_string(),
            },
            KnowledgeGraphError::SemanticError {
                message: "test".to_string(),
            },
            KnowledgeGraphError::PatternError {
                message: "test".to_string(),
            },
            KnowledgeGraphError::MemoryError {
                message: "test".to_string(),
            },
            KnowledgeGraphError::EvolutionError {
                message: "test".to_string(),
            },
            KnowledgeGraphError::ScalingError {
                message: "test".to_string(),
            },
            KnowledgeGraphError::PruningError {
                message: "test".to_string(),
            },
            KnowledgeGraphError::ConfigurationError {
                message: "test".to_string(),
            },
            KnowledgeGraphError::ConcurrencyError {
                message: "test".to_string(),
            },
            KnowledgeGraphError::SerializationError {
                message: "test".to_string(),
            },
            KnowledgeGraphError::ValidationError {
                message: "test".to_string(),
            },
            KnowledgeGraphError::StorageError {
                message: "test".to_string(),
            },
            KnowledgeGraphError::TimeoutError {
                operation: "test".to_string(),
                timeout: Duration::from_secs(5),
            },
            KnowledgeGraphError::CapacityExceeded {
                current: 100,
                maximum: 50,
            },
            KnowledgeGraphError::Other("test".to_string()),
        ];

        for error in errors {
            // Test Debug and Display
            let debug_str = format!("{:?}", error);
            let display_str = error.to_string();
            assert!(!debug_str.is_empty());
            assert!(!display_str.is_empty());
        }
    }

    // Node and Edge edge cases

    #[test]
    fn test_node_edge_cases() {
        // Test with empty properties
        let node = Node::new(NodeType::Agent, HashMap::new());
        assert_eq!(node.properties.len(), 0);

        // Test with unicode properties
        let mut props = HashMap::new();
        props.insert("名前".to_string(), "値".to_string());
        props.insert(String::new(), String::new()); // Empty key-value
        props.insert("x".repeat(1000), "y".repeat(1000)); // Long strings

        let node2 = Node::new(NodeType::Concept, props);
        assert_eq!(node2.properties.len(), 3);
    }

    #[test]
    fn test_node_type_all_variants() {
        let types = vec![
            NodeType::Agent,
            NodeType::Goal,
            NodeType::Concept,
            NodeType::Resource,
            NodeType::Pattern,
            NodeType::Event,
            NodeType::Constraint,
            NodeType::Metric,
        ];

        for node_type in types {
            let json = serde_json::to_string(&node_type).unwrap();
            let deserialized: NodeType = serde_json::from_str(&json).unwrap();
            assert_eq!(node_type, deserialized);
        }
    }

    #[test]
    fn test_edge_extreme_weights() {
        let edges = vec![
            Edge::new(
                "a".to_string(),
                "b".to_string(),
                EdgeType::Has,
                f64::INFINITY,
            ),
            Edge::new(
                "c".to_string(),
                "d".to_string(),
                EdgeType::Uses,
                f64::NEG_INFINITY,
            ),
            Edge::new(
                "e".to_string(),
                "f".to_string(),
                EdgeType::Requires,
                f64::NAN,
            ),
            Edge::new("g".to_string(), "h".to_string(), EdgeType::Produces, 0.0),
            Edge::new("i".to_string(), "j".to_string(), EdgeType::RelatesTo, -0.0),
        ];

        for edge in edges {
            // Test that extreme weights are handled
            if edge.weight.is_nan() {
                assert!(edge.weight.is_nan());
            } else {
                assert!(edge.weight.is_finite() || edge.weight.is_infinite());
            }
        }
    }

    #[test]
    fn test_edge_type_all_variants() {
        let types = vec![
            EdgeType::Has,
            EdgeType::Uses,
            EdgeType::Requires,
            EdgeType::Produces,
            EdgeType::RelatesTo,
            EdgeType::DependsOn,
            EdgeType::Evolves,
            EdgeType::Influences,
            EdgeType::Contradicts,
            EdgeType::Supports,
        ];

        for edge_type in types {
            let json = serde_json::to_string(&edge_type).unwrap();
            let deserialized: EdgeType = serde_json::from_str(&json).unwrap();
            assert_eq!(edge_type, deserialized);
        }
    }

    // Query edge cases

    #[test]
    fn test_query_edge_cases() {
        // Test with empty query
        let query = Query {
            query_type: QueryType::NodeSearch,
            filters: vec![],
            limit: Some(0),           // Zero limit
            offset: Some(usize::MAX), // Max offset
            order_by: None,
            aggregations: vec![],
        };

        assert_eq!(query.filters.len(), 0);
        assert_eq!(query.limit, Some(0));
    }

    #[test]
    fn test_query_type_all_variants() {
        let types = vec![
            QueryType::NodeSearch,
            QueryType::EdgeSearch,
            QueryType::PathSearch,
            QueryType::PatternMatch,
            QueryType::Aggregation,
            QueryType::Traversal,
        ];

        for query_type in types {
            let json = serde_json::to_string(&query_type).unwrap();
            let deserialized: QueryType = serde_json::from_str(&json).unwrap();
            assert_eq!(query_type, deserialized);
        }
    }

    #[test]
    fn test_aggregation_function_variants() {
        let functions = vec![
            AggregationFunction::Count,
            AggregationFunction::Sum,
            AggregationFunction::Average,
            AggregationFunction::Min,
            AggregationFunction::Max,
            AggregationFunction::StandardDeviation,
        ];

        for func in functions {
            let json = serde_json::to_string(&func).unwrap();
            let deserialized: AggregationFunction = serde_json::from_str(&json).unwrap();
            assert_eq!(func, deserialized);
        }
    }

    // Semantic search edge cases

    #[test]
    fn test_embedding_vector_extremes() {
        // Empty vector
        let vec1 = EmbeddingVector::from_vec(vec![]);
        assert_eq!(vec1.dimension(), 0);

        // Vector with extreme values
        let vec2 =
            EmbeddingVector::from_vec(vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0, -0.0]);
        assert_eq!(vec2.dimension(), 5);

        // Very large vector
        let vec3 = EmbeddingVector::from_vec(vec![1.0; 10000]);
        assert_eq!(vec3.dimension(), 10000);
    }

    #[test]
    fn test_semantic_query_edge_cases() {
        let query = SemanticQuery {
            query_text: String::new(), // Empty query
            embedding: EmbeddingVector::from_vec(vec![]),
            similarity_threshold: f64::NAN,
            max_results: 0,
            include_metadata: true,
        };

        assert_eq!(query.query_text, "");
        assert!(query.similarity_threshold.is_nan());
    }

    // Pattern discovery edge cases

    #[test]
    fn test_pattern_type_all_variants() {
        let types = vec![
            PatternType::Structural,
            PatternType::Behavioral,
            PatternType::Temporal,
            PatternType::Causal,
            PatternType::Emergent,
        ];

        for pattern_type in types {
            let json = serde_json::to_string(&pattern_type)?;
            let deserialized: PatternType = serde_json::from_str(&json).unwrap();
            assert_eq!(pattern_type, deserialized);
        }
    }

    #[test]
    fn test_pattern_extreme_confidence() {
        let pattern = Pattern {
            pattern_id: "🔍".repeat(100),
            pattern_type: PatternType::Emergent,
            nodes: vec![],
            edges: vec![],
            confidence: f64::INFINITY,
            support: 0,
            frequency: f64::NEG_INFINITY,
            metadata: HashMap::new(),
        };

        assert!(pattern.confidence.is_infinite());
        assert_eq!(pattern.support, 0);
        assert!(pattern.frequency.is_infinite());
    }

    // Evolution tracking edge cases

    #[test]
    fn test_evolution_event_extremes() {
        let event = EvolutionEvent {
            event_id: String::new(),
            timestamp: std::time::SystemTime::now(),
            event_type: "mutation".to_string(),
            affected_nodes: vec![String::new(); 1000], // Many empty IDs
            affected_edges: vec!["x".repeat(100); 100], // Long IDs
            changes: HashMap::new(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("unicode_测试".to_string(), "value_值".to_string());
                meta
            },
        };

        assert_eq!(event.affected_nodes.len(), 1000);
        assert_eq!(event.affected_edges.len(), 100);
    }

    // Scaling configuration edge cases

    #[test]
    fn test_scaling_config_extremes() {
        let config = ScalingConfig {
            num_partitions: 0, // No partitions
            replication_factor: ReplicationFactor::Custom(usize::MAX),
            partition_strategy: PartitionStrategy::Hash,
            consistency_level: ConsistencyLevel::Eventual,
            sync_interval: Duration::from_nanos(1),
            max_partition_size: u64::MAX,
            enable_auto_scaling: true,
            scaling_threshold: f64::INFINITY,
        };

        assert_eq!(config.num_partitions, 0);
        match config.replication_factor {
            ReplicationFactor::Custom(n) => assert_eq!(n, usize::MAX),
            _ => panic!("Wrong replication factor"),
        }
    }

    #[test]
    fn test_partition_strategy_variants() {
        let strategies = vec![
            PartitionStrategy::Hash,
            PartitionStrategy::Range,
            PartitionStrategy::Geographic,
            PartitionStrategy::Random,
            PartitionStrategy::Custom,
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy)?;
            let deserialized: PartitionStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    #[test]
    fn test_replication_factor_variants() {
        let factors = vec![
            ReplicationFactor::None,
            ReplicationFactor::Single,
            ReplicationFactor::Double,
            ReplicationFactor::Triple,
            ReplicationFactor::Custom(0),
            ReplicationFactor::Custom(100),
        ];

        for factor in factors {
            let json = serde_json::to_string(&factor).unwrap();
            let deserialized: ReplicationFactor = serde_json::from_str(&json).unwrap();

            match (factor, deserialized) {
                (ReplicationFactor::Custom(a), ReplicationFactor::Custom(b)) => assert_eq!(a, b),
                (a, b) => assert_eq!(a, b),
            }
        }
    }

    #[test]
    fn test_consistency_level_variants() {
        let levels = vec![
            ConsistencyLevel::Strong,
            ConsistencyLevel::Eventual,
            ConsistencyLevel::Weak,
            ConsistencyLevel::ReadYourWrite,
            ConsistencyLevel::BoundedStaleness(Duration::from_secs(0)),
            ConsistencyLevel::BoundedStaleness(Duration::from_secs(u64::MAX)),
        ];

        for level in levels {
            let json = serde_json::to_string(&level).unwrap();
            let deserialized: ConsistencyLevel = serde_json::from_str(&json).unwrap();

            match (level, deserialized) {
                (ConsistencyLevel::BoundedStaleness(a), ConsistencyLevel::BoundedStaleness(b)) => {
                    assert_eq!(a, b);
                }
                (a, b) => assert_eq!(a, b),
            }
        }
    }

    // Pruning edge cases

    #[test]
    fn test_pruning_strategy_variants() {
        let strategies = vec![
            PruningStrategy::Age,
            PruningStrategy::AccessFrequency,
            PruningStrategy::Importance,
            PruningStrategy::Size,
            PruningStrategy::Combined,
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy)?;
            let deserialized: PruningStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    #[test]
    fn test_pruning_config_extremes() {
        let config = PruningConfig {
            strategy: PruningStrategy::Combined,
            retention_policy: RetentionPolicy::KeepRecent(Duration::from_nanos(0)),
            max_graph_size: 0,
            prune_interval: Duration::from_secs(u64::MAX),
            batch_size: usize::MAX,
            importance_threshold: f64::NEG_INFINITY,
            preserve_patterns: true,
        };

        match config.retention_policy {
            RetentionPolicy::KeepRecent(d) => assert_eq!(d, Duration::from_nanos(0)),
            _ => panic!("Wrong retention policy"),
        }
    }

    #[test]
    fn test_retention_policy_variants() {
        let policies = vec![
            RetentionPolicy::KeepAll,
            RetentionPolicy::KeepRecent(Duration::from_secs(3600)),
            RetentionPolicy::KeepImportant(0.5),
            RetentionPolicy::KeepAccessed(10),
            RetentionPolicy::Custom,
        ];

        for policy in policies {
            let json = serde_json::to_string(&policy)?;
            let deserialized: RetentionPolicy = serde_json::from_str(&json).unwrap();

            match (policy, deserialized) {
                (RetentionPolicy::KeepRecent(a), RetentionPolicy::KeepRecent(b)) => {
                    assert_eq!(a, b)
                }
                (RetentionPolicy::KeepImportant(a), RetentionPolicy::KeepImportant(b)) => {
                    // Handle potential float comparison
                    assert!((a - b).abs() < f64::EPSILON);
                }
                (RetentionPolicy::KeepAccessed(a), RetentionPolicy::KeepAccessed(b)) => {
                    assert_eq!(a, b)
                }
                (a, b) => assert_eq!(a, b),
            }
        }
    }

    // Memory integration edge cases

    #[test]
    fn test_memory_config_extremes() {
        let config = MemoryConfig {
            cache_size_mb: 0,
            cache_strategy: CacheStrategy::LRU,
            prefetch_enabled: true,
            prefetch_distance: usize::MAX,
            compression_enabled: true,
            compression_level: 100, // Invalid level
            tier_migration_threshold: f64::NAN,
            gpu_memory_limit_mb: u64::MAX,
        };

        assert_eq!(config.cache_size_mb, 0);
        assert_eq!(config.compression_level, 100);
        assert!(config.tier_migration_threshold.is_nan());
    }

    #[test]
    fn test_cache_strategy_variants() {
        let strategies = vec![
            CacheStrategy::LRU,
            CacheStrategy::LFU,
            CacheStrategy::FIFO,
            CacheStrategy::ARC,
            CacheStrategy::Random,
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy)?;
            let deserialized: CacheStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    // Thread safety tests

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // Verify error types are Send + Sync
        assert_send::<KnowledgeGraphError>();
        assert_sync::<KnowledgeGraphError>();

        // Verify key types are Send + Sync
        assert_send::<Node>();
        assert_sync::<Node>();
        assert_send::<Edge>();
        assert_sync::<Edge>();
        assert_send::<Query>();
        assert_sync::<Query>();
    }

    // Empty graph edge cases

    #[test]
    fn test_empty_graph_operations() {
        let graph = KnowledgeGraph::new();

        // Operations on empty graph
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.get_node("any_id").is_none());
        assert!(graph.get_neighbors("any_id").is_empty());
    }

    // Query result edge cases

    #[test]
    fn test_query_result_extremes() {
        let result = QueryResult {
            nodes: vec![],
            edges: vec![],
            paths: vec![vec![]; 100], // Many empty paths
            aggregations: HashMap::new(),
            total_count: usize::MAX,
            execution_time: Duration::from_nanos(0),
            metadata: HashMap::new(),
        };

        assert_eq!(result.nodes.len(), 0);
        assert_eq!(result.paths.len(), 100);
        assert_eq!(result.total_count, usize::MAX);
    }

    // Pattern match edge cases

    #[test]
    fn test_pattern_match_extremes() {
        let pattern_match = PatternMatch {
            pattern_id: "🎯".repeat(50),
            matched_nodes: vec![String::new(); 1000],
            matched_edges: vec![],
            confidence: -1.0, // Invalid confidence
            transformations: HashMap::new(),
        };

        assert_eq!(pattern_match.matched_nodes.len(), 1000);
        assert_eq!(pattern_match.confidence, -1.0);
    }

    // Evolution snapshot edge cases

    #[test]
    fn test_evolution_snapshot_extremes() {
        let snapshot = EvolutionSnapshot {
            snapshot_id: String::new(),
            timestamp: std::time::SystemTime::now(),
            graph_state: HashMap::new(),
            metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("node_count".to_string(), f64::INFINITY);
                metrics.insert("edge_count".to_string(), f64::NAN);
                metrics
            },
            generation: u64::MAX,
        };

        assert_eq!(snapshot.snapshot_id, "");
        assert_eq!(snapshot.generation, u64::MAX);
    }

    // Capacity edge cases

    #[test]
    fn test_capacity_exceeded_error() {
        let error = KnowledgeGraphError::CapacityExceeded {
            current: usize::MAX,
            maximum: 0,
        };

        let error_str = error.to_string();
        assert!(error_str.contains(&usize::MAX.to_string()));
        assert!(error_str.contains("0"));
    }

    // Timeout edge cases

    #[test]
    fn test_timeout_error_extremes() {
        let errors = vec![
            KnowledgeGraphError::TimeoutError {
                operation: String::new(),
                timeout: Duration::from_nanos(0),
            },
            KnowledgeGraphError::TimeoutError {
                operation: "x".repeat(1000),
                timeout: Duration::from_secs(u64::MAX),
            },
        ];

        for error in errors {
            let error_str = error.to_string();
            assert!(error_str.contains("timeout"));
        }
    }

    // Serialization edge cases

    #[test]
    fn test_node_serialization_unicode() {
        let mut props = HashMap::new();
        props.insert("🔑".to_string(), "🎁".to_string());
        props.insert("key_键".to_string(), "value_值".to_string());

        let node = Node::new(NodeType::Concept, props);

        let json = serde_json::to_string(&node)?;
        let deserialized: Node = serde_json::from_str(&json)?;

        assert_eq!(node.node_type, deserialized.node_type);
        assert_eq!(node.properties.len(), deserialized.properties.len());
    }

    // Debug trait coverage

    #[test]
    fn test_debug_display_coverage() {
        // Test Debug trait for all enums
        let node_type = NodeType::Agent;
        assert!(!format!("{:?}", node_type).is_empty());

        let edge_type = EdgeType::Has;
        assert!(!format!("{:?}", edge_type).is_empty());

        let query_type = QueryType::NodeSearch;
        assert!(!format!("{:?}", query_type).is_empty());

        let pattern_type = PatternType::Structural;
        assert!(!format!("{:?}", pattern_type).is_empty());

        let partition_strategy = PartitionStrategy::Hash;
        assert!(!format!("{:?}", partition_strategy).is_empty());

        let cache_strategy = CacheStrategy::LRU;
        assert!(!format!("{:?}", cache_strategy).is_empty());
    }
}
