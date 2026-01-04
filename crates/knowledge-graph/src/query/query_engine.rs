//! Main query engine implementation
//!
//! This module contains the QueryEngine struct and its core functionality
//! for managing queries, statistics, and coordination between different
//! execution strategies.

use super::query_execution::QueryExecutor;
use super::query_types::*;
use crate::error::KnowledgeGraphResult;
use crate::graph::KnowledgeGraph;
use std::time::Instant;

/// High-performance query engine
pub struct QueryEngine {
    /// Query executor for handling specific query types
    executor: QueryExecutor,
    /// Engine statistics
    stats: QueryStats,
    /// GPU enablement flag
    gpu_enabled: bool,
}

/// Query execution statistics
#[derive(Debug, Clone, Default)]
pub struct QueryStats {
    /// Total queries executed
    pub total_queries: u64,
    /// GPU-accelerated queries
    pub gpu_queries: u64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Cache hits (for future caching implementation)
    pub cache_hits: u64,
    /// Query type statistics
    pub query_type_counts: std::collections::HashMap<String, u64>,
    /// Error count
    pub error_count: u64,
}

impl QueryEngine {
    /// Create a new query engine
    pub async fn new(gpu_enabled: bool) -> KnowledgeGraphResult<Self> {
        let executor = QueryExecutor::new(gpu_enabled).await?;

        Ok(Self {
            executor,
            stats: QueryStats::default(),
            gpu_enabled,
        })
    }

    /// Execute a query against the knowledge graph
    pub async fn execute(
        &mut self,
        graph: &KnowledgeGraph,
        query: Query,
    ) -> KnowledgeGraphResult<QueryResult> {
        let start_time = Instant::now();
        let query_type_name = self.get_query_type_name(&query.query_type);

        // Update query type statistics
        *self
            .stats
            .query_type_counts
            .entry(query_type_name.clone())
            .or_insert(0) += 1;

        let result = match self.execute_query_internal(graph, &query).await {
            Ok(mut result) => {
                // Update success statistics
                self.stats.total_queries += 1;
                if result.gpu_accelerated {
                    self.stats.gpu_queries += 1;
                }

                let execution_time_ms = start_time.elapsed().as_millis() as u64;
                result.execution_time_ms = execution_time_ms;

                // Update average execution time
                self.update_average_execution_time(execution_time_ms);

                Ok(result)
            }
            Err(error) => {
                // Update error statistics
                self.stats.error_count += 1;
                Err(error)
            }
        };

        result
    }

    /// Internal query execution dispatcher
    async fn execute_query_internal(
        &mut self,
        graph: &KnowledgeGraph,
        query: &Query,
    ) -> KnowledgeGraphResult<QueryResult> {
        match &query.query_type {
            QueryType::FindNodes {
                node_type,
                properties,
            } => {
                self.executor
                    .execute_find_nodes(graph, node_type.clone(), properties.clone(), query)
                    .await
            }
            QueryType::FindEdges {
                edge_type,
                weight_range,
            } => {
                self.executor
                    .execute_find_edges(graph, edge_type.clone(), *weight_range, query)
                    .await
            }
            QueryType::FindPath {
                source_id,
                target_id,
                max_length,
            } => {
                self.executor
                    .execute_find_path(graph, source_id, target_id, *max_length, query)
                    .await
            }
            QueryType::Neighborhood {
                node_id,
                radius,
                edge_types,
            } => {
                self.executor
                    .execute_neighborhood(graph, node_id, *radius, edge_types.clone(), query)
                    .await
            }
            QueryType::Subgraph {
                seed_nodes,
                max_depth,
            } => {
                self.executor
                    .execute_subgraph(graph, seed_nodes.clone(), *max_depth, query)
                    .await
            }
            QueryType::PatternMatch { pattern } => {
                self.executor
                    .execute_pattern_match(graph, pattern.clone(), query)
                    .await
            }
            QueryType::Ranking {
                algorithm,
                node_type,
            } => {
                self.executor
                    .execute_ranking(graph, algorithm.clone(), node_type.clone(), query)
                    .await
            }
        }
    }

    /// Get query statistics
    pub fn stats(&self) -> &QueryStats {
        &self.stats
    }

    /// Reset query statistics
    pub fn reset_stats(&mut self) {
        self.stats = QueryStats::default();
        self.executor.reset_stats();
    }

    /// Check if GPU is enabled
    pub fn is_gpu_enabled(&self) -> bool {
        self.gpu_enabled
    }

    /// Get executor statistics
    pub fn executor_stats(&self) -> &crate::query::query_execution::ExecutionStats {
        self.executor.stats()
    }

    /// Execute multiple queries in batch
    pub async fn execute_batch(
        &mut self,
        graph: &KnowledgeGraph,
        queries: Vec<Query>,
    ) -> Vec<KnowledgeGraphResult<QueryResult>> {
        let mut results = Vec::with_capacity(queries.len());

        for query in queries {
            let result = self.execute(graph, query).await;
            results.push(result);
        }

        results
    }

    /// Execute query with timeout
    pub async fn execute_with_timeout(
        &mut self,
        graph: &KnowledgeGraph,
        query: Query,
        timeout_ms: u64,
    ) -> KnowledgeGraphResult<QueryResult> {
        let query_with_timeout = Query {
            timeout_ms: Some(timeout_ms),
            ..query
        };

        // For now, we don't implement actual timeout handling
        // This would require tokio::time::timeout in a real implementation
        self.execute(graph, query_with_timeout).await
    }

    /// Validate query before execution
    pub fn validate_query(&self, query: &Query) -> KnowledgeGraphResult<()> {
        match &query.query_type {
            QueryType::FindNodes { properties, .. } => {
                if properties.len() > 100 {
                    return Err(crate::error::KnowledgeGraphError::Other(
                        "Too many property filters".to_string(),
                    ));
                }
            }
            QueryType::FindPath {
                source_id,
                target_id,
                max_length,
            } => {
                if source_id.is_empty() || target_id.is_empty() {
                    return Err(crate::error::KnowledgeGraphError::Other(
                        "Source and target IDs cannot be empty".to_string(),
                    ));
                }
                if *max_length == 0 || *max_length > 1000 {
                    return Err(crate::error::KnowledgeGraphError::Other(
                        "Invalid path length constraint".to_string(),
                    ));
                }
            }
            QueryType::Neighborhood { radius, .. } => {
                if *radius == 0 || *radius > 100 {
                    return Err(crate::error::KnowledgeGraphError::Other(
                        "Invalid neighborhood radius".to_string(),
                    ));
                }
            }
            QueryType::Subgraph {
                seed_nodes,
                max_depth,
            } => {
                if seed_nodes.is_empty() {
                    return Err(crate::error::KnowledgeGraphError::Other(
                        "Subgraph requires at least one seed node".to_string(),
                    ));
                }
                if *max_depth == 0 || *max_depth > 50 {
                    return Err(crate::error::KnowledgeGraphError::Other(
                        "Invalid subgraph depth".to_string(),
                    ));
                }
            }
            QueryType::PatternMatch { pattern } => {
                if !pattern.is_valid() {
                    return Err(crate::error::KnowledgeGraphError::Other(
                        "Invalid graph pattern".to_string(),
                    ));
                }
            }
            _ => {} // Other query types don't need validation
        }

        // Validate common constraints
        if let Some(limit) = query.limit {
            if limit == 0 || limit > 10000 {
                return Err(crate::error::KnowledgeGraphError::Other(
                    "Invalid result limit".to_string(),
                ));
            }
        }

        if let Some(timeout) = query.timeout_ms {
            if timeout == 0 || timeout > 300000 {
                // 5 minutes max
                return Err(crate::error::KnowledgeGraphError::Other(
                    "Invalid timeout value".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Get query type name for statistics
    fn get_query_type_name(&self, query_type: &QueryType) -> String {
        match query_type {
            QueryType::FindNodes { .. } => "find_nodes".to_string(),
            QueryType::FindEdges { .. } => "find_edges".to_string(),
            QueryType::FindPath { .. } => "find_path".to_string(),
            QueryType::Neighborhood { .. } => "neighborhood".to_string(),
            QueryType::Subgraph { .. } => "subgraph".to_string(),
            QueryType::PatternMatch { .. } => "pattern_match".to_string(),
            QueryType::Ranking { .. } => "ranking".to_string(),
        }
    }

    /// Update average execution time
    fn update_average_execution_time(&mut self, execution_time_ms: u64) {
        let current_avg = self.stats.avg_execution_time_ms;
        let total_queries = self.stats.total_queries;

        if total_queries == 1 {
            self.stats.avg_execution_time_ms = execution_time_ms as f64;
        } else {
            self.stats.avg_execution_time_ms = (current_avg * (total_queries - 1) as f64
                + execution_time_ms as f64)
                / total_queries as f64;
        }
    }
}

impl QueryStats {
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            (self.total_queries - self.error_count) as f64 / self.total_queries as f64
        }
    }

    /// Get GPU utilization ratio
    pub fn gpu_utilization(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.gpu_queries as f64 / self.total_queries as f64
        }
    }

    /// Get most frequent query type
    pub fn most_frequent_query_type(&self) -> Option<String> {
        self.query_type_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(query_type, _)| query_type.clone())
    }

    /// Get total execution time estimate
    pub fn total_execution_time_estimate_ms(&self) -> f64 {
        self.avg_execution_time_ms * self.total_queries as f64
    }

    /// Export statistics as JSON
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "total_queries": self.total_queries,
            "gpu_queries": self.gpu_queries,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "cache_hits": self.cache_hits,
            "error_count": self.error_count,
            "success_rate": self.success_rate(),
            "gpu_utilization": self.gpu_utilization(),
            "query_type_counts": self.query_type_counts,
            "most_frequent_query_type": self.most_frequent_query_type(),
            "total_execution_time_estimate_ms": self.total_execution_time_estimate_ms()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{KnowledgeGraphConfig, Node, NodeType};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_query_engine_creation() {
        let cpu_engine = QueryEngine::new(false).await;
        assert!(cpu_engine.is_ok());

        let engine = cpu_engine.unwrap();
        assert!(!engine.is_gpu_enabled());
        assert_eq!(engine.stats().total_queries, 0);

        let gpu_engine = QueryEngine::new(true).await;
        assert!(gpu_engine.is_ok());

        let engine = gpu_engine.unwrap();
        assert!(engine.is_gpu_enabled());
    }

    #[tokio::test]
    async fn test_query_execution_and_stats() {
        let mut engine = QueryEngine::new(false).await.unwrap();
        let mut graph = KnowledgeGraph::new(KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        })
        .await?;

        // Add test node
        let node = Node::new(NodeType::Agent, HashMap::new());
        graph.add_node(node).unwrap();

        let query = Query::new(QueryType::FindNodes {
            node_type: Some(NodeType::Agent),
            properties: HashMap::new(),
        });

        let result = engine.execute(&graph, query).await;
        assert!(result.is_ok());

        let stats = engine.stats();
        assert_eq!(stats.total_queries, 1);
        assert_eq!(stats.gpu_queries, 0);
        assert!(stats.avg_execution_time_ms >= 0.0);
        assert_eq!(stats.error_count, 0);
        assert_eq!(stats.success_rate(), 1.0);
    }

    #[tokio::test]
    async fn test_batch_execution() {
        let mut engine = QueryEngine::new(false).await.unwrap();
        let graph = KnowledgeGraph::new(KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        })
        .await?;

        let queries = vec![
            Query::new(QueryType::FindNodes {
                node_type: Some(NodeType::Agent),
                properties: HashMap::new(),
            }),
            Query::new(QueryType::FindEdges {
                edge_type: None,
                weight_range: None,
            }),
        ];

        let results = engine.execute_batch(&graph, queries).await;
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());

        let stats = engine.stats();
        assert_eq!(stats.total_queries, 2);
    }

    #[tokio::test]
    async fn test_query_validation() {
        let engine = QueryEngine::new(false).await.unwrap();

        // Valid query
        let valid_query = Query::new(QueryType::FindPath {
            source_id: "a".to_string(),
            target_id: "b".to_string(),
            max_length: 5,
        });
        assert!(engine.validate_query(&valid_query).is_ok());

        // Invalid query - empty source ID
        let invalid_query = Query::new(QueryType::FindPath {
            source_id: "".to_string(),
            target_id: "b".to_string(),
            max_length: 5,
        });
        assert!(engine.validate_query(&invalid_query).is_err());

        // Invalid query - too large radius
        let invalid_query = Query::new(QueryType::Neighborhood {
            node_id: "a".to_string(),
            radius: 200,
            edge_types: None,
        });
        assert!(engine.validate_query(&invalid_query).is_err());

        // Invalid query - empty subgraph seeds
        let invalid_query = Query::new(QueryType::Subgraph {
            seed_nodes: vec![],
            max_depth: 3,
        });
        assert!(engine.validate_query(&invalid_query).is_err());
    }

    #[tokio::test]
    async fn test_error_handling() {
        let mut engine = QueryEngine::new(false).await.unwrap();
        let graph = KnowledgeGraph::new(KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        })
        .await?;

        // This should trigger validation error
        let invalid_query = Query::new(QueryType::FindPath {
            source_id: "".to_string(),
            target_id: "b".to_string(),
            max_length: 5,
        });

        // First validate, then try to execute
        assert!(engine.validate_query(&invalid_query).is_err());

        // Stats should not be affected by validation errors
        let stats = engine.stats();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.error_count, 0);
    }

    #[test]
    fn test_query_stats_methods() {
        let mut stats = QueryStats::default();

        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.gpu_utilization(), 0.0);
        assert!(stats.most_frequent_query_type().is_none());

        stats.total_queries = 10;
        stats.gpu_queries = 3;
        stats.error_count = 1;
        stats.avg_execution_time_ms = 150.0;
        stats.query_type_counts.insert("find_nodes".to_string(), 5);
        stats.query_type_counts.insert("find_path".to_string(), 3);

        assert_eq!(stats.success_rate(), 0.9);
        assert_eq!(stats.gpu_utilization(), 0.3);
        assert_eq!(
            stats.most_frequent_query_type(),
            Some("find_nodes".to_string())
        );
        assert_eq!(stats.total_execution_time_estimate_ms(), 1500.0);

        let json = stats.to_json();
        assert!(json.is_object());
        assert_eq!(json["total_queries"], 10);
        assert_eq!(json["success_rate"], 0.9);
    }

    #[test]
    fn test_query_type_name_extraction() {
        let engine = QueryEngine {
            executor: QueryExecutor {
                gpu_kernels: None,
                stats: crate::query::query_execution::ExecutionStats::default(),
            },
            stats: QueryStats::default(),
            gpu_enabled: false,
        };

        let query_types = vec![
            (
                QueryType::FindNodes {
                    node_type: None,
                    properties: HashMap::new(),
                },
                "find_nodes",
            ),
            (
                QueryType::FindEdges {
                    edge_type: None,
                    weight_range: None,
                },
                "find_edges",
            ),
            (
                QueryType::FindPath {
                    source_id: "a".to_string(),
                    target_id: "b".to_string(),
                    max_length: 5,
                },
                "find_path",
            ),
            (
                QueryType::Neighborhood {
                    node_id: "a".to_string(),
                    radius: 1,
                    edge_types: None,
                },
                "neighborhood",
            ),
            (
                QueryType::Subgraph {
                    seed_nodes: vec!["a".to_string()],
                    max_depth: 2,
                },
                "subgraph",
            ),
            (
                QueryType::PatternMatch {
                    pattern: GraphPattern::new(),
                },
                "pattern_match",
            ),
            (
                QueryType::Ranking {
                    algorithm: RankingAlgorithm::DegreeCentrality,
                    node_type: None,
                },
                "ranking",
            ),
        ];

        for (query_type, expected_name) in query_types {
            assert_eq!(engine.get_query_type_name(&query_type), expected_name);
        }
    }

    #[tokio::test]
    async fn test_stats_reset() {
        let mut engine = QueryEngine::new(false).await.unwrap();
        let graph = KnowledgeGraph::new(KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        })
        .await?;

        // Execute a query to generate stats
        let query = Query::new(QueryType::FindNodes {
            node_type: Some(NodeType::Agent),
            properties: HashMap::new(),
        });
        let _ = engine.execute(&graph, query).await;

        assert_eq!(engine.stats().total_queries, 1);

        // Reset stats
        engine.reset_stats();
        assert_eq!(engine.stats().total_queries, 0);
        assert_eq!(engine.stats().error_count, 0);
        assert!(engine.stats().query_type_counts.is_empty());
    }
}
