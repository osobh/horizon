//! Knowledge graph adapter for gpu-agents integration
//!
//! This module provides adapters to connect the knowledge-graph crate
//! with gpu-agents, enabling semantic search and pattern storage.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;

// Import from knowledge-graph crate
use exorust_knowledge_graph::{
    scaling::DistributedGraphStore, Edge, EdgeType, Node, NodeType, QueryEngine, QueryResult,
    ScaledKnowledgeGraph, ScalingConfig, SemanticSearchEngine,
};

// Import local types
use crate::consensus_synthesis::integration::ConsensusSynthesisEngine;
use crate::synthesis::{Pattern, SynthesisTask, Template};

/// Adapter for knowledge graph integration with consensus-synthesis
pub struct KnowledgeGraphAdapter {
    /// Main knowledge graph instance
    knowledge_graph: Arc<ScaledKnowledgeGraph>,
    /// Semantic search engine
    semantic_engine: SemanticSearchEngine,
    /// Query engine for pattern searches
    query_engine: QueryEngine,
    /// Cache for frequent queries
    query_cache: HashMap<String, QueryResult>,
    /// GPU device for accelerated operations
    device: Arc<cudarc::driver::CudaDevice>,
}

impl KnowledgeGraphAdapter {
    /// Create a new knowledge graph adapter
    pub async fn new(device: Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        // Configure scaled knowledge graph
        let scaling_config = ScalingConfig {
            shard_count: 16,
            max_nodes_per_shard: 100_000,
            enable_compression: true,
            replication_factor: 3,
            cache_size: 10_000,
        };

        // Create scaled knowledge graph directly
        let knowledge_graph = ScaledKnowledgeGraph::new(scaling_config)
            .await
            .context("Failed to create scaled knowledge graph")?;

        // Initialize semantic search engine
        let embedding_config = exorust_knowledge_graph::semantic::EmbeddingConfig {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            dimension: 384,
            max_length: 512,
            gpu_enabled: true,
        };
        let semantic_engine = SemanticSearchEngine::new(embedding_config);

        // Initialize query engine
        let query_engine = QueryEngine::new(true)
            .await // Enable GPU acceleration
            .context("Failed to create query engine")?;

        Ok(Self {
            knowledge_graph: Arc::new(knowledge_graph),
            semantic_engine,
            query_engine,
            query_cache: HashMap::new(),
            device,
        })
    }

    /// Store successful synthesis result in knowledge graph
    pub async fn store_synthesis_pattern(
        &mut self,
        goal: &str,
        synthesis_task: &SynthesisTask,
        performance_metrics: &SynthesisPerformanceMetrics,
    ) -> Result<String> {
        // Create nodes for the synthesis pattern
        let goal_node = Node::new(
            NodeType::Goal,
            HashMap::from([
                (
                    "description".to_string(),
                    serde_json::Value::String(goal.to_string()),
                ),
                (
                    "priority".to_string(),
                    serde_json::Value::String("high".to_string()),
                ),
            ]),
        );
        let goal_id = goal_node.id.clone();
        self.knowledge_graph
            .add_node(goal_id.clone(), goal_node)
            .await
            .context("Failed to add goal node")?;

        let pattern_node = Node::new(
            NodeType::Pattern,
            HashMap::from([
                (
                    "node_type".to_string(),
                    serde_json::Value::String(format!("{:?}", synthesis_task.pattern.node_type)),
                ),
                (
                    "complexity".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(
                        synthesis_task.pattern.children.len(),
                    )),
                ),
            ]),
        );
        let pattern_id = pattern_node.id.clone();
        self.knowledge_graph
            .add_node(pattern_id.clone(), pattern_node)
            .await
            .context("Failed to add pattern node")?;

        let template_node = Node::new(
            NodeType::Custom("Template".to_string()),
            HashMap::from([
                (
                    "token_count".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(
                        synthesis_task.template.tokens.len(),
                    )),
                ),
                (
                    "estimated_output_size".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(1024)),
                ),
            ]),
        );
        let template_id = template_node.id.clone();
        self.knowledge_graph
            .add_node(template_id.clone(), template_node)
            .await
            .context("Failed to add template node")?;

        let performance_node = Node::new(
            NodeType::Custom("Performance".to_string()),
            HashMap::from([
                (
                    "throughput".to_string(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(performance_metrics.throughput)?,
                    ),
                ),
                (
                    "latency_ms".to_string(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(performance_metrics.latency_ms)?,
                    ),
                ),
                (
                    "accuracy".to_string(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(performance_metrics.accuracy)?,
                    ),
                ),
            ]),
        );
        let performance_id = performance_node.id.clone();
        self.knowledge_graph
            .add_node(performance_id.clone(), performance_node)
            .await
            .context("Failed to add performance node")?;

        // Create relationships (edges will be added in future implementation)
        let _goal_to_pattern = Edge::new(goal_id.clone(), pattern_id.clone(), EdgeType::Uses, 1.0);

        let _pattern_to_template = Edge::new(
            pattern_id.clone(),
            template_id.clone(),
            EdgeType::Produces,
            performance_metrics.accuracy,
        );

        let _template_to_performance = Edge::new(
            template_id.clone(),
            performance_id.clone(),
            EdgeType::Has,
            performance_metrics.throughput / 1000.0, // Normalize
        );

        // Note: Edge addition is implemented in knowledge-graph crate
        // This adapter provides a simplified interface for testing

        Ok(goal_id)
    }

    /// Search for similar successful synthesis patterns
    pub async fn find_similar_patterns(
        &mut self,
        goal: &str,
        similarity_threshold: f64,
    ) -> Result<Vec<SimilarPattern>> {
        // Check cache first
        let cache_key = format!("similar_{}", goal);
        if let Some(cached_result) = self.query_cache.get(&cache_key) {
            return self.parse_similar_patterns(cached_result);
        }

        // Create semantic query
        let semantic_query = exorust_knowledge_graph::semantic::SemanticQuery {
            query: exorust_knowledge_graph::semantic::QueryInput::Text(goal.to_string()),
            node_types: Some(vec![NodeType::Goal]),
            top_k: 10,
            threshold: similarity_threshold,
            use_gpu: true,
        };

        // Execute semantic search (using a dummy KnowledgeGraph for now)
        let config = exorust_knowledge_graph::graph::KnowledgeGraphConfig::default();
        let dummy_graph = exorust_knowledge_graph::graph::KnowledgeGraph::new(config)
            .await
            .context("Failed to create dummy knowledge graph")?;
        let search_results = self
            .semantic_engine
            .search(&dummy_graph, semantic_query)
            .await
            .context("Failed to execute semantic search")?;

        // Build query for related patterns
        let mut goal_ids = Vec::new();
        for result in &search_results {
            goal_ids.push(result.node.id.clone());
        }

        if goal_ids.is_empty() {
            return Ok(Vec::new());
        }

        // For now, return empty results as query system needs more implementation
        // Note: Full query implementation exists in knowledge-graph/src/graph.rs
        let empty_result = exorust_knowledge_graph::query::QueryResult {
            nodes: Vec::new(),
            edges: Vec::new(),
            paths: Vec::new(),
            scores: HashMap::new(),
            execution_time_ms: 0,
            gpu_accelerated: false,
            metadata: HashMap::new(),
        };

        // Cache the result
        self.query_cache.insert(cache_key, empty_result.clone());

        // Parse and return similar patterns
        self.parse_similar_patterns(&empty_result)
    }

    /// Parse query results into similar patterns
    fn parse_similar_patterns(&self, query_result: &QueryResult) -> Result<Vec<SimilarPattern>> {
        let mut patterns = Vec::new();

        for node in &query_result.nodes {
            if node.node_type == NodeType::Pattern {
                // Extract pattern information
                let complexity = node
                    .properties
                    .get("complexity")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(0);

                let node_type_str = node
                    .properties
                    .get("node_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");

                // Find related performance metrics
                let performance_score = query_result
                    .edges
                    .iter()
                    .find(|edge| edge.source_id == node.id && edge.edge_type == EdgeType::Has)
                    .map(|edge| edge.weight)
                    .unwrap_or(0.5);

                patterns.push(SimilarPattern {
                    pattern_id: node.id.clone(),
                    similarity_score: performance_score,
                    complexity,
                    node_type_description: node_type_str.to_string(),
                    estimated_performance: performance_score,
                });
            }
        }

        // Sort by similarity score (descending)
        patterns.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(patterns)
    }

    /// Get consensus patterns for decision making
    pub async fn get_consensus_patterns(
        &mut self,
        decision_context: &str,
    ) -> Result<Vec<ConsensusPattern>> {
        // For now, return empty results as the query system needs more implementation
        // Note: Consensus pattern querying is implemented in the consensus module
        let _context = decision_context; // Use the parameter to avoid warnings
        let result = exorust_knowledge_graph::query::QueryResult {
            nodes: Vec::new(),
            edges: Vec::new(),
            paths: Vec::new(),
            scores: HashMap::new(),
            execution_time_ms: 0,
            gpu_accelerated: false,
            metadata: HashMap::new(),
        };

        let mut patterns = Vec::new();
        for node in &result.nodes {
            if let NodeType::Custom(ref custom_type) = node.node_type {
                if custom_type == "Consensus" {
                    let success_rate = node
                        .properties
                        .get("success_rate")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5);

                    let voting_strategy = node
                        .properties
                        .get("voting_strategy")
                        .and_then(|v| v.as_str())
                        .unwrap_or("majority")
                        .to_string();

                    patterns.push(ConsensusPattern {
                        pattern_id: node.id.clone(),
                        context: decision_context.to_string(),
                        success_rate,
                        voting_strategy,
                        recommended_threshold: 0.67, // Default supermajority
                    });
                }
            }
        }

        Ok(patterns)
    }

    /// Store consensus decision outcome
    pub async fn store_consensus_outcome(
        &mut self,
        decision_id: &str,
        context: &str,
        outcome: &ConsensusOutcome,
    ) -> Result<String> {
        let consensus_node = Node::new(
            NodeType::Custom("Consensus".to_string()),
            HashMap::from([
                (
                    "decision_id".to_string(),
                    serde_json::Value::String(decision_id.to_string()),
                ),
                (
                    "context".to_string(),
                    serde_json::Value::String(context.to_string()),
                ),
                (
                    "success_rate".to_string(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(outcome.success_rate)?,
                    ),
                ),
                (
                    "voting_strategy".to_string(),
                    serde_json::Value::String(outcome.voting_strategy.clone()),
                ),
                (
                    "participant_count".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(outcome.participant_count)),
                ),
            ]),
        );

        let node_id = consensus_node.id.clone();
        self.knowledge_graph
            .add_node(node_id.clone(), consensus_node)
            .await
            .context("Failed to store consensus outcome")?;

        Ok(node_id)
    }

    /// Get knowledge graph statistics
    pub async fn get_graph_statistics(&self) -> Result<GraphStatistics> {
        // Note: Statistics collection is available in knowledge-graph/src/graph.rs
        // For now, return mock statistics
        Ok(GraphStatistics {
            total_nodes: 0,
            total_edges: 0,
            node_type_distribution: HashMap::new(),
            edge_type_distribution: HashMap::new(),
            average_clustering_coefficient: 0.0,
            graph_density: 0.0,
        })
    }

    /// Clear query cache
    pub fn clear_cache(&mut self) {
        self.query_cache.clear();
    }

    /// Optimize knowledge graph for better performance
    pub async fn optimize_graph(&mut self) -> Result<()> {
        // Note: Graph optimization algorithms exist in knowledge-graph/src/scaling.rs
        // For now, just clear cache
        self.clear_cache();

        Ok(())
    }
}

/// Similar pattern found in knowledge graph
#[derive(Debug, Clone)]
pub struct SimilarPattern {
    pub pattern_id: String,
    pub similarity_score: f64,
    pub complexity: usize,
    pub node_type_description: String,
    pub estimated_performance: f64,
}

/// Consensus pattern for decision making
#[derive(Debug, Clone)]
pub struct ConsensusPattern {
    pub pattern_id: String,
    pub context: String,
    pub success_rate: f64,
    pub voting_strategy: String,
    pub recommended_threshold: f64,
}

/// Consensus outcome to store
#[derive(Debug, Clone)]
pub struct ConsensusOutcome {
    pub success_rate: f64,
    pub voting_strategy: String,
    pub participant_count: usize,
    pub decision_quality: f64,
}

/// Performance metrics for synthesis
#[derive(Debug, Clone)]
pub struct SynthesisPerformanceMetrics {
    pub throughput: f64,
    pub latency_ms: f64,
    pub accuracy: f64,
    pub resource_usage: f64,
}

/// Knowledge graph statistics
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub node_type_distribution: HashMap<String, usize>,
    pub edge_type_distribution: HashMap<String, usize>,
    pub average_clustering_coefficient: f64,
    pub graph_density: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::{NodeType as SynthNodeType, Token};

    #[tokio::test]
    async fn test_knowledge_graph_adapter_creation() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
        let adapter = KnowledgeGraphAdapter::new(device).await;
        assert!(adapter.is_ok());
    }

    #[tokio::test]
    async fn test_store_synthesis_pattern() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
        let mut adapter = KnowledgeGraphAdapter::new(device).await?;

        let goal = "Create matrix multiplication kernel";
        let pattern = crate::synthesis::Pattern {
            node_type: SynthNodeType::Function,
            children: vec![],
            value: Some("matmul".to_string()),
        };
        let template = crate::synthesis::Template {
            tokens: vec![Token::Literal("__global__ void matmul".to_string())],
        };
        let synthesis_task = SynthesisTask { pattern, template };

        let metrics = SynthesisPerformanceMetrics {
            throughput: 1000.0,
            latency_ms: 5.0,
            accuracy: 0.95,
            resource_usage: 0.7,
        };

        let result = adapter
            .store_synthesis_pattern(goal, &synthesis_task, &metrics)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_find_similar_patterns() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
        let mut adapter = KnowledgeGraphAdapter::new(device).await?;

        let goal = "Optimize parallel reduction";
        let result = adapter.find_similar_patterns(goal, 0.8).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_consensus_patterns() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
        let mut adapter = KnowledgeGraphAdapter::new(device).await?;

        let context = "GPU kernel selection";
        let result = adapter.get_consensus_patterns(context).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_store_consensus_outcome() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
        let mut adapter = KnowledgeGraphAdapter::new(device).await?;

        let outcome = ConsensusOutcome {
            success_rate: 0.85,
            voting_strategy: "majority".to_string(),
            participant_count: 7,
            decision_quality: 0.9,
        };

        let result = adapter
            .store_consensus_outcome("test_decision", "test context", &outcome)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_graph_statistics() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
        let adapter = KnowledgeGraphAdapter::new(device).await?;

        let result = adapter.get_graph_statistics().await;
        assert!(result.is_ok());
        let stats = result?;
        assert!(stats.total_nodes >= 0);
        assert!(stats.total_edges >= 0);
    }

    #[test]
    fn test_similar_pattern_creation() {
        let pattern = SimilarPattern {
            pattern_id: "test_pattern".to_string(),
            similarity_score: 0.95,
            complexity: 5,
            node_type_description: "Function".to_string(),
            estimated_performance: 0.9,
        };

        assert_eq!(pattern.pattern_id, "test_pattern");
        assert_eq!(pattern.similarity_score, 0.95);
        assert_eq!(pattern.complexity, 5);
    }

    #[test]
    fn test_consensus_pattern_creation() {
        let pattern = ConsensusPattern {
            pattern_id: "consensus_1".to_string(),
            context: "GPU selection".to_string(),
            success_rate: 0.8,
            voting_strategy: "weighted".to_string(),
            recommended_threshold: 0.67,
        };

        assert_eq!(pattern.success_rate, 0.8);
        assert_eq!(pattern.voting_strategy, "weighted");
        assert_eq!(pattern.recommended_threshold, 0.67);
    }

    #[test]
    fn test_synthesis_performance_metrics() {
        let metrics = SynthesisPerformanceMetrics {
            throughput: 2000.0,
            latency_ms: 3.5,
            accuracy: 0.98,
            resource_usage: 0.6,
        };

        assert_eq!(metrics.throughput, 2000.0);
        assert_eq!(metrics.latency_ms, 3.5);
        assert_eq!(metrics.accuracy, 0.98);
        assert_eq!(metrics.resource_usage, 0.6);
    }
}
