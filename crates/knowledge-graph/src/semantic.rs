//! Semantic search system for knowledge graph

use crate::error::{KnowledgeGraphError, KnowledgeGraphResult};
use crate::graph::{KnowledgeGraph, Node, NodeType};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Embedding vector type
pub type EmbeddingVector = Vec<f32>;

/// Semantic query specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticQuery {
    /// Query text or embedding
    pub query: QueryInput,
    /// Node type filter
    pub node_types: Option<Vec<NodeType>>,
    /// Number of results to return
    pub top_k: usize,
    /// Similarity threshold
    pub threshold: f64,
    /// Enable GPU acceleration
    pub use_gpu: bool,
}

/// Query input types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryInput {
    /// Text query (will be embedded)
    Text(String),
    /// Pre-computed embedding vector
    Embedding(EmbeddingVector),
}

/// Semantic search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticResult {
    /// Matching node
    pub node: Node,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f64,
}

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model name or path
    pub model_name: String,
    /// Embedding dimension
    pub dimension: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Enable GPU for embedding computation
    pub gpu_enabled: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            dimension: 384,
            max_length: 512,
            gpu_enabled: true,
        }
    }
}

/// Semantic search engine
pub struct SemanticSearchEngine {
    /// Embedding configuration
    config: EmbeddingConfig,
    /// Node embeddings cache
    embedding_cache: HashMap<String, EmbeddingVector>,
    /// Embedding statistics
    stats: EmbeddingStats,
}

/// Embedding statistics
#[derive(Debug, Clone, Default)]
struct EmbeddingStats {
    /// Total embeddings computed
    total_embeddings: u64,
    /// Cache hits
    cache_hits: u64,
    /// GPU computations
    gpu_computations: u64,
    /// Average embedding time (ms)
    avg_embedding_time_ms: f64,
}

impl SemanticSearchEngine {
    /// Create a new semantic search engine
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            embedding_cache: HashMap::new(),
            stats: EmbeddingStats::default(),
        }
    }

    /// Search for semantically similar nodes
    pub async fn search(
        &mut self,
        graph: &KnowledgeGraph,
        query: SemanticQuery,
    ) -> KnowledgeGraphResult<Vec<SemanticResult>> {
        let start_time = std::time::Instant::now();

        // Get query embedding
        let query_embedding = match query.query {
            QueryInput::Text(ref text) => self.text_to_embedding(text).await?,
            QueryInput::Embedding(ref embedding) => embedding.clone(),
        };

        // Get candidate nodes
        let candidates = self.get_candidate_nodes(graph, &query.node_types);

        // Ensure all candidate nodes have embeddings
        for node in &candidates {
            if !self.embedding_cache.contains_key(&node.id) {
                let embedding = self.compute_node_embedding(node).await?;
                self.embedding_cache.insert(node.id.clone(), embedding);
            }
        }

        // Compute similarities
        let mut results = Vec::new();
        for node in candidates {
            if let Some(node_embedding) = self.embedding_cache.get(&node.id) {
                let similarity = self.compute_similarity(&query_embedding, node_embedding)?;

                if similarity >= query.threshold {
                    results.push(SemanticResult {
                        node: node.clone(),
                        similarity,
                    });
                }
            }
        }

        // Sort by similarity (descending) and take top-k
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(query.top_k);

        // Update stats
        let elapsed = start_time.elapsed().as_millis() as f64;
        self.stats.avg_embedding_time_ms =
            (self.stats.avg_embedding_time_ms * self.stats.total_embeddings as f64 + elapsed)
                / (self.stats.total_embeddings + 1) as f64;

        Ok(results)
    }

    /// Get candidate nodes for search
    fn get_candidate_nodes(
        &self,
        graph: &KnowledgeGraph,
        node_types: &Option<Vec<NodeType>>,
    ) -> Vec<Node> {
        match node_types {
            Some(types) => {
                let mut candidates = Vec::new();
                for node_type in types {
                    candidates.extend(graph.get_nodes_by_type(node_type).into_iter().cloned());
                }
                candidates
            }
            None => {
                // Would need to get all nodes from graph
                vec![]
            }
        }
    }

    /// Convert text to embedding vector
    pub async fn text_to_embedding(&mut self, text: &str) -> KnowledgeGraphResult<EmbeddingVector> {
        // Mock implementation - real implementation would use a transformer model
        self.stats.total_embeddings += 1;

        if self.config.gpu_enabled {
            self.stats.gpu_computations += 1;
        }

        // Generate mock embedding based on text hash
        let hash = self.hash_text(text);
        let mut embedding = vec![0.0; self.config.dimension];

        for (i, value) in embedding.iter_mut().enumerate() {
            *value = ((hash.wrapping_add(i as u64)) as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }

        // Normalize to unit vector
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut embedding {
                *value /= norm;
            }
        }

        Ok(embedding)
    }

    /// Compute embedding for a node based on its properties
    async fn compute_node_embedding(
        &mut self,
        node: &Node,
    ) -> KnowledgeGraphResult<EmbeddingVector> {
        // Combine node type and properties into text
        let mut text_parts = vec![format!("{:?}", node.node_type)];

        for (key, value) in &node.properties {
            text_parts.push(format!("{}: {}, {}", key, key, value));
        }

        let combined_text = text_parts.join(" ");
        self.text_to_embedding(&combined_text).await
    }

    /// Compute cosine similarity between two embeddings
    fn compute_similarity(
        &self,
        embedding1: &EmbeddingVector,
        embedding2: &EmbeddingVector,
    ) -> KnowledgeGraphResult<f64> {
        if embedding1.len() != embedding2.len() {
            return Err(KnowledgeGraphError::SemanticError {
                message: "Embedding dimensions do not match".to_string(),
            });
        }

        if self.config.gpu_enabled {
            // GPU-accelerated computation
            self.compute_similarity_gpu(embedding1, embedding2)
        } else {
            // CPU computation
            self.compute_similarity_cpu(embedding1, embedding2)
        }
    }

    /// CPU-based cosine similarity
    fn compute_similarity_cpu(
        &self,
        embedding1: &EmbeddingVector,
        embedding2: &EmbeddingVector,
    ) -> KnowledgeGraphResult<f64> {
        let dot_product: f32 = embedding1
            .iter()
            .zip(embedding2.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }

        Ok((dot_product / (norm1 * norm2)) as f64)
    }

    /// GPU-accelerated cosine similarity
    fn compute_similarity_gpu(
        &self,
        embedding1: &EmbeddingVector,
        embedding2: &EmbeddingVector,
    ) -> KnowledgeGraphResult<f64> {
        // Mock GPU computation - fall back to CPU for now
        self.compute_similarity_cpu(embedding1, embedding2)
    }

    /// Simple hash function for text
    fn hash_text(&self, text: &str) -> u64 {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Update node embedding in cache
    pub async fn update_node_embedding(
        &mut self,
        node_id: &str,
        node: &Node,
    ) -> KnowledgeGraphResult<()> {
        let embedding = self.compute_node_embedding(node).await?;
        self.embedding_cache.insert(node_id.to_string(), embedding);
        Ok(())
    }

    /// Remove node embedding from cache
    pub fn remove_node_embedding(&mut self, node_id: &str) {
        self.embedding_cache.remove(node_id);
    }

    /// Get embedding statistics
    pub fn stats(&self) -> &EmbeddingStats {
        &self.stats
    }

    /// Find similar nodes to a given node
    pub async fn find_similar_nodes(
        &mut self,
        graph: &KnowledgeGraph,
        node_id: &str,
        top_k: usize,
        threshold: f64,
    ) -> KnowledgeGraphResult<Vec<SemanticResult>> {
        let node = graph.get_node(node_id)?;

        // Get or compute embedding for the target node
        let node_embedding = if let Some(embedding) = self.embedding_cache.get(node_id) {
            self.stats.cache_hits += 1;
            embedding.clone()
        } else {
            let embedding = self.compute_node_embedding(node).await?;
            self.embedding_cache
                .insert(node_id.to_string(), embedding.clone());
            embedding
        };

        let query = SemanticQuery {
            query: QueryInput::Embedding(node_embedding),
            node_types: Some(vec![node.node_type.clone()]),
            top_k,
            threshold,
            use_gpu: self.config.gpu_enabled,
        };

        let mut results = self.search(graph, query).await?;

        // Remove the original node from results
        results.retain(|result| result.node.id != node_id);

        Ok(results)
    }

    /// Batch update embeddings for multiple nodes
    pub async fn batch_update_embeddings(&mut self, nodes: Vec<&Node>) -> KnowledgeGraphResult<()> {
        for node in nodes {
            let embedding = self.compute_node_embedding(node).await?;
            self.embedding_cache.insert(node.id.clone(), embedding);
        }
        Ok(())
    }

    /// Clear embedding cache
    pub fn clear_cache(&mut self) {
        self.embedding_cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.embedding_cache.len()
    }

    /// Embed a batch of nodes by their IDs
    pub async fn embed_batch(
        &mut self,
        graph: &KnowledgeGraph,
        node_ids: &[String],
    ) -> KnowledgeGraphResult<()> {
        for node_id in node_ids {
            let node = graph.get_node(node_id)?;
            let embedding = self.compute_node_embedding(node).await?;
            self.embedding_cache.insert(node_id.clone(), embedding);
        }
        Ok(())
    }

    /// Get or compute embedding for a node
    pub async fn get_or_compute_embedding(
        &mut self,
        node: &Node,
    ) -> KnowledgeGraphResult<EmbeddingVector> {
        if let Some(embedding) = self.embedding_cache.get(&node.id) {
            self.stats.cache_hits += 1;
            Ok(embedding.clone())
        } else {
            let embedding = self.compute_node_embedding(node).await?;
            self.embedding_cache
                .insert(node.id.clone(), embedding.clone());
            Ok(embedding)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{KnowledgeGraphConfig, Node};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_semantic_search_engine_creation() {
        let config = EmbeddingConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let engine = SemanticSearchEngine::new(config);
        assert_eq!(engine.cache_size(), 0);
    }

    #[tokio::test]
    async fn test_text_to_embedding() {
        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 128,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        let embedding = engine.text_to_embedding("hello world").await?;
        assert_eq!(embedding.len(), 128);

        // Test that same text produces same embedding
        let embedding2 = engine.text_to_embedding("hello world").await.unwrap();
        assert_eq!(embedding, embedding2);

        // Test that different text produces different embedding
        let embedding3 = engine.text_to_embedding("goodbye world").await.unwrap();
        assert_ne!(embedding, embedding3);
    }

    #[tokio::test]
    async fn test_node_embedding() {
        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 64,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            serde_json::Value::String("test_node".to_string()),
        );
        properties.insert(
            "type".to_string(),
            serde_json::Value::String("example".to_string()),
        );

        let node = Node::new(NodeType::Agent, properties);
        let embedding = engine.compute_node_embedding(&node).await.unwrap();

        assert_eq!(embedding.len(), 64);

        // Verify it's normalized
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_cosine_similarity() {
        let config = EmbeddingConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let engine = SemanticSearchEngine::new(config);

        // Test identical vectors
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let similarity = engine.compute_similarity(&vec1, &vec2)?;
        assert!((similarity - 1.0).abs() < 1e-6);

        // Test orthogonal vectors
        let vec3 = vec![1.0, 0.0, 0.0];
        let vec4 = vec![0.0, 1.0, 0.0];
        let similarity = engine.compute_similarity(&vec3, &vec4).unwrap();
        assert!(similarity.abs() < 1e-6);

        // Test opposite vectors
        let vec5 = vec![1.0, 0.0, 0.0];
        let vec6 = vec![-1.0, 0.0, 0.0];
        let similarity = engine.compute_similarity(&vec5, &vec6).unwrap();
        assert!((similarity + 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_semantic_search() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        // Add test nodes
        let mut node1 = Node::new(NodeType::Agent, HashMap::new());
        node1.update_property(
            "description".to_string(),
            serde_json::Value::String("artificial intelligence agent".to_string()),
        );
        graph.add_node(node1).unwrap();

        let mut node2 = Node::new(NodeType::Agent, HashMap::new());
        node2.update_property(
            "description".to_string(),
            serde_json::Value::String("machine learning model".to_string()),
        );
        graph.add_node(node2).unwrap();

        let mut node3 = Node::new(NodeType::Goal, HashMap::new());
        node3.update_property(
            "description".to_string(),
            serde_json::Value::String("cooking recipe".to_string()),
        );
        graph.add_node(node3).unwrap();

        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 64,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        let query = SemanticQuery {
            query: QueryInput::Text("AI system".to_string()),
            node_types: Some(vec![NodeType::Agent]),
            top_k: 10,
            threshold: 0.0,
            use_gpu: false,
        };

        let results = engine.search(&graph, query).await.unwrap();

        // Should find the agent nodes, with AI-related content scoring higher
        assert!(!results.is_empty());

        // Verify results are sorted by similarity
        for i in 1..results.len() {
            assert!(results[i - 1].similarity >= results[i].similarity);
        }
    }

    #[tokio::test]
    async fn test_find_similar_nodes() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        // Add similar nodes
        let mut node1 = Node::new(NodeType::Agent, HashMap::new());
        node1.update_property(
            "task".to_string(),
            serde_json::Value::String("data analysis".to_string()),
        );
        let node1_id = node1.id.clone();
        graph.add_node(node1).unwrap();

        let mut node2 = Node::new(NodeType::Agent, HashMap::new());
        node2.update_property(
            "task".to_string(),
            serde_json::Value::String("data processing".to_string()),
        );
        graph.add_node(node2).unwrap();

        let mut node3 = Node::new(NodeType::Agent, HashMap::new());
        node3.update_property(
            "task".to_string(),
            serde_json::Value::String("image generation".to_string()),
        );
        graph.add_node(node3).unwrap();

        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 32,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        let similar = engine
            .find_similar_nodes(&graph, &node1_id, 5, 0.0)
            .await
            .unwrap();

        // Should find other nodes but not the original
        assert!(!similar.is_empty());
        assert!(similar.iter().all(|result| result.node.id != node1_id));
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 16,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        let node = Node::new(NodeType::Concept, HashMap::new());
        let node_id = node.id.clone();

        assert_eq!(engine.cache_size(), 0);

        // Update embedding should add to cache
        engine.update_node_embedding(&node_id, &node).await.unwrap();
        assert_eq!(engine.cache_size(), 1);

        // Remove embedding should remove from cache
        engine.remove_node_embedding(&node_id);
        assert_eq!(engine.cache_size(), 0);

        // Test cache clear
        engine.update_node_embedding(&node_id, &node).await.unwrap();
        engine.clear_cache();
        assert_eq!(engine.cache_size(), 0);
    }

    #[tokio::test]
    async fn test_batch_node_embeddings() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        // Add multiple nodes
        let mut node_ids = Vec::new();
        for i in 0..10 {
            let mut node = Node::new(NodeType::Agent, HashMap::new());
            node.update_property(
                "description".to_string(),
                serde_json::Value::String(format!("Test agent number {i}")),
            );
            node_ids.push(node.id.clone());
            graph.add_node(node).unwrap();
        }

        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 32,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        // Process batch
        engine.embed_batch(&graph, &node_ids).await.unwrap();

        // All nodes should have embeddings
        assert_eq!(engine.cache_size(), 10);
        for node_id in &node_ids {
            assert!(engine.embedding_cache.contains_key(node_id));
        }
    }

    #[test]
    fn test_embedding_config_serialization() {
        let config = EmbeddingConfig {
            model_name: "custom-model".to_string(),
            dimension: 768,
            max_length: 256,
            gpu_enabled: false,
        };

        let json = serde_json::to_string(&config)?;
        let deserialized: EmbeddingConfig = serde_json::from_str(&json)?;

        assert_eq!(config.model_name, deserialized.model_name);
        assert_eq!(config.dimension, deserialized.dimension);
        assert_eq!(config.max_length, deserialized.max_length);
        assert_eq!(config.gpu_enabled, deserialized.gpu_enabled);
    }

    #[test]
    fn test_semantic_query_construction() {
        // Test with text query
        let text_query = SemanticQuery {
            query: QueryInput::Text("machine learning".to_string()),
            node_types: Some(vec![NodeType::Agent, NodeType::Goal]),
            top_k: 20,
            threshold: 0.5,
            use_gpu: true,
        };

        match &text_query.query {
            QueryInput::Text(t) => assert_eq!(t, "machine learning"),
            _ => panic!("Expected text query"),
        }

        // Test with embedding query
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let embedding_query = SemanticQuery {
            query: QueryInput::Embedding(embedding.clone()),
            node_types: None,
            top_k: 5,
            threshold: 0.8,
            use_gpu: false,
        };

        match &embedding_query.query {
            QueryInput::Embedding(e) => assert_eq!(e, &embedding),
            _ => panic!("Expected embedding query"),
        }
    }

    #[tokio::test]
    async fn test_similarity_threshold_filtering() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        // Add nodes with varying content
        for i in 0..5 {
            let mut node = Node::new(NodeType::Agent, HashMap::new());
            let content = if i < 2 {
                "artificial intelligence and machine learning"
            } else {
                "completely unrelated content about cooking"
            };
            node.update_property(
                "content".to_string(),
                serde_json::Value::String(content.to_string()),
            );
            graph.add_node(node).unwrap();
        }

        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 64,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        let query = SemanticQuery {
            query: QueryInput::Text("AI and ML research".to_string()),
            node_types: None,
            top_k: 10,
            threshold: 0.5, // Filter out low similarity results
            use_gpu: false,
        };

        let results = engine.search(&graph, query).await.unwrap();

        // All results should meet the threshold
        for result in &results {
            assert!(result.similarity >= 0.5);
        }
    }

    #[tokio::test]
    async fn test_empty_graph_search() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let graph = KnowledgeGraph::new(graph_config).await?;

        let config = EmbeddingConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        let query = SemanticQuery {
            query: QueryInput::Text("test query".to_string()),
            node_types: None,
            top_k: 10,
            threshold: 0.0,
            use_gpu: false,
        };

        let results = engine.search(&graph, query).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_node_type_filtering() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        // Add nodes of different types
        let types = vec![NodeType::Agent, NodeType::Goal, NodeType::Concept];
        for (i, node_type) in types.iter().enumerate() {
            let mut node = Node::new(node_type.clone(), HashMap::new());
            node.update_property(
                "content".to_string(),
                serde_json::Value::String("test content".to_string()),
            );
            graph.add_node(node).unwrap();
        }

        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 32,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        // Search only for Agent nodes
        let query = SemanticQuery {
            query: QueryInput::Text("test".to_string()),
            node_types: Some(vec![NodeType::Agent]),
            top_k: 10,
            threshold: 0.0,
            use_gpu: false,
        };

        let results = engine.search(&graph, query).await.unwrap();

        // Should only find Agent nodes
        assert!(!results.is_empty());
        for result in &results {
            assert_eq!(result.node.node_type, NodeType::Agent);
        }
    }

    #[tokio::test]
    async fn test_top_k_limiting() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        // Add many nodes
        for i in 0..20 {
            let mut node = Node::new(NodeType::Agent, HashMap::new());
            node.update_property(
                "content".to_string(),
                serde_json::Value::String(format!("content {i}")),
            );
            graph.add_node(node).unwrap();
        }

        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 32,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        let query = SemanticQuery {
            query: QueryInput::Text("content".to_string()),
            node_types: None,
            top_k: 5, // Limit to 5 results
            threshold: 0.0,
            use_gpu: false,
        };

        let results = engine.search(&graph, query).await.unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_embedding_normalization() {
        let config = EmbeddingConfig::default();
        let engine = SemanticSearchEngine::new(config);

        // Test normalization of various vectors
        let test_vectors = vec![
            vec![3.0, 4.0],           // Should normalize to [0.6, 0.8]
            vec![1.0, 0.0, 0.0],      // Should remain [1.0, 0.0, 0.0]
            vec![1.0, 1.0, 1.0, 1.0], // Should normalize to [0.5, 0.5, 0.5, 0.5]
        ];

        for vec in test_vectors {
            let normalized = normalize_embedding(vec.clone());
            let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }

    fn normalize_embedding(mut vec: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vec {
                *value /= norm;
            }
        }
        vec
    }

    #[test]
    fn test_zero_vector_handling() {
        let config = EmbeddingConfig::default();
        let engine = SemanticSearchEngine::new(config);

        let zero_vec = vec![0.0, 0.0, 0.0];
        let result = engine.compute_similarity(&zero_vec, &zero_vec);

        // Should handle zero vectors gracefully - returns 0.0 for zero vectors
        assert_eq!(result?, 0.0);
    }

    #[tokio::test]
    async fn test_cache_statistics() {
        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 16,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        let node1 = Node::new(NodeType::Agent, HashMap::new());
        let node1_id = node1.id.clone();

        // First computation - cache miss
        let initial_cache_hits = engine.stats.cache_hits;
        engine
            .update_node_embedding(&node1_id, &node1)
            .await
            .unwrap();
        assert_eq!(engine.stats.cache_hits, initial_cache_hits);

        // Second computation - should be cache hit
        let embedding = engine.get_or_compute_embedding(&node1).await.unwrap();
        assert_eq!(engine.stats.cache_hits, initial_cache_hits + 1);
    }

    #[tokio::test]
    async fn test_similarity_with_self() {
        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 32,
            ..Default::default()
        };
        let engine = SemanticSearchEngine::new(config);

        let vec = vec![0.5, 0.5, 0.5, 0.5];
        let normalized = normalize_embedding(vec);

        let similarity = engine.compute_similarity(&normalized, &normalized).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_large_batch_processing() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        // Add many nodes for batch processing
        let mut node_ids = Vec::new();
        for i in 0..100 {
            let mut node = Node::new(NodeType::Agent, HashMap::new());
            node.update_property("id".to_string(), serde_json::Value::Number(i.into()));
            node_ids.push(node.id.clone());
            graph.add_node(node).unwrap();
        }

        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 16, // Small dimension for speed
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        // Process large batch
        let start = std::time::Instant::now();
        engine.embed_batch(&graph, &node_ids).await.unwrap();
        let duration = start.elapsed();

        // Should complete in reasonable time
        assert!(duration.as_secs() < 10);
        assert_eq!(engine.cache_size(), 100);
    }

    #[test]
    fn test_semantic_result_ordering() {
        let node1 = Node::new(NodeType::Agent, HashMap::new());
        let node2 = Node::new(NodeType::Agent, HashMap::new());

        let result1 = SemanticResult {
            node: node1,
            similarity: 0.9,
        };

        let result2 = SemanticResult {
            node: node2,
            similarity: 0.7,
        };

        // Higher similarity should come first
        let mut results = vec![result2.clone(), result1.clone()];
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        assert_eq!(results[0].similarity, 0.9);
        assert_eq!(results[1].similarity, 0.7);
    }

    #[tokio::test]
    async fn test_embedding_with_empty_properties() {
        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 32,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        // Node with no properties
        let node = Node::new(NodeType::Agent, HashMap::new());
        let embedding = engine.compute_node_embedding(&node).await?;

        // Should still produce a valid embedding
        assert_eq!(embedding.len(), 32);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_find_similar_with_invalid_node() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let graph = KnowledgeGraph::new(graph_config).await?;

        let config = EmbeddingConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        // Try to find similar nodes for non-existent node
        let result = engine
            .find_similar_nodes(&graph, "non-existent-id", 5, 0.0)
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_query_input_variants() {
        // Test text input
        let text_input = QueryInput::Text("test query".to_string());
        match text_input {
            QueryInput::Text(ref t) => assert_eq!(t, "test query"),
            _ => panic!("Wrong variant"),
        }

        // Test embedding input
        let embedding_input = QueryInput::Embedding(vec![0.1, 0.2, 0.3]);
        match embedding_input {
            QueryInput::Embedding(ref e) => assert_eq!(e.len(), 3),
            _ => panic!("Wrong variant"),
        }
    }

    #[tokio::test]
    async fn test_concurrent_cache_access() {
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let config = EmbeddingConfig {
            gpu_enabled: false,
            dimension: 16,
            ..Default::default()
        };
        let engine = Arc::new(Mutex::new(SemanticSearchEngine::new(config)));

        let mut handles = vec![];

        // Spawn multiple tasks updating cache
        for i in 0..10 {
            let engine_clone = Arc::clone(&engine);
            let handle = tokio::spawn(async move {
                let node = Node::new(
                    NodeType::Agent,
                    HashMap::from([("id".to_string(), serde_json::Value::Number(i.into()))]),
                );
                let mut eng = engine_clone.lock().await;
                eng.update_node_embedding(&node.id, &node).await.unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let eng = engine.lock().await;
        assert_eq!(eng.cache_size(), 10);
    }

    #[test]
    fn test_embedding_vector_type() {
        let vec: EmbeddingVector = vec![0.1, 0.2, 0.3];
        assert_eq!(vec.len(), 3);

        // Test conversion and operations
        let sum: f32 = vec.iter().sum();
        assert!((sum - 0.6).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_gpu_acceleration_flag() {
        let config = EmbeddingConfig {
            gpu_enabled: true,
            ..Default::default()
        };
        let mut engine = SemanticSearchEngine::new(config);

        // Create query with GPU enabled
        let query = SemanticQuery {
            query: QueryInput::Text("test".to_string()),
            node_types: None,
            top_k: 10,
            threshold: 0.0,
            use_gpu: true,
        };

        // In mock mode, GPU operations should still work
        assert!(query.use_gpu);
        assert!(engine.config.gpu_enabled);
    }
}
