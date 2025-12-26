//! Text embeddings for semantic understanding

use crate::error::AssistantResult;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Embedding vector for semantic similarity
pub type Embedding = Array1<f32>;

/// Embedding engine for text semantic understanding
pub struct EmbeddingEngine {
    /// Dimension of embeddings
    dimension: usize,
    /// Cached embeddings for common phrases
    cache: HashMap<String, Embedding>,
    /// Whether to use local or cloud model
    use_local: bool,
}

/// Semantic similarity result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub score: f32,
    pub text_a: String,
    pub text_b: String,
}

impl EmbeddingEngine {
    pub fn new(dimension: usize, use_local: bool) -> AssistantResult<Self> {
        Ok(Self {
            dimension,
            cache: HashMap::new(),
            use_local,
        })
    }

    /// Generate embedding for text
    pub async fn embed(&mut self, text: &str) -> AssistantResult<Embedding> {
        // Check cache first
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached.clone());
        }

        let embedding = if self.use_local {
            self.generate_local_embedding(text).await?
        } else {
            self.generate_cloud_embedding(text).await?
        };

        // Cache the result
        self.cache.insert(text.to_string(), embedding.clone());

        Ok(embedding)
    }

    /// Calculate cosine similarity between two texts
    pub async fn similarity(&mut self, text_a: &str, text_b: &str) -> AssistantResult<f32> {
        let embedding_a = self.embed(text_a).await?;
        let embedding_b = self.embed(text_b).await?;

        let dot_product = embedding_a.dot(&embedding_b);
        let norm_a = embedding_a.dot(&embedding_a).sqrt();
        let norm_b = embedding_b.dot(&embedding_b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Find most similar texts from a collection
    pub async fn find_most_similar(
        &mut self,
        query: &str,
        candidates: &[String],
        top_k: usize,
    ) -> AssistantResult<Vec<SimilarityResult>> {
        let query_embedding = self.embed(query).await?;
        let mut results = Vec::new();

        for candidate in candidates {
            let candidate_embedding = self.embed(candidate).await?;
            let score = self.cosine_similarity(&query_embedding, &candidate_embedding);

            results.push(SimilarityResult {
                score,
                text_a: query.to_string(),
                text_b: candidate.clone(),
            });
        }

        // Sort by similarity score (descending)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top k results
        results.truncate(top_k);
        Ok(results)
    }

    /// Generate embedding using local model (simplified TF-IDF approach)
    async fn generate_local_embedding(&self, text: &str) -> AssistantResult<Embedding> {
        // Simple bag-of-words approach with predefined vocabulary
        let vocab = self.get_vocabulary();
        let mut embedding = Array1::zeros(self.dimension);

        let lowercase_text = text.to_lowercase();
        let words: Vec<&str> = lowercase_text.split_whitespace().collect();
        let word_count = words.len() as f32;

        if word_count == 0.0 {
            return Ok(embedding);
        }

        // Count word frequencies
        let mut word_freq = HashMap::new();
        for word in &words {
            *word_freq.entry(*word).or_insert(0) += 1;
        }

        // Generate embedding based on vocabulary
        for (word, freq) in word_freq {
            if let Some(&idx) = vocab.get(word) {
                if idx < self.dimension {
                    embedding[idx] = freq as f32 / word_count;
                }
            }
        }

        // Normalize the embedding
        let norm = embedding.dot(&embedding).sqrt();
        if norm > 0.0 {
            embedding /= norm;
        }

        Ok(embedding)
    }

    /// Generate embedding using cloud API (placeholder)
    async fn generate_cloud_embedding(&self, text: &str) -> AssistantResult<Embedding> {
        // In a real implementation, this would call a cloud API like OpenAI
        // For now, fall back to local generation
        self.generate_local_embedding(text).await
    }

    /// Get predefined vocabulary for local embeddings
    fn get_vocabulary(&self) -> HashMap<&'static str, usize> {
        // Common StratoSwarm and infrastructure terms
        let terms = vec![
            // Actions
            "deploy",
            "scale",
            "rollback",
            "debug",
            "optimize",
            "evolve",
            "start",
            "stop",
            "restart",
            "pause",
            "resume",
            "migrate",
            "create",
            "delete",
            "update",
            "list",
            "show",
            "get",
            "find",
            // Resources
            "agent",
            "node",
            "application",
            "service",
            "container",
            "pod",
            "gpu",
            "cpu",
            "memory",
            "storage",
            "network",
            "cluster",
            "tier",
            "mesh",
            "swarm",
            "evolution",
            "personality",
            // States
            "running",
            "stopped",
            "failed",
            "pending",
            "ready",
            "healthy",
            "evolving",
            "optimizing",
            "scaling",
            "migrating",
            // Attributes
            "conservative",
            "aggressive",
            "balanced",
            "explorer",
            "cooperative",
            "performance",
            "latency",
            "throughput",
            "utilization",
            "efficiency",
            "high",
            "low",
            "medium",
            "fast",
            "slow",
            "stable",
            // Infrastructure
            "datacenter",
            "workstation",
            "laptop",
            "edge",
            "cloud",
            "kubernetes",
            "docker",
            "nvidia",
            "cuda",
            "nvme",
            "ssd",
            // Operations
            "help",
            "status",
            "logs",
            "metrics",
            "monitor",
            "alert",
            "backup",
            "restore",
            "snapshot",
            "checkpoint",
            "migrate",
        ];

        terms
            .into_iter()
            .enumerate()
            .map(|(i, term)| (term, i))
            .collect()
    }

    /// Calculate cosine similarity between two embeddings
    fn cosine_similarity(&self, a: &Embedding, b: &Embedding) -> f32 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Clear the embedding cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.dimension)
    }
}

/// Semantic query matcher using embeddings
pub struct SemanticMatcher {
    engine: EmbeddingEngine,
    intent_examples: HashMap<String, Vec<String>>,
}

impl SemanticMatcher {
    pub fn new(use_local: bool) -> AssistantResult<Self> {
        let engine = EmbeddingEngine::new(256, use_local)?;
        let intent_examples = Self::initialize_intent_examples();

        Ok(Self {
            engine,
            intent_examples,
        })
    }

    /// Find the most likely intent for a query using semantic similarity
    pub async fn match_intent(&mut self, query: &str) -> AssistantResult<Vec<(String, f32)>> {
        let mut intent_scores = Vec::new();

        for (intent, examples) in &self.intent_examples {
            let mut max_score: f32 = 0.0;

            // Find the best matching example for this intent
            for example in examples {
                let score = self.engine.similarity(query, example).await?;
                max_score = max_score.max(score);
            }

            intent_scores.push((intent.clone(), max_score));
        }

        // Sort by score (descending)
        intent_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(intent_scores)
    }

    /// Initialize example phrases for each intent
    fn initialize_intent_examples() -> HashMap<String, Vec<String>> {
        let mut examples = HashMap::new();

        examples.insert(
            "Deploy".to_string(),
            vec![
                "deploy my application".to_string(),
                "launch the service".to_string(),
                "start the app from repository".to_string(),
                "run my code".to_string(),
                "create deployment".to_string(),
            ],
        );

        examples.insert(
            "Scale".to_string(),
            vec![
                "scale up the service".to_string(),
                "increase replicas".to_string(),
                "add more resources".to_string(),
                "resize the application".to_string(),
                "scale to 5 instances".to_string(),
            ],
        );

        examples.insert(
            "Query".to_string(),
            vec![
                "show me all agents".to_string(),
                "list running services".to_string(),
                "get application status".to_string(),
                "find healthy nodes".to_string(),
                "display gpu usage".to_string(),
            ],
        );

        examples.insert(
            "Debug".to_string(),
            vec![
                "debug this issue".to_string(),
                "troubleshoot the problem".to_string(),
                "diagnose application failure".to_string(),
                "fix the error".to_string(),
                "investigate performance issue".to_string(),
            ],
        );

        examples.insert(
            "Optimize".to_string(),
            vec![
                "optimize performance".to_string(),
                "improve efficiency".to_string(),
                "enhance throughput".to_string(),
                "reduce latency".to_string(),
                "make it faster".to_string(),
            ],
        );

        examples.insert(
            "Help".to_string(),
            vec![
                "help me".to_string(),
                "how do I".to_string(),
                "what is".to_string(),
                "explain this".to_string(),
                "documentation".to_string(),
            ],
        );

        examples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embedding_generation() {
        let mut engine = EmbeddingEngine::new(128, true).unwrap();

        let embedding = engine.embed("deploy my application").await.unwrap();
        assert_eq!(embedding.len(), 128);

        // Check that it's normalized
        let norm = embedding.dot(&embedding).sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_similarity_calculation() {
        let mut engine = EmbeddingEngine::new(128, true).unwrap();

        let similarity = engine
            .similarity("deploy my app", "deploy my application")
            .await
            .unwrap();
        assert!(similarity > 0.5);

        let similarity = engine
            .similarity("deploy app", "scale service")
            .await
            .unwrap();
        assert!(similarity < 0.8);
    }

    #[tokio::test]
    async fn test_find_most_similar() {
        let mut engine = EmbeddingEngine::new(128, true).unwrap();

        let candidates = vec![
            "deploy my application".to_string(),
            "scale the service".to_string(),
            "launch the app".to_string(),
            "show me status".to_string(),
        ];

        let results = engine
            .find_most_similar("deploy the app", &candidates, 2)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
    }

    #[tokio::test]
    async fn test_semantic_matcher() {
        let mut matcher = SemanticMatcher::new(true).unwrap();

        let intent_scores = matcher.match_intent("deploy my application").await.unwrap();
        assert!(!intent_scores.is_empty());

        // Deploy should be the top match
        assert_eq!(intent_scores[0].0, "Deploy");
        assert!(intent_scores[0].1 > 0.5);
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let mut engine = EmbeddingEngine::new(64, true).unwrap();

        // First embedding
        let embed1 = engine.embed("test phrase").await.unwrap();
        let (cache_size, _) = engine.cache_stats();
        assert_eq!(cache_size, 1);

        // Second embedding (should use cache)
        let embed2 = engine.embed("test phrase").await.unwrap();
        assert_eq!(embed1, embed2);

        // Clear cache
        engine.clear_cache();
        let (cache_size, _) = engine.cache_stats();
        assert_eq!(cache_size, 0);
    }

    #[tokio::test]
    async fn test_empty_text_embedding() {
        let mut engine = EmbeddingEngine::new(64, true).unwrap();

        let embedding = engine.embed("").await.unwrap();
        assert_eq!(embedding.len(), 64);

        // Should be zero vector
        let sum: f32 = embedding.sum();
        assert_eq!(sum, 0.0);
    }
}
