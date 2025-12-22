//! Configuration for intent orchestrator

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    /// Model configuration
    pub model_config: ModelConfig,
    /// Classification threshold
    pub classification_threshold: f32,
    /// Maximum concurrent executions
    pub max_concurrent_executions: usize,
    /// Execution timeout
    pub execution_timeout: Duration,
    /// Enable caching
    pub enable_caching: bool,
    /// Cache TTL
    pub cache_ttl: Duration,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Debug mode
    pub debug_mode: bool,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type (transformer, bert, etc.)
    pub model_type: String,
    /// Model path or identifier
    pub model_path: String,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Batch size for inference
    pub batch_size: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            model_config: ModelConfig::default(),
            classification_threshold: 0.7,
            max_concurrent_executions: 10,
            execution_timeout: Duration::from_secs(300),
            enable_caching: true,
            cache_ttl: Duration::from_secs(3600),
            enable_metrics: true,
            debug_mode: false,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: "transformer".to_string(),
            model_path: "models/intent_classifier".to_string(),
            vocab_size: 10000,
            embedding_dim: 128,
            hidden_dim: 256,
            num_heads: 8,
            num_layers: 6,
            dropout_rate: 0.1,
            max_sequence_length: 512,
            batch_size: 32,
        }
    }
}

impl OrchestratorConfig {
    /// Create config for production environment
    pub fn production() -> Self {
        Self {
            model_config: ModelConfig {
                model_type: "bert".to_string(),
                model_path: "models/production/intent_bert".to_string(),
                vocab_size: 30000,
                embedding_dim: 768,
                hidden_dim: 3072,
                num_heads: 12,
                num_layers: 12,
                dropout_rate: 0.1,
                max_sequence_length: 512,
                batch_size: 64,
            },
            classification_threshold: 0.85,
            max_concurrent_executions: 50,
            execution_timeout: Duration::from_secs(600),
            enable_caching: true,
            cache_ttl: Duration::from_secs(7200),
            enable_metrics: true,
            debug_mode: false,
        }
    }

    /// Create config for development environment
    pub fn development() -> Self {
        Self {
            model_config: ModelConfig {
                model_type: "transformer".to_string(),
                model_path: "models/dev/intent_transformer".to_string(),
                vocab_size: 5000,
                embedding_dim: 64,
                hidden_dim: 128,
                num_heads: 4,
                num_layers: 3,
                dropout_rate: 0.2,
                max_sequence_length: 256,
                batch_size: 16,
            },
            classification_threshold: 0.6,
            max_concurrent_executions: 5,
            execution_timeout: Duration::from_secs(120),
            enable_caching: false,
            cache_ttl: Duration::from_secs(60),
            enable_metrics: true,
            debug_mode: true,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.classification_threshold <= 0.0 || self.classification_threshold > 1.0 {
            return Err("Classification threshold must be between 0 and 1".to_string());
        }

        if self.max_concurrent_executions == 0 {
            return Err("Must allow at least one concurrent execution".to_string());
        }

        if self.model_config.vocab_size == 0 {
            return Err("Vocabulary size must be greater than 0".to_string());
        }

        if self.model_config.embedding_dim == 0 {
            return Err("Embedding dimension must be greater than 0".to_string());
        }

        if self.model_config.num_heads == 0 {
            return Err("Number of attention heads must be greater than 0".to_string());
        }

        if self.model_config.num_layers == 0 {
            return Err("Number of layers must be greater than 0".to_string());
        }

        Ok(())
    }
}