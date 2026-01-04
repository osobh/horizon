//! Neural network models for intent processing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::config::ModelConfig;
use super::types::ActivationType;

/// Transformer model for NLP tasks
#[derive(Debug, Clone)]
pub struct TransformerModel {
    /// Model configuration
    pub config: TransformerConfig,
    /// Embedding layer
    pub embeddings: EmbeddingLayer,
    /// Attention layers
    pub attention_layers: Vec<AttentionLayer>,
    /// Feed-forward layers
    pub ff_layers: Vec<FeedForwardLayer>,
    /// Layer normalization
    pub layer_norm: LayerNorm,
    /// Model weights
    pub weights: HashMap<String, Vec<f32>>,
}

/// Transformer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Maximum sequence length
    pub max_seq_length: usize,
}

/// Attention layer for transformer
#[derive(Debug, Clone)]
pub struct AttentionLayer {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Query projection
    pub query_proj: Vec<Vec<f32>>,
    /// Key projection
    pub key_proj: Vec<Vec<f32>>,
    /// Value projection
    pub value_proj: Vec<Vec<f32>>,
    /// Output projection
    pub output_proj: Vec<Vec<f32>>,
}

/// Feed-forward layer
#[derive(Debug, Clone)]
pub struct FeedForwardLayer {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function
    pub activation: ActivationType,
    /// Weights
    pub weights: Vec<Vec<f32>>,
    /// Bias
    pub bias: Vec<f32>,
}

/// Layer normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Normalized dimension
    pub normalized_dim: usize,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Scale parameter
    pub gamma: Vec<f32>,
    /// Shift parameter
    pub beta: Vec<f32>,
}

/// Embedding layer
#[derive(Debug, Clone)]
pub struct EmbeddingLayer {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Embedding matrix
    pub embeddings: Vec<Vec<f32>>,
    /// Position embeddings
    pub position_embeddings: Option<Vec<Vec<f32>>>,
}

/// BERT model for entity extraction
#[derive(Debug, Clone)]
pub struct BertModel {
    /// Model configuration
    pub config: BertConfig,
    /// Embeddings
    pub embeddings: BertEmbeddings,
    /// Encoder layers
    pub encoder_layers: Vec<BertEncoderLayer>,
    /// Pooler
    pub pooler: Option<PoolerLayer>,
}

/// BERT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Intermediate size
    pub intermediate_size: usize,
    /// Hidden activation
    pub hidden_act: ActivationType,
    /// Hidden dropout probability
    pub hidden_dropout_prob: f32,
    /// Attention dropout probability
    pub attention_probs_dropout_prob: f32,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// Type vocabulary size
    pub type_vocab_size: usize,
}

/// BERT embeddings
#[derive(Debug, Clone)]
pub struct BertEmbeddings {
    /// Word embeddings
    pub word_embeddings: Vec<Vec<f32>>,
    /// Position embeddings
    pub position_embeddings: Vec<Vec<f32>>,
    /// Token type embeddings
    pub token_type_embeddings: Vec<Vec<f32>>,
    /// Layer normalization
    pub layer_norm: LayerNorm,
}

/// BERT encoder layer
#[derive(Debug, Clone)]
pub struct BertEncoderLayer {
    /// Self-attention
    pub self_attention: AttentionLayer,
    /// Intermediate layer
    pub intermediate: FeedForwardLayer,
    /// Output layer
    pub output: FeedForwardLayer,
}

/// Pooler layer for BERT
#[derive(Debug, Clone)]
pub struct PoolerLayer {
    /// Dense layer
    pub dense: FeedForwardLayer,
    /// Activation
    pub activation: ActivationType,
}

/// Graph attention layer for relation extraction
#[derive(Debug, Clone)]
pub struct GraphAttentionLayer {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Attention weights
    pub attention_weights: Vec<Vec<f32>>,
}

/// Sequence model for action planning
#[derive(Debug, Clone)]
pub struct SequenceModel {
    /// Model type (LSTM, GRU, etc.)
    pub model_type: String,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Bidirectional
    pub bidirectional: bool,
    /// Attention mechanism
    pub attention: Option<AttentionMechanism>,
}

/// Attention mechanism for sequence models
#[derive(Debug, Clone)]
pub struct AttentionMechanism {
    /// Attention type (bahdanau, luong, etc.)
    pub attention_type: String,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Context dimension
    pub context_dim: usize,
    /// Attention weights
    pub weights: Vec<Vec<f32>>,
}

// Implementation helpers
impl TransformerModel {
    /// Create new transformer model
    pub fn new(config: TransformerConfig) -> Self {
        // Save values before moving config
        let vocab_size = config.vocab_size;
        let hidden_dim = config.hidden_dim;
        let num_heads = config.num_heads;
        let num_layers = config.num_layers;
        let max_seq_length = config.max_seq_length;

        let embeddings = EmbeddingLayer {
            vocab_size,
            embedding_dim: hidden_dim,
            embeddings: vec![vec![0.0; hidden_dim]; vocab_size],
            position_embeddings: Some(vec![vec![0.0; hidden_dim]; max_seq_length]),
        };

        let mut attention_layers = Vec::new();
        let mut ff_layers = Vec::new();

        for _ in 0..num_layers {
            attention_layers.push(AttentionLayer::new(hidden_dim, num_heads));
            ff_layers.push(FeedForwardLayer::new(
                hidden_dim,
                hidden_dim * 4,
                hidden_dim,
                ActivationType::GELU,
            ));
        }

        Self {
            config,
            embeddings,
            attention_layers,
            ff_layers,
            layer_norm: LayerNorm::new(hidden_dim),
            weights: HashMap::new(),
        }
    }
}

impl AttentionLayer {
    /// Create new attention layer
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        Self {
            hidden_dim,
            num_heads,
            query_proj: vec![vec![0.0; head_dim]; hidden_dim],
            key_proj: vec![vec![0.0; head_dim]; hidden_dim],
            value_proj: vec![vec![0.0; head_dim]; hidden_dim],
            output_proj: vec![vec![0.0; hidden_dim]; hidden_dim],
        }
    }
}

impl FeedForwardLayer {
    /// Create new feed-forward layer
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        activation: ActivationType,
    ) -> Self {
        Self {
            input_dim,
            hidden_dim,
            output_dim,
            activation,
            weights: vec![vec![0.0; hidden_dim]; input_dim],
            bias: vec![0.0; hidden_dim],
        }
    }
}

impl LayerNorm {
    /// Create new layer normalization
    pub fn new(normalized_dim: usize) -> Self {
        Self {
            normalized_dim,
            epsilon: 1e-6,
            gamma: vec![1.0; normalized_dim],
            beta: vec![0.0; normalized_dim],
        }
    }
}

// Conversions from ModelConfig
impl From<ModelConfig> for TransformerConfig {
    fn from(config: ModelConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            num_layers: config.num_layers,
            dropout_rate: config.dropout_rate,
            max_seq_length: config.max_sequence_length,
        }
    }
}

impl From<ModelConfig> for BertConfig {
    fn from(config: ModelConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_dim,
            num_hidden_layers: config.num_layers,
            num_attention_heads: config.num_heads,
            intermediate_size: config.hidden_dim * 4,
            hidden_act: ActivationType::GELU,
            hidden_dropout_prob: config.dropout_rate,
            attention_probs_dropout_prob: config.dropout_rate,
            max_position_embeddings: config.max_sequence_length,
            type_vocab_size: 2,
        }
    }
}

impl BertModel {
    /// Create new BERT model from config
    pub fn new(config: ModelConfig) -> Self {
        let bert_config: BertConfig = config.into();

        let embeddings = BertEmbeddings {
            word_embeddings: vec![vec![0.0; bert_config.hidden_size]; bert_config.vocab_size],
            position_embeddings: vec![
                vec![0.0; bert_config.hidden_size];
                bert_config.max_position_embeddings
            ],
            token_type_embeddings: vec![
                vec![0.0; bert_config.hidden_size];
                bert_config.type_vocab_size
            ],
            layer_norm: LayerNorm::new(bert_config.hidden_size),
        };

        let mut encoder_layers = Vec::new();
        for _ in 0..bert_config.num_hidden_layers {
            encoder_layers.push(BertEncoderLayer {
                self_attention: AttentionLayer::new(
                    bert_config.hidden_size,
                    bert_config.num_attention_heads,
                ),
                intermediate: FeedForwardLayer::new(
                    bert_config.hidden_size,
                    bert_config.intermediate_size,
                    bert_config.hidden_size,
                    bert_config.hidden_act,
                ),
                output: FeedForwardLayer::new(
                    bert_config.intermediate_size,
                    bert_config.hidden_size,
                    bert_config.hidden_size,
                    ActivationType::ReLU,
                ),
            });
        }

        Self {
            config: bert_config.clone(),
            embeddings,
            encoder_layers,
            pooler: Some(PoolerLayer {
                dense: FeedForwardLayer::new(
                    bert_config.hidden_size,
                    bert_config.hidden_size,
                    bert_config.hidden_size,
                    ActivationType::Tanh,
                ),
                activation: ActivationType::Tanh,
            }),
        }
    }
}
