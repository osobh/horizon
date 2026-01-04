//! Core types for intent orchestration

use serde::{Deserialize, Serialize};

/// Agent identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub uuid::Uuid);

impl AgentId {
    /// Create new agent ID
    pub fn new() -> Self {
        AgentId(uuid::Uuid::new_v4())
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

/// Activation function types for neural networks
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActivationType {
    /// Rectified Linear Unit
    ReLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Hyperbolic tangent
    Tanh,
    /// Sigmoid
    Sigmoid,
    /// Softmax
    Softmax,
}

/// Comparison operators for success criteria
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Equal to
    Equal,
    /// Not equal to
    NotEqual,
    /// Greater than
    GreaterThan,
    /// Greater than or equal to
    GreaterThanOrEqual,
    /// Less than
    LessThan,
    /// Less than or equal to
    LessThanOrEqual,
    /// Contains
    Contains,
    /// Does not contain
    NotContains,
}

/// Scale direction for scaling intents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleDirection {
    /// Scale up resources
    Up,
    /// Scale down resources
    Down,
    /// Scale horizontally (out)
    Out,
    /// Scale horizontally (in)
    In,
}

/// Retry backoff strategies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed,
    /// Linear increase in delay
    Linear,
    /// Exponential increase in delay
    Exponential,
    /// Fibonacci sequence delays
    Fibonacci,
    /// Random jitter added to delays
    Jitter,
}

/// Success criterion for evaluating execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriterion {
    /// Metric to evaluate
    pub metric: String,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Target value
    pub target_value: f64,
    /// Optional tolerance
    pub tolerance: Option<f64>,
}

impl SuccessCriterion {
    /// Create new success criterion
    pub fn new(metric: String, operator: ComparisonOperator, target_value: f64) -> Self {
        Self {
            metric,
            operator,
            target_value,
            tolerance: None,
        }
    }

    /// Set tolerance for comparison
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    /// Evaluate if criterion is met
    pub fn evaluate(&self, actual_value: f64) -> bool {
        let tolerance = self.tolerance.unwrap_or(0.0);

        match self.operator {
            ComparisonOperator::Equal => (actual_value - self.target_value).abs() <= tolerance,
            ComparisonOperator::NotEqual => (actual_value - self.target_value).abs() > tolerance,
            ComparisonOperator::GreaterThan => actual_value > self.target_value,
            ComparisonOperator::GreaterThanOrEqual => actual_value >= self.target_value,
            ComparisonOperator::LessThan => actual_value < self.target_value,
            ComparisonOperator::LessThanOrEqual => actual_value <= self.target_value,
            _ => false, // Contains/NotContains not applicable for numeric values
        }
    }
}

/// Vocabulary for NLP processing
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Word to index mapping
    pub word_to_idx: std::collections::HashMap<String, usize>,
    /// Index to word mapping
    pub idx_to_word: std::collections::HashMap<usize, String>,
    /// Special tokens
    pub special_tokens: SpecialTokens,
    /// Vocabulary size
    pub size: usize,
}

/// Special tokens for NLP
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Padding token
    pub pad: String,
    /// Unknown token
    pub unk: String,
    /// Start of sequence
    pub bos: String,
    /// End of sequence
    pub eos: String,
    /// Mask token
    pub mask: String,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            pad: "[PAD]".to_string(),
            unk: "[UNK]".to_string(),
            bos: "[BOS]".to_string(),
            eos: "[EOS]".to_string(),
            mask: "[MASK]".to_string(),
        }
    }
}

impl Vocabulary {
    /// Create new vocabulary
    pub fn new() -> Self {
        let special_tokens = SpecialTokens::default();
        let mut word_to_idx = std::collections::HashMap::new();
        let mut idx_to_word = std::collections::HashMap::new();

        // Add special tokens
        word_to_idx.insert(special_tokens.pad.clone(), 0);
        word_to_idx.insert(special_tokens.unk.clone(), 1);
        word_to_idx.insert(special_tokens.bos.clone(), 2);
        word_to_idx.insert(special_tokens.eos.clone(), 3);
        word_to_idx.insert(special_tokens.mask.clone(), 4);

        idx_to_word.insert(0, special_tokens.pad.clone());
        idx_to_word.insert(1, special_tokens.unk.clone());
        idx_to_word.insert(2, special_tokens.bos.clone());
        idx_to_word.insert(3, special_tokens.eos.clone());
        idx_to_word.insert(4, special_tokens.mask.clone());

        Self {
            word_to_idx,
            idx_to_word,
            special_tokens,
            size: 5,
        }
    }

    /// Add word to vocabulary
    pub fn add_word(&mut self, word: String) -> usize {
        if let Some(&idx) = self.word_to_idx.get(&word) {
            idx
        } else {
            let idx = self.size;
            self.word_to_idx.insert(word.clone(), idx);
            self.idx_to_word.insert(idx, word);
            self.size += 1;
            idx
        }
    }

    /// Get index for word
    pub fn get_idx(&self, word: &str) -> usize {
        self.word_to_idx
            .get(word)
            .copied()
            .unwrap_or_else(|| self.word_to_idx[&self.special_tokens.unk])
    }

    /// Get word for index
    pub fn get_word(&self, idx: usize) -> Option<&String> {
        self.idx_to_word.get(&idx)
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Knowledge graph for entity relationships
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// Entities in the graph
    pub entities: std::collections::HashMap<String, super::entities::Entity>,
    /// Relations between entities
    pub relations: Vec<super::entities::EntityRelation>,
}

/// Entity value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
}
