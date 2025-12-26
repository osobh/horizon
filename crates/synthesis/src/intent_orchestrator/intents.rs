//! Intent definition and classification

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::entities::Entity;
use super::types::ScaleDirection;

/// Intent representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    /// Intent type
    pub intent_type: IntentType,
    /// Confidence score
    pub confidence: f32,
    /// Extracted entities
    pub entities: Vec<Entity>,
    /// Raw input text
    pub raw_text: String,
    /// Processed tokens
    pub tokens: Vec<String>,
    /// Intent metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Intent types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentType {
    /// Deploy a new agent or service
    Deploy,
    /// Scale existing resources
    Scale(ScaleDirection),
    /// Query system status
    Query,
    /// Update configuration
    Configure,
    /// Monitor metrics
    Monitor,
    /// Optimize performance
    Optimize,
    /// Debug issues
    Debug,
    /// Migrate resources
    Migrate,
    /// Backup data
    Backup,
    /// Restore from backup
    Restore,
    /// Execute custom action
    Custom(String),
    /// Unknown intent
    Unknown,
}

impl Intent {
    /// Create new intent
    pub fn new(intent_type: IntentType, raw_text: String) -> Self {
        Self {
            intent_type,
            confidence: 0.0,
            entities: Vec::new(),
            raw_text,
            tokens: Vec::new(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Add entity
    pub fn add_entity(&mut self, entity: Entity) {
        self.entities.push(entity);
    }

    /// Set tokens
    pub fn with_tokens(mut self, tokens: Vec<String>) -> Self {
        self.tokens = tokens;
        self
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Check if intent is actionable
    pub fn is_actionable(&self) -> bool {
        self.confidence > 0.7 && !matches!(self.intent_type, IntentType::Unknown)
    }

    /// Get primary action from intent
    pub fn get_primary_action(&self) -> String {
        match &self.intent_type {
            IntentType::Deploy => "deploy".to_string(),
            IntentType::Scale(_) => "scale".to_string(),
            IntentType::Query => "query".to_string(),
            IntentType::Configure => "configure".to_string(),
            IntentType::Monitor => "monitor".to_string(),
            IntentType::Optimize => "optimize".to_string(),
            IntentType::Debug => "debug".to_string(),
            IntentType::Migrate => "migrate".to_string(),
            IntentType::Backup => "backup".to_string(),
            IntentType::Restore => "restore".to_string(),
            IntentType::Custom(action) => action.clone(),
            IntentType::Unknown => "unknown".to_string(),
        }
    }

    /// Get required entities for intent
    pub fn get_required_entities(&self) -> Vec<String> {
        match &self.intent_type {
            IntentType::Deploy => vec!["service".to_string(), "environment".to_string()],
            IntentType::Scale(_) => vec!["service".to_string(), "count".to_string()],
            IntentType::Query => vec!["metric".to_string()],
            IntentType::Configure => vec!["service".to_string(), "setting".to_string()],
            IntentType::Monitor => vec!["service".to_string(), "metric".to_string()],
            IntentType::Optimize => vec!["service".to_string(), "resource".to_string()],
            IntentType::Debug => vec!["service".to_string()],
            IntentType::Migrate => vec!["source".to_string(), "destination".to_string()],
            IntentType::Backup => vec!["service".to_string()],
            IntentType::Restore => vec!["backup_id".to_string()],
            IntentType::Custom(_) => vec![],
            IntentType::Unknown => vec![],
        }
    }

    /// Validate intent has required entities
    pub fn validate(&self) -> Result<(), String> {
        let required = self.get_required_entities();
        let entity_types: Vec<String> = self.entities.iter()
            .map(|e| e.entity_type.to_string())
            .collect();
        
        for req in required {
            if !entity_types.contains(&req) {
                return Err(format!("Missing required entity: {}", req));
            }
        }
        
        if self.confidence < 0.5 {
            return Err(format!("Confidence too low: {}", self.confidence));
        }
        
        Ok(())
    }
}

/// Classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// Predicted intent type
    pub intent_type: IntentType,
    /// Confidence scores for each intent type
    pub scores: HashMap<String, f32>,
    /// Top k predictions
    pub top_k: Vec<(IntentType, f32)>,
    /// Classification metadata
    pub metadata: HashMap<String, String>,
}

impl ClassificationResult {
    /// Create new classification result
    pub fn new(intent_type: IntentType) -> Self {
        Self {
            intent_type,
            scores: HashMap::new(),
            top_k: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add score for intent type
    pub fn add_score(&mut self, intent_type: String, score: f32) {
        self.scores.insert(intent_type, score);
    }

    /// Calculate top k predictions
    pub fn calculate_top_k(&mut self, k: usize) {
        let mut scores: Vec<(IntentType, f32)> = self.scores.iter()
            .map(|(intent_str, &score)| {
                let intent_type = match intent_str.as_str() {
                    "deploy" => IntentType::Deploy,
                    "scale_up" => IntentType::Scale(ScaleDirection::Up),
                    "scale_down" => IntentType::Scale(ScaleDirection::Down),
                    "query" => IntentType::Query,
                    "configure" => IntentType::Configure,
                    "monitor" => IntentType::Monitor,
                    "optimize" => IntentType::Optimize,
                    "debug" => IntentType::Debug,
                    "migrate" => IntentType::Migrate,
                    "backup" => IntentType::Backup,
                    "restore" => IntentType::Restore,
                    custom => IntentType::Custom(custom.to_string()),
                };
                (intent_type, score)
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        self.top_k = scores.into_iter().take(k).collect();
    }

    /// Get confidence for predicted intent
    pub fn get_confidence(&self) -> f32 {
        self.top_k.first()
            .map(|(_, score)| *score)
            .unwrap_or(0.0)
    }
}

/// Training example for intent classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Input text
    pub text: String,
    /// True intent label
    pub label: IntentType,
    /// Entities in the text
    pub entities: Vec<Entity>,
    /// Optional weight for example
    pub weight: f32,
}

impl TrainingExample {
    /// Create new training example
    pub fn new(text: String, label: IntentType) -> Self {
        Self {
            text,
            label,
            entities: Vec::new(),
            weight: 1.0,
        }
    }

    /// Add entity annotation
    pub fn add_entity(&mut self, entity: Entity) {
        self.entities.push(entity);
    }

    /// Set example weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}