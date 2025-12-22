//! Entity extraction and relation mapping

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Entity representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity text
    pub text: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Entity value
    pub value: EntityValue,
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
    /// Confidence score
    pub confidence: f32,
    /// Entity metadata
    pub metadata: HashMap<String, String>,
}

/// Entity types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    /// Service or application name
    Service,
    /// Environment (dev, staging, prod)
    Environment,
    /// Numeric value
    Number,
    /// Date/time reference
    DateTime,
    /// Resource type (CPU, memory, disk)
    Resource,
    /// Metric name
    Metric,
    /// Location or region
    Location,
    /// User or role
    User,
    /// Configuration key
    ConfigKey,
    /// Configuration value
    ConfigValue,
    /// Action verb
    Action,
    /// Custom entity type
    Custom(String),
}

impl EntityType {
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match self {
            EntityType::Service => "service".to_string(),
            EntityType::Environment => "environment".to_string(),
            EntityType::Number => "number".to_string(),
            EntityType::DateTime => "datetime".to_string(),
            EntityType::Resource => "resource".to_string(),
            EntityType::Metric => "metric".to_string(),
            EntityType::Location => "location".to_string(),
            EntityType::User => "user".to_string(),
            EntityType::ConfigKey => "config_key".to_string(),
            EntityType::ConfigValue => "config_value".to_string(),
            EntityType::Action => "action".to_string(),
            EntityType::Custom(s) => s.clone(),
        }
    }
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
    /// List of values
    List(Vec<EntityValue>),
    /// Key-value pairs
    Map(HashMap<String, EntityValue>),
}

impl Entity {
    /// Create new entity
    pub fn new(text: String, entity_type: EntityType, start: usize, end: usize) -> Self {
        Self {
            value: EntityValue::String(text.clone()),
            text,
            entity_type,
            start,
            end,
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Set entity value
    pub fn with_value(mut self, value: EntityValue) -> Self {
        self.value = value;
        self
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Check if entity is valid
    pub fn is_valid(&self) -> bool {
        !self.text.is_empty() && self.end > self.start && self.confidence > 0.0
    }

    /// Get entity length
    pub fn length(&self) -> usize {
        self.end - self.start
    }
}

/// Entity relation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelation {
    /// Source entity
    pub source: Entity,
    /// Target entity
    pub target: Entity,
    /// Relation type
    pub relation_type: RelationType,
    /// Confidence score
    pub confidence: f32,
}

/// Relation types between entities
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationType {
    /// Belongs to
    BelongsTo,
    /// Contains
    Contains,
    /// Depends on
    DependsOn,
    /// Modifies
    Modifies,
    /// References
    References,
    /// Precedes
    Precedes,
    /// Follows
    Follows,
    /// Equivalent to
    EquivalentTo,
    /// Custom relation
    Custom(String),
}

impl EntityRelation {
    /// Create new relation
    pub fn new(source: Entity, target: Entity, relation_type: RelationType) -> Self {
        Self {
            source,
            target,
            relation_type,
            confidence: 1.0,
        }
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Check if relation is valid
    pub fn is_valid(&self) -> bool {
        self.source.is_valid() && self.target.is_valid() && self.confidence > 0.0
    }

    /// Get relation strength (based on confidence and entity confidences)
    pub fn strength(&self) -> f32 {
        self.confidence * self.source.confidence * self.target.confidence
    }
}

/// Entity extraction result
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Extracted entities
    pub entities: Vec<Entity>,
    /// Entity relations
    pub relations: Vec<EntityRelation>,
    /// Extraction metadata
    pub metadata: HashMap<String, String>,
}

impl ExtractionResult {
    /// Create new extraction result
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            relations: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add entity
    pub fn add_entity(&mut self, entity: Entity) {
        self.entities.push(entity);
    }

    /// Add relation
    pub fn add_relation(&mut self, relation: EntityRelation) {
        self.relations.push(relation);
    }

    /// Get entities by type
    pub fn get_entities_by_type(&self, entity_type: &EntityType) -> Vec<&Entity> {
        self.entities.iter()
            .filter(|e| &e.entity_type == entity_type)
            .collect()
    }

    /// Get relations by type
    pub fn get_relations_by_type(&self, relation_type: &RelationType) -> Vec<&EntityRelation> {
        self.relations.iter()
            .filter(|r| &r.relation_type == relation_type)
            .collect()
    }

    /// Merge with another extraction result
    pub fn merge(&mut self, other: ExtractionResult) {
        self.entities.extend(other.entities);
        self.relations.extend(other.relations);
        self.metadata.extend(other.metadata);
    }
}

impl Default for ExtractionResult {
    fn default() -> Self {
        Self::new()
    }
}