//! Pattern type definitions and structures

use serde::{Deserialize, Serialize};

/// A recognized pattern in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizedPattern {
    pub pattern_type: PatternType,
    pub name: String,
    pub confidence: f32,
    pub description: String,
    pub recommendations: Vec<String>,
    pub tags: Vec<String>,
}

/// Types of patterns that can be recognized
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    Language,
    Framework,
    Dependency,
    Architecture,
    Complexity,
    Resource,
}
