//! Mutation operations for evolution

use serde::{Deserialize, Serialize};

/// Types of mutations available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationType {
    ParameterTweak,
    StructuralChange,
    CodeModification,
}

/// Mutation operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mutation {
    pub mutation_type: MutationType,
    pub intensity: f32,
    pub target: String,
}
