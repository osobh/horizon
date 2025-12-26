//! Error types for the Universal Agent

use thiserror::Error;
use uuid::Uuid;

use crate::dna::DNAVersion;

/// Result type for Universal Agent operations
pub type Result<T> = std::result::Result<T, UniversalAgentError>;

/// Errors that can occur in the Universal Agent
#[derive(Error, Debug)]
pub enum UniversalAgentError {
    #[error("Skill not found: {0}")]
    SkillNotFound(String),

    #[error("Handler not found: {0}")]
    HandlerNotFound(String),

    #[error("Unknown handler: {0}")]
    UnknownHandler(String),

    #[error("Prerequisites not met for skill {skill_id}: missing {missing:?}")]
    PrerequisitesNotMet {
        skill_id: String,
        missing: Vec<String>,
    },

    #[error("Invalid autonomy level transition from {from} to {to}")]
    InvalidAutonomyTransition { from: String, to: String },

    #[error("Agent not operational: current state is {0}")]
    NotOperational(String),

    #[error("DNA not found: {0}")]
    DNANotFound(Uuid),

    #[error("Version not found: {0} @ {1}")]
    VersionNotFound(Uuid, DNAVersion),

    #[error("Invalid version upgrade: current {current}, proposed {proposed}")]
    InvalidVersionUpgrade {
        current: DNAVersion,
        proposed: DNAVersion,
    },

    #[error("DNA validation failed: {0}")]
    ValidationFailed(String),

    #[error("Learning engine error: {0}")]
    LearningError(String),

    #[error("Registry error: {0}")]
    RegistryError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Core agent error: {0}")]
    CoreError(#[from] horizon_agents_core::AgentError),

    #[error("Internal error: {0}")]
    Internal(String),
}
