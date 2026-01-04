//! Error types for the bootstrap crate

use thiserror::Error;

/// Bootstrap-specific errors
#[derive(Debug, Error)]
pub enum BootstrapError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Agent initialization error
    #[error("Agent initialization failed: {0}")]
    AgentInit(String),

    /// DNA synthesis error
    #[error("DNA synthesis failed: {0}")]
    DnaSynthesis(String),

    /// Population generation error
    #[error("Population generation failed: {0}")]
    PopulationGen(String),

    /// Monitoring setup error
    #[error("Monitoring setup failed: {0}")]
    MonitoringSetup(String),

    /// Genesis creation error
    #[error("Genesis creation failed: {0}")]
    GenesisCreation(String),

    /// Resource allocation error
    #[error("Resource allocation failed: {0}")]
    ResourceAllocation(String),

    /// Validation error
    #[error("Validation failed: {0}")]
    Validation(String),

    /// IO error
    #[error("IO operation failed")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization failed")]
    Serialization(#[from] serde_json::Error),

    /// Parse error
    #[error("Parse error: {0}")]
    Parse(String),

    /// Generic error for unexpected conditions
    #[error("Unexpected error: {0}")]
    Unexpected(String),
}

/// Result type for bootstrap operations
pub type BootstrapResult<T> = Result<T, BootstrapError>;

impl BootstrapError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::Config(msg.into())
    }

    /// Create an agent initialization error
    pub fn agent_init<S: Into<String>>(msg: S) -> Self {
        Self::AgentInit(msg.into())
    }

    /// Create a DNA synthesis error
    pub fn dna_synthesis<S: Into<String>>(msg: S) -> Self {
        Self::DnaSynthesis(msg.into())
    }

    /// Create a population generation error
    pub fn population_gen<S: Into<String>>(msg: S) -> Self {
        Self::PopulationGen(msg.into())
    }

    /// Create a monitoring setup error
    pub fn monitoring_setup<S: Into<String>>(msg: S) -> Self {
        Self::MonitoringSetup(msg.into())
    }

    /// Create a genesis creation error
    pub fn genesis_creation<S: Into<String>>(msg: S) -> Self {
        Self::GenesisCreation(msg.into())
    }

    /// Create a resource allocation error
    pub fn resource_allocation<S: Into<String>>(msg: S) -> Self {
        Self::ResourceAllocation(msg.into())
    }

    /// Create a validation error
    pub fn validation<S: Into<String>>(msg: S) -> Self {
        Self::Validation(msg.into())
    }

    /// Create an unexpected error
    pub fn unexpected<S: Into<String>>(msg: S) -> Self {
        Self::Unexpected(msg.into())
    }
}
