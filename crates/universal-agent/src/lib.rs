//! Universal Agent with Versioned AgentDNA
//!
//! This crate provides a single, evolvable agent implementation that replaces
//! multiple specialized agent crates. The UniversalAgent is configured via
//! AgentDNA, which contains versioned capabilities that can evolve through
//! learning from reconciliation, postprocessing, and other operations.
//!
//! # Architecture
//!
//! - **AgentDNA**: The versioned capability blueprint containing skills, behavior,
//!   and architecture configuration.
//! - **UniversalAgent**: The runtime that executes skills based on DNA configuration.
//! - **AgentRegistry**: Service discovery for finding the best versioned agent.
//! - **LearningEngine**: Upskilling mechanism for DNA evolution.
//!
//! # Example
//!
//! ```ignore
//! use universal_agent::{AgentDNA, UniversalAgent, AgentRegistry};
//!
//! // Create a DNA with efficiency hunting capabilities
//! let dna = AgentDNA::efficiency_hunter_template();
//!
//! // Create an agent from the DNA
//! let agent = UniversalAgent::from_dna(dna, config).await?;
//!
//! // Execute a skill
//! let response = agent.execute_skill("detect_idle_resources", request).await?;
//! ```

pub mod dna;
pub mod agent;
pub mod learning;
pub mod registry;
pub mod integration;
mod error;

// Re-exports for convenience
pub use dna::{
    AgentDNA, DNAId, DNAVersion, DNALineage, Modification, ModificationType, ModificationSource,
    Skill, SkillCategory, SkillEvidence, SkillExecution, SkillRepertoire,
    BehaviorGenome, ArchitectureGenome,
    Capability, FitnessProfile, FitnessDimension, BenchmarkEvidence,
};
pub use agent::{UniversalAgent, UniversalAgentConfig};
pub use learning::LearningEngine;
pub use registry::{AgentRegistry, CapabilityRequirements};
pub use error::{UniversalAgentError, Result};

// Re-export core agent types for compatibility
pub use horizon_agents_core::{
    Agent, AgentConfig, AgentRequest, AgentResponse, AutonomyLevel,
    HealthStatus, Lifecycle, AgentState, SafetyThresholds,
};
