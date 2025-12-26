//! Scenario configuration system for stress testing GPU agent swarms
//!
//! This module provides a flexible configuration system to define and execute
//! different types of agent scenarios including simple reactive agents,
//! LLM-reasoning agents, and knowledge graph agents.

pub mod config;
pub mod knowledge;
pub mod reasoning;
pub mod runner;
pub mod simple;

pub use config::{
    AgentType, KnowledgeConfig, MemoryPattern, PromptComplexity, ReasoningConfig, ScenarioConfig,
    ScenarioType, SimpleBehavior,
};
pub use runner::{ScenarioResult, ScenarioRunner};

// Re-export concrete scenario implementations
pub use knowledge::KnowledgeAgentScenario;
pub use reasoning::ReasoningAgentScenario;
pub use simple::SimpleAgentScenario;
