//! Intent orchestration for agent synthesis
//!
//! This module provides comprehensive intent understanding and orchestration including:
//! - Natural language intent classification using transformer models
//! - Entity extraction and relation mapping
//! - Action planning and resource allocation
//! - Execution orchestration with retry policies
//! - Context-aware decision making
//! - Multi-agent coordination

// Public exports
pub use config::*;
pub use context::*;
pub use engine::*;
pub use entities::*;
pub use execution::*;
pub use intents::*;
pub use metrics::*;
pub use models::*;
pub use orchestrator::*;
pub use planning::*;
pub use types::*;

// Module declarations
mod config;
mod context;
mod engine;
mod entities;
mod execution;
mod intents;
mod metrics;
mod models;
mod orchestrator;
mod planning;
mod types;

#[cfg(test)]
mod tests;
