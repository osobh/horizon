//! ADAS (Automated Design of Agentic Systems) evolution engine
//!
//! This engine focuses on meta-level agent architecture search with XP integration

mod config;
mod engine;
mod meta_search;
mod search_spaces;
mod xp_bridge;

#[cfg(test)]
mod tests;

pub use config::AdasConfig;
pub use engine::AdasEngine;
pub use search_spaces::{ArchitectureSearchSpace, BehaviorSearchSpace};
pub use xp_bridge::{AdasXPEngine, AdasXPFitnessFunction, AdasXPStats};
