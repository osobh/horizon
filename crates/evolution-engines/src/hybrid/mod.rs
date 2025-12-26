//! Hybrid evolution system combining multiple engines
//!
//! This system orchestrates multiple evolution engines to leverage their strengths

mod config;
mod engine_impl;
mod performance;
mod system;

#[cfg(test)]
mod tests;

pub use config::{EngineStrategy, HybridConfig};
pub use performance::EnginePerformance;
pub use system::{EngineType, HybridEvolutionSystem};
