//! Darwin-Gödel Integration Layer for Dynamic Mode Switching
//!
//! This module provides seamless integration between Darwin-style empirical validation
//! and traditional Gödel Machine formal proofs, enabling dynamic mode switching
//! based on context, available resources, and validation requirements.

pub mod context_analyzer;
pub mod mode_controller;
pub mod types;
pub mod validation_bridge;

pub use context_analyzer::ContextAnalyzer;
pub use mode_controller::DarwinGodelController;
pub use types::{
    ContextMetrics, IntegrationConfig, ModeDecision, ModeSwitch, ValidationMode, ValidationRequest,
    ValidationResponse,
};
pub use validation_bridge::ValidationBridge;

#[cfg(test)]
mod tests;
