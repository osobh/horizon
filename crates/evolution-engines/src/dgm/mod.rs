//! DGM (Discovered Agent Growth Mode) evolution engine for self-improvement with XP integration
//!
//! This engine focuses on incremental self-improvement through discovered growth patterns
//! combined with XP-based agent progression

// Re-export everything from the dgm module
pub use self::config::{DgmConfig, GrowthPatterns, ImprovementParameters};
pub use self::engine::DgmEngine;
pub use self::improvement::{DiscoveredPattern, GrowthHistory, PatternType};
pub use self::patterns::PatternDiscovery;
pub use self::xp_bridge::{DgmXPEngine, DgmXPFitnessFunction, DgmXPStats};

mod config;
mod engine;
mod improvement;
mod patterns;
mod xp_bridge;

#[cfg(test)]
mod tests;
