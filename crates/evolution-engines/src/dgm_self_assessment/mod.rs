//! Self-assessment functionality for DGM (Discovered Agent Growth Mode)
//!
//! This module implements self-assessment capabilities inspired by the Darwin Gödel Machine paper,
//! enabling agents to evaluate their own performance, track improvements, and assess their
//! self-modification capabilities.

mod assessment_engine;
mod config;
mod lineage_tracker;
mod performance_tracker;
mod types;

pub use assessment_engine::SelfAssessmentEngine;
pub use config::SelfAssessmentConfig;
pub use lineage_tracker::LineageTracker;
pub use performance_tracker::PerformanceTracker;
pub use types::*;

#[cfg(test)]
mod tests;
