//! Performance analysis and anomaly detection for GPU container debugging
//!
//! This module provides advanced analysis capabilities for debugging data, including
//! performance pattern detection, anomaly detection, and regression analysis.

mod analysis_impl;
mod engine;
mod manager;
mod types;

pub use engine::{AnalysisConfig, DefaultAnalysisEngine};
pub use manager::{AnalysisManager, AnalysisManagerConfig};
pub use types::*;
