//! Recovery Planning Module
//!
//! Comprehensive disaster recovery planning system with RTO/RPO modeling,
//! dependency resolution, and priority-based orchestration.
//!
//! This module provides:
//! - Recovery Time/Point Objective (RTO/RPO) planning
//! - Multi-tier recovery strategies with priority orchestration
//! - Service dependency resolution and critical path analysis
//! - Resource allocation and capacity planning
//! - Recovery testing and validation frameworks
//! - Dynamic plan adaptation and cross-region coordination

pub mod config;
pub mod dependencies;
pub mod execution;
pub mod metrics;
pub mod objectives;
pub mod optimization;
pub mod planner;
pub mod resources;
pub mod strategies;
pub mod types;
pub mod validation;

// Re-export main types for convenience
pub use config::*;
pub use dependencies::*;
pub use execution::*;
pub use metrics::*;
pub use objectives::*;
pub use optimization::*;
pub use planner::*;
pub use resources::*;
pub use strategies::*;
pub use types::*;
pub use validation::*;
