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
pub mod types;
pub mod objectives;
pub mod strategies;
pub mod dependencies;
pub mod resources;
pub mod execution;
pub mod validation;
pub mod metrics;
pub mod planner;
pub mod optimization;

// Re-export main types for convenience
pub use config::*;
pub use types::*;
pub use objectives::*;
pub use strategies::*;
pub use dependencies::*;
pub use resources::*;
pub use execution::*;
pub use validation::*;
pub use metrics::*;
pub use planner::*;
pub use optimization::*;
