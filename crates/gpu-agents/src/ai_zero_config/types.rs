//! AI Assistant + Zero-Config Integration Types
//!
//! Type definitions for natural language deployment and intelligent configuration generation.
//! This module provides the core data structures for the AI-driven zero-config pipeline.

mod deployment_types;
mod analysis_types;
mod config_generation_types;
mod infrastructure_types;
mod security_types;
mod monitoring_types;
mod network_types;
mod storage_types;
mod scaling_types;
mod cost_types;

pub use deployment_types::*;
pub use analysis_types::*;
pub use config_generation_types::*;
pub use infrastructure_types::*;
pub use security_types::*;
pub use monitoring_types::*;
pub use network_types::*;
pub use storage_types::*;
pub use scaling_types::*;
pub use cost_types::*;