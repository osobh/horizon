//! ADAS Meta Agent Search Implementation
//!
//! Implements the Meta Agent Search framework from ADAS (Hu et al., 2024)
//! This includes the meta-agent that rewrites candidate code, prompts, and tool calls,
//! maintains an evolving archive of workflows, and optimizes agent collaboration structures.

mod meta_agent;
mod types;
mod workflow_archive;

pub use meta_agent::*;
pub use types::*;
pub use workflow_archive::*;
