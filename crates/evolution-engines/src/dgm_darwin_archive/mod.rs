//! Darwin-style archive system for DGM open-ended exploration
//!
//! This module implements the archive-based exploration system inspired by the Darwin Gödel Machine
//! paper, maintaining all discovered agents (not just successful ones) to enable stepping stones
//! and avoid local optima.

mod archive;
mod config;
mod types;

pub use archive::DarwinArchive;
pub use config::DarwinArchiveConfig;
pub use types::{ArchivedAgent, DiversityMetrics};

#[cfg(test)]
mod tests;
