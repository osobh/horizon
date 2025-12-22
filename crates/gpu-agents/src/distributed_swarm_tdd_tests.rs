//! Distributed SwarmAgentic Integration Tests (TDD Implementation) - Refactored
//!
//! This module has been refactored into multiple smaller, focused modules to improve
//! maintainability and reduce file size. The original implementation has been split into:
//!
//! - `distributed_swarm_tdd::shared`: Common types and utilities
//! - `distributed_swarm_tdd::consensus_tests`: Multi-region consensus tests
//! - `distributed_swarm_tdd::security_tests`: Security and disaster recovery tests  
//! - `distributed_swarm_tdd::performance_tests`: Performance and scaling tests
//!
//! For backward compatibility, all types and functions are re-exported here.

pub use crate::distributed_swarm_tdd::*;

// Re-export the main test suite for backward compatibility
pub use crate::distributed_swarm_tdd::performance_tests::DistributedSwarmTddTests;
