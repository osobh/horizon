//! Distributed SwarmAgentic Integration Tests (TDD Implementation)
//! 
//! This module implements comprehensive integration tests for distributed runtime
//! using Test-Driven Development methodology:
//! 1. RED Phase: Write failing tests defining distributed runtime requirements
//! 2. GREEN Phase: Implement minimal functionality to make tests pass
//! 3. REFACTOR Phase: Optimize for production-ready distributed scenarios

pub mod shared;
pub mod consensus_tests;
pub mod security_tests;
pub mod performance_tests;

// Re-export main types and functions for backward compatibility
pub use consensus_tests::*;
pub use security_tests::*;
pub use performance_tests::*;
pub use shared::*;