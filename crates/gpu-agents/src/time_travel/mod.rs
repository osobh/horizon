//! Time-Travel Debugging Module for Evolution Systems
//!
//! This module provides comprehensive time-travel debugging capabilities for agent evolution,
//! enabling state capture, navigation, rollback, and analysis of evolutionary processes.

pub mod analyzer;
pub mod evolution_debugger;
pub mod navigator;
pub mod rollback_manager;
pub mod session;
pub mod snapshot;

// Use specific imports to avoid ambiguity
pub use analyzer::StateAnalyzer;
pub use evolution_debugger::EvolutionTimelineDebugger;
pub use evolution_debugger::{
    AgentGenome, ArchitectureGenes, BehaviorGenes, DebugSessionConfig, EvolutionState,
    PerformanceMetrics,
};
pub use navigator::TimeNavigator;
pub use rollback_manager::RollbackManager;
pub use session::DebugSession;
pub use snapshot::{EvolutionSnapshot, FitnessMetrics};
