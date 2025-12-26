//! Execution replay functionality for time-travel debugging
//!
//! This module provides the ability to replay GPU container executions from previously
//! captured memory snapshots, enabling time-travel debugging and regression analysis.
//!
//! The module is organized into several sub-modules:
//! - `types`: Core data structures and configuration types
//! - `engine`: ReplayEngine trait and implementations
//! - `manager`: Higher-level ReplayManager for coordinating replay operations
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use crate::replay::{ReplayManager, ReplayManagerConfig, MockReplayEngine};
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let engine = Arc::new(MockReplayEngine::new());
//! let config = ReplayManagerConfig::default();
//! let manager = ReplayManager::new(engine, config);
//!
//! // Start a replay session
//! // let session_id = manager.start_replay(snapshot, None).await?;
//! # Ok(())
//! # }
//! ```

#[path = "replay_engine.rs"]
pub mod engine;
#[path = "replay_manager.rs"]
pub mod manager;
#[path = "replay_types.rs"]
pub mod types;

// Re-export all types and traits for backward compatibility
pub use engine::{MockReplayEngine, ReplayEngine};
pub use manager::ReplayManager;
pub use types::*;
