//! Evolution streaming pipeline modules
//!
//! This module provides two implementations:
//! - `core`: Original Arc<RwLock> based pipeline (for backwards compatibility)
//! - `actor`: Cancel-safe actor-based pipeline (recommended for new code)

pub mod actor;
pub mod builder;
pub mod core;
pub mod events;
pub mod result;
pub mod stats;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use builder::EvolutionStreamingPipelineBuilder;
pub use core::EvolutionStreamingPipeline;
pub use events::EvolutionEventProcessor;
pub use result::EvolutionCycleResult;
pub use stats::PipelineStats;

// Re-export actor types for new code
pub use actor::{
    create_pipeline_actor, EvolutionPipelineActor, EvolutionPipelineHandle, PipelineRequest,
};
