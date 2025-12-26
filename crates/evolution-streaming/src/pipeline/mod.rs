//! Evolution streaming pipeline modules

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
