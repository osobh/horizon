//! Behavioral learning and pattern collection for zero-config intelligence
//!
//! This module learns from deployment patterns to improve future agent configurations
//! through pattern recognition and knowledge transfer.

pub mod behavioral_learner;
pub mod knowledge_transfer;
pub mod pattern_collector;
pub mod pattern_store;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export main types
pub use behavioral_learner::BehavioralLearner;
pub use knowledge_transfer::KnowledgeTransfer;
pub use pattern_collector::PatternCollector;
pub use pattern_store::PatternStore;
pub use types::{
    DependencyFeatures, DeploymentOutcome, DeploymentPattern, FeatureVector, LanguageFeatures,
    LearningStatistics, PersonalityFeatures, ResourceFeatures, ScalingFeatures,
};
