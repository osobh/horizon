//! Code Modification Engine for Darwin Gödel Machine
//!
//! This module implements the self-referential code modification capabilities
//! inspired by the Darwin Gödel Machine paper, enabling agents to modify
//! their own code to improve performance.

mod analyzer;
mod modifier;
mod types;
mod validator;

pub use analyzer::CodeAnalyzer;
pub use modifier::CodeModifier;
pub use types::{
    CodeAnalysis, CodeModification, ModificationProposal, ModificationResult, ModificationType,
    PerformanceFeedback,
};
pub use validator::ModificationValidator;

#[cfg(test)]
mod tests;
