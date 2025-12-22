//! Causal chain analysis and pathways

use chrono::Duration as ChronoDuration;

use super::types::CausalRelationship;

/// Complete causal chain from root cause to final effect
#[derive(Debug, Clone)]
pub struct CausalChain {
    /// Chain identifier
    pub id: String,
    /// Ordered sequence of nodes in causal chain
    pub nodes: Vec<String>,
    /// Causal relationships connecting the nodes
    pub relationships: Vec<CausalRelationship>,
    /// Overall chain strength (weakest link)
    pub chain_strength: f64,
    /// Chain confidence
    pub chain_confidence: f64,
    /// Total temporal delay for full chain
    pub total_delay: ChronoDuration,
    /// Alternative causal pathways
    pub alternative_pathways: Vec<AlternativePathway>,
}

/// Alternative causal pathway
#[derive(Debug, Clone)]
pub struct AlternativePathway {
    pub nodes: Vec<String>,
    pub strength: f64,
    pub confidence: f64,
    pub delay: ChronoDuration,
}
