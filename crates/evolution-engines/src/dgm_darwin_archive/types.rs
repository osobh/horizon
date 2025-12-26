//! Type definitions for Darwin archive system

use crate::traits::AgentGenome;
use serde::{Deserialize, Serialize};

/// Represents an agent stored in the archive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivedAgent {
    /// Unique identifier
    pub id: String,
    /// Agent genome
    pub genome: AgentGenome,
    /// Performance score on benchmark
    pub performance_score: f64,
    /// Number of children with editing capability
    pub children_count: usize,
    /// Whether this agent retains code editing capability
    pub has_editing_capability: bool,
    /// Generation when discovered
    pub discovery_generation: u32,
    /// Parent agent ID (None for initial agent)
    pub parent_id: Option<String>,
    /// Type of modification that created this agent
    pub modification_type: String,
    /// Timestamp of creation
    pub created_at: u64,
}

/// Tracks stepping stone relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SteppingStone {
    /// Agent that served as stepping stone
    pub from_agent: String,
    /// Agent that benefited
    pub to_agent: String,
    /// Performance improvement
    pub improvement: f64,
    /// Generation gap
    pub generation_gap: u32,
}

/// Diversity metrics for archive analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityMetrics {
    /// Number of unique modification types
    pub modification_diversity: f64,
    /// Performance variance
    pub performance_variance: f64,
    /// Genome diversity score
    pub genome_diversity: f64,
    /// Number of unique lineages
    pub lineage_count: usize,
}
