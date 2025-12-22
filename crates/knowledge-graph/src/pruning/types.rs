//! Type definitions for the pruning system
//!
//! This module contains all the types, enums, and configuration structures
//! used by the knowledge graph pruning system.

use crate::graph::NodeType;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pruning strategy types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PruningStrategy {
    /// Time-based pruning (remove old nodes/edges)
    TimeBased {
        /// Maximum age before pruning
        max_age_hours: i64,
        /// Node types to prune
        node_types: Vec<NodeType>,
    },
    /// Size-based pruning (maintain maximum graph size)
    SizeBased {
        /// Maximum number of nodes
        max_nodes: usize,
        /// Maximum number of edges
        max_edges: usize,
        /// Removal priority (oldest first, lowest degree first, etc.)
        priority: RemovalPriority,
    },
    /// Importance-based pruning (remove low-importance entities)
    ImportanceBased {
        /// Minimum importance score to keep
        min_importance: f64,
        /// Importance calculation method
        method: ImportanceMethod,
    },
    /// Usage-based pruning (remove rarely accessed entities)
    UsageBased {
        /// Minimum access count to keep
        min_access_count: u32,
        /// Time window for access tracking (hours)
        time_window_hours: i64,
    },
    /// Redundancy-based pruning (remove duplicate/similar entities)
    RedundancyBased {
        /// Similarity threshold for considering duplicates
        similarity_threshold: f64,
        /// Maximum similar entities to keep
        max_similar: usize,
    },
}

/// Node/edge removal priority
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RemovalPriority {
    /// Remove oldest first
    OldestFirst,
    /// Remove lowest degree first
    LowestDegreeFirst,
    /// Remove lowest importance first
    LowestImportanceFirst,
    /// Remove least accessed first
    LeastAccessedFirst,
}

/// Importance calculation methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImportanceMethod {
    /// Based on node degree (centrality)
    DegreeCentrality,
    /// Based on PageRank score
    PageRank,
    /// Based on betweenness centrality
    BetweennessCentrality,
    /// Based on access frequency
    AccessFrequency,
    /// Composite score combining multiple factors
    Composite,
}

/// Pruning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Enabled pruning strategies
    pub strategies: Vec<PruningStrategy>,
    /// Pruning interval (hours)
    pub pruning_interval_hours: i64,
    /// Enable automatic pruning
    pub auto_pruning: bool,
    /// Backup removed entities
    pub backup_removed: bool,
    /// Maximum backup size
    pub max_backup_size: usize,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            strategies: vec![
                PruningStrategy::TimeBased {
                    max_age_hours: 24 * 7, // 1 week
                    node_types: vec![NodeType::Memory],
                },
                PruningStrategy::SizeBased {
                    max_nodes: 100_000,
                    max_edges: 1_000_000,
                    priority: RemovalPriority::OldestFirst,
                },
            ],
            pruning_interval_hours: 24,
            auto_pruning: true,
            backup_removed: true,
            max_backup_size: 10_000,
        }
    }
}

/// Pruning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningStats {
    /// Total pruning operations
    pub total_operations: u64,
    /// Total nodes pruned
    pub nodes_pruned: usize,
    /// Total edges pruned
    pub edges_pruned: usize,
    /// Pruned nodes by type
    pub pruned_by_type: HashMap<NodeType, usize>,
    /// Last pruning timestamp
    pub last_pruning: Option<DateTime<Utc>>,
    /// Average pruning duration (ms)
    pub avg_duration_ms: f64,
    /// Space saved (estimated bytes)
    pub space_saved_bytes: u64,
}

/// Entity access tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EntityAccess {
    /// Entity ID
    pub entity_id: String,
    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,
    /// Total access count
    pub access_count: u32,
    /// Recent access timestamps
    pub recent_accesses: Vec<DateTime<Utc>>,
}

/// Removed entity backup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RemovedEntity {
    /// Entity data
    pub entity: serde_json::Value,
    /// Entity type
    pub entity_type: String,
    /// Removal timestamp
    pub removed_at: DateTime<Utc>,
    /// Removal reason
    pub reason: String,
}
