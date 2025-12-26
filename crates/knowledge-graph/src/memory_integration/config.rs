//! Memory integration configuration types

use serde::{Deserialize, Serialize};

/// Memory integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryIntegrationConfig {
    /// Enable automatic memory synchronization
    pub auto_sync: bool,
    /// Sync interval in seconds
    pub sync_interval_seconds: u64,
    /// Maximum memory entries to sync at once
    pub max_sync_batch: usize,
    /// Enable semantic indexing of memories
    pub semantic_indexing: bool,
    /// Memory retention policy
    pub retention_policy: MemoryRetentionPolicy,
}

impl Default for MemoryIntegrationConfig {
    fn default() -> Self {
        Self {
            auto_sync: true,
            sync_interval_seconds: 300, // 5 minutes
            max_sync_batch: 100,
            semantic_indexing: true,
            retention_policy: MemoryRetentionPolicy::default(),
        }
    }
}

/// Memory retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRetentionPolicy {
    /// Working memory retention (hours)
    pub working_memory_hours: i64,
    /// Episodic memory retention (days)
    pub episodic_memory_days: i64,
    /// Semantic memory retention (permanent if None)
    pub semantic_memory_days: Option<i64>,
    /// Procedural memory retention (permanent if None)
    pub procedural_memory_days: Option<i64>,
}

impl Default for MemoryRetentionPolicy {
    fn default() -> Self {
        Self {
            working_memory_hours: 24,
            episodic_memory_days: 30,
            semantic_memory_days: None,   // Permanent
            procedural_memory_days: None, // Permanent
        }
    }
}
