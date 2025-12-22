//! Memory system types and identifiers

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Memory entry ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(pub Uuid);

impl MemoryId {
    /// Create new memory ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// Working memory (short-term)
    Working,
    /// Episodic memory (experiences)
    Episodic,
    /// Semantic memory (facts and knowledge)
    Semantic,
    /// Procedural memory (skills and procedures)
    Procedural,
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total entries by type
    pub entries_by_type: std::collections::HashMap<MemoryType, usize>,
    /// Total memory usage in bytes
    pub total_memory_bytes: usize,
    /// Last cleanup time
    pub last_cleanup: chrono::DateTime<chrono::Utc>,
    /// Cleanup count
    pub cleanup_count: usize,
    /// Eviction count
    pub eviction_count: usize,
    /// Total entries across all stores
    pub total_entries: usize,
    /// Total memory used
    pub memory_used: usize,
    /// Total stores
    pub total_stores: u64,
    /// Total retrieves  
    pub total_retrieves: u64,
    /// Cache hit rate
    pub hit_rate: f32,
}
