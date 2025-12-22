//! Memory integration statistics

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Memory integration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryIntegrationStats {
    /// Total memories synced
    pub total_synced: usize,
    /// Memories by type
    pub by_type: HashMap<String, usize>,
    /// Last sync timestamp
    pub last_sync: Option<DateTime<Utc>>,
    /// Sync errors
    pub sync_errors: usize,
    /// Average sync time (ms)
    pub avg_sync_time_ms: f64,
}
