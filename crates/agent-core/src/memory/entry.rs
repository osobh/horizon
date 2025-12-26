//! Memory entry implementation

use super::types::{MemoryId, MemoryType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Entry ID
    pub id: MemoryId,
    /// Memory type
    pub memory_type: MemoryType,
    /// Entry key
    pub key: String,
    /// Entry value
    pub value: serde_json::Value,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Last access time
    pub last_accessed: DateTime<Utc>,
    /// Access count
    pub access_count: u64,
    /// Importance score (0.0 - 1.0)
    pub importance: f32,
    /// Time-to-live (None = permanent)
    pub ttl: Option<std::time::Duration>,
}

impl MemoryEntry {
    /// Create new memory entry
    pub fn new(memory_type: MemoryType, key: String, value: serde_json::Value) -> Self {
        let now = Utc::now();
        Self {
            id: MemoryId::new(),
            memory_type,
            key,
            value,
            metadata: HashMap::new(),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            importance: 0.5,
            ttl: None,
        }
    }

    /// Check if entry has expired
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            let elapsed = Utc::now() - self.created_at;
            elapsed.to_std().unwrap_or_default() > ttl
        } else {
            false
        }
    }

    /// Update access time and count
    pub fn record_access(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count += 1;
    }

    /// Update importance score
    pub fn set_importance(&mut self, importance: f32) {
        self.importance = importance.clamp(0.0, 1.0);
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: serde_json::Value) {
        self.metadata.insert(key, value);
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Set TTL
    pub fn set_ttl(&mut self, ttl: std::time::Duration) {
        self.ttl = Some(ttl);
    }

    /// Clear TTL (make permanent)
    pub fn clear_ttl(&mut self) {
        self.ttl = None;
    }

    /// Get age in seconds
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.created_at).num_seconds()
    }

    /// Get time since last access in seconds
    pub fn idle_seconds(&self) -> i64 {
        (Utc::now() - self.last_accessed).num_seconds()
    }

    /// Calculate memory score for eviction
    pub fn memory_score(&self) -> f32 {
        let age_factor = {
            let age_minutes = (Utc::now() - self.last_accessed).num_minutes() as f32;
            1.0 / (1.0 + age_minutes / 60.0) // Decay over hours
        };

        let access_factor = (self.access_count as f32).ln() / 10.0;
        let importance_factor = self.importance;

        // Combined score (higher = more important to keep)
        age_factor * 0.3 + access_factor * 0.3 + importance_factor * 0.4
    }
}
