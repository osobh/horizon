//! Memory store implementation for specific memory types

use super::entry::MemoryEntry;
use super::types::MemoryType;
use crate::error::{AgentError, AgentResult};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Memory store for a specific memory type
pub struct MemoryStore {
    /// Store type
    memory_type: MemoryType,
    /// Entries indexed by key
    entries: DashMap<String, Arc<RwLock<MemoryEntry>>>,
    /// Maximum entries
    max_entries: usize,
    /// Total memory used (bytes)
    memory_used: Arc<RwLock<usize>>,
}

impl MemoryStore {
    /// Create new memory store
    pub fn new(memory_type: MemoryType, max_entries: usize) -> Self {
        Self {
            memory_type,
            entries: DashMap::new(),
            max_entries,
            memory_used: Arc::new(RwLock::new(0)),
        }
    }

    /// Get memory type
    pub fn memory_type(&self) -> MemoryType {
        self.memory_type
    }

    /// Get current entry count
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Get current memory usage
    pub async fn memory_usage(&self) -> usize {
        *self.memory_used.read().await
    }

    /// Store an entry
    pub async fn store(&self, mut entry: MemoryEntry) -> AgentResult<()> {
        entry.memory_type = self.memory_type;

        // Check if we need to evict
        if self.entries.len() >= self.max_entries {
            self.evict_lowest_score().await?;
        }

        let key = entry.key.clone();
        let entry_size = Self::estimate_size(&entry.value);

        // Update memory usage
        {
            let mut used = self.memory_used.write().await;
            *used += entry_size;
        }

        self.entries.insert(key, Arc::new(RwLock::new(entry)));
        Ok(())
    }

    /// Retrieve an entry
    pub async fn retrieve(&self, key: &str) -> Option<serde_json::Value> {
        if let Some(entry) = self.entries.get(key) {
            let mut entry = entry.write().await;

            // Check expiration
            if entry.is_expired() {
                drop(entry);
                self.entries.remove(key);
                return None;
            }

            entry.record_access();
            Some(entry.value.clone())
        } else {
            None
        }
    }

    /// Update an existing entry
    pub async fn update(&self, key: &str, value: serde_json::Value) -> AgentResult<()> {
        if let Some(entry) = self.entries.get(key) {
            let mut entry = entry.write().await;

            if entry.is_expired() {
                return Err(AgentError::MemoryError {
                    message: "Entry has expired".to_string(),
                });
            }

            let old_size = Self::estimate_size(&entry.value);
            let new_size = Self::estimate_size(&value);

            entry.value = value;
            entry.record_access();

            // Update memory usage
            let mut used = self.memory_used.write().await;
            *used = used.saturating_sub(old_size) + new_size;

            Ok(())
        } else {
            Err(AgentError::MemoryError {
                message: format!("Entry not found: {}", key),
            })
        }
    }

    /// Remove an entry
    pub async fn remove(&self, key: &str) -> AgentResult<()> {
        if let Some((_, entry)) = self.entries.remove(key) {
            let entry = entry.read().await;
            let entry_size = Self::estimate_size(&entry.value);

            // Update memory usage
            let mut used = self.memory_used.write().await;
            *used = used.saturating_sub(entry_size);

            Ok(())
        } else {
            Err(AgentError::MemoryError {
                message: format!("Entry not found: {}", key),
            })
        }
    }

    /// Check if entry exists
    pub async fn contains(&self, key: &str) -> bool {
        if let Some(entry) = self.entries.get(key) {
            let entry = entry.read().await;
            !entry.is_expired()
        } else {
            false
        }
    }

    /// Search entries
    pub async fn search<F>(&self, predicate: F) -> Vec<MemoryEntry>
    where
        F: Fn(&MemoryEntry) -> bool,
    {
        let mut results = Vec::new();

        for entry in self.entries.iter() {
            let entry = entry.read().await;
            if !entry.is_expired() && predicate(&entry) {
                results.push(entry.clone());
            }
        }

        results
    }

    /// Get all keys
    pub async fn keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        for entry in self.entries.iter() {
            let entry = entry.read().await;
            if !entry.is_expired() {
                keys.push(entry.key.clone());
            }
        }
        keys
    }

    /// Get all entries
    pub async fn all_entries(&self) -> Vec<MemoryEntry> {
        let mut entries = Vec::new();
        for entry in self.entries.iter() {
            let entry = entry.read().await;
            if !entry.is_expired() {
                entries.push(entry.clone());
            }
        }
        entries
    }

    /// Clear all entries
    pub async fn clear(&self) {
        self.entries.clear();
        let mut used = self.memory_used.write().await;
        *used = 0;
    }

    /// Evict entry with lowest score
    async fn evict_lowest_score(&self) -> AgentResult<()> {
        let mut lowest_score = f32::MAX;
        let mut lowest_key = None;

        for entry in self.entries.iter() {
            let entry = entry.read().await;
            let score = entry.memory_score();

            if score < lowest_score {
                lowest_score = score;
                lowest_key = Some(entry.key.clone());
            }
        }

        if let Some(key) = lowest_key {
            self.remove(&key).await?;
        }

        Ok(())
    }

    /// Estimate size of a value in bytes
    fn estimate_size(value: &serde_json::Value) -> usize {
        serde_json::to_vec(value).map(|v| v.len()).unwrap_or(0)
    }

    /// Clean up expired entries
    pub async fn cleanup(&self) -> usize {
        let mut removed = 0;
        let mut to_remove = Vec::new();

        for entry in self.entries.iter() {
            let entry = entry.read().await;
            if entry.is_expired() {
                to_remove.push(entry.key.clone());
            }
        }

        for key in to_remove {
            if self.remove(&key).await.is_ok() {
                removed += 1;
            }
        }

        removed
    }

    /// Get entry by key (for internal use)
    pub async fn get_entry(&self, key: &str) -> Option<MemoryEntry> {
        if let Some(entry) = self.entries.get(key) {
            let entry = entry.read().await;
            if !entry.is_expired() {
                Some(entry.clone())
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Update entry importance
    pub async fn update_importance(&self, key: &str, importance: f32) -> AgentResult<()> {
        if let Some(entry) = self.entries.get(key) {
            let mut entry = entry.write().await;
            if entry.is_expired() {
                return Err(AgentError::MemoryError {
                    message: "Entry has expired".to_string(),
                });
            }
            entry.set_importance(importance);
            Ok(())
        } else {
            Err(AgentError::MemoryError {
                message: format!("Entry not found: {}", key),
            })
        }
    }

    /// Set entry TTL
    pub async fn set_ttl(&self, key: &str, ttl: std::time::Duration) -> AgentResult<()> {
        if let Some(entry) = self.entries.get(key) {
            let mut entry = entry.write().await;
            if entry.is_expired() {
                return Err(AgentError::MemoryError {
                    message: "Entry has expired".to_string(),
                });
            }
            entry.set_ttl(ttl);
            Ok(())
        } else {
            Err(AgentError::MemoryError {
                message: format!("Entry not found: {}", key),
            })
        }
    }
}
