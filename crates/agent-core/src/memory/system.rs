//! Agent memory system implementation

use super::entry::MemoryEntry;
use super::store::MemoryStore;
use super::types::{MemoryId, MemoryStats, MemoryType};
use crate::agent::AgentId;
use crate::error::{AgentError, AgentResult};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Agent memory system
pub struct AgentMemory {
    /// Agent ID
    agent_id: AgentId,
    /// Memory stores by type
    stores: HashMap<MemoryType, Arc<MemoryStore>>,
    /// Total memory limit
    memory_limit: usize,
    /// Statistics
    stats: Arc<RwLock<MemoryStats>>,
}

impl AgentMemory {
    /// Create new agent memory
    pub fn new(agent_id: AgentId, memory_limit: usize) -> AgentResult<Self> {
        let mut stores = HashMap::new();

        // Create stores for each memory type
        stores.insert(
            MemoryType::Working,
            Arc::new(MemoryStore::new(MemoryType::Working, 100)),
        );
        stores.insert(
            MemoryType::Episodic,
            Arc::new(MemoryStore::new(MemoryType::Episodic, 1000)),
        );
        stores.insert(
            MemoryType::Semantic,
            Arc::new(MemoryStore::new(MemoryType::Semantic, 10000)),
        );
        stores.insert(
            MemoryType::Procedural,
            Arc::new(MemoryStore::new(MemoryType::Procedural, 500)),
        );

        let stats = MemoryStats {
            entries_by_type: HashMap::new(),
            total_memory_bytes: 0,
            last_cleanup: Utc::now(),
            cleanup_count: 0,
            eviction_count: 0,
            total_entries: 0,
            memory_used: 0,
            total_stores: 0,
            total_retrieves: 0,
            hit_rate: 0.0,
        };

        Ok(Self {
            agent_id,
            stores,
            memory_limit,
            stats: Arc::new(RwLock::new(stats)),
        })
    }

    /// Get agent ID
    pub fn agent_id(&self) -> AgentId {
        self.agent_id
    }

    /// Get memory limit
    pub fn memory_limit(&self) -> usize {
        self.memory_limit
    }

    /// Initialize memory system
    pub async fn initialize(&self) -> AgentResult<()> {
        // Perform any initialization needed
        Ok(())
    }

    /// Store a memory entry
    pub async fn store(&self, entry: MemoryEntry) -> AgentResult<MemoryId> {
        let store = self
            .stores
            .get(&entry.memory_type)
            .ok_or_else(|| AgentError::MemoryError {
                message: format!("Unknown memory type: {:?}", entry.memory_type),
            })?;

        let id = entry.id;
        store.store(entry).await?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            *stats.entries_by_type.entry(id.into()).or_insert(0) += 1;
        }

        Ok(id)
    }

    /// Store a memory with type, key, and value
    pub async fn store_memory(
        &self,
        memory_type: MemoryType,
        key: String,
        value: serde_json::Value,
    ) -> AgentResult<MemoryId> {
        let entry = MemoryEntry::new(memory_type, key, value);
        self.store(entry).await
    }

    /// Retrieve a memory
    pub async fn retrieve(&self, memory_type: MemoryType, key: &str) -> Option<serde_json::Value> {
        let store = self.stores.get(&memory_type)?;
        store.retrieve(key).await
    }

    /// Update a memory
    pub async fn update(
        &self,
        memory_type: MemoryType,
        key: &str,
        value: serde_json::Value,
    ) -> AgentResult<()> {
        let store = self
            .stores
            .get(&memory_type)
            .ok_or_else(|| AgentError::MemoryError {
                message: format!("Unknown memory type: {:?}", memory_type),
            })?;

        store.update(key, value).await
    }

    /// Remove a memory
    pub async fn remove(&self, memory_type: MemoryType, key: &str) -> AgentResult<()> {
        let store = self
            .stores
            .get(&memory_type)
            .ok_or_else(|| AgentError::MemoryError {
                message: format!("Unknown memory type: {:?}", memory_type),
            })?;

        store.remove(key).await
    }

    /// Check if memory exists
    pub async fn contains(&self, memory_type: MemoryType, key: &str) -> bool {
        if let Some(store) = self.stores.get(&memory_type) {
            store.contains(key).await
        } else {
            false
        }
    }

    /// Search memories
    pub async fn search<F>(&self, memory_type: MemoryType, predicate: F) -> Vec<MemoryEntry>
    where
        F: Fn(&MemoryEntry) -> bool,
    {
        if let Some(store) = self.stores.get(&memory_type) {
            store.search(predicate).await
        } else {
            Vec::new()
        }
    }

    /// Get all keys for a memory type
    pub async fn keys(&self, memory_type: MemoryType) -> Vec<String> {
        if let Some(store) = self.stores.get(&memory_type) {
            store.keys().await
        } else {
            Vec::new()
        }
    }

    /// Get all entries for a memory type
    pub async fn all_entries(&self, memory_type: MemoryType) -> Vec<MemoryEntry> {
        if let Some(store) = self.stores.get(&memory_type) {
            store.all_entries().await
        } else {
            Vec::new()
        }
    }

    /// Store episodic memory
    pub async fn store_episode(
        &self,
        description: String,
        context: serde_json::Value,
    ) -> AgentResult<MemoryId> {
        let episode = serde_json::json!({
            "description": description,
            "context": context,
            "timestamp": Utc::now(),
        });

        self.store_memory(
            MemoryType::Episodic,
            format!("episode_{}", Uuid::new_v4()),
            episode,
        )
        .await
    }

    /// Store fact in semantic memory
    pub async fn store_fact(
        &self,
        subject: String,
        fact: serde_json::Value,
    ) -> AgentResult<MemoryId> {
        self.store_memory(MemoryType::Semantic, subject, fact).await
    }

    /// Store procedure
    pub async fn store_procedure(
        &self,
        name: String,
        procedure: serde_json::Value,
    ) -> AgentResult<MemoryId> {
        self.store_memory(MemoryType::Procedural, name, procedure)
            .await
    }

    /// Store working memory
    pub async fn store_working(
        &self,
        key: String,
        value: serde_json::Value,
    ) -> AgentResult<MemoryId> {
        self.store_memory(MemoryType::Working, key, value).await
    }

    /// Get entry count for a memory type
    pub async fn entry_count(&self, memory_type: MemoryType) -> usize {
        if let Some(store) = self.stores.get(&memory_type) {
            store.entry_count()
        } else {
            0
        }
    }

    /// Get total entry count across all types  
    pub async fn total_entries(&self) -> usize {
        let mut total = 0;
        for store in self.stores.values() {
            total += store.entry_count();
        }
        total
    }

    /// Get memory usage for a specific type
    pub async fn memory_usage(&self, memory_type: MemoryType) -> usize {
        if let Some(store) = self.stores.get(&memory_type) {
            store.memory_usage().await
        } else {
            0
        }
    }

    /// Get total memory usage across all types
    pub async fn total_memory_usage(&self) -> usize {
        let mut total = 0;
        for store in self.stores.values() {
            total += store.memory_usage().await;
        }
        total
    }

    /// Get memory statistics
    pub async fn stats(&self) -> MemoryStats {
        let mut stats = self.stats.write().await;

        // Update current counts
        stats.entries_by_type.clear();
        stats.total_memory_bytes = 0;

        for (memory_type, store) in &self.stores {
            stats
                .entries_by_type
                .insert(*memory_type, store.entry_count());
            stats.total_memory_bytes += store.memory_usage().await;
        }

        stats.clone()
    }

    /// Clean up expired memories
    pub async fn cleanup(&self) -> AgentResult<usize> {
        let mut total_removed = 0;

        for store in self.stores.values() {
            total_removed += store.cleanup().await;
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.last_cleanup = Utc::now();
            stats.cleanup_count += 1;
        }

        Ok(total_removed)
    }

    /// Clear all memories of a specific type
    pub async fn clear_type(&self, memory_type: MemoryType) -> AgentResult<()> {
        if let Some(store) = self.stores.get(&memory_type) {
            store.clear().await;
            Ok(())
        } else {
            Err(AgentError::MemoryError {
                message: format!("Unknown memory type: {:?}", memory_type),
            })
        }
    }

    /// Clear all memories
    pub async fn clear_all(&self) -> AgentResult<()> {
        for store in self.stores.values() {
            store.clear().await;
        }
        Ok(())
    }

    /// Update memory importance
    pub async fn update_importance(
        &self,
        memory_type: MemoryType,
        key: &str,
        importance: f32,
    ) -> AgentResult<()> {
        let store = self
            .stores
            .get(&memory_type)
            .ok_or_else(|| AgentError::MemoryError {
                message: format!("Unknown memory type: {:?}", memory_type),
            })?;

        store.update_importance(key, importance).await
    }

    /// Set memory TTL
    pub async fn set_ttl(
        &self,
        memory_type: MemoryType,
        key: &str,
        ttl: std::time::Duration,
    ) -> AgentResult<()> {
        let store = self
            .stores
            .get(&memory_type)
            .ok_or_else(|| AgentError::MemoryError {
                message: format!("Unknown memory type: {:?}", memory_type),
            })?;

        store.set_ttl(key, ttl).await
    }

    /// Get a specific entry
    pub async fn get_entry(&self, memory_type: MemoryType, key: &str) -> Option<MemoryEntry> {
        if let Some(store) = self.stores.get(&memory_type) {
            store.get_entry(key).await
        } else {
            None
        }
    }
}

// Implement From trait for MemoryId -> MemoryType conversion for stats
impl From<MemoryId> for MemoryType {
    fn from(_: MemoryId) -> Self {
        // This is a placeholder - in practice you'd need to track this mapping
        MemoryType::Working
    }
}
