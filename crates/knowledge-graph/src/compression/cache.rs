//! Compression cache for frequently accessed graphs

use super::CompressedKnowledgeGraph;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// LRU cache for compressed knowledge graphs
pub struct CompressionCache {
    capacity: usize,
    entries: HashMap<Uuid, CacheEntry>,
    access_order: VecDeque<Uuid>,
    stats: CacheStats,
}

/// Cache entry with metadata
struct CacheEntry {
    compressed: CompressedKnowledgeGraph,
    access_count: usize,
    last_accessed: std::time::Instant,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
}

impl CompressionCache {
    /// Create new cache with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            stats: CacheStats::default(),
        }
    }

    /// Get compressed graph from cache
    pub fn get(&mut self, id: &Uuid) -> Option<&CompressedKnowledgeGraph> {
        if let Some(entry) = self.entries.get_mut(id) {
            entry.access_count += 1;
            entry.last_accessed = std::time::Instant::now();
            self.stats.hits += 1;
            
            // Move to front of access order
            self.access_order.retain(|&x| x != *id);
            self.access_order.push_front(*id);
            
            Some(&entry.compressed)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert compressed graph into cache
    pub fn insert(&mut self, id: Uuid, compressed: CompressedKnowledgeGraph) {
        // Evict if at capacity
        if self.entries.len() >= self.capacity {
            self.evict_lru();
        }

        let entry = CacheEntry {
            compressed,
            access_count: 1,
            last_accessed: std::time::Instant::now(),
        };

        self.entries.insert(id, entry);
        self.access_order.push_front(id);
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    fn evict_lru(&mut self) {
        if let Some(id) = self.access_order.pop_back() {
            self.entries.remove(&id);
            self.stats.evictions += 1;
        }
    }
}

/// Thread-safe cache wrapper
pub struct ThreadSafeCache {
    inner: Arc<RwLock<CompressionCache>>,
}

impl ThreadSafeCache {
    /// Create new thread-safe cache
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CompressionCache::new(capacity))),
        }
    }

    /// Get from cache
    pub async fn get(&self, id: &Uuid) -> Option<CompressedKnowledgeGraph> {
        let mut cache = self.inner.write().await;
        cache.get(id).cloned()
    }

    /// Insert into cache
    pub async fn insert(&self, id: Uuid, compressed: CompressedKnowledgeGraph) {
        let mut cache = self.inner.write().await;
        cache.insert(id, compressed);
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let cache = self.inner.read().await;
        cache.stats().clone()
    }
}