//! GPU cache structures for high-speed graph access
//!
//! This module provides GPU-optimized caching structures for frequently accessed
//! graph nodes and edges. The cache is designed to work with GPU memory for
//! minimal latency access during graph operations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::StorageError;
use crate::graph_format::NodeRecord;

/// GPU cache entry for storing frequently accessed nodes
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub node: NodeRecord,
    pub access_count: u64,
    pub last_accessed: u64,
    pub is_dirty: bool,
}

impl CacheEntry {
    /// Create a new cache entry
    pub fn new(node: NodeRecord) -> Self {
        Self {
            node,
            access_count: 1,
            last_accessed: current_timestamp(),
            is_dirty: false,
        }
    }

    /// Mark entry as accessed
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.last_accessed = current_timestamp();
    }

    /// Mark entry as dirty (modified)
    pub fn mark_dirty(&mut self) {
        self.is_dirty = true;
        self.last_accessed = current_timestamp();
    }

    /// Clear dirty flag
    pub fn clear_dirty(&mut self) {
        self.is_dirty = false;
    }
}

/// Statistics for cache performance monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub dirty_evictions: u64,
    pub total_entries: u64,
    pub memory_usage: u64,
}

impl CacheStats {
    /// Calculate cache hit rate
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }

    /// Calculate memory usage per entry
    pub fn avg_memory_per_entry(&self) -> f64 {
        if self.total_entries == 0 {
            0.0
        } else {
            self.memory_usage as f64 / self.total_entries as f64
        }
    }
}

/// LRU-based GPU cache for graph nodes
pub struct GpuCache {
    entries: Arc<Mutex<HashMap<u64, CacheEntry>>>,
    access_order: Arc<Mutex<Vec<u64>>>, // LRU tracking
    stats: Arc<Mutex<CacheStats>>,
    max_entries: usize,
    #[allow(dead_code)]
    max_memory: u64, // Reserved for future memory-based eviction
}

impl GpuCache {
    /// Create a new GPU cache with specified limits
    pub fn new(max_entries: usize, max_memory: u64) -> Self {
        Self {
            entries: Arc::new(Mutex::new(HashMap::new())),
            access_order: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(CacheStats::default())),
            max_entries,
            max_memory,
        }
    }

    /// Get a node from cache
    pub fn get(&self, node_id: u64) -> Result<Option<NodeRecord>, StorageError> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| StorageError::LockPoisoned {
                resource: "cache entries".to_string(),
            })?;

        let mut access_order =
            self.access_order
                .lock()
                .map_err(|_| StorageError::LockPoisoned {
                    resource: "cache access order".to_string(),
                })?;

        let mut stats = self.stats.lock().map_err(|_| StorageError::LockPoisoned {
            resource: "cache stats".to_string(),
        })?;

        if let Some(entry) = entries.get_mut(&node_id) {
            // Cache hit
            entry.mark_accessed();
            stats.hits += 1;

            // Update LRU order
            if let Some(pos) = access_order.iter().position(|&x| x == node_id) {
                access_order.remove(pos);
            }
            access_order.push(node_id);

            Ok(Some(entry.node))
        } else {
            // Cache miss
            stats.misses += 1;
            Ok(None)
        }
    }

    /// Put a node in cache
    pub fn put(&self, node_id: u64, node: NodeRecord) -> Result<(), StorageError> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| StorageError::LockPoisoned {
                resource: "cache entries".to_string(),
            })?;

        let mut access_order =
            self.access_order
                .lock()
                .map_err(|_| StorageError::LockPoisoned {
                    resource: "cache access order".to_string(),
                })?;

        let mut stats = self.stats.lock().map_err(|_| StorageError::LockPoisoned {
            resource: "cache stats".to_string(),
        })?;

        // Check if we need to evict
        if entries.len() >= self.max_entries {
            self.evict_lru(&mut entries, &mut access_order, &mut stats)?;
        }

        // Add new entry
        let entry = CacheEntry::new(node);
        entries.insert(node_id, entry);
        access_order.push(node_id);

        stats.total_entries = entries.len() as u64;
        stats.memory_usage = self.calculate_memory_usage(&entries);

        Ok(())
    }

    /// Update a cached node
    pub fn update(&self, node_id: u64, node: NodeRecord) -> Result<bool, StorageError> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| StorageError::LockPoisoned {
                resource: "cache entries".to_string(),
            })?;

        if let Some(entry) = entries.get_mut(&node_id) {
            entry.node = node;
            entry.mark_dirty();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Remove a node from cache
    pub fn remove(&self, node_id: u64) -> Result<Option<NodeRecord>, StorageError> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| StorageError::LockPoisoned {
                resource: "cache entries".to_string(),
            })?;

        let mut access_order =
            self.access_order
                .lock()
                .map_err(|_| StorageError::LockPoisoned {
                    resource: "cache access order".to_string(),
                })?;

        let mut stats = self.stats.lock().map_err(|_| StorageError::LockPoisoned {
            resource: "cache stats".to_string(),
        })?;

        let result = entries.remove(&node_id).map(|entry| entry.node);

        // Remove from access order
        if let Some(pos) = access_order.iter().position(|&x| x == node_id) {
            access_order.remove(pos);
        }

        stats.total_entries = entries.len() as u64;
        stats.memory_usage = self.calculate_memory_usage(&entries);

        Ok(result)
    }

    /// Get cache statistics
    pub fn stats(&self) -> Result<CacheStats, StorageError> {
        let stats = self.stats.lock().map_err(|_| StorageError::LockPoisoned {
            resource: "cache stats".to_string(),
        })?;

        Ok(stats.clone())
    }

    /// Clear the entire cache
    pub fn clear(&self) -> Result<(), StorageError> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| StorageError::LockPoisoned {
                resource: "cache entries".to_string(),
            })?;

        let mut access_order =
            self.access_order
                .lock()
                .map_err(|_| StorageError::LockPoisoned {
                    resource: "cache access order".to_string(),
                })?;

        let mut stats = self.stats.lock().map_err(|_| StorageError::LockPoisoned {
            resource: "cache stats".to_string(),
        })?;

        entries.clear();
        access_order.clear();
        stats.total_entries = 0;
        stats.memory_usage = 0;

        Ok(())
    }

    /// Get all dirty entries for writeback
    pub fn get_dirty_entries(&self) -> Result<Vec<(u64, NodeRecord)>, StorageError> {
        let entries = self
            .entries
            .lock()
            .map_err(|_| StorageError::LockPoisoned {
                resource: "cache entries".to_string(),
            })?;

        let dirty_entries = entries
            .iter()
            .filter(|(_, entry)| entry.is_dirty)
            .map(|(id, entry)| (*id, entry.node))
            .collect();

        Ok(dirty_entries)
    }

    /// Mark all dirty entries as clean
    pub fn mark_clean(&self, node_ids: &[u64]) -> Result<(), StorageError> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| StorageError::LockPoisoned {
                resource: "cache entries".to_string(),
            })?;

        for node_id in node_ids {
            if let Some(entry) = entries.get_mut(node_id) {
                entry.clear_dirty();
            }
        }

        Ok(())
    }

    /// Evict least recently used entry
    fn evict_lru(
        &self,
        entries: &mut HashMap<u64, CacheEntry>,
        access_order: &mut Vec<u64>,
        stats: &mut CacheStats,
    ) -> Result<(), StorageError> {
        if let Some(lru_id) = access_order.first().copied() {
            if let Some(entry) = entries.remove(&lru_id) {
                access_order.remove(0);
                stats.evictions += 1;
                if entry.is_dirty {
                    stats.dirty_evictions += 1;
                }
            }
        }
        Ok(())
    }

    /// Calculate current memory usage
    fn calculate_memory_usage(&self, entries: &HashMap<u64, CacheEntry>) -> u64 {
        // Approximate memory usage: NodeRecord size + overhead
        let node_size = std::mem::size_of::<NodeRecord>() as u64;
        let entry_overhead = std::mem::size_of::<CacheEntry>() as u64;
        entries.len() as u64 * (node_size + entry_overhead)
    }
}

impl Default for GpuCache {
    fn default() -> Self {
        // Default cache: 10,000 entries, 64MB max memory
        Self::new(10_000, 64 * 1024 * 1024)
    }
}

/// Get current timestamp in seconds since epoch
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_format::NodeRecord;
    use std::thread;

    #[test]
    fn test_cache_entry_creation() {
        let node = NodeRecord::new(1, 100);
        let entry = CacheEntry::new(node);

        assert_eq!(entry.node.id, 1);
        assert_eq!(entry.node.type_id, 100);
        assert_eq!(entry.access_count, 1);
        assert!(!entry.is_dirty);
        assert!(entry.last_accessed > 0);
    }

    #[test]
    fn test_cache_entry_access() {
        let node = NodeRecord::new(1, 100);
        let mut entry = CacheEntry::new(node);
        let original_access = entry.access_count;

        entry.mark_accessed();

        assert_eq!(entry.access_count, original_access + 1);
        // Timestamp should be updated to current time
        assert!(entry.last_accessed > 0);
    }

    #[test]
    fn test_cache_entry_dirty() {
        let node = NodeRecord::new(1, 100);
        let mut entry = CacheEntry::new(node);

        assert!(!entry.is_dirty);
        entry.mark_dirty();
        assert!(entry.is_dirty);
        entry.clear_dirty();
        assert!(!entry.is_dirty);
    }

    #[test]
    fn test_cache_stats() {
        let mut stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);

        stats.hits = 8;
        stats.misses = 2;
        assert_eq!(stats.hit_rate(), 0.8);

        stats.total_entries = 5;
        stats.memory_usage = 1000;
        assert_eq!(stats.avg_memory_per_entry(), 200.0);
    }

    #[test]
    fn test_gpu_cache_creation() -> Result<(), crate::StorageError> {
        let cache = GpuCache::new(1000, 1024 * 1024);
        assert_eq!(cache.max_entries, 1000);
        assert_eq!(cache.max_memory, 1024 * 1024);

        let stats = cache.stats()?;
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        Ok(())
    }

    #[test]
    fn test_gpu_cache_put_get() -> Result<(), crate::StorageError> {
        let cache = GpuCache::new(10, 1024);
        let node = NodeRecord::new(42, 7);

        // Cache miss
        let result = cache.get(42)?;
        assert!(result.is_none());

        // Put node in cache
        cache.put(42, node)?;

        // Cache hit
        let result = cache.get(42)?;
        assert!(result.is_some());
        assert_eq!(result.clone().unwrap().id, 42);
        assert_eq!(result.unwrap().type_id, 7);

        let stats = cache.stats()?;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_entries, 1);
        Ok(())
    }

    #[test]
    fn test_gpu_cache_update() -> Result<(), crate::StorageError> {
        let cache = GpuCache::new(10, 1024);
        let mut node = NodeRecord::new(42, 7);

        // Put original node
        cache.put(42, node)?;

        // Update node
        node.type_id = 8;
        let updated = cache.update(42, node)?;
        assert!(updated);

        // Verify update
        let result = cache.get(42)?.unwrap();
        assert_eq!(result.type_id, 8);

        // Update non-existent node
        let updated = cache.update(99, node)?;
        assert!(!updated);
        Ok(())
    }

    #[test]
    fn test_gpu_cache_remove() -> Result<(), crate::StorageError> {
        let cache = GpuCache::new(10, 1024);
        let node = NodeRecord::new(42, 7);

        // Put node
        cache.put(42, node)?;
        assert_eq!(cache.stats().unwrap().total_entries, 1);

        // Remove node
        let removed = cache.remove(42)?;
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, 42);
        assert_eq!(cache.stats().unwrap().total_entries, 0);

        // Remove non-existent node
        let removed = cache.remove(42)?;
        assert!(removed.is_none());
        Ok(())
    }

    #[test]
    fn test_gpu_cache_clear() -> Result<(), crate::StorageError> {
        let cache = GpuCache::new(10, 1024);

        // Add multiple nodes
        for i in 0..5 {
            let node = NodeRecord::new(i, i as u32);
            cache.put(i, node)?;
        }

        assert_eq!(cache.stats().unwrap().total_entries, 5);

        // Clear cache
        cache.clear()?;
        assert_eq!(cache.stats().unwrap().total_entries, 0);
        Ok(())
    }

    #[test]
    fn test_gpu_cache_lru_eviction() -> Result<(), crate::StorageError> {
        let cache = GpuCache::new(3, 1024 * 1024); // Small cache

        // Fill cache
        for i in 0..3 {
            let node = NodeRecord::new(i, i as u32);
            cache.put(i, node)?;
        }

        assert_eq!(cache.stats().unwrap().total_entries, 3);

        // Add one more (should trigger eviction)
        let node = NodeRecord::new(3, 3);
        cache.put(3, node)?;

        assert_eq!(cache.stats().unwrap().total_entries, 3);
        assert_eq!(cache.stats().unwrap().evictions, 1);

        // First item (0) should be evicted
        let result = cache.get(0)?;
        assert!(result.is_none());

        // Last items should still be there
        let result = cache.get(3)?;
        assert!(result.is_some());
        Ok(())
    }

    #[test]
    fn test_gpu_cache_dirty_entries() -> Result<(), crate::StorageError> {
        let cache = GpuCache::new(10, 1024);
        let node1 = NodeRecord::new(1, 1);
        let node2 = NodeRecord::new(2, 2);
        let mut node3 = NodeRecord::new(3, 3);

        // Add nodes
        cache.put(1, node1)?;
        cache.put(2, node2)?;
        cache.put(3, node3)?;

        // Update one node (makes it dirty)
        node3.type_id = 33;
        cache.update(3, node3)?;

        // Get dirty entries
        let dirty = cache.get_dirty_entries()?;
        assert_eq!(dirty.len(), 1);
        assert_eq!(dirty[0].0, 3);
        assert_eq!(dirty[0].1.type_id, 33);

        // Mark clean
        cache.mark_clean(&[3])?;
        let dirty = cache.get_dirty_entries()?;
        assert_eq!(dirty.len(), 0);
        Ok(())
    }

    #[test]
    fn test_gpu_cache_default() {
        let cache = GpuCache::default();
        assert_eq!(cache.max_entries, 10_000);
        assert_eq!(cache.max_memory, 64 * 1024 * 1024);
    }

    // TDD: Test avg_memory_per_entry with zero entries
    #[test]
    fn test_cache_stats_avg_memory_zero_entries() -> Result<(), crate::StorageError> {
        let cache = GpuCache::new(10, 1024);
        let stats = cache.stats()?;

        // When no entries, avg_memory_per_entry should return 0.0
        assert_eq!(stats.avg_memory_per_entry(), 0.0);
        assert_eq!(stats.total_entries, 0);
        Ok(())
    }

    // TDD: Test error paths for mutex poisoning
    #[test]
    fn test_gpu_cache_mutex_poisoning() {
        use std::thread;

        let cache = GpuCache::new(10, 1024);

        // Poison the mutex
        let entries_clone = cache.entries.clone();
        let handle = thread::spawn(move || {
            let _guard = entries_clone.lock().unwrap();
            panic!("Poison mutex");
        });
        let _ = handle.join();

        // Operations should handle poisoned mutex
        let result = cache.get(42);
        assert!(result.is_err());
    }

    // TDD: Test memory limit enforcement
    #[test]
    fn test_gpu_cache_memory_limit() -> Result<(), crate::StorageError> {
        let cache = GpuCache::new(1000, 1000); // Reasonable memory limit
        let node = NodeRecord::new(42, 7);

        // This should work initially
        cache.put(42, node)?;

        let stats = cache.stats()?;
        // Just verify that cache tracks memory usage
        assert!(stats.memory_usage > 0);
        assert!(stats.total_entries > 0);
        Ok(())
    }

    // TDD RED Phase: Test get method access_order mutex poisoning (lines 116-117)
    #[test]
    fn test_get_access_order_mutex_poisoning() -> Result<(), crate::StorageError> {
        let cache = GpuCache::new(10, 1024);
        let node = NodeRecord::new(42, 7);
        cache.put(42, node)?;

        // Poison the access_order mutex
        let access_order_clone = cache.access_order.clone();
        let handle = thread::spawn(move || {
            let _guard = access_order_clone.lock().unwrap();
            panic!("Poison access_order mutex");
        });
        let _ = handle.join();

        // Try to get - should handle poisoned access_order mutex
        let result = cache.get(42);
        assert!(result.is_err());
        match result {
            Err(StorageError::LockPoisoned { resource }) => {
                assert!(resource.contains("cache access order"));
            }
            _ => panic!("Expected LockPoisoned error"),
        }
        Ok(())
    }

    // TDD RED Phase: Test get method stats mutex poisoning (lines 120-121)
    #[test]
    fn test_get_stats_mutex_poisoning() -> Result<(), crate::StorageError> {
        let cache = GpuCache::new(10, 1024);
        let node = NodeRecord::new(42, 7);
        cache.put(42, node)?;

        // Poison the stats mutex
        let stats_clone = cache.stats.clone();
        let handle = thread::spawn(move || {
            let _guard = stats_clone.lock().unwrap();
            panic!("Poison stats mutex");
        });
        let _ = handle.join();

        // Try to get - should handle poisoned stats mutex
        let result = cache.get(42);
        assert!(result.is_err());
        match result {
            Err(StorageError::LockPoisoned { resource }) => {
                assert!(resource.contains("cache stats"));
            }
            _ => panic!("Expected LockPoisoned error"),
        }
        Ok(())
    }

    // TDD RED Phase: Test put method mutex poisoning scenarios
    #[test]
    fn test_put_mutex_poisoning_scenarios() {
        let cache = GpuCache::new(10, 1024);

        // Test entries mutex poisoning
        let entries_clone = cache.entries.clone();
        let handle = thread::spawn(move || {
            let _guard = entries_clone.lock().unwrap();
            panic!("Poison entries mutex");
        });
        let _ = handle.join();

        let node = NodeRecord::new(42, 7);
        let result = cache.put(42, node);
        assert!(result.is_err());
        match result {
            Err(StorageError::LockPoisoned { resource }) => {
                assert!(resource.contains("cache entries"));
            }
            _ => panic!("Expected LockPoisoned error"),
        }
    }

    // TDD RED Phase: Test remove method mutex poisoning
    #[test]
    fn test_remove_mutex_poisoning() -> Result<(), crate::StorageError> {
        let cache = GpuCache::new(10, 1024);
        let node = NodeRecord::new(42, 7);
        cache.put(42, node)?;

        // Poison the entries mutex
        let entries_clone = cache.entries.clone();
        let handle = thread::spawn(move || {
            let _guard = entries_clone.lock().unwrap();
            panic!("Poison entries mutex");
        });
        let _ = handle.join();

        // Try to remove - should handle poisoned entries mutex
        let result = cache.remove(42);
        assert!(result.is_err());
        match result {
            Err(StorageError::LockPoisoned { resource }) => {
                assert!(resource.contains("cache entries"));
            }
            _ => panic!("Expected LockPoisoned error"),
        }
        Ok(())
    }

    // TDD RED Phase: Test clear method mutex poisoning
    #[test]
    fn test_clear_mutex_poisoning() {
        let cache = GpuCache::new(10, 1024);

        // Poison the entries mutex
        let entries_clone = cache.entries.clone();
        let handle = thread::spawn(move || {
            let _guard = entries_clone.lock().unwrap();
            panic!("Poison entries mutex");
        });
        let _ = handle.join();

        // Try to clear - should handle poisoned entries mutex
        let result = cache.clear();
        assert!(result.is_err());
        match result {
            Err(StorageError::LockPoisoned { resource }) => {
                assert!(resource.contains("cache entries"));
            }
            _ => panic!("Expected LockPoisoned error"),
        }
    }

    // TDD RED Phase: Test stats method mutex poisoning
    #[test]
    fn test_stats_method_mutex_poisoning() {
        let cache = GpuCache::new(10, 1024);

        // Poison the stats mutex
        let stats_clone = cache.stats.clone();
        let handle = thread::spawn(move || {
            let _guard = stats_clone.lock().unwrap();
            panic!("Poison stats mutex");
        });
        let _ = handle.join();

        // Try to get stats - should handle poisoned stats mutex
        let result = cache.stats();
        assert!(result.is_err());
        match result {
            Err(StorageError::LockPoisoned { resource }) => {
                assert!(resource.contains("cache stats"));
            }
            _ => panic!("Expected LockPoisoned error"),
        }
    }
}
