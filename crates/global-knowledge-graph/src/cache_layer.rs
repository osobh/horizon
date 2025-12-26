//! Multi-tier caching system

use crate::error::{GlobalKnowledgeGraphError, GlobalKnowledgeGraphResult};
use async_trait::async_trait;
use dashmap::DashMap;
use lru::LruCache;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::interval;

/// Cache layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// L1 cache size (hot data)
    pub l1_size: usize,
    /// L2 cache size (warm data)
    pub l2_size: usize,
    /// L3 cache size (cold data)
    pub l3_size: usize,
    /// Default TTL for cached items
    pub default_ttl: Duration,
    /// Enable cache warming
    pub enable_warming: bool,
    /// Cache warming interval
    pub warming_interval: Duration,
    /// Enable regional synchronization
    pub enable_sync: bool,
    /// Sync interval
    pub sync_interval: Duration,
    /// Hit rate threshold for optimization
    pub hit_rate_threshold: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_size: 10_000,
            l2_size: 100_000,
            l3_size: 1_000_000,
            default_ttl: Duration::from_secs(300),
            enable_warming: true,
            warming_interval: Duration::from_secs(60),
            enable_sync: true,
            sync_interval: Duration::from_secs(30),
            hit_rate_threshold: 0.8,
        }
    }
}

/// Cache tier
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CacheTier {
    /// L1 - Hot data (fastest)
    L1,
    /// L2 - Warm data
    L2,
    /// L3 - Cold data
    L3,
}

/// Cached item
#[derive(Debug, Clone)]
pub struct CachedItem<T> {
    /// The cached value
    pub value: T,
    /// Expiration time
    pub expires_at: Instant,
    /// Access count
    pub access_count: u64,
    /// Last access time
    pub last_access: Instant,
    /// Cache tier
    pub tier: CacheTier,
}

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Key
    pub key: String,
    /// Size in bytes
    pub size_bytes: usize,
    /// Access frequency
    pub access_frequency: f64,
    /// TTL remaining
    pub ttl_remaining_secs: u64,
    /// Current tier
    pub tier: CacheTier,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total requests
    pub total_requests: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Hit rate
    pub hit_rate: f64,
    /// L1 hits
    pub l1_hits: u64,
    /// L2 hits
    pub l2_hits: u64,
    /// L3 hits
    pub l3_hits: u64,
    /// Evictions
    pub evictions: u64,
    /// Average latency
    pub avg_latency_us: f64,
}

/// Cache warming strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WarmingStrategy {
    /// Warm most frequently accessed
    MostFrequent,
    /// Warm most recently accessed
    MostRecent,
    /// Warm based on prediction
    Predictive,
    /// Custom warming function
    Custom,
}

/// Regional cache synchronizer trait
#[async_trait]
pub trait CacheSynchronizer: Send + Sync {
    /// Sync cache entry to region
    async fn sync_entry(
        &self,
        region: &str,
        key: &str,
        value: &[u8],
    ) -> GlobalKnowledgeGraphResult<()>;

    /// Get synced entry from region
    async fn get_synced_entry(
        &self,
        region: &str,
        key: &str,
    ) -> GlobalKnowledgeGraphResult<Option<Vec<u8>>>;

    /// Invalidate entry across regions
    async fn invalidate_entry(&self, key: &str) -> GlobalKnowledgeGraphResult<()>;
}

/// Multi-tier cache layer with LRU eviction
pub struct CacheLayer {
    config: Arc<CacheConfig>,
    l1_cache: Arc<RwLock<LruCache<String, CachedItem<Vec<u8>>>>>,
    l2_cache: Arc<RwLock<LruCache<String, CachedItem<Vec<u8>>>>>,
    l3_cache: Arc<RwLock<LruCache<String, CachedItem<Vec<u8>>>>>,
    stats: Arc<DashMap<String, CacheStats>>,
    access_frequency: Arc<DashMap<String, f64>>,
    warming_queue: Arc<RwLock<Vec<String>>>,
    synchronizer: Option<Arc<dyn CacheSynchronizer>>,
    shutdown_tx: mpsc::Sender<()>,
    shutdown_rx: Arc<RwLock<Option<mpsc::Receiver<()>>>>,
}

impl CacheLayer {
    /// Create new cache layer
    pub fn new(config: CacheConfig) -> GlobalKnowledgeGraphResult<Self> {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        let l1_size = NonZeroUsize::new(config.l1_size).ok_or_else(|| {
            GlobalKnowledgeGraphError::ConfigurationError {
                parameter: "l1_size".to_string(),
                reason: "Cache size must be greater than 0".to_string(),
            }
        })?;

        let l2_size = NonZeroUsize::new(config.l2_size).ok_or_else(|| {
            GlobalKnowledgeGraphError::ConfigurationError {
                parameter: "l2_size".to_string(),
                reason: "Cache size must be greater than 0".to_string(),
            }
        })?;

        let l3_size = NonZeroUsize::new(config.l3_size).ok_or_else(|| {
            GlobalKnowledgeGraphError::ConfigurationError {
                parameter: "l3_size".to_string(),
                reason: "Cache size must be greater than 0".to_string(),
            }
        })?;

        Ok(Self {
            config: Arc::new(config),
            l1_cache: Arc::new(RwLock::new(LruCache::new(l1_size))),
            l2_cache: Arc::new(RwLock::new(LruCache::new(l2_size))),
            l3_cache: Arc::new(RwLock::new(LruCache::new(l3_size))),
            stats: Arc::new(DashMap::new()),
            access_frequency: Arc::new(DashMap::new()),
            warming_queue: Arc::new(RwLock::new(Vec::new())),
            synchronizer: None,
            shutdown_tx,
            shutdown_rx: Arc::new(RwLock::new(Some(shutdown_rx))),
        })
    }

    /// Set cache synchronizer
    pub fn set_synchronizer(&mut self, synchronizer: Arc<dyn CacheSynchronizer>) {
        self.synchronizer = Some(synchronizer);
    }

    /// Start cache layer
    pub async fn start(&self) {
        let config = self.config.clone();
        let stats = self.stats.clone();
        let warming_queue = self.warming_queue.clone();

        // Take ownership of the receiver
        let mut shutdown_rx = self
            .shutdown_rx
            .write()
            .take()
            .expect("start called multiple times");

        tokio::spawn(async move {
            let mut warming_interval = interval(config.warming_interval);
            let mut stats_interval = interval(Duration::from_secs(10));

            loop {
                tokio::select! {
                    _ = warming_interval.tick() => {
                        if config.enable_warming {
                            Self::perform_cache_warming(&warming_queue).await;
                        }
                    }
                    _ = stats_interval.tick() => {
                        Self::update_cache_stats(&stats).await;
                    }
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                }
            }
        });
    }

    /// Stop cache layer
    pub async fn stop(&self) {
        let _ = self.shutdown_tx.send(()).await;
    }

    /// Get item from cache
    pub async fn get<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> GlobalKnowledgeGraphResult<Option<T>> {
        let start = Instant::now();
        let mut region_stats = self.get_or_create_stats("global");

        // Update access frequency
        self.update_access_frequency(key);

        // Check L1 cache
        if let Some(item) = self.get_from_tier(key, CacheTier::L1) {
            region_stats.l1_hits += 1;
            region_stats.cache_hits += 1;
            self.update_stats_timing(&mut region_stats, start);

            if item.expires_at > Instant::now() {
                return self.deserialize_item(&item.value);
            }
        }

        // Check L2 cache
        if let Some(item) = self.get_from_tier(key, CacheTier::L2) {
            region_stats.l2_hits += 1;
            region_stats.cache_hits += 1;
            self.update_stats_timing(&mut region_stats, start);

            if item.expires_at > Instant::now() {
                // Promote to L1
                self.promote_item(key, item.clone(), CacheTier::L1).await;
                return self.deserialize_item(&item.value);
            }
        }

        // Check L3 cache
        if let Some(item) = self.get_from_tier(key, CacheTier::L3) {
            region_stats.l3_hits += 1;
            region_stats.cache_hits += 1;
            self.update_stats_timing(&mut region_stats, start);

            if item.expires_at > Instant::now() {
                // Promote to L2
                self.promote_item(key, item.clone(), CacheTier::L2).await;
                return self.deserialize_item(&item.value);
            }
        }

        // Cache miss
        region_stats.cache_misses += 1;
        self.update_stats_timing(&mut region_stats, start);

        Ok(None)
    }

    /// Put item into cache
    pub async fn put<T: Serialize>(
        &self,
        key: String,
        value: T,
        ttl: Option<Duration>,
    ) -> GlobalKnowledgeGraphResult<()> {
        let serialized = self.serialize_item(&value)?;
        let ttl = ttl.unwrap_or(self.config.default_ttl);

        let item = CachedItem {
            value: serialized,
            expires_at: Instant::now() + ttl,
            access_count: 0,
            last_access: Instant::now(),
            tier: CacheTier::L1,
        };

        // Add to L1 cache
        let item_value = item.value.clone();
        self.put_to_tier(key.clone(), item, CacheTier::L1)?;

        // Sync to other regions if enabled
        if self.config.enable_sync {
            if let Some(ref synchronizer) = self.synchronizer {
                let _ = synchronizer.sync_entry("global", &key, &item_value).await;
            }
        }

        // Add to warming queue if frequently accessed
        if self.get_access_frequency(&key) > 5.0 {
            self.warming_queue.write().push(key);
        }

        Ok(())
    }

    /// Invalidate cache entry
    pub async fn invalidate(&self, key: &str) -> GlobalKnowledgeGraphResult<()> {
        // Remove from all tiers
        self.l1_cache.write().pop(key);
        self.l2_cache.write().pop(key);
        self.l3_cache.write().pop(key);

        // Remove access frequency
        self.access_frequency.remove(key);

        // Invalidate across regions if sync is enabled
        if self.config.enable_sync {
            if let Some(ref synchronizer) = self.synchronizer {
                synchronizer.invalidate_entry(key).await?;
            }
        }

        Ok(())
    }

    /// Clear all caches
    pub fn clear(&self) {
        self.l1_cache.write().clear();
        self.l2_cache.write().clear();
        self.l3_cache.write().clear();
        self.access_frequency.clear();
        self.stats.clear();
    }

    /// Get cache statistics
    pub fn get_stats(&self, region: &str) -> Option<CacheStats> {
        self.stats.get(region).map(|s| s.clone())
    }

    /// Get cache metadata
    pub fn get_metadata(&self, limit: usize) -> Vec<CacheMetadata> {
        let mut metadata = Vec::new();

        // Collect from L1
        let l1_cache = self.l1_cache.read();
        for (key, item) in l1_cache.iter().take(limit) {
            metadata.push(CacheMetadata {
                key: key.clone(),
                size_bytes: item.value.len(),
                access_frequency: self.get_access_frequency(key),
                ttl_remaining_secs: item
                    .expires_at
                    .saturating_duration_since(Instant::now())
                    .as_secs(),
                tier: CacheTier::L1,
            });
        }

        metadata
    }

    /// Warm cache with specific keys
    pub async fn warm_cache(
        &self,
        keys: Vec<String>,
        strategy: WarmingStrategy,
    ) -> GlobalKnowledgeGraphResult<()> {
        match strategy {
            WarmingStrategy::MostFrequent => {
                let mut keys_with_freq: Vec<(String, f64)> = keys
                    .into_iter()
                    .map(|k| {
                        let freq = self.get_access_frequency(&k);
                        (k, freq)
                    })
                    .collect();

                keys_with_freq.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                for (key, _) in keys_with_freq.into_iter().take(self.config.l1_size / 10) {
                    // In production, would fetch and cache the data
                    self.warming_queue.write().push(key);
                }
            }
            _ => {
                // Other strategies would have different implementations
                for key in keys.into_iter().take(self.config.l1_size / 10) {
                    self.warming_queue.write().push(key);
                }
            }
        }

        Ok(())
    }

    /// Optimize cache based on hit rate
    pub fn optimize_cache(&self) {
        let stats = self.get_stats("global").unwrap_or_default();

        if stats.hit_rate < self.config.hit_rate_threshold {
            tracing::info!(
                "Cache hit rate {} below threshold {}, optimizing...",
                stats.hit_rate,
                self.config.hit_rate_threshold
            );

            // In production, would implement cache optimization strategies
            // such as adjusting tier sizes, TTLs, or eviction policies
        }
    }

    /// Get item from specific tier
    fn get_from_tier(&self, key: &str, tier: CacheTier) -> Option<CachedItem<Vec<u8>>> {
        match tier {
            CacheTier::L1 => {
                let mut cache = self.l1_cache.write();
                cache.get_mut(key).map(|item| {
                    item.access_count += 1;
                    item.last_access = Instant::now();
                    item.clone()
                })
            }
            CacheTier::L2 => {
                let mut cache = self.l2_cache.write();
                cache.get_mut(key).map(|item| {
                    item.access_count += 1;
                    item.last_access = Instant::now();
                    item.clone()
                })
            }
            CacheTier::L3 => {
                let mut cache = self.l3_cache.write();
                cache.get_mut(key).map(|item| {
                    item.access_count += 1;
                    item.last_access = Instant::now();
                    item.clone()
                })
            }
        }
    }

    /// Put item to specific tier
    fn put_to_tier(
        &self,
        key: String,
        item: CachedItem<Vec<u8>>,
        tier: CacheTier,
    ) -> GlobalKnowledgeGraphResult<()> {
        match tier {
            CacheTier::L1 => {
                let mut cache = self.l1_cache.write();
                if let Some(evicted) = cache.push(key.clone(), item) {
                    // Demote evicted item to L2
                    drop(cache);
                    let mut demoted_item = evicted.1;
                    demoted_item.tier = CacheTier::L2;
                    self.put_to_tier(evicted.0, demoted_item, CacheTier::L2)?;
                    self.update_eviction_stats();
                }
            }
            CacheTier::L2 => {
                let mut cache = self.l2_cache.write();
                if let Some(evicted) = cache.push(key.clone(), item) {
                    // Demote evicted item to L3
                    drop(cache);
                    let mut demoted_item = evicted.1;
                    demoted_item.tier = CacheTier::L3;
                    self.put_to_tier(evicted.0, demoted_item, CacheTier::L3)?;
                    self.update_eviction_stats();
                }
            }
            CacheTier::L3 => {
                let mut cache = self.l3_cache.write();
                if cache.push(key, item).is_some() {
                    self.update_eviction_stats();
                }
            }
        }
        Ok(())
    }

    /// Promote item to higher tier
    async fn promote_item(&self, key: &str, mut item: CachedItem<Vec<u8>>, target_tier: CacheTier) {
        item.access_count += 1;
        item.last_access = Instant::now();
        item.tier = target_tier;

        let _ = self.put_to_tier(key.to_string(), item, target_tier);
    }

    /// Update access frequency
    fn update_access_frequency(&self, key: &str) {
        self.access_frequency
            .entry(key.to_string())
            .and_modify(|freq| *freq = (*freq * 0.9) + 1.0)
            .or_insert(1.0);
    }

    /// Get access frequency
    fn get_access_frequency(&self, key: &str) -> f64 {
        self.access_frequency.get(key).map(|f| *f).unwrap_or(0.0)
    }

    /// Serialize item
    fn serialize_item<T: Serialize>(&self, item: &T) -> GlobalKnowledgeGraphResult<Vec<u8>> {
        bincode::serialize(item).map_err(|e| GlobalKnowledgeGraphError::SerializationError {
            context: "cache_serialize".to_string(),
            details: e.to_string(),
        })
    }

    /// Deserialize item
    fn deserialize_item<T: serde::de::DeserializeOwned>(
        &self,
        data: &[u8],
    ) -> GlobalKnowledgeGraphResult<Option<T>> {
        bincode::deserialize(data).map(Some).map_err(|e| {
            GlobalKnowledgeGraphError::SerializationError {
                context: "cache_deserialize".to_string(),
                details: e.to_string(),
            }
        })
    }

    /// Get or create stats for region
    fn get_or_create_stats(
        &self,
        region: &str,
    ) -> dashmap::mapref::one::RefMut<String, CacheStats> {
        self.stats
            .entry(region.to_string())
            .or_insert_with(CacheStats::default)
    }

    /// Update stats timing
    fn update_stats_timing(&self, stats: &mut CacheStats, start: Instant) {
        stats.total_requests += 1;
        let latency = start.elapsed().as_micros() as f64;

        if stats.total_requests == 1 {
            stats.avg_latency_us = latency;
        } else {
            stats.avg_latency_us = (stats.avg_latency_us * (stats.total_requests - 1) as f64
                + latency)
                / stats.total_requests as f64;
        }

        stats.hit_rate = if stats.total_requests > 0 {
            stats.cache_hits as f64 / stats.total_requests as f64
        } else {
            0.0
        };
    }

    /// Update eviction stats
    fn update_eviction_stats(&self) {
        let mut stats = self.get_or_create_stats("global");
        stats.evictions += 1;
    }

    /// Perform cache warming
    async fn perform_cache_warming(warming_queue: &Arc<RwLock<Vec<String>>>) {
        let keys: Vec<String> = {
            let mut queue = warming_queue.write();
            let keys = queue.clone();
            queue.clear();
            keys
        };

        if !keys.is_empty() {
            tracing::debug!("Warming cache with {} keys", keys.len());
            // In production, would fetch and cache the data for these keys
        }
    }

    /// Update cache statistics
    async fn update_cache_stats(stats: &Arc<DashMap<String, CacheStats>>) {
        for mut entry in stats.iter_mut() {
            let stats = entry.value_mut();
            if stats.total_requests > 0 {
                stats.hit_rate = stats.cache_hits as f64 / stats.total_requests as f64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_layer_creation() {
        let config = CacheConfig::default();
        let cache = CacheLayer::new(config);
        assert!(cache.is_ok());
    }

    #[tokio::test]
    async fn test_cache_layer_invalid_config() {
        let config = CacheConfig {
            l1_size: 0, // Invalid
            ..Default::default()
        };
        let cache = CacheLayer::new(config);
        assert!(cache.is_err());
    }

    #[tokio::test]
    async fn test_put_and_get() {
        let cache = CacheLayer::new(CacheConfig::default()).unwrap();

        let key = "test-key".to_string();
        let value = "test-value".to_string();

        cache.put(key.clone(), value.clone(), None).await?;

        let retrieved: Option<String> = cache.get(&key).await?;
        assert_eq!(retrieved, Some(value));
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache = CacheLayer::new(CacheConfig::default()).unwrap();

        let result: Option<String> = cache.get("non-existent").await.unwrap();
        assert!(result.is_none());

        let stats = cache.get_stats("global")?;
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 0);
    }

    #[tokio::test]
    async fn test_ttl_expiration() {
        let cache = CacheLayer::new(CacheConfig::default()).unwrap();

        let key = "ttl-test".to_string();
        let value = "value".to_string();
        let ttl = Duration::from_millis(50);

        cache
            .put(key.clone(), value.clone(), Some(ttl))
            .await
            ?;

        // Should exist immediately
        let result: Option<String> = cache.get(&key).await.unwrap();
        assert_eq!(result, Some(value));

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should be expired
        let result: Option<String> = cache.get(&key).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_cache_invalidation() {
        let cache = CacheLayer::new(CacheConfig::default()).unwrap();

        let key = "invalidate-test".to_string();
        let value = "value".to_string();

        cache.put(key.clone(), value, None).await?;
        cache.invalidate(&key).await?;

        let result: Option<String> = cache.get(&key).await?;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let cache = CacheLayer::new(CacheConfig::default()).unwrap();

        // Add multiple items
        for i in 0..10 {
            cache
                .put(format!("key-{}", i), format!("value-{}", i), None)
                .await
                ?;
        }

        cache.clear();

        // All should be gone
        for i in 0..10 {
            let result: Option<String> = cache.get(&format!("key-{}", i)).await.unwrap();
            assert!(result.is_none());
        }
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = CacheLayer::new(CacheConfig::default()).unwrap();

        // Generate some cache activity
        cache.put("key1".to_string(), "value1", None).await.unwrap();
        let _: Option<String> = cache.get("key1").await?; // Hit
        let _: Option<String> = cache.get("key2").await?; // Miss

        let stats = cache.get_stats("global")?;
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[tokio::test]
    async fn test_tier_promotion() {
        let config = CacheConfig {
            l1_size: 2,
            l2_size: 2,
            l3_size: 2,
            ..Default::default()
        };
        let cache = CacheLayer::new(config)?;

        // Fill L1 cache to trigger demotion
        cache.put("key1".to_string(), "value1", None).await?;
        cache.put("key2".to_string(), "value2", None).await.unwrap();
        cache.put("key3".to_string(), "value3", None).await.unwrap(); // Should demote key1 to L2

        // Access key1 multiple times to trigger promotion
        let _: Option<String> = cache.get("key1").await.unwrap();
        let _: Option<String> = cache.get("key1").await.unwrap();

        let stats = cache.get_stats("global").unwrap();
        assert!(stats.l2_hits > 0); // Should have L2 hits
    }

    #[tokio::test]
    async fn test_access_frequency() {
        let cache = CacheLayer::new(CacheConfig::default()).unwrap();

        let key = "freq-test".to_string();
        cache.put(key.clone(), "value", None).await.unwrap();

        // Access multiple times
        for _ in 0..5 {
            let _: Option<String> = cache.get(&key).await?;
        }

        let freq = cache.get_access_frequency(&key);
        assert!(freq > 1.0);
    }

    #[tokio::test]
    async fn test_cache_metadata() {
        let cache = CacheLayer::new(CacheConfig::default()).unwrap();

        cache
            .put("meta-key".to_string(), "meta-value", None)
            .await
            ?;

        let metadata = cache.get_metadata(10);
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].key, "meta-key");
        assert_eq!(metadata[0].tier, CacheTier::L1);
    }

    #[tokio::test]
    async fn test_cache_warming() {
        let cache = CacheLayer::new(CacheConfig::default()).unwrap();

        let keys = vec![
            "warm1".to_string(),
            "warm2".to_string(),
            "warm3".to_string(),
        ];

        cache
            .warm_cache(keys, WarmingStrategy::MostFrequent)
            .await
            .unwrap();

        // Check warming queue was populated
        let queue = cache.warming_queue.read();
        assert!(queue.len() > 0);
    }

    #[tokio::test]
    async fn test_complex_data_types() {
        let cache = CacheLayer::new(CacheConfig::default()).unwrap();

        #[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
        struct ComplexData {
            id: String,
            values: Vec<i32>,
            metadata: HashMap<String, String>,
        }

        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "test".to_string());

        let data = ComplexData {
            id: "complex-1".to_string(),
            values: vec![1, 2, 3, 4, 5],
            metadata,
        };

        cache
            .put("complex-key".to_string(), data.clone(), None)
            .await
            .unwrap();

        let retrieved: Option<ComplexData> = cache.get("complex-key").await.unwrap();
        assert_eq!(retrieved, Some(data));
    }

    #[tokio::test]
    async fn test_eviction_stats() {
        let config = CacheConfig {
            l1_size: 2,
            ..Default::default()
        };
        let cache = CacheLayer::new(config)?;

        // Fill cache beyond capacity
        cache.put("key1".to_string(), "value1", None).await?;
        cache.put("key2".to_string(), "value2", None).await?;
        cache.put("key3".to_string(), "value3", None).await?; // Should trigger eviction

        let stats = cache.get_stats("global").unwrap();
        assert!(stats.evictions > 0);
    }

    #[tokio::test]
    async fn test_start_stop_cache() {
        let cache = CacheLayer::new(CacheConfig::default()).unwrap();

        cache.start().await;
        tokio::time::sleep(Duration::from_millis(100)).await;

        cache.stop().await;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_cache_optimization() {
        let cache = CacheLayer::new(CacheConfig::default()).unwrap();

        // Create low hit rate scenario
        for i in 0..100 {
            let _: Option<String> = cache.get(&format!("miss-{}", i)).await?;
        }

        cache.optimize_cache();
        // In production, would verify optimization effects
    }

    #[tokio::test]
    async fn test_tier_specific_hits() {
        let config = CacheConfig {
            l1_size: 1,
            l2_size: 1,
            l3_size: 1,
            ..Default::default()
        };
        let cache = CacheLayer::new(config)?;

        // Add items to fill tiers
        cache.put("key1".to_string(), "value1", None).await?;
        cache.put("key2".to_string(), "value2", None).await.unwrap(); // key1 -> L2
        cache.put("key3".to_string(), "value3", None).await.unwrap(); // key2 -> L2, key1 -> L3

        // Access from different tiers
        let _: Option<String> = cache.get("key3").await.unwrap(); // L1 hit
        let _: Option<String> = cache.get("key2").await.unwrap(); // L2 hit
        let _: Option<String> = cache.get("key1").await.unwrap(); // L3 hit

        let stats = cache.get_stats("global").unwrap();
        assert_eq!(stats.l1_hits, 1);
        assert_eq!(stats.l2_hits, 1);
        assert_eq!(stats.l3_hits, 1);
    }
}
