//! Actor-based multi-tier caching system for cancel-safe async operations
//!
//! This module provides an actor-based implementation that eliminates
//! parking_lot RwLock usage in async context and provides cancel-safe operations.
//!
//! # Cancel Safety
//!
//! The actor model ensures cancel safety by:
//! 1. Actor owns all mutable state exclusively - no Arc<RwLock<...>>
//! 2. No parking_lot locks in async context
//! 3. All cache operations are message-based
//! 4. Background tasks (warming, stats) integrated into actor loop

use crate::cache_layer::{
    CacheConfig, CacheMetadata, CacheStats, CacheTier, CachedItem, WarmingStrategy,
};
use crate::error::{GlobalKnowledgeGraphError, GlobalKnowledgeGraphResult};
use lru::LruCache;
use serde::{de::DeserializeOwned, Serialize};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::interval;

/// Requests that can be sent to the cache actor
pub enum CacheRequest {
    /// Get item from cache
    Get {
        key: String,
        reply: oneshot::Sender<GlobalKnowledgeGraphResult<Option<Vec<u8>>>>,
    },
    /// Put item into cache
    Put {
        key: String,
        value: Vec<u8>,
        ttl: Option<Duration>,
        reply: oneshot::Sender<GlobalKnowledgeGraphResult<()>>,
    },
    /// Invalidate cache entry
    Invalidate {
        key: String,
        reply: oneshot::Sender<GlobalKnowledgeGraphResult<()>>,
    },
    /// Clear all caches
    Clear,
    /// Get cache statistics
    GetStats {
        region: String,
        reply: oneshot::Sender<Option<CacheStats>>,
    },
    /// Get cache metadata
    GetMetadata {
        limit: usize,
        reply: oneshot::Sender<Vec<CacheMetadata>>,
    },
    /// Warm cache with keys
    WarmCache {
        keys: Vec<String>,
        strategy: WarmingStrategy,
        reply: oneshot::Sender<GlobalKnowledgeGraphResult<()>>,
    },
    /// Optimize cache
    OptimizeCache,
    /// Graceful shutdown
    Shutdown,
}

/// Cache actor that owns all mutable state
///
/// This actor processes requests sequentially, eliminating the need for
/// parking_lot RwLock and providing inherent cancel safety.
pub struct CacheActor {
    /// Configuration
    config: CacheConfig,
    /// L1 cache - owned, not shared
    l1_cache: LruCache<String, CachedItem<Vec<u8>>>,
    /// L2 cache - owned, not shared
    l2_cache: LruCache<String, CachedItem<Vec<u8>>>,
    /// L3 cache - owned, not shared
    l3_cache: LruCache<String, CachedItem<Vec<u8>>>,
    /// Stats per region - owned, not shared
    stats: std::collections::HashMap<String, CacheStats>,
    /// Access frequency per key - owned, not shared
    access_frequency: std::collections::HashMap<String, f64>,
    /// Warming queue - owned, not shared
    warming_queue: Vec<String>,
    /// Request receiver
    inbox: mpsc::Receiver<CacheRequest>,
}

impl CacheActor {
    /// Create a new cache actor
    pub fn new(
        config: CacheConfig,
        inbox: mpsc::Receiver<CacheRequest>,
    ) -> GlobalKnowledgeGraphResult<Self> {
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
            config,
            l1_cache: LruCache::new(l1_size),
            l2_cache: LruCache::new(l2_size),
            l3_cache: LruCache::new(l3_size),
            stats: std::collections::HashMap::new(),
            access_frequency: std::collections::HashMap::new(),
            warming_queue: Vec::new(),
            inbox,
        })
    }

    /// Run the actor's message processing loop
    pub async fn run(mut self) {
        let mut warming_interval = interval(self.config.warming_interval);
        let mut stats_interval = interval(Duration::from_secs(10));

        loop {
            tokio::select! {
                // Process incoming requests - highest priority
                Some(request) = self.inbox.recv() => {
                    match request {
                        CacheRequest::Get { key, reply } => {
                            let result = self.get(&key);
                            let _ = reply.send(result);
                        }
                        CacheRequest::Put { key, value, ttl, reply } => {
                            let result = self.put(key, value, ttl);
                            let _ = reply.send(result);
                        }
                        CacheRequest::Invalidate { key, reply } => {
                            let result = self.invalidate(&key);
                            let _ = reply.send(result);
                        }
                        CacheRequest::Clear => {
                            self.clear();
                        }
                        CacheRequest::GetStats { region, reply } => {
                            let _ = reply.send(self.stats.get(&region).cloned());
                        }
                        CacheRequest::GetMetadata { limit, reply } => {
                            let _ = reply.send(self.get_metadata(limit));
                        }
                        CacheRequest::WarmCache { keys, strategy, reply } => {
                            let result = self.warm_cache(keys, strategy);
                            let _ = reply.send(result);
                        }
                        CacheRequest::OptimizeCache => {
                            self.optimize_cache();
                        }
                        CacheRequest::Shutdown => {
                            tracing::info!("CacheActor shutting down");
                            break;
                        }
                    }
                }
                // Cache warming timer
                _ = warming_interval.tick() => {
                    if self.config.enable_warming {
                        self.perform_cache_warming();
                    }
                }
                // Stats update timer
                _ = stats_interval.tick() => {
                    self.update_cache_stats();
                }
            }
        }
        tracing::debug!("CacheActor terminated gracefully");
    }

    /// Get item from cache (internal implementation)
    fn get(&mut self, key: &str) -> GlobalKnowledgeGraphResult<Option<Vec<u8>>> {
        let start = Instant::now();

        // Update access frequency
        self.update_access_frequency(key);

        // Get or create stats
        let stats = self.stats.entry("global".to_string()).or_default();

        // Check L1 cache
        if let Some(item) = self.l1_cache.get_mut(key) {
            item.access_count += 1;
            item.last_access = Instant::now();
            stats.l1_hits += 1;
            stats.cache_hits += 1;
            Self::update_stats_timing(stats, start);

            if item.expires_at > Instant::now() {
                return Ok(Some(item.value.clone()));
            }
        }

        // Check L2 cache
        if let Some(item) = self.l2_cache.get_mut(key) {
            item.access_count += 1;
            item.last_access = Instant::now();
            let item_clone = item.clone();
            stats.l2_hits += 1;
            stats.cache_hits += 1;
            Self::update_stats_timing(stats, start);

            if item_clone.expires_at > Instant::now() {
                // Promote to L1
                self.promote_item(key, item_clone.clone(), CacheTier::L1);
                return Ok(Some(item_clone.value));
            }
        }

        // Check L3 cache
        if let Some(item) = self.l3_cache.get_mut(key) {
            item.access_count += 1;
            item.last_access = Instant::now();
            let item_clone = item.clone();
            stats.l3_hits += 1;
            stats.cache_hits += 1;
            Self::update_stats_timing(stats, start);

            if item_clone.expires_at > Instant::now() {
                // Promote to L2
                self.promote_item(key, item_clone.clone(), CacheTier::L2);
                return Ok(Some(item_clone.value));
            }
        }

        // Cache miss
        let stats = self.stats.entry("global".to_string()).or_default();
        stats.cache_misses += 1;
        Self::update_stats_timing(stats, start);

        Ok(None)
    }

    /// Put item into cache (internal implementation)
    fn put(
        &mut self,
        key: String,
        value: Vec<u8>,
        ttl: Option<Duration>,
    ) -> GlobalKnowledgeGraphResult<()> {
        let ttl = ttl.unwrap_or(self.config.default_ttl);

        let item = CachedItem {
            value,
            expires_at: Instant::now() + ttl,
            access_count: 0,
            last_access: Instant::now(),
            tier: CacheTier::L1,
        };

        // Add to L1 cache
        self.put_to_tier(key.clone(), item, CacheTier::L1)?;

        // Add to warming queue if frequently accessed
        if self.get_access_frequency(&key) > 5.0 {
            self.warming_queue.push(key);
        }

        Ok(())
    }

    /// Invalidate cache entry (internal implementation)
    fn invalidate(&mut self, key: &str) -> GlobalKnowledgeGraphResult<()> {
        self.l1_cache.pop(key);
        self.l2_cache.pop(key);
        self.l3_cache.pop(key);
        self.access_frequency.remove(key);
        Ok(())
    }

    /// Clear all caches (internal implementation)
    fn clear(&mut self) {
        self.l1_cache.clear();
        self.l2_cache.clear();
        self.l3_cache.clear();
        self.access_frequency.clear();
        self.stats.clear();
    }

    /// Get cache metadata (internal implementation)
    fn get_metadata(&self, limit: usize) -> Vec<CacheMetadata> {
        let mut metadata = Vec::new();

        for (key, item) in self.l1_cache.iter().take(limit) {
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

    /// Warm cache with keys (internal implementation)
    fn warm_cache(
        &mut self,
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

                keys_with_freq.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                for (key, _) in keys_with_freq.into_iter().take(self.config.l1_size / 10) {
                    self.warming_queue.push(key);
                }
            }
            _ => {
                for key in keys.into_iter().take(self.config.l1_size / 10) {
                    self.warming_queue.push(key);
                }
            }
        }

        Ok(())
    }

    /// Optimize cache (internal implementation)
    fn optimize_cache(&self) {
        let stats = self.stats.get("global").cloned().unwrap_or_default();

        if stats.hit_rate < self.config.hit_rate_threshold {
            tracing::info!(
                "Cache hit rate {} below threshold {}, would optimize...",
                stats.hit_rate,
                self.config.hit_rate_threshold
            );
        }
    }

    /// Put item to specific tier (internal implementation)
    fn put_to_tier(
        &mut self,
        key: String,
        item: CachedItem<Vec<u8>>,
        tier: CacheTier,
    ) -> GlobalKnowledgeGraphResult<()> {
        match tier {
            CacheTier::L1 => {
                if let Some((evicted_key, evicted_item)) = self.l1_cache.push(key, item) {
                    // Demote evicted item to L2
                    let mut demoted = evicted_item;
                    demoted.tier = CacheTier::L2;
                    self.put_to_tier(evicted_key, demoted, CacheTier::L2)?;
                    self.update_eviction_stats();
                }
            }
            CacheTier::L2 => {
                if let Some((evicted_key, evicted_item)) = self.l2_cache.push(key, item) {
                    // Demote evicted item to L3
                    let mut demoted = evicted_item;
                    demoted.tier = CacheTier::L3;
                    self.put_to_tier(evicted_key, demoted, CacheTier::L3)?;
                    self.update_eviction_stats();
                }
            }
            CacheTier::L3 => {
                if self.l3_cache.push(key, item).is_some() {
                    self.update_eviction_stats();
                }
            }
        }
        Ok(())
    }

    /// Promote item to higher tier (internal implementation)
    fn promote_item(&mut self, key: &str, mut item: CachedItem<Vec<u8>>, target_tier: CacheTier) {
        item.access_count += 1;
        item.last_access = Instant::now();
        item.tier = target_tier;
        let _ = self.put_to_tier(key.to_string(), item, target_tier);
    }

    /// Update access frequency (internal implementation)
    fn update_access_frequency(&mut self, key: &str) {
        self.access_frequency
            .entry(key.to_string())
            .and_modify(|freq| *freq = (*freq * 0.9) + 1.0)
            .or_insert(1.0);
    }

    /// Get access frequency (internal implementation)
    fn get_access_frequency(&self, key: &str) -> f64 {
        self.access_frequency.get(key).copied().unwrap_or(0.0)
    }

    /// Update stats timing
    fn update_stats_timing(stats: &mut CacheStats, start: Instant) {
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
    fn update_eviction_stats(&mut self) {
        let stats = self.stats.entry("global".to_string()).or_default();
        stats.evictions += 1;
    }

    /// Perform cache warming (internal background task)
    fn perform_cache_warming(&mut self) {
        if !self.warming_queue.is_empty() {
            tracing::debug!("Warming cache with {} keys", self.warming_queue.len());
            self.warming_queue.clear();
        }
    }

    /// Update cache statistics (internal background task)
    fn update_cache_stats(&mut self) {
        for stats in self.stats.values_mut() {
            if stats.total_requests > 0 {
                stats.hit_rate = stats.cache_hits as f64 / stats.total_requests as f64;
            }
        }
    }
}

/// Handle for interacting with the cache actor
///
/// This handle is cheap to clone and can be used from multiple tasks.
/// All operations are cancel-safe.
#[derive(Clone)]
pub struct CacheHandle {
    sender: mpsc::Sender<CacheRequest>,
}

impl CacheHandle {
    /// Create a new handle from a sender
    pub fn new(sender: mpsc::Sender<CacheRequest>) -> Self {
        Self { sender }
    }

    /// Get item from cache
    ///
    /// This operation is cancel-safe.
    pub async fn get<T: DeserializeOwned>(
        &self,
        key: &str,
    ) -> GlobalKnowledgeGraphResult<Option<T>> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(CacheRequest::Get {
                key: key.to_string(),
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        let result = rx
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))??;

        match result {
            Some(data) => {
                let value: T = bincode::deserialize(&data).map_err(|e| {
                    GlobalKnowledgeGraphError::SerializationError {
                        context: "cache_deserialize".to_string(),
                        details: e.to_string(),
                    }
                })?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    /// Put item into cache
    pub async fn put<T: Serialize>(
        &self,
        key: String,
        value: T,
        ttl: Option<Duration>,
    ) -> GlobalKnowledgeGraphResult<()> {
        let serialized = bincode::serialize(&value).map_err(|e| {
            GlobalKnowledgeGraphError::SerializationError {
                context: "cache_serialize".to_string(),
                details: e.to_string(),
            }
        })?;

        let (tx, rx) = oneshot::channel();
        self.sender
            .send(CacheRequest::Put {
                key,
                value: serialized,
                ttl,
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))?
    }

    /// Invalidate cache entry
    pub async fn invalidate(&self, key: &str) -> GlobalKnowledgeGraphResult<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(CacheRequest::Invalidate {
                key: key.to_string(),
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))?
    }

    /// Clear all caches
    pub async fn clear(&self) -> GlobalKnowledgeGraphResult<()> {
        self.sender
            .send(CacheRequest::Clear)
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))
    }

    /// Get cache statistics
    pub async fn get_stats(&self, region: &str) -> GlobalKnowledgeGraphResult<Option<CacheStats>> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(CacheRequest::GetStats {
                region: region.to_string(),
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))
    }

    /// Get cache metadata
    pub async fn get_metadata(
        &self,
        limit: usize,
    ) -> GlobalKnowledgeGraphResult<Vec<CacheMetadata>> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(CacheRequest::GetMetadata { limit, reply: tx })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))
    }

    /// Warm cache with keys
    pub async fn warm_cache(
        &self,
        keys: Vec<String>,
        strategy: WarmingStrategy,
    ) -> GlobalKnowledgeGraphResult<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(CacheRequest::WarmCache {
                keys,
                strategy,
                reply: tx,
            })
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor dropped".to_string()))?
    }

    /// Optimize cache
    pub async fn optimize_cache(&self) -> GlobalKnowledgeGraphResult<()> {
        self.sender
            .send(CacheRequest::OptimizeCache)
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor stopped".to_string()))
    }

    /// Request graceful shutdown
    pub async fn shutdown(&self) -> GlobalKnowledgeGraphResult<()> {
        self.sender
            .send(CacheRequest::Shutdown)
            .await
            .map_err(|_| GlobalKnowledgeGraphError::Internal("Actor already stopped".to_string()))
    }
}

/// Create a cache actor and its handle
///
/// # Example
/// ```ignore
/// let (actor, handle) = create_cache_actor(config)?;
///
/// // Spawn the actor
/// tokio::spawn(actor.run());
///
/// // Use the handle
/// handle.put("key".to_string(), "value", None).await?;
/// let value: Option<String> = handle.get("key").await?;
///
/// // Shutdown
/// handle.shutdown().await?;
/// ```
pub fn create_cache_actor(
    config: CacheConfig,
) -> GlobalKnowledgeGraphResult<(CacheActor, CacheHandle)> {
    let (tx, rx) = mpsc::channel(256); // Larger buffer for cache operations

    let actor = CacheActor::new(config, rx)?;
    let handle = CacheHandle::new(tx);

    Ok((actor, handle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_actor_creation() {
        let config = CacheConfig::default();
        let result = create_cache_actor(config);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_actor_put_and_get() {
        let (actor, handle) = create_cache_actor(CacheConfig::default()).unwrap();
        tokio::spawn(actor.run());

        handle.put("key1".to_string(), "value1".to_string(), None).await.unwrap();
        let result: Option<String> = handle.get("key1").await.unwrap();
        assert_eq!(result, Some("value1".to_string()));

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_cache_miss() {
        let (actor, handle) = create_cache_actor(CacheConfig::default()).unwrap();
        tokio::spawn(actor.run());

        let result: Option<String> = handle.get("nonexistent").await.unwrap();
        assert!(result.is_none());

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_invalidate() {
        let (actor, handle) = create_cache_actor(CacheConfig::default()).unwrap();
        tokio::spawn(actor.run());

        handle.put("key1".to_string(), "value1".to_string(), None).await.unwrap();
        handle.invalidate("key1").await.unwrap();
        let result: Option<String> = handle.get("key1").await.unwrap();
        assert!(result.is_none());

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_actor_cancel_safety() {
        let (actor, handle) = create_cache_actor(CacheConfig::default()).unwrap();
        tokio::spawn(actor.run());

        // Start a request but cancel it
        let handle_clone = handle.clone();
        let future = handle_clone.get::<String>("key1");
        drop(future);

        // Give actor time to process
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Actor should still be responsive
        handle.put("key2".to_string(), "value2".to_string(), None).await.unwrap();
        let result: Option<String> = handle.get("key2").await.unwrap();
        assert_eq!(result, Some("value2".to_string()));

        handle.shutdown().await.unwrap();
    }
}
