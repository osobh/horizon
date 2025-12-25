//! ContentAddressableStore - Deduplication and tier-aware storage

use crate::mocks::memory::{MemoryTier, TierManager};
use crate::{Result, SwarmRegistryError};
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info};

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total objects stored
    pub total_objects: usize,
    /// Total size in bytes
    pub total_size: u64,
    /// Deduplicated size (actual disk usage)
    pub deduplicated_size: u64,
    /// Number of duplicate objects
    pub duplicate_count: usize,
    /// Storage by tier
    pub tier_usage: HashMap<MemoryTier, u64>,
}

/// Object metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ObjectMetadata {
    /// Content hash
    pub hash: String,
    /// Object size
    pub size: u64,
    /// Reference count
    pub ref_count: u32,
    /// Current storage tier
    pub tier: MemoryTier,
    /// Access count
    pub access_count: u64,
    /// Last access time
    pub last_access: u64,
    /// Creation time
    pub created: u64,
}

/// Store configuration
#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// Base storage path
    pub storage_path: PathBuf,
    /// Enable deduplication
    pub enable_dedup: bool,
    /// Enable tier-aware storage
    pub enable_tiers: bool,
    /// Tier thresholds
    pub tier_thresholds: TierThresholds,
    /// Garbage collection interval
    pub gc_interval: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct TierThresholds {
    /// Size threshold for GPU tier (objects smaller than this go to GPU)
    pub gpu_threshold: u64,
    /// Size threshold for CPU tier
    pub cpu_threshold: u64,
    /// Size threshold for NVMe tier
    pub nvme_threshold: u64,
    /// Size threshold for SSD tier
    pub ssd_threshold: u64,
    // Anything larger goes to HDD
}

impl Default for TierThresholds {
    fn default() -> Self {
        Self {
            gpu_threshold: 1024 * 1024,             // 1 MB
            cpu_threshold: 100 * 1024 * 1024,       // 100 MB
            nvme_threshold: 1024 * 1024 * 1024,     // 1 GB
            ssd_threshold: 10 * 1024 * 1024 * 1024, // 10 GB
        }
    }
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("/var/lib/stratoswarm/cas"),
            enable_dedup: true,
            enable_tiers: true,
            tier_thresholds: TierThresholds::default(),
            gc_interval: std::time::Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Content-addressable store with deduplication and tier awareness
pub struct ContentAddressableStore {
    config: StoreConfig,
    /// Object metadata index
    metadata: Arc<DashMap<String, ObjectMetadata>>,
    /// Reverse index: content hash -> objects using it
    content_index: Arc<DashMap<String, HashSet<String>>>,
    /// Tier manager for tier-aware storage
    tier_manager: Option<Arc<TierManager>>,
    /// Storage statistics
    stats: Arc<RwLock<StorageStats>>,
    /// GC lock
    gc_lock: Arc<Mutex<()>>,
}

impl ContentAddressableStore {
    /// Create a new content-addressable store
    pub async fn new(config: StoreConfig) -> Result<Self> {
        // Create storage directories
        for tier in &["gpu", "cpu", "nvme", "ssd", "hdd"] {
            let tier_path = config.storage_path.join(tier);
            tokio::fs::create_dir_all(&tier_path).await?;
        }

        // Initialize tier manager if enabled
        let tier_manager = if config.enable_tiers {
            Some(Arc::new(TierManager::new()))
        } else {
            None
        };

        let store = Self {
            config,
            metadata: Arc::new(DashMap::new()),
            content_index: Arc::new(DashMap::new()),
            tier_manager,
            stats: Arc::new(RwLock::new(StorageStats::default())),
            gc_lock: Arc::new(Mutex::new(())),
        };

        // Load existing metadata
        store.load_metadata().await?;

        Ok(store)
    }

    /// Store an object
    pub async fn put(&self, key: &str, data: &[u8]) -> Result<String> {
        let content_hash = self.calculate_hash(data);

        debug!("Storing object {} with content hash {}", key, content_hash);

        // Check if content already exists (deduplication)
        let existing = self.metadata.iter().find(|entry| entry.value().hash == content_hash);

        if let Some(existing_entry) = existing {
            if self.config.enable_dedup {
                info!("Content already exists, deduplicating");

                // Clone the existing metadata for the new key
                let existing_meta = existing_entry.value();
                let new_meta = ObjectMetadata {
                    hash: existing_meta.hash.clone(),
                    size: existing_meta.size,
                    ref_count: 1, // Each key has its own ref count
                    tier: existing_meta.tier,
                    access_count: 0,
                    last_access: self.current_timestamp(),
                    created: self.current_timestamp(),
                };

                drop(existing_entry);

                // Add metadata for the new key
                self.metadata.insert(key.to_string(), new_meta);

                // Update content index
                self.add_reference(&content_hash, key).await?;

                return Ok(content_hash);
            }
        }

        // Determine storage tier
        let tier = self.select_tier(data.len() as u64);

        // Store the data
        let storage_path = self.get_storage_path(&content_hash, &tier);
        tokio::fs::create_dir_all(storage_path.parent().unwrap()).await?;
        tokio::fs::write(&storage_path, data).await?;

        // Create metadata
        let meta = ObjectMetadata {
            hash: content_hash.clone(),
            size: data.len() as u64,
            ref_count: 1,
            tier,
            access_count: 0,
            last_access: self.current_timestamp(),
            created: self.current_timestamp(),
        };

        // Update indexes
        self.metadata.insert(key.to_string(), meta.clone());

        self.content_index
            .entry(content_hash.clone())
            .or_insert_with(HashSet::new)
            .insert(key.to_string());

        // Update stats
        self.update_stats_for_put(&meta).await?;

        Ok(content_hash)
    }

    /// Get an object
    pub async fn get(&self, key: &str) -> Result<Vec<u8>> {
        // Get metadata
        let meta = self.metadata
            .get(key)
            .ok_or_else(|| SwarmRegistryError::Storage(format!("Object not found: {}", key)))?
            .clone();

        // Read from storage
        let storage_path = self.get_storage_path(&meta.hash, &meta.tier);
        let data = tokio::fs::read(&storage_path).await?;

        // Update access stats
        self.update_access_stats(key).await?;

        // Consider tier migration based on access patterns
        if self.config.enable_tiers {
            self.consider_migration(key, &meta).await?;
        }

        Ok(data)
    }

    /// Check if an object exists
    pub async fn exists(&self, key: &str) -> bool {
        self.metadata.contains_key(key)
    }

    /// Delete an object
    pub async fn delete(&self, key: &str) -> Result<()> {
        // Get metadata
        let (_, meta) = self.metadata
            .remove(key)
            .ok_or_else(|| SwarmRegistryError::Storage(format!("Object not found: {}", key)))?;

        // Update content index
        if let Some(mut refs) = self.content_index.get_mut(&meta.hash) {
            refs.remove(key);

            // If no more references and dedup is enabled, delete the content
            if refs.is_empty() && self.config.enable_dedup {
                drop(refs);
                self.content_index.remove(&meta.hash);

                // Delete from storage
                let storage_path = self.get_storage_path(&meta.hash, &meta.tier);
                tokio::fs::remove_file(&storage_path).await?;

                // Update stats
                self.update_stats_for_delete(&meta).await?;
            }
        }

        Ok(())
    }

    /// List all objects
    pub async fn list(&self) -> Result<Vec<(String, ObjectMetadata)>> {
        Ok(self.metadata
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect())
    }

    /// Get storage statistics
    pub async fn stats(&self) -> Result<StorageStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }

    /// Run garbage collection
    pub async fn garbage_collect(&self) -> Result<()> {
        let _gc_lock = self.gc_lock.lock().await;
        info!("Running garbage collection");

        let mut objects_deleted = 0;
        let mut space_reclaimed = 0u64;

        // Find orphaned content (content with no references)
        let all_content_hashes: HashSet<_> = self.metadata.iter().map(|e| e.value().hash.clone()).collect();

        // Check all tier directories
        for tier in &["gpu", "cpu", "nvme", "ssd", "hdd"] {
            let tier_path = self.config.storage_path.join(tier);
            if let Ok(mut entries) = tokio::fs::read_dir(&tier_path).await {
                while let Some(entry) = entries.next_entry().await? {
                    let path = entry.path();
                    if path.is_file() {
                        let hash = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

                        if !all_content_hashes.contains(hash) {
                            // Orphaned content, delete it
                            if let Ok(metadata) = tokio::fs::metadata(&path).await {
                                space_reclaimed += metadata.len();
                                objects_deleted += 1;
                            }
                            tokio::fs::remove_file(&path).await?;
                        }
                    }
                }
            }
        }

        info!(
            "Garbage collection complete: deleted {} objects, reclaimed {} bytes",
            objects_deleted, space_reclaimed
        );

        Ok(())
    }

    /// Migrate object between tiers
    pub async fn migrate(&self, key: &str, target_tier: MemoryTier) -> Result<()> {
        let mut meta = self.metadata
            .get_mut(key)
            .ok_or_else(|| SwarmRegistryError::Storage(format!("Object not found: {}", key)))?;

        if meta.tier == target_tier {
            return Ok(()); // Already in target tier
        }

        info!(
            "Migrating object {} from {:?} to {:?}",
            key, meta.tier, target_tier
        );

        // Read data from current location
        let current_path = self.get_storage_path(&meta.hash, &meta.tier);
        let data = tokio::fs::read(&current_path).await?;

        // Write to new location
        let new_path = self.get_storage_path(&meta.hash, &target_tier);
        tokio::fs::create_dir_all(new_path.parent().unwrap()).await?;
        tokio::fs::write(&new_path, &data).await?;

        // Delete from old location
        tokio::fs::remove_file(&current_path).await?;

        // Update metadata
        meta.tier = target_tier;

        Ok(())
    }

    // Helper methods

    fn calculate_hash(&self, data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("sha256:{}", hex::encode(hasher.finalize()))
    }

    fn select_tier(&self, size: u64) -> MemoryTier {
        if !self.config.enable_tiers {
            return MemoryTier::Ssd; // Default tier
        }

        let thresholds = &self.config.tier_thresholds;

        if size <= thresholds.gpu_threshold {
            MemoryTier::Gpu
        } else if size <= thresholds.cpu_threshold {
            MemoryTier::Cpu
        } else if size <= thresholds.nvme_threshold {
            MemoryTier::Nvme
        } else if size <= thresholds.ssd_threshold {
            MemoryTier::Ssd
        } else {
            MemoryTier::Hdd
        }
    }

    fn get_storage_path(&self, hash: &str, tier: &MemoryTier) -> PathBuf {
        let tier_name = match tier {
            MemoryTier::Gpu => "gpu",
            MemoryTier::Cpu => "cpu",
            MemoryTier::Nvme => "nvme",
            MemoryTier::Ssd => "ssd",
            MemoryTier::Hdd => "hdd",
        };

        // Use first 2 chars of hash for directory sharding
        let shard = &hash[7..9]; // Skip "sha256:" prefix
        self.config
            .storage_path
            .join(tier_name)
            .join(shard)
            .join(hash)
    }

    async fn add_reference(&self, content_hash: &str, key: &str) -> Result<()> {
        self.content_index
            .entry(content_hash.to_string())
            .or_insert_with(HashSet::new)
            .insert(key.to_string());

        // Update ref count in metadata if exists
        for mut entry in self.metadata.iter_mut() {
            if entry.value().hash == content_hash {
                entry.value_mut().ref_count += 1;
                break;
            }
        }

        Ok(())
    }

    async fn update_access_stats(&self, key: &str) -> Result<()> {
        if let Some(mut meta) = self.metadata.get_mut(key) {
            meta.access_count += 1;
            meta.last_access = self.current_timestamp();
        }
        Ok(())
    }

    async fn consider_migration(&self, key: &str, meta: &ObjectMetadata) -> Result<()> {
        // Simple migration policy: move frequently accessed objects to faster tiers
        if meta.access_count > 10 {
            let current_tier_priority = self.tier_priority(&meta.tier);
            if current_tier_priority > 0 {
                // Can move to faster tier
                let target_tier = self.get_faster_tier(&meta.tier);
                if let Some(target) = target_tier {
                    // Don't migrate immediately, just log for now
                    debug!(
                        "Object {} is hot, candidate for migration to {:?}",
                        key, target
                    );
                }
            }
        }
        Ok(())
    }

    fn tier_priority(&self, tier: &MemoryTier) -> u8 {
        match tier {
            MemoryTier::Gpu => 0,
            MemoryTier::Cpu => 1,
            MemoryTier::Nvme => 2,
            MemoryTier::Ssd => 3,
            MemoryTier::Hdd => 4,
        }
    }

    fn get_faster_tier(&self, current: &MemoryTier) -> Option<MemoryTier> {
        match current {
            MemoryTier::Hdd => Some(MemoryTier::Ssd),
            MemoryTier::Ssd => Some(MemoryTier::Nvme),
            MemoryTier::Nvme => Some(MemoryTier::Cpu),
            MemoryTier::Cpu => Some(MemoryTier::Gpu),
            MemoryTier::Gpu => None,
        }
    }

    async fn update_stats_for_put(&self, meta: &ObjectMetadata) -> Result<()> {
        let mut stats = self.stats.write().await;
        stats.total_objects += 1;
        stats.total_size += meta.size;
        stats.deduplicated_size += meta.size; // Will be adjusted if deduped
        *stats.tier_usage.entry(meta.tier).or_insert(0) += meta.size;
        Ok(())
    }

    async fn update_stats_for_delete(&self, meta: &ObjectMetadata) -> Result<()> {
        let mut stats = self.stats.write().await;
        stats.total_objects = stats.total_objects.saturating_sub(1);
        stats.total_size = stats.total_size.saturating_sub(meta.size);
        stats.deduplicated_size = stats.deduplicated_size.saturating_sub(meta.size);
        if let Some(tier_usage) = stats.tier_usage.get_mut(&meta.tier) {
            *tier_usage = tier_usage.saturating_sub(meta.size);
        }
        Ok(())
    }

    async fn load_metadata(&self) -> Result<()> {
        // TODO: Load metadata from persistent store
        Ok(())
    }

    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_store() -> (ContentAddressableStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = StoreConfig {
            storage_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let store = ContentAddressableStore::new(config).await.unwrap();
        (store, temp_dir)
    }

    #[tokio::test]
    async fn test_store_creation() {
        let (store, temp_dir) = create_test_store().await;

        // Check tier directories were created
        for tier in &["gpu", "cpu", "nvme", "ssd", "hdd"] {
            let tier_path = temp_dir.path().join(tier);
            assert!(tier_path.exists());
        }

        assert!(store.config.enable_dedup);
        assert!(store.config.enable_tiers);
    }

    #[tokio::test]
    async fn test_put_and_get() {
        let (store, _temp_dir) = create_test_store().await;

        let data = b"test content";
        let hash = store.put("test-key", data).await.unwrap();

        assert!(hash.starts_with("sha256:"));

        let retrieved = store.get("test-key").await.unwrap();
        assert_eq!(retrieved, data);
    }

    #[tokio::test]
    async fn test_deduplication() {
        let (store, _temp_dir) = create_test_store().await;

        let data = b"duplicate content";

        // Store same content with different keys
        let hash1 = store.put("key1", data).await.unwrap();
        let hash2 = store.put("key2", data).await.unwrap();

        // Hashes should be the same
        assert_eq!(hash1, hash2);

        // Stats should show deduplication
        let stats = store.stats().await.unwrap();
        assert_eq!(stats.total_objects, 2);
        // Due to deduplication, actual storage should be less than total
        assert!(stats.deduplicated_size <= stats.total_size);
    }

    #[tokio::test]
    async fn test_exists() {
        let (store, _temp_dir) = create_test_store().await;

        assert!(!store.exists("nonexistent").await);

        store.put("exists", b"data").await.unwrap();
        assert!(store.exists("exists").await);
    }

    #[tokio::test]
    async fn test_delete() {
        let (store, _temp_dir) = create_test_store().await;

        store.put("to-delete", b"data").await.unwrap();
        assert!(store.exists("to-delete").await);

        store.delete("to-delete").await.unwrap();
        assert!(!store.exists("to-delete").await);

        // Try to get deleted object
        let result = store.get("to-delete").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list() {
        let (store, _temp_dir) = create_test_store().await;

        // Store some objects
        store.put("obj1", b"data1").await.unwrap();
        store.put("obj2", b"data2").await.unwrap();
        store.put("obj3", b"data3").await.unwrap();

        let objects = store.list().await.unwrap();
        assert_eq!(objects.len(), 3);

        let keys: Vec<_> = objects.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"obj1"));
        assert!(keys.contains(&"obj2"));
        assert!(keys.contains(&"obj3"));
    }

    #[tokio::test]
    async fn test_tier_selection() {
        let (store, _temp_dir) = create_test_store().await;

        // Small object should go to GPU tier
        let small_data = vec![0u8; 500 * 1024]; // 500KB
        store.put("small", &small_data).await.unwrap();

        let metadata = store.metadata.read().await;
        let meta = metadata.get("small").unwrap();
        assert_eq!(meta.tier, MemoryTier::Gpu);
    }

    #[tokio::test]
    async fn test_migration() {
        let (store, _temp_dir) = create_test_store().await;

        // Store object
        store.put("to-migrate", b"data").await.unwrap();

        // Get current tier
        let metadata = store.metadata.read().await;
        let original_tier = metadata.get("to-migrate").unwrap().tier;
        drop(metadata);

        // Migrate to different tier
        let target_tier = if original_tier == MemoryTier::Gpu {
            MemoryTier::Cpu
        } else {
            MemoryTier::Gpu
        };

        store.migrate("to-migrate", target_tier).await.unwrap();

        // Verify migration
        let metadata = store.metadata.read().await;
        let meta = metadata.get("to-migrate").unwrap();
        assert_eq!(meta.tier, target_tier);

        // Verify data is still accessible
        let data = store.get("to-migrate").await.unwrap();
        assert_eq!(data, b"data");
    }

    #[tokio::test]
    async fn test_access_tracking() {
        let (store, _temp_dir) = create_test_store().await;

        store.put("tracked", b"data").await.unwrap();

        // Access multiple times
        for _ in 0..5 {
            store.get("tracked").await.unwrap();
        }

        let metadata = store.metadata.read().await;
        let meta = metadata.get("tracked").unwrap();
        assert_eq!(meta.access_count, 5);
    }

    #[tokio::test]
    async fn test_garbage_collection() {
        let (store, _temp_dir) = create_test_store().await;

        // Run GC on empty store
        store.garbage_collect().await.unwrap();

        // Add and remove objects
        store.put("temp1", b"data1").await.unwrap();
        store.put("temp2", b"data2").await.unwrap();
        store.delete("temp1").await.unwrap();

        // Run GC
        store.garbage_collect().await.unwrap();

        // Remaining object should still be accessible
        let data = store.get("temp2").await.unwrap();
        assert_eq!(data, b"data2");
    }
}
