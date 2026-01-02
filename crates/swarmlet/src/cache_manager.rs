//! Build cache management
//!
//! This module manages shared caches for Rust builds:
//! - Cargo registry (crates.io index and downloaded crates)
//! - sccache (compilation cache)
//! - Target directory caches (per-project)

use crate::build_backend::CacheMount;
use crate::build_job::CacheConfig;
use crate::{Result, SwarmletError};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Manages shared caches for build jobs
pub struct CacheManager {
    /// Base cache directory
    cache_dir: PathBuf,
    /// Cargo registry cache path
    registry_path: PathBuf,
    /// Cargo git cache path
    git_path: PathBuf,
    /// sccache directory
    sccache_path: PathBuf,
    /// Project-specific target caches
    target_caches: Arc<RwLock<HashMap<String, TargetCacheInfo>>>,
    /// Maximum cache size in bytes
    max_cache_bytes: u64,
}

/// Information about a target cache
#[derive(Debug, Clone)]
pub struct TargetCacheInfo {
    /// Path to the cache
    pub path: PathBuf,
    /// Cache key (project hash)
    pub key: String,
    /// When the cache was last used
    pub last_used: std::time::Instant,
    /// Approximate size in bytes
    pub size_bytes: u64,
}

impl CacheManager {
    /// Create a new cache manager
    pub async fn new(cache_dir: PathBuf) -> Result<Self> {
        let registry_path = cache_dir.join("cargo-registry");
        let git_path = cache_dir.join("cargo-git");
        let sccache_path = cache_dir.join("sccache");
        let targets_dir = cache_dir.join("targets");

        // Create cache directories
        for dir in [&registry_path, &git_path, &sccache_path, &targets_dir] {
            tokio::fs::create_dir_all(dir).await.map_err(|e| {
                SwarmletError::Configuration(format!(
                    "Failed to create cache directory {}: {}",
                    dir.display(),
                    e
                ))
            })?;
        }

        info!("Cache manager initialized at {}", cache_dir.display());

        Ok(Self {
            cache_dir,
            registry_path,
            git_path,
            sccache_path,
            target_caches: Arc::new(RwLock::new(HashMap::new())),
            max_cache_bytes: 50 * 1024 * 1024 * 1024, // 50GB default
        })
    }

    /// Get cache mounts for a build based on configuration
    pub async fn get_mounts(&self, config: &CacheConfig) -> Result<Vec<CacheMount>> {
        let mut mounts = Vec::new();

        if config.use_cargo_registry {
            // Mount cargo registry
            mounts.push(CacheMount::new(
                self.registry_path.clone(),
                PathBuf::from("/root/.cargo/registry"),
            ));

            // Mount cargo git cache
            mounts.push(CacheMount::new(
                self.git_path.clone(),
                PathBuf::from("/root/.cargo/git"),
            ));
        }

        if config.use_sccache {
            mounts.push(CacheMount::new(
                self.sccache_path.clone(),
                PathBuf::from("/root/.cache/sccache"),
            ));
        }

        if config.cache_target {
            if let Some(key) = &config.cache_key {
                let target_path = self.get_or_create_target_cache(key).await?;
                mounts.push(CacheMount::new(
                    target_path,
                    PathBuf::from("/workspace/target"),
                ));
            }
        }

        debug!("Created {} cache mounts", mounts.len());
        Ok(mounts)
    }

    /// Get or create a target cache for a project
    async fn get_or_create_target_cache(&self, cache_key: &str) -> Result<PathBuf> {
        let mut caches = self.target_caches.write().await;

        if let Some(info) = caches.get_mut(cache_key) {
            info.last_used = std::time::Instant::now();
            return Ok(info.path.clone());
        }

        let path = self.cache_dir.join("targets").join(cache_key);
        tokio::fs::create_dir_all(&path).await.map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create target cache: {e}"),
            ))
        })?;

        let info = TargetCacheInfo {
            path: path.clone(),
            key: cache_key.to_string(),
            last_used: std::time::Instant::now(),
            size_bytes: 0,
        };

        caches.insert(cache_key.to_string(), info);
        debug!("Created new target cache: {}", cache_key);

        Ok(path)
    }

    /// Get the cargo registry path
    pub fn registry_path(&self) -> &PathBuf {
        &self.registry_path
    }

    /// Get the sccache path
    pub fn sccache_path(&self) -> &PathBuf {
        &self.sccache_path
    }

    /// Calculate total cache size
    pub async fn total_size(&self) -> Result<u64> {
        let mut total = 0u64;

        for path in [&self.registry_path, &self.git_path, &self.sccache_path] {
            if path.exists() {
                total += Self::dir_size(path).await?;
            }
        }

        // Add target caches
        let caches = self.target_caches.read().await;
        for info in caches.values() {
            if info.path.exists() {
                total += Self::dir_size(&info.path).await?;
            }
        }

        Ok(total)
    }

    /// Calculate directory size recursively
    async fn dir_size(path: &PathBuf) -> Result<u64> {
        let mut size = 0u64;

        let mut entries = tokio::fs::read_dir(path).await.map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read directory: {e}"),
            ))
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read entry: {e}"),
            ))
        })? {
            let metadata = entry.metadata().await?;
            if metadata.is_file() {
                size += metadata.len();
            } else if metadata.is_dir() {
                size += Box::pin(Self::dir_size(&entry.path())).await?;
            }
        }

        Ok(size)
    }

    /// Cleanup old cache entries to free space
    pub async fn cleanup(&self, max_age_secs: u64) -> Result<CleanupResult> {
        let mut result = CleanupResult::default();

        // Cleanup old target caches
        let mut caches = self.target_caches.write().await;
        let now = std::time::Instant::now();

        let to_remove: Vec<String> = caches
            .iter()
            .filter(|(_, info)| now.duration_since(info.last_used).as_secs() > max_age_secs)
            .map(|(key, _)| key.clone())
            .collect();

        for key in to_remove {
            if let Some(info) = caches.remove(&key) {
                if let Err(e) = tokio::fs::remove_dir_all(&info.path).await {
                    warn!("Failed to remove target cache {}: {}", key, e);
                } else {
                    info!("Removed stale target cache: {}", key);
                    result.target_caches_removed += 1;
                    result.bytes_freed += info.size_bytes;
                }
            }
        }

        // Cleanup sccache if needed
        let total_size = self.total_size().await?;
        if total_size > self.max_cache_bytes {
            info!(
                "Cache size ({:.2} GB) exceeds limit ({:.2} GB), cleaning up",
                total_size as f64 / 1024.0 / 1024.0 / 1024.0,
                self.max_cache_bytes as f64 / 1024.0 / 1024.0 / 1024.0
            );

            // Run sccache --show-stats to potentially trigger cleanup
            let _ = tokio::process::Command::new("sccache")
                .arg("--show-stats")
                .env("SCCACHE_DIR", &self.sccache_path)
                .output()
                .await;
        }

        Ok(result)
    }

    /// Clear all caches
    pub async fn clear_all(&self) -> Result<()> {
        info!("Clearing all caches");

        for path in [&self.registry_path, &self.git_path, &self.sccache_path] {
            if path.exists() {
                tokio::fs::remove_dir_all(path).await?;
                tokio::fs::create_dir_all(path).await?;
            }
        }

        // Clear target caches
        {
            let mut caches = self.target_caches.write().await;
            for info in caches.values() {
                if info.path.exists() {
                    let _ = tokio::fs::remove_dir_all(&info.path).await;
                }
            }
            caches.clear();
        }

        Ok(())
    }

    /// Get sccache statistics
    pub async fn sccache_stats(&self) -> Result<SccacheStats> {
        let output = tokio::process::Command::new("sccache")
            .arg("--show-stats")
            .arg("--stats-format=json")
            .env("SCCACHE_DIR", &self.sccache_path)
            .output()
            .await
            .map_err(|e| {
                SwarmletError::WorkloadExecution(format!("Failed to run sccache: {e}"))
            })?;

        if !output.status.success() {
            return Ok(SccacheStats::default());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse JSON output (simplified)
        Ok(SccacheStats::from_json(&stdout).unwrap_or_default())
    }

    /// Set maximum cache size
    pub fn set_max_cache_bytes(&mut self, bytes: u64) {
        self.max_cache_bytes = bytes;
    }
}

/// Result of a cache cleanup operation
#[derive(Debug, Default)]
pub struct CleanupResult {
    /// Number of target caches removed
    pub target_caches_removed: usize,
    /// Bytes freed
    pub bytes_freed: u64,
}

/// sccache statistics
#[derive(Debug, Default)]
pub struct SccacheStats {
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Hit rate (0.0 - 1.0)
    pub hit_rate: f64,
    /// Cache size in bytes
    pub cache_size_bytes: u64,
}

impl SccacheStats {
    fn from_json(json: &str) -> Option<Self> {
        // Simplified JSON parsing
        // In production, use serde_json
        let hits = json
            .find("cache_hits")
            .and_then(|i| json[i..].find(':'))
            .and_then(|_| Some(0u64))?;

        Some(Self {
            cache_hits: hits,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_manager_creation() {
        let temp_dir = std::env::temp_dir().join("test-cache-manager");
        let manager = CacheManager::new(temp_dir.clone()).await;
        assert!(manager.is_ok());

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_get_mounts_with_all_caches() {
        let temp_dir = std::env::temp_dir().join("test-cache-mounts");
        let manager = CacheManager::new(temp_dir.clone()).await.unwrap();

        let config = CacheConfig {
            use_cargo_registry: true,
            use_sccache: true,
            cache_target: true,
            cache_key: Some("test-project".to_string()),
        };

        let mounts = manager.get_mounts(&config).await.unwrap();
        assert_eq!(mounts.len(), 4); // registry, git, sccache, target

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_get_mounts_minimal() {
        let temp_dir = std::env::temp_dir().join("test-cache-minimal");
        let manager = CacheManager::new(temp_dir.clone()).await.unwrap();

        let config = CacheConfig::no_cache();
        let mounts = manager.get_mounts(&config).await.unwrap();
        assert!(mounts.is_empty());

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }
}
