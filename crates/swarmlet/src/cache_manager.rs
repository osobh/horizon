//! Build cache management
//!
//! This module manages shared caches for Rust builds:
//! - Cargo registry (crates.io index and downloaded crates)
//! - sccache (compilation cache)
//! - Target directory caches (per-project)
//! - Source caches (cached build sources by hash)

use crate::build_backend::CacheMount;
use crate::build_job::CacheConfig;
use crate::{Result, SwarmletError};
use chrono::{DateTime, Utc};
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tar::Builder;
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
    /// Source caches directory (cached build sources by hash)
    sources_dir: PathBuf,
    /// Project-specific target caches
    target_caches: Arc<RwLock<HashMap<String, TargetCacheInfo>>>,
    /// Source cache metadata
    source_caches: Arc<RwLock<HashMap<String, SourceCacheMetadata>>>,
    /// Maximum cache size in bytes
    max_cache_bytes: u64,
}

/// Information about a target cache (runtime tracking)
#[derive(Debug, Clone)]
pub struct TargetCacheInfo {
    /// Path to the cache
    pub path: PathBuf,
    /// Cache key (project hash)
    pub key: String,
    /// When the cache was last used (runtime)
    pub last_used: std::time::Instant,
    /// Approximate size in bytes
    pub size_bytes: u64,
}

/// Metadata for artifact cache (persisted to disk)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactCacheMetadata {
    /// Cache key (project identifier)
    pub cache_key: String,
    /// When the cache was created
    pub created_at: DateTime<Utc>,
    /// Size in bytes
    pub size_bytes: u64,
    /// When the cache was last used
    pub last_used: DateTime<Utc>,
    /// Number of times the cache has been used
    pub use_count: u32,
    /// Associated build job IDs that used this cache
    pub build_jobs: Vec<String>,
}

/// Metadata for cached source code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceCacheMetadata {
    /// SHA256 hash of the source
    pub hash: String,
    /// Type of source (git, archive, local)
    pub source_type: String,
    /// When the cache was created
    pub created_at: DateTime<Utc>,
    /// Size in bytes
    pub size_bytes: u64,
    /// When the cache was last used
    pub last_used: DateTime<Utc>,
    /// Original source info (URL or path)
    pub source_info: String,
}

impl CacheManager {
    /// Create a new cache manager
    pub async fn new(cache_dir: PathBuf) -> Result<Self> {
        let registry_path = cache_dir.join("cargo-registry");
        let git_path = cache_dir.join("cargo-git");
        let sccache_path = cache_dir.join("sccache");
        let targets_dir = cache_dir.join("targets");
        let sources_dir = cache_dir.join("sources");

        // Create cache directories
        for dir in [&registry_path, &git_path, &sccache_path, &targets_dir, &sources_dir] {
            tokio::fs::create_dir_all(dir).await.map_err(|e| {
                SwarmletError::Configuration(format!(
                    "Failed to create cache directory {}: {}",
                    dir.display(),
                    e
                ))
            })?;
        }

        // Load existing source cache metadata
        let source_caches = Self::load_source_cache_metadata(&sources_dir).await;

        info!("Cache manager initialized at {}", cache_dir.display());

        Ok(Self {
            cache_dir,
            registry_path,
            git_path,
            sccache_path,
            sources_dir,
            target_caches: Arc::new(RwLock::new(HashMap::new())),
            source_caches: Arc::new(RwLock::new(source_caches)),
            max_cache_bytes: 50 * 1024 * 1024 * 1024, // 50GB default
        })
    }

    /// Load source cache metadata from disk
    async fn load_source_cache_metadata(sources_dir: &PathBuf) -> HashMap<String, SourceCacheMetadata> {
        let mut metadata_map = HashMap::new();

        if let Ok(mut entries) = tokio::fs::read_dir(sources_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let metadata_path = entry.path().join("metadata.json");
                if metadata_path.exists() {
                    if let Ok(contents) = tokio::fs::read_to_string(&metadata_path).await {
                        if let Ok(metadata) = serde_json::from_str::<SourceCacheMetadata>(&contents) {
                            metadata_map.insert(metadata.hash.clone(), metadata);
                        }
                    }
                }
            }
        }

        debug!("Loaded {} source cache entries", metadata_map.len());
        metadata_map
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

    /// Get the sources cache path
    pub fn sources_path(&self) -> &PathBuf {
        &self.sources_dir
    }

    // ==================== Source Caching ====================

    /// Cache source code from a workspace directory and return its hash
    ///
    /// This computes a hash of the directory contents and stores a compressed
    /// tarball that can be retrieved later using `get_cached_source()`.
    pub async fn cache_source(
        &self,
        workspace: &PathBuf,
        source_type: &str,
        source_info: &str,
    ) -> Result<String> {
        // Compute hash of the directory
        let hash = self.compute_dir_hash(workspace).await?;

        // Check if already cached
        if self.has_cached_source(&hash).await {
            debug!("Source already cached: {}", hash);
            self.touch_source(&hash).await?;
            return Ok(hash);
        }

        let cache_path = self.sources_dir.join(&hash);
        tokio::fs::create_dir_all(&cache_path).await.map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create source cache directory: {e}"),
            ))
        })?;

        // Create tarball
        let archive_path = cache_path.join("source.tar.gz");
        let workspace_clone = workspace.clone();
        let archive_path_clone = archive_path.clone();

        // Run synchronous tar operations in blocking task
        let size_bytes = tokio::task::spawn_blocking(move || -> Result<u64> {
            let file = std::fs::File::create(&archive_path_clone).map_err(|e| {
                SwarmletError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create archive file: {e}"),
                ))
            })?;
            let encoder = GzEncoder::new(file, Compression::default());
            let mut builder = Builder::new(encoder);

            // Add all files from workspace
            builder.append_dir_all(".", &workspace_clone).map_err(|e| {
                SwarmletError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to add files to archive: {e}"),
                ))
            })?;

            let encoder = builder.into_inner().map_err(|e| {
                SwarmletError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to finish archive: {e}"),
                ))
            })?;
            encoder.finish().map_err(|e| {
                SwarmletError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to finish gzip compression: {e}"),
                ))
            })?;

            // Get archive size
            let metadata = std::fs::metadata(&archive_path_clone)?;
            Ok(metadata.len())
        })
        .await
        .map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to spawn blocking task: {e}"),
            ))
        })??;

        // Create and save metadata
        let now = Utc::now();
        let metadata = SourceCacheMetadata {
            hash: hash.clone(),
            source_type: source_type.to_string(),
            created_at: now,
            size_bytes,
            last_used: now,
            source_info: source_info.to_string(),
        };

        let metadata_path = cache_path.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata).map_err(|e| {
            SwarmletError::Configuration(format!("Failed to serialize metadata: {e}"))
        })?;
        tokio::fs::write(&metadata_path, metadata_json).await?;

        // Update in-memory cache
        {
            let mut caches = self.source_caches.write().await;
            caches.insert(hash.clone(), metadata);
        }

        info!(
            "Cached source {} ({} bytes, type: {})",
            hash, size_bytes, source_type
        );

        Ok(hash)
    }

    /// Get the path to a cached source by hash
    ///
    /// Returns None if the source is not cached.
    pub async fn get_cached_source(&self, hash: &str) -> Option<PathBuf> {
        let caches = self.source_caches.read().await;
        if caches.contains_key(hash) {
            let path = self.sources_dir.join(hash);
            if path.exists() {
                return Some(path);
            }
        }
        None
    }

    /// Check if a source is cached
    pub async fn has_cached_source(&self, hash: &str) -> bool {
        let caches = self.source_caches.read().await;
        caches.contains_key(hash)
    }

    /// Update last_used timestamp for a cached source
    pub async fn touch_source(&self, hash: &str) -> Result<()> {
        let mut caches = self.source_caches.write().await;
        if let Some(metadata) = caches.get_mut(hash) {
            metadata.last_used = Utc::now();

            // Persist updated metadata
            let metadata_path = self.sources_dir.join(hash).join("metadata.json");
            let metadata_json = serde_json::to_string_pretty(metadata).map_err(|e| {
                SwarmletError::Configuration(format!("Failed to serialize metadata: {e}"))
            })?;
            tokio::fs::write(&metadata_path, metadata_json).await?;
        }
        Ok(())
    }

    /// Compute a SHA256 hash of a directory's contents
    ///
    /// This performs a deterministic traversal (sorted by path) and hashes
    /// file names and contents to produce a reproducible hash.
    pub async fn compute_dir_hash(&self, path: &PathBuf) -> Result<String> {
        let path_clone = path.clone();

        tokio::task::spawn_blocking(move || -> Result<String> {
            let mut hasher = Sha256::new();

            // Collect all file paths and sort them for deterministic hashing
            let mut entries: Vec<PathBuf> = Vec::new();
            Self::collect_files_sync(&path_clone, &mut entries)?;
            entries.sort();

            for entry_path in entries {
                // Hash the relative path
                let relative = entry_path
                    .strip_prefix(&path_clone)
                    .unwrap_or(&entry_path);
                hasher.update(relative.to_string_lossy().as_bytes());
                hasher.update(b"\0");

                // Hash file contents
                let contents = std::fs::read(&entry_path).map_err(|e| {
                    SwarmletError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to read file {}: {e}", entry_path.display()),
                    ))
                })?;
                hasher.update(&contents);
                hasher.update(b"\0");
            }

            Ok(format!("{:x}", hasher.finalize()))
        })
        .await
        .map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to spawn blocking task: {e}"),
            ))
        })?
    }

    /// Synchronously collect all files in a directory (for use in blocking context)
    fn collect_files_sync(path: &PathBuf, entries: &mut Vec<PathBuf>) -> Result<()> {
        if path.is_file() {
            entries.push(path.clone());
            return Ok(());
        }

        let dir_entries = std::fs::read_dir(path).map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read directory {}: {e}", path.display()),
            ))
        })?;

        for entry in dir_entries {
            let entry = entry.map_err(|e| {
                SwarmletError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to read entry: {e}"),
                ))
            })?;
            let entry_path = entry.path();

            if entry_path.is_file() {
                entries.push(entry_path);
            } else if entry_path.is_dir() {
                Self::collect_files_sync(&entry_path, entries)?;
            }
        }

        Ok(())
    }

    /// Cleanup old source caches
    pub async fn cleanup_sources(&self, max_age_secs: u64) -> Result<SourceCleanupResult> {
        let mut result = SourceCleanupResult::default();
        let now = Utc::now();

        let mut caches = self.source_caches.write().await;
        let to_remove: Vec<String> = caches
            .iter()
            .filter(|(_, metadata)| {
                let age = now.signed_duration_since(metadata.last_used);
                age.num_seconds() as u64 > max_age_secs
            })
            .map(|(hash, _)| hash.clone())
            .collect();

        for hash in to_remove {
            if let Some(metadata) = caches.remove(&hash) {
                let cache_path = self.sources_dir.join(&hash);
                if let Err(e) = tokio::fs::remove_dir_all(&cache_path).await {
                    warn!("Failed to remove source cache {}: {}", hash, e);
                } else {
                    info!("Removed stale source cache: {}", hash);
                    result.sources_removed += 1;
                    result.bytes_freed += metadata.size_bytes;
                }
            }
        }

        Ok(result)
    }

    /// Get metadata for a cached source
    pub async fn get_source_metadata(&self, hash: &str) -> Option<SourceCacheMetadata> {
        let caches = self.source_caches.read().await;
        caches.get(hash).cloned()
    }

    /// List all cached sources
    pub async fn list_cached_sources(&self) -> Vec<SourceCacheMetadata> {
        let caches = self.source_caches.read().await;
        caches.values().cloned().collect()
    }

    /// Delete a cached source by hash
    ///
    /// Returns the metadata of the deleted source, or None if not found.
    pub async fn delete_cached_source(&self, hash: &str) -> Result<Option<SourceCacheMetadata>> {
        let mut caches = self.source_caches.write().await;

        if let Some(metadata) = caches.remove(hash) {
            let cache_path = self.sources_dir.join(hash);
            if cache_path.exists() {
                tokio::fs::remove_dir_all(&cache_path).await.map_err(|e| {
                    // Re-insert metadata on failure
                    caches.insert(hash.to_string(), metadata.clone());
                    SwarmletError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to remove source cache directory: {e}"),
                    ))
                })?;
            }
            info!("Deleted source cache: {} ({} bytes)", hash, metadata.size_bytes);
            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }

    // ==================== End Source Caching ====================

    // ==================== Artifact Caching ====================

    /// Get the artifacts cache directory path
    pub fn artifacts_path(&self) -> PathBuf {
        self.cache_dir.join("targets")
    }

    /// Record artifact cache usage for a build
    ///
    /// This updates or creates metadata for an artifact cache, tracking
    /// which builds have used it for incremental compilation.
    pub async fn record_artifact_usage(&self, cache_key: &str, build_job_id: &str) -> Result<()> {
        let targets_dir = self.artifacts_path();
        let cache_path = targets_dir.join(cache_key);
        let metadata_path = cache_path.join("artifact_metadata.json");

        // Calculate current size
        let size_bytes = if cache_path.exists() {
            Self::dir_size(&cache_path).await.unwrap_or(0)
        } else {
            0
        };

        let now = Utc::now();

        // Load existing metadata or create new
        let mut metadata = if metadata_path.exists() {
            let contents = tokio::fs::read_to_string(&metadata_path).await?;
            serde_json::from_str::<ArtifactCacheMetadata>(&contents).unwrap_or_else(|_| {
                ArtifactCacheMetadata {
                    cache_key: cache_key.to_string(),
                    created_at: now,
                    size_bytes,
                    last_used: now,
                    use_count: 0,
                    build_jobs: Vec::new(),
                }
            })
        } else {
            ArtifactCacheMetadata {
                cache_key: cache_key.to_string(),
                created_at: now,
                size_bytes,
                last_used: now,
                use_count: 0,
                build_jobs: Vec::new(),
            }
        };

        // Update metadata
        metadata.last_used = now;
        metadata.use_count += 1;
        metadata.size_bytes = size_bytes;

        // Add build job ID if not already present (keep last 10)
        if !metadata.build_jobs.contains(&build_job_id.to_string()) {
            metadata.build_jobs.push(build_job_id.to_string());
            if metadata.build_jobs.len() > 10 {
                metadata.build_jobs.remove(0);
            }
        }

        // Save metadata
        let metadata_json = serde_json::to_string_pretty(&metadata).map_err(|e| {
            SwarmletError::Configuration(format!("Failed to serialize artifact metadata: {e}"))
        })?;
        tokio::fs::write(&metadata_path, metadata_json).await?;

        debug!(
            "Recorded artifact cache usage: {} (use_count: {}, size: {} bytes)",
            cache_key, metadata.use_count, metadata.size_bytes
        );

        Ok(())
    }

    /// List all artifact caches
    pub async fn list_artifact_caches(&self) -> Vec<ArtifactCacheMetadata> {
        let targets_dir = self.artifacts_path();
        let mut caches = Vec::new();

        if let Ok(mut entries) = tokio::fs::read_dir(&targets_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let metadata_path = entry.path().join("artifact_metadata.json");
                if metadata_path.exists() {
                    if let Ok(contents) = tokio::fs::read_to_string(&metadata_path).await {
                        if let Ok(metadata) = serde_json::from_str::<ArtifactCacheMetadata>(&contents) {
                            caches.push(metadata);
                        }
                    }
                } else if entry.path().is_dir() {
                    // Directory exists but no metadata - create basic entry
                    let cache_key = entry.file_name().to_string_lossy().to_string();
                    let size_bytes = Self::dir_size(&entry.path()).await.unwrap_or(0);
                    caches.push(ArtifactCacheMetadata {
                        cache_key,
                        created_at: Utc::now(),
                        size_bytes,
                        last_used: Utc::now(),
                        use_count: 0,
                        build_jobs: Vec::new(),
                    });
                }
            }
        }

        caches
    }

    /// Get metadata for a specific artifact cache
    pub async fn get_artifact_metadata(&self, cache_key: &str) -> Option<ArtifactCacheMetadata> {
        let metadata_path = self.artifacts_path().join(cache_key).join("artifact_metadata.json");

        if metadata_path.exists() {
            if let Ok(contents) = tokio::fs::read_to_string(&metadata_path).await {
                return serde_json::from_str(&contents).ok();
            }
        }

        // Check if directory exists but has no metadata
        let cache_path = self.artifacts_path().join(cache_key);
        if cache_path.is_dir() {
            let size_bytes = Self::dir_size(&cache_path).await.unwrap_or(0);
            return Some(ArtifactCacheMetadata {
                cache_key: cache_key.to_string(),
                created_at: Utc::now(),
                size_bytes,
                last_used: Utc::now(),
                use_count: 0,
                build_jobs: Vec::new(),
            });
        }

        None
    }

    /// Delete an artifact cache
    pub async fn delete_artifact_cache(&self, cache_key: &str) -> Result<Option<ArtifactCacheMetadata>> {
        let cache_path = self.artifacts_path().join(cache_key);

        if !cache_path.exists() {
            return Ok(None);
        }

        // Get metadata before deletion
        let metadata = self.get_artifact_metadata(cache_key).await;

        // Remove from runtime cache
        {
            let mut caches = self.target_caches.write().await;
            caches.remove(cache_key);
        }

        // Delete directory
        tokio::fs::remove_dir_all(&cache_path).await.map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to remove artifact cache directory: {e}"),
            ))
        })?;

        if let Some(ref m) = metadata {
            info!(
                "Deleted artifact cache: {} ({} bytes, {} builds)",
                cache_key,
                m.size_bytes,
                m.build_jobs.len()
            );
        }

        Ok(metadata)
    }

    /// Cleanup old artifact caches that haven't been used recently
    pub async fn cleanup_artifacts(&self, max_age_secs: u64) -> Result<ArtifactCleanupResult> {
        let mut result = ArtifactCleanupResult::default();
        let now = Utc::now();

        let caches = self.list_artifact_caches().await;

        for cache in caches {
            let age = now.signed_duration_since(cache.last_used);
            if age.num_seconds() as u64 > max_age_secs {
                if let Ok(Some(deleted)) = self.delete_artifact_cache(&cache.cache_key).await {
                    result.caches_removed += 1;
                    result.bytes_freed += deleted.size_bytes;
                }
            }
        }

        Ok(result)
    }

    // ==================== End Artifact Caching ====================

    /// Calculate total cache size
    pub async fn total_size(&self) -> Result<u64> {
        let mut total = 0u64;

        for path in [&self.registry_path, &self.git_path, &self.sccache_path, &self.sources_dir] {
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

        for path in [&self.registry_path, &self.git_path, &self.sccache_path, &self.sources_dir] {
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

        // Clear source caches
        {
            let mut caches = self.source_caches.write().await;
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

/// Result of a source cache cleanup operation
#[derive(Debug, Default)]
pub struct SourceCleanupResult {
    /// Number of source caches removed
    pub sources_removed: usize,
    /// Bytes freed
    pub bytes_freed: u64,
}

/// Result of an artifact cache cleanup operation
#[derive(Debug, Default)]
pub struct ArtifactCleanupResult {
    /// Number of artifact caches removed
    pub caches_removed: usize,
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

    #[tokio::test]
    async fn test_cache_source_and_retrieve() {
        let temp_dir = std::env::temp_dir().join("test-source-cache");
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;

        let manager = CacheManager::new(temp_dir.clone()).await.unwrap();

        // Create a test workspace with some files
        let workspace = temp_dir.join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("main.rs"), "fn main() {}").await.unwrap();
        tokio::fs::write(workspace.join("Cargo.toml"), "[package]\nname = \"test\"").await.unwrap();

        // Cache the source
        let hash = manager.cache_source(&workspace, "local", "/test/path").await.unwrap();
        assert!(!hash.is_empty());

        // Verify it's cached
        assert!(manager.has_cached_source(&hash).await);

        // Get cached source path
        let cache_path = manager.get_cached_source(&hash).await;
        assert!(cache_path.is_some());

        let cache_path = cache_path.unwrap();
        assert!(cache_path.join("source.tar.gz").exists());
        assert!(cache_path.join("metadata.json").exists());

        // Verify metadata
        let metadata = manager.get_source_metadata(&hash).await.unwrap();
        assert_eq!(metadata.hash, hash);
        assert_eq!(metadata.source_type, "local");
        assert_eq!(metadata.source_info, "/test/path");

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_compute_dir_hash_deterministic() {
        let temp_dir = std::env::temp_dir().join("test-hash-deterministic");
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;

        let manager = CacheManager::new(temp_dir.clone()).await.unwrap();

        // Create test directory
        let workspace = temp_dir.join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("file1.txt"), "content1").await.unwrap();
        tokio::fs::write(workspace.join("file2.txt"), "content2").await.unwrap();

        // Hash should be deterministic
        let hash1 = manager.compute_dir_hash(&workspace).await.unwrap();
        let hash2 = manager.compute_dir_hash(&workspace).await.unwrap();
        assert_eq!(hash1, hash2);

        // Different content should produce different hash
        tokio::fs::write(workspace.join("file1.txt"), "modified").await.unwrap();
        let hash3 = manager.compute_dir_hash(&workspace).await.unwrap();
        assert_ne!(hash1, hash3);

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_list_cached_sources() {
        let temp_dir = std::env::temp_dir().join("test-list-sources");
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;

        let manager = CacheManager::new(temp_dir.clone()).await.unwrap();

        // Initially empty
        assert!(manager.list_cached_sources().await.is_empty());

        // Create and cache some workspaces
        let workspace1 = temp_dir.join("ws1");
        tokio::fs::create_dir_all(&workspace1).await.unwrap();
        tokio::fs::write(workspace1.join("main.rs"), "fn main() { /* ws1 */ }").await.unwrap();

        let workspace2 = temp_dir.join("ws2");
        tokio::fs::create_dir_all(&workspace2).await.unwrap();
        tokio::fs::write(workspace2.join("main.rs"), "fn main() { /* ws2 */ }").await.unwrap();

        manager.cache_source(&workspace1, "local", "ws1").await.unwrap();
        manager.cache_source(&workspace2, "archive", "ws2").await.unwrap();

        let sources = manager.list_cached_sources().await;
        assert_eq!(sources.len(), 2);

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_touch_source_updates_timestamp() {
        let temp_dir = std::env::temp_dir().join("test-touch-source");
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;

        let manager = CacheManager::new(temp_dir.clone()).await.unwrap();

        // Create and cache workspace
        let workspace = temp_dir.join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("main.rs"), "fn main() {}").await.unwrap();

        let hash = manager.cache_source(&workspace, "local", "/test").await.unwrap();

        let metadata_before = manager.get_source_metadata(&hash).await.unwrap();

        // Wait a moment and touch
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        manager.touch_source(&hash).await.unwrap();

        let metadata_after = manager.get_source_metadata(&hash).await.unwrap();
        assert!(metadata_after.last_used >= metadata_before.last_used);

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_delete_cached_source() {
        let temp_dir = std::env::temp_dir().join("test-delete-source");
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;

        let manager = CacheManager::new(temp_dir.clone()).await.unwrap();

        // Create and cache workspace
        let workspace = temp_dir.join("workspace");
        tokio::fs::create_dir_all(&workspace).await.unwrap();
        tokio::fs::write(workspace.join("main.rs"), "fn main() {}").await.unwrap();

        let hash = manager.cache_source(&workspace, "local", "/test").await.unwrap();

        // Verify it's cached
        assert!(manager.has_cached_source(&hash).await);
        let cache_path = manager.get_cached_source(&hash).await.unwrap();
        assert!(cache_path.exists());

        // Delete the source
        let deleted = manager.delete_cached_source(&hash).await.unwrap();
        assert!(deleted.is_some());
        assert_eq!(deleted.unwrap().hash, hash);

        // Verify it's no longer cached
        assert!(!manager.has_cached_source(&hash).await);
        assert!(manager.get_cached_source(&hash).await.is_none());
        assert!(!cache_path.exists());

        // Deleting non-existent source returns None
        let deleted_again = manager.delete_cached_source(&hash).await.unwrap();
        assert!(deleted_again.is_none());

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_artifact_cache_record_and_list() {
        let temp_dir = std::env::temp_dir().join("test-artifact-cache");
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;

        let manager = CacheManager::new(temp_dir.clone()).await.unwrap();

        // Initially empty
        assert!(manager.list_artifact_caches().await.is_empty());

        // Create artifact cache directory and record usage
        let cache_key = "my-project-abc123";
        let artifacts_path = manager.artifacts_path().join(cache_key);
        tokio::fs::create_dir_all(&artifacts_path).await.unwrap();
        tokio::fs::write(artifacts_path.join("some-artifact.rlib"), "binary data").await.unwrap();

        // Record usage
        manager.record_artifact_usage(cache_key, "job-1").await.unwrap();

        // List should now contain the cache
        let caches = manager.list_artifact_caches().await;
        assert_eq!(caches.len(), 1);
        assert_eq!(caches[0].cache_key, cache_key);
        assert_eq!(caches[0].use_count, 1);
        assert!(caches[0].build_jobs.contains(&"job-1".to_string()));

        // Record another usage
        manager.record_artifact_usage(cache_key, "job-2").await.unwrap();

        let metadata = manager.get_artifact_metadata(cache_key).await.unwrap();
        assert_eq!(metadata.use_count, 2);
        assert!(metadata.build_jobs.contains(&"job-1".to_string()));
        assert!(metadata.build_jobs.contains(&"job-2".to_string()));

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_artifact_cache_delete() {
        let temp_dir = std::env::temp_dir().join("test-artifact-delete");
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;

        let manager = CacheManager::new(temp_dir.clone()).await.unwrap();

        // Create and record artifact cache
        let cache_key = "delete-test-project";
        let artifacts_path = manager.artifacts_path().join(cache_key);
        tokio::fs::create_dir_all(&artifacts_path).await.unwrap();
        tokio::fs::write(artifacts_path.join("artifact.o"), "obj data").await.unwrap();

        manager.record_artifact_usage(cache_key, "job-1").await.unwrap();

        // Verify it exists
        assert!(manager.get_artifact_metadata(cache_key).await.is_some());
        assert!(artifacts_path.exists());

        // Delete it
        let deleted = manager.delete_artifact_cache(cache_key).await.unwrap();
        assert!(deleted.is_some());
        assert_eq!(deleted.unwrap().cache_key, cache_key);

        // Verify it's gone
        assert!(manager.get_artifact_metadata(cache_key).await.is_none());
        assert!(!artifacts_path.exists());

        // Delete again returns None
        let deleted_again = manager.delete_artifact_cache(cache_key).await.unwrap();
        assert!(deleted_again.is_none());

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }
}
