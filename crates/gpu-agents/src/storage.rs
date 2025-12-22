//! GPU Agent Storage Tier
//!
//! Provides high-performance storage for GPU agents using /magikdev/gpu path
//! with NVMe optimization and GPU memory mapping capabilities.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

// Import from the storage crate
use exorust_storage::{NvmeConfig, NvmeStorage, Storage as StorageTrait};

/// GPU storage configuration
#[derive(Debug, Clone)]
pub struct GpuStorageConfig {
    /// Base path for storage (default: /magikdev/gpu)
    pub base_path: PathBuf,
    /// Cache directory path
    pub cache_path: PathBuf,
    /// Write-ahead log path
    pub wal_path: PathBuf,
    /// Enable GPU memory caching
    pub enable_gpu_cache: bool,
    /// GPU cache size in MB
    pub cache_size_mb: usize,
    /// Enable compression for stored data
    pub enable_compression: bool,
    /// Sync interval in milliseconds
    pub sync_interval_ms: u64,
    /// Enable GPUDirect Storage
    pub enable_gpudirect: bool,
}

impl Default for GpuStorageConfig {
    fn default() -> Self {
        let base_path = PathBuf::from("/magikdev/gpu");
        Self {
            cache_path: base_path.join("cache"),
            wal_path: base_path.join("wal"),
            base_path,
            enable_gpu_cache: true,
            cache_size_mb: 1024, // 1GB default
            enable_compression: true,
            sync_interval_ms: 100,
            enable_gpudirect: false, // Disabled by default, enable when available
        }
    }
}

impl GpuStorageConfig {
    /// Create production configuration using /magikdev/gpu
    pub fn production() -> Self {
        Self::default()
    }

    /// Create development configuration using local directory
    pub fn development() -> Self {
        Self {
            base_path: PathBuf::from("./gpu_storage"),
            cache_path: PathBuf::from("./gpu_storage/cache"),
            wal_path: PathBuf::from("./gpu_storage/wal"),
            ..Default::default()
        }
    }

    /// Create configuration with custom base path
    pub fn with_base_path<P: Into<PathBuf>>(path: P) -> Self {
        let base_path = path.into();
        Self {
            cache_path: base_path.join("cache"),
            wal_path: base_path.join("wal"),
            base_path,
            ..Default::default()
        }
    }

    /// Create test configuration
    #[cfg(test)]
    pub fn test_config<P: Into<PathBuf>>(path: P) -> Self {
        let base_path = path.into();
        Self {
            cache_path: base_path.join("cache"),
            wal_path: base_path.join("wal"),
            base_path,
            enable_gpu_cache: true,
            cache_size_mb: 128,        // Smaller for tests
            enable_compression: false, // Faster tests
            sync_interval_ms: 10,
        }
    }
}

/// GPU agent data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAgentData {
    /// Agent unique identifier
    pub id: String,
    /// Neural network state vector
    pub state: Vec<f32>,
    /// Agent memory buffer
    pub memory: Vec<f32>,
    /// Generation number
    pub generation: u64,
    /// Fitness score
    pub fitness: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl GpuAgentData {
    /// Create random agent data for testing
    #[cfg(test)]
    pub fn random(id: &str) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        Self {
            id: id.to_string(),
            state: (0..256).map(|_| rng.gen()).collect(),
            memory: (0..128).map(|_| rng.gen()).collect(),
            generation: rng.gen_range(0..100),
            fitness: rng.gen(),
            metadata: HashMap::new(),
        }
    }
}

/// GPU knowledge graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuKnowledgeGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

/// Graph node with embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

/// Graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub weight: f32,
    pub edge_type: String,
}

/// Swarm data for GPU memory mapping
#[derive(Debug, Clone)]
pub struct SwarmData {
    pub agent_count: usize,
    pub state_dimension: usize,
    pub data: Vec<f32>,
}

/// GPU memory handle for mapped data
pub struct GpuMemoryHandle {
    size: usize,
    mapped: bool,
}

impl GpuMemoryHandle {
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    pub fn size_bytes(&self) -> usize {
        self.size
    }
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub cached_agents: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_size_bytes: usize,
}

/// GPU agent storage implementation
pub struct GpuAgentStorage {
    pub(crate) config: GpuStorageConfig,
    nvme_storage: Arc<NvmeStorage>,
    agent_cache: Arc<RwLock<HashMap<String, GpuAgentData>>>,
    cache_stats: Arc<RwLock<CacheStats>>,
    pub(crate) gds_manager: Option<crate::gpudirect::GpuDirectManager>,
    initialized: bool,
}

impl GpuAgentStorage {
    /// Create new GPU agent storage
    pub fn new(config: GpuStorageConfig) -> Result<Self> {
        // Create directories
        std::fs::create_dir_all(&config.base_path)
            .context("Failed to create base storage directory")?;
        std::fs::create_dir_all(&config.cache_path).context("Failed to create cache directory")?;
        std::fs::create_dir_all(&config.wal_path).context("Failed to create WAL directory")?;

        // Configure NVMe storage
        let nvme_config = NvmeConfig {
            base_path: config.base_path.clone(),
            block_size: 4096,
            cache_size: config.cache_size_mb * 1024 * 1024,
            sync_writes: true,
        };

        let nvme_storage = futures::executor::block_on(NvmeStorage::with_config(nvme_config))?;

        // Initialize GPUDirect Storage if enabled
        let gds_manager = if config.enable_gpudirect {
            match crate::gpudirect::GpuDirectManager::new(
                crate::gpudirect::GpuDirectConfig::default(),
            ) {
                Ok(manager) => Some(manager),
                Err(e) => {
                    eprintln!(
                        "Failed to initialize GPUDirect Storage: {}, using fallback",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            nvme_storage: Arc::new(nvme_storage),
            agent_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_stats: Arc::new(RwLock::new(CacheStats::default())),
            gds_manager,
            initialized: true,
        })
    }

    /// Check if storage is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Store agent data
    pub async fn store_agent(&self, agent_id: &str, data: &GpuAgentData) -> Result<()> {
        // Serialize agent data
        let serialized = bincode::serialize(data).context("Failed to serialize agent data")?;

        // Store to NVMe
        let key = format!("agent/{}", agent_id);
        self.nvme_storage
            .store(&key, &serialized)
            .await
            .context("Failed to store agent data")?;

        // Update cache if enabled
        if self.config.enable_gpu_cache {
            let mut cache = self.agent_cache.write().await;
            cache.insert(agent_id.to_string(), data.clone());

            let mut stats = self.cache_stats.write().await;
            stats.cached_agents = cache.len();
        }

        Ok(())
    }

    /// Retrieve agent data
    pub async fn retrieve_agent(&self, agent_id: &str) -> Result<GpuAgentData> {
        // Check cache first
        if self.config.enable_gpu_cache {
            let cache = self.agent_cache.read().await;
            if let Some(data) = cache.get(agent_id) {
                let mut stats = self.cache_stats.write().await;
                stats.cache_hits += 1;
                return Ok(data.clone());
            }
        }

        // Cache miss - retrieve from storage
        let mut stats = self.cache_stats.write().await;
        stats.cache_misses += 1;
        drop(stats);

        let key = format!("agent/{}", agent_id);
        let data = self
            .nvme_storage
            .retrieve(&key)
            .await
            .context("Failed to retrieve agent data")?;

        let agent_data: GpuAgentData =
            bincode::deserialize(&data).context("Failed to deserialize agent data")?;

        Ok(agent_data)
    }

    /// Cache an agent for fast access
    pub async fn cache_agent(&self, agent_id: &str) -> Result<()> {
        let agent_data = self.retrieve_agent(agent_id).await?;

        let mut cache = self.agent_cache.write().await;
        cache.insert(agent_id.to_string(), agent_data);

        let mut stats = self.cache_stats.write().await;
        stats.cached_agents = cache.len();

        Ok(())
    }

    /// Retrieve agent from cache only
    pub async fn retrieve_agent_cached(&self, agent_id: &str) -> Result<GpuAgentData> {
        let cache = self.agent_cache.read().await;
        cache
            .get(agent_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Agent not in cache"))
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> Result<CacheStats> {
        let stats = self.cache_stats.read().await;
        Ok((*stats).clone())
    }

    /// Store knowledge graph
    pub async fn store_knowledge_graph(
        &self,
        graph_id: &str,
        graph: &GpuKnowledgeGraph,
    ) -> Result<()> {
        let serialized =
            bincode::serialize(graph).context("Failed to serialize knowledge graph")?;

        let key = format!("graph/{}", graph_id);
        self.nvme_storage
            .store(&key, &serialized)
            .await
            .context("Failed to store knowledge graph")?;

        Ok(())
    }

    /// Retrieve knowledge graph
    pub async fn retrieve_knowledge_graph(&self, graph_id: &str) -> Result<GpuKnowledgeGraph> {
        let key = format!("graph/{}", graph_id);
        let data = self
            .nvme_storage
            .retrieve(&key)
            .await
            .context("Failed to retrieve knowledge graph")?;

        let graph: GpuKnowledgeGraph =
            bincode::deserialize(&data).context("Failed to deserialize knowledge graph")?;

        Ok(graph)
    }

    /// Map swarm data to GPU memory
    pub async fn map_to_gpu_memory(
        &self,
        swarm_id: &str,
        swarm_data: &SwarmData,
    ) -> Result<GpuMemoryHandle> {
        // In a real implementation, this would use CUDA memory mapping
        // For now, we simulate the mapping
        let size_bytes = swarm_data.data.len() * std::mem::size_of::<f32>();

        // Store swarm data for persistence
        let key = format!("swarm/{}", swarm_id);
        let serialized =
            bincode::serialize(&swarm_data.data).context("Failed to serialize swarm data")?;

        self.nvme_storage
            .store(&key, &serialized)
            .await
            .context("Failed to store swarm data")?;

        Ok(GpuMemoryHandle {
            size: size_bytes,
            mapped: true,
        })
    }

    /// Check if agent exists
    pub async fn agent_exists(&self, agent_id: &str) -> Result<bool> {
        let key = format!("agent/{}", agent_id);
        let keys = self.nvme_storage.list_keys("agent/").await?;
        Ok(keys.contains(&key))
    }

    /// Get agent file path
    pub(crate) fn get_agent_path(&self, agent_id: &str) -> PathBuf {
        self.config
            .base_path
            .join("agents")
            .join(format!("{}.bin", agent_id))
    }

    /// Load agent data (compatibility method)
    pub async fn load_agent(&self, agent_id: &str) -> Result<Option<GpuAgentData>> {
        match self.retrieve_agent(agent_id).await {
            Ok(data) => Ok(Some(data)),
            Err(_) => Ok(None),
        }
    }
}

#[cfg(test)]
#[path = "storage_tests.rs"]
mod storage_tests;
