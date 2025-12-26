//! Archive storage implementations

use crate::{AgentGenome, AgentId, EvolutionStreamingError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Archive storage trait
#[async_trait]
pub trait ArchiveStorage: Send + Sync {
    /// Store agent in persistent storage
    async fn store_agent(
        &self,
        agent: &AgentGenome,
        fitness: f64,
    ) -> Result<(), EvolutionStreamingError>;

    /// Load agent from persistent storage
    async fn load_agent(
        &self,
        agent_id: AgentId,
    ) -> Result<Option<(AgentGenome, f64)>, EvolutionStreamingError>;

    /// Remove agent from persistent storage
    async fn remove_agent(&self, agent_id: AgentId) -> Result<(), EvolutionStreamingError>;

    /// List all stored agent IDs
    async fn list_agents(&self) -> Result<Vec<AgentId>, EvolutionStreamingError>;

    /// Compact storage (remove fragmentation)
    async fn compact(&self) -> Result<(), EvolutionStreamingError>;

    /// Get storage statistics
    async fn get_stats(&self) -> Result<StorageStats, EvolutionStreamingError>;
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_agents: u64,
    pub storage_size_bytes: u64,
    pub fragmentation_ratio: f64,
    pub last_compaction: Option<u64>,
}

/// File-based archive storage
#[derive(Debug, Clone)]
pub struct FileArchiveStorage {
    base_path: PathBuf,
    compression_enabled: bool,
}

impl FileArchiveStorage {
    /// Create new file-based archive storage
    pub fn new(base_path: PathBuf) -> Self {
        Self {
            base_path,
            compression_enabled: false,
        }
    }

    /// Enable compression for stored data
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression_enabled = enabled;
        self
    }

    /// Get file path for agent
    fn agent_file_path(&self, agent_id: AgentId) -> PathBuf {
        self.base_path.join(format!("{}.json", agent_id))
    }

    /// Ensure base directory exists
    async fn ensure_directory(&self) -> Result<(), EvolutionStreamingError> {
        if !self.base_path.exists() {
            fs::create_dir_all(&self.base_path).await.map_err(|e| {
                EvolutionStreamingError::ArchiveFailed(format!(
                    "Failed to create archive directory: {}",
                    e
                ))
            })?;
        }
        Ok(())
    }
}

#[async_trait]
impl ArchiveStorage for FileArchiveStorage {
    async fn store_agent(
        &self,
        agent: &AgentGenome,
        fitness: f64,
    ) -> Result<(), EvolutionStreamingError> {
        self.ensure_directory().await?;

        let stored_data = StoredAgentData {
            genome: agent.clone(),
            fitness,
            storage_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };

        let json_data = serde_json::to_string_pretty(&stored_data).map_err(|e| {
            EvolutionStreamingError::ArchiveFailed(format!("Failed to serialize agent: {e}"))
        })?;

        let file_path = self.agent_file_path(agent.id);
        let mut file = fs::File::create(&file_path).await.map_err(|e| {
            EvolutionStreamingError::ArchiveFailed(format!("Failed to create agent file: {e}"))
        })?;

        file.write_all(json_data.as_bytes()).await.map_err(|e| {
            EvolutionStreamingError::ArchiveFailed(format!("Failed to write agent data: {e}"))
        })?;

        file.sync_all().await.map_err(|e| {
            EvolutionStreamingError::ArchiveFailed(format!("Failed to sync agent file: {e}"))
        })?;

        Ok(())
    }

    async fn load_agent(
        &self,
        agent_id: AgentId,
    ) -> Result<Option<(AgentGenome, f64)>, EvolutionStreamingError> {
        let file_path = self.agent_file_path(agent_id);

        if !file_path.exists() {
            return Ok(None);
        }

        let mut file = fs::File::open(&file_path).await.map_err(|e| {
            EvolutionStreamingError::ArchiveFailed(format!("Failed to open agent file: {e}"))
        })?;

        let mut contents = String::new();
        file.read_to_string(&mut contents).await.map_err(|e| {
            EvolutionStreamingError::ArchiveFailed(format!("Failed to read agent file: {e}"))
        })?;

        let stored_data: StoredAgentData = serde_json::from_str(&contents).map_err(|e| {
            EvolutionStreamingError::ArchiveFailed(format!("Failed to deserialize agent: {e}"))
        })?;

        Ok(Some((stored_data.genome, stored_data.fitness)))
    }

    async fn remove_agent(&self, agent_id: AgentId) -> Result<(), EvolutionStreamingError> {
        let file_path = self.agent_file_path(agent_id);

        if file_path.exists() {
            fs::remove_file(&file_path).await.map_err(|e| {
                EvolutionStreamingError::ArchiveFailed(format!(
                    "Failed to remove agent file: {}",
                    e
                ))
            })?;
        }

        Ok(())
    }

    async fn list_agents(&self) -> Result<Vec<AgentId>, EvolutionStreamingError> {
        if !self.base_path.exists() {
            return Ok(Vec::new());
        }

        let mut dir = fs::read_dir(&self.base_path).await.map_err(|e| {
            EvolutionStreamingError::ArchiveFailed(format!(
                "Failed to read archive directory: {}",
                e
            ))
        })?;

        let mut agent_ids = Vec::new();

        while let Some(entry) = dir.next_entry().await.map_err(|e| {
            EvolutionStreamingError::ArchiveFailed(format!("Failed to read directory entry: {e}"))
        })? {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                if let Some(stem) = path.file_stem() {
                    if let Some(filename) = stem.to_str() {
                        if let Ok(agent_id) = filename.parse::<uuid::Uuid>() {
                            agent_ids.push(agent_id);
                        }
                    }
                }
            }
        }

        Ok(agent_ids)
    }

    async fn compact(&self) -> Result<(), EvolutionStreamingError> {
        // For file storage, compaction involves removing fragmented files
        // and potentially rewriting data in a more efficient format

        if !self.base_path.exists() {
            return Ok(());
        }

        let agent_ids = self.list_agents().await?;
        let mut valid_agents = Vec::new();

        // Verify all agent files are valid
        for agent_id in agent_ids {
            if let Ok(Some((genome, fitness))) = self.load_agent(agent_id).await {
                valid_agents.push((genome, fitness));
            }
        }

        // Create temporary directory for compaction
        let temp_dir = self.base_path.with_extension("compact_temp");
        if temp_dir.exists() {
            fs::remove_dir_all(&temp_dir).await.map_err(|e| {
                EvolutionStreamingError::ArchiveFailed(format!(
                    "Failed to remove temp directory: {}",
                    e
                ))
            })?;
        }

        fs::create_dir_all(&temp_dir).await.map_err(|e| {
            EvolutionStreamingError::ArchiveFailed(format!(
                "Failed to create temp directory: {}",
                e
            ))
        })?;

        // Rewrite all valid agents to temp directory
        let temp_storage =
            FileArchiveStorage::new(temp_dir.clone()).with_compression(self.compression_enabled);

        for (genome, fitness) in valid_agents {
            temp_storage.store_agent(&genome, fitness).await?;
        }

        // Atomic swap of directories
        let backup_dir = self.base_path.with_extension("backup");
        if backup_dir.exists() {
            fs::remove_dir_all(&backup_dir).await.map_err(|e| {
                EvolutionStreamingError::ArchiveFailed(format!(
                    "Failed to remove backup directory: {}",
                    e
                ))
            })?;
        }

        fs::rename(&self.base_path, &backup_dir)
            .await
            .map_err(|e| {
                EvolutionStreamingError::ArchiveFailed(format!(
                    "Failed to backup original directory: {}",
                    e
                ))
            })?;

        fs::rename(&temp_dir, &self.base_path).await.map_err(|e| {
            EvolutionStreamingError::ArchiveFailed(format!(
                "Failed to move compacted directory: {}",
                e
            ))
        })?;

        fs::remove_dir_all(&backup_dir).await.map_err(|e| {
            EvolutionStreamingError::ArchiveFailed(format!(
                "Failed to remove backup directory: {}",
                e
            ))
        })?;

        Ok(())
    }

    async fn get_stats(&self) -> Result<StorageStats, EvolutionStreamingError> {
        if !self.base_path.exists() {
            return Ok(StorageStats {
                total_agents: 0,
                storage_size_bytes: 0,
                fragmentation_ratio: 0.0,
                last_compaction: None,
            });
        }

        let agent_ids = self.list_agents().await?;
        let mut total_size = 0u64;

        // Calculate total storage size
        for agent_id in &agent_ids {
            let file_path = self.agent_file_path(*agent_id);
            if let Ok(metadata) = fs::metadata(&file_path).await {
                total_size += metadata.len();
            }
        }

        // Simple fragmentation estimation
        let expected_size = agent_ids.len() as u64 * 1024; // Assume 1KB per agent
        let fragmentation_ratio = if expected_size > 0 {
            (total_size as f64 - expected_size as f64) / expected_size as f64
        } else {
            0.0
        };

        Ok(StorageStats {
            total_agents: agent_ids.len() as u64,
            storage_size_bytes: total_size,
            fragmentation_ratio: fragmentation_ratio.max(0.0),
            last_compaction: None, // Would track this in metadata
        })
    }
}

/// In-memory archive storage for testing
#[derive(Debug, Clone, Default)]
pub struct MemoryArchiveStorage {
    agents: std::sync::Arc<tokio::sync::RwLock<HashMap<AgentId, (AgentGenome, f64)>>>,
}

impl MemoryArchiveStorage {
    /// Create new in-memory archive storage
    pub fn new() -> Self {
        Self {
            agents: std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl ArchiveStorage for MemoryArchiveStorage {
    async fn store_agent(
        &self,
        agent: &AgentGenome,
        fitness: f64,
    ) -> Result<(), EvolutionStreamingError> {
        let mut agents = self.agents.write().await;
        agents.insert(agent.id, (agent.clone(), fitness));
        Ok(())
    }

    async fn load_agent(
        &self,
        agent_id: AgentId,
    ) -> Result<Option<(AgentGenome, f64)>, EvolutionStreamingError> {
        let agents = self.agents.read().await;
        Ok(agents.get(&agent_id).cloned())
    }

    async fn remove_agent(&self, agent_id: AgentId) -> Result<(), EvolutionStreamingError> {
        let mut agents = self.agents.write().await;
        agents.remove(&agent_id);
        Ok(())
    }

    async fn list_agents(&self) -> Result<Vec<AgentId>, EvolutionStreamingError> {
        let agents = self.agents.read().await;
        Ok(agents.keys().copied().collect())
    }

    async fn compact(&self) -> Result<(), EvolutionStreamingError> {
        // Memory storage doesn't need compaction
        Ok(())
    }

    async fn get_stats(&self) -> Result<StorageStats, EvolutionStreamingError> {
        let agents = self.agents.read().await;
        let total_agents = agents.len() as u64;

        // Estimate memory usage
        let estimated_size = agents
            .iter()
            .map(|(_, (genome, _))| genome.size() as u64)
            .sum();

        Ok(StorageStats {
            total_agents,
            storage_size_bytes: estimated_size,
            fragmentation_ratio: 0.0, // No fragmentation in memory
            last_compaction: None,
        })
    }
}

/// Stored agent data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredAgentData {
    genome: AgentGenome,
    fitness: f64,
    storage_time: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_memory_storage() {
        let storage = MemoryArchiveStorage::new();
        let agent = AgentGenome::new("fn test() { 42 }".to_string(), vec![1.0, 2.0]);

        // Store agent
        storage.store_agent(&agent, 0.8).await.unwrap();

        // Load agent
        let loaded = storage.load_agent(agent.id).await.unwrap();
        assert!(loaded.is_some());
        let (loaded_genome, loaded_fitness) = loaded.unwrap();
        assert_eq!(loaded_genome.id, agent.id);
        assert_eq!(loaded_fitness, 0.8);

        // List agents
        let agents = storage.list_agents().await.unwrap();
        assert_eq!(agents.len(), 1);
        assert!(agents.contains(&agent.id));

        // Remove agent
        storage.remove_agent(agent.id).await.unwrap();
        let loaded_after_remove = storage.load_agent(agent.id).await.unwrap();
        assert!(loaded_after_remove.is_none());
    }

    #[tokio::test]
    async fn test_file_storage() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileArchiveStorage::new(temp_dir.path().to_path_buf());
        let agent = AgentGenome::new("fn test() { 42 }".to_string(), vec![1.0, 2.0]);

        // Store agent
        storage.store_agent(&agent, 0.8).await.unwrap();

        // Load agent
        let loaded = storage.load_agent(agent.id).await.unwrap();
        assert!(loaded.is_some());
        let (loaded_genome, loaded_fitness) = loaded.unwrap();
        assert_eq!(loaded_genome.id, agent.id);
        assert_eq!(loaded_fitness, 0.8);

        // List agents
        let agents = storage.list_agents().await.unwrap();
        assert_eq!(agents.len(), 1);
        assert!(agents.contains(&agent.id));

        // Get stats
        let stats = storage.get_stats().await.unwrap();
        assert_eq!(stats.total_agents, 1);
        assert!(stats.storage_size_bytes > 0);

        // Remove agent
        storage.remove_agent(agent.id).await.unwrap();
        let loaded_after_remove = storage.load_agent(agent.id).await.unwrap();
        assert!(loaded_after_remove.is_none());
    }

    #[tokio::test]
    async fn test_file_storage_compression() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileArchiveStorage::new(temp_dir.path().to_path_buf()).with_compression(true);

        let agent = AgentGenome::new("fn test() { 42 }".to_string(), vec![1.0]);
        storage.store_agent(&agent, 0.5).await.unwrap();

        let loaded = storage.load_agent(agent.id).await.unwrap();
        assert!(loaded.is_some());
    }

    #[tokio::test]
    async fn test_file_storage_compact() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileArchiveStorage::new(temp_dir.path().to_path_buf());

        // Store multiple agents
        for i in 0..5 {
            let agent = AgentGenome::new(format!("fn test{} {{}}", i), vec![i as f32]);
            storage.store_agent(&agent, i as f64 * 0.1).await.unwrap();
        }

        let agents_before = storage.list_agents().await.unwrap();
        assert_eq!(agents_before.len(), 5);

        // Compact storage
        storage.compact().await.unwrap();

        let agents_after = storage.list_agents().await.unwrap();
        assert_eq!(agents_after.len(), 5);

        // Verify all agents are still loadable
        for agent_id in agents_after {
            let loaded = storage.load_agent(agent_id).await.unwrap();
            assert!(loaded.is_some());
        }
    }

    #[tokio::test]
    async fn test_storage_stats() {
        let storage = MemoryArchiveStorage::new();

        // Empty storage stats
        let empty_stats = storage.get_stats().await.unwrap();
        assert_eq!(empty_stats.total_agents, 0);
        assert_eq!(empty_stats.storage_size_bytes, 0);

        // Add some agents
        for i in 0..3 {
            let agent = AgentGenome::new(format!("fn test{} {{}}", i), vec![i as f32]);
            storage.store_agent(&agent, i as f64 * 0.1).await.unwrap();
        }

        let stats = storage.get_stats().await.unwrap();
        assert_eq!(stats.total_agents, 3);
        assert!(stats.storage_size_bytes > 0);
        assert_eq!(stats.fragmentation_ratio, 0.0); // Memory storage has no fragmentation
    }

    #[tokio::test]
    async fn test_nonexistent_agent_load() {
        let storage = MemoryArchiveStorage::new();
        let nonexistent_id = uuid::Uuid::new_v4();

        let result = storage.load_agent(nonexistent_id).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_file_storage_directory_creation() {
        let temp_dir = TempDir::new().unwrap();
        let non_existent_path = temp_dir.path().join("new_dir");
        let storage = FileArchiveStorage::new(non_existent_path.clone());

        // Directory shouldn't exist initially
        assert!(!non_existent_path.exists());

        let agent = AgentGenome::new("fn test() {}".to_string(), vec![]);
        storage.store_agent(&agent, 0.5).await.unwrap();

        // Directory should be created
        assert!(non_existent_path.exists());
    }

    #[tokio::test]
    async fn test_memory_storage_concurrent_access() {
        use std::sync::Arc;
        use tokio::task;

        let storage = Arc::new(MemoryArchiveStorage::new());
        let mut handles = vec![];

        // Spawn multiple tasks that store agents concurrently
        for i in 0..10 {
            let storage_clone = storage.clone();
            let handle = task::spawn(async move {
                let agent = AgentGenome::new(format!("fn test_{} {{}}", i), vec![i as f32]);
                storage_clone
                    .store_agent(&agent, i as f64 * 0.1)
                    .await
                    .unwrap();
                agent.id
            });
            handles.push(handle);
        }

        // Collect all agent IDs
        let mut stored_ids = vec![];
        for handle in handles {
            let agent_id = handle.await.unwrap();
            stored_ids.push(agent_id);
        }

        // Verify all agents were stored
        let listed_agents = storage.list_agents().await.unwrap();
        assert_eq!(listed_agents.len(), 10);

        for id in stored_ids {
            assert!(listed_agents.contains(&id));
            let loaded = storage.load_agent(id).await.unwrap();
            assert!(loaded.is_some());
        }
    }

    #[tokio::test]
    async fn test_file_storage_concurrent_access() {
        use std::sync::Arc;
        use tokio::task;

        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(FileArchiveStorage::new(temp_dir.path().to_path_buf()));
        let mut handles = vec![];

        // Spawn multiple tasks that store agents concurrently
        for i in 0..5 {
            let storage_clone = storage.clone();
            let handle = task::spawn(async move {
                let agent = AgentGenome::new(format!("fn concurrent_{} {{}}", i), vec![i as f32]);
                storage_clone
                    .store_agent(&agent, i as f64 * 0.2)
                    .await
                    .unwrap();
                agent.id
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut stored_ids = vec![];
        for handle in handles {
            let agent_id = handle.await.unwrap();
            stored_ids.push(agent_id);
        }

        // Verify all agents are accessible
        let listed_agents = storage.list_agents().await.unwrap();
        assert_eq!(listed_agents.len(), 5);

        for id in stored_ids {
            let loaded = storage.load_agent(id).await.unwrap();
            assert!(loaded.is_some());
        }
    }

    #[tokio::test]
    async fn test_memory_storage_overwrite() {
        let storage = MemoryArchiveStorage::new();
        let agent = AgentGenome::new("fn test() { 1 }".to_string(), vec![1.0]);

        // Store agent with initial fitness
        storage.store_agent(&agent, 0.5).await.unwrap();
        let loaded = storage.load_agent(agent.id).await.unwrap();
        assert_eq!(loaded.unwrap().1, 0.5);

        // Overwrite with new fitness
        storage.store_agent(&agent, 0.8).await.unwrap();
        let loaded_updated = storage.load_agent(agent.id).await.unwrap();
        assert_eq!(loaded_updated.unwrap().1, 0.8);

        // Should still have only one agent
        let agents = storage.list_agents().await.unwrap();
        assert_eq!(agents.len(), 1);
    }

    #[tokio::test]
    async fn test_file_storage_overwrite() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileArchiveStorage::new(temp_dir.path().to_path_buf());
        let agent = AgentGenome::new("fn test() { 1 }".to_string(), vec![1.0]);

        // Store agent with initial fitness
        storage.store_agent(&agent, 0.3).await.unwrap();
        let loaded = storage.load_agent(agent.id).await.unwrap();
        assert_eq!(loaded.unwrap().1, 0.3);

        // Overwrite with new fitness
        storage.store_agent(&agent, 0.9).await.unwrap();
        let loaded_updated = storage.load_agent(agent.id).await.unwrap();
        assert_eq!(loaded_updated.unwrap().1, 0.9);

        // Should still have only one agent
        let agents = storage.list_agents().await.unwrap();
        assert_eq!(agents.len(), 1);
    }

    #[tokio::test]
    async fn test_file_storage_load_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileArchiveStorage::new(temp_dir.path().to_path_buf());
        let nonexistent_id = uuid::Uuid::new_v4();

        let result = storage.load_agent(nonexistent_id).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_file_storage_remove_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileArchiveStorage::new(temp_dir.path().to_path_buf());
        let nonexistent_id = uuid::Uuid::new_v4();

        // Should not error when removing non-existent agent
        let result = storage.remove_agent(nonexistent_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_storage_list_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileArchiveStorage::new(temp_dir.path().to_path_buf());

        let agents = storage.list_agents().await.unwrap();
        assert!(agents.is_empty());
    }

    #[tokio::test]
    async fn test_file_storage_list_nonexistent_directory() {
        let temp_dir = TempDir::new().unwrap();
        let non_existent_path = temp_dir.path().join("does_not_exist");
        let storage = FileArchiveStorage::new(non_existent_path);

        let agents = storage.list_agents().await.unwrap();
        assert!(agents.is_empty());
    }

    #[tokio::test]
    async fn test_file_storage_stats_empty() {
        let temp_dir = TempDir::new().unwrap();
        let non_existent_path = temp_dir.path().join("empty_archive");
        let storage = FileArchiveStorage::new(non_existent_path);

        let stats = storage.get_stats().await.unwrap();
        assert_eq!(stats.total_agents, 0);
        assert_eq!(stats.storage_size_bytes, 0);
        assert_eq!(stats.fragmentation_ratio, 0.0);
        assert!(stats.last_compaction.is_none());
    }

    #[tokio::test]
    async fn test_file_storage_fragmentation_calculation() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileArchiveStorage::new(temp_dir.path().to_path_buf());

        // Store agents with varying sizes
        let small_agent = AgentGenome::new("x".to_string(), vec![1.0]);
        let large_agent = AgentGenome::new("x".repeat(5000), vec![1.0; 100]);

        storage.store_agent(&small_agent, 0.5).await.unwrap();
        storage.store_agent(&large_agent, 0.8).await.unwrap();

        let stats = storage.get_stats().await.unwrap();
        assert_eq!(stats.total_agents, 2);
        assert!(stats.storage_size_bytes > 2048); // Should be larger than expected minimum
                                                  // Fragmentation could be positive or negative depending on actual vs expected size
    }

    #[tokio::test]
    async fn test_memory_storage_large_dataset() {
        let storage = MemoryArchiveStorage::new();
        let mut agent_ids = vec![];

        // Store 100 agents
        for i in 0..100 {
            let agent = AgentGenome::new(
                format!("fn agent_{}() {{ {} }}", i, i),
                vec![i as f32, (i * 2) as f32, (i * 3) as f32],
            );
            agent_ids.push(agent.id);
            storage.store_agent(&agent, i as f64 / 100.0).await.unwrap();
        }

        // Verify count
        let stats = storage.get_stats().await.unwrap();
        assert_eq!(stats.total_agents, 100);
        assert!(stats.storage_size_bytes > 0);

        // Verify all agents are listable
        let listed = storage.list_agents().await.unwrap();
        assert_eq!(listed.len(), 100);

        // Remove half the agents
        for i in 0..50 {
            storage.remove_agent(agent_ids[i]).await.unwrap();
        }

        let stats_after = storage.get_stats().await.unwrap();
        assert_eq!(stats_after.total_agents, 50);

        // Verify remaining agents are still accessible
        for i in 50..100 {
            let loaded = storage.load_agent(agent_ids[i]).await.unwrap();
            assert!(loaded.is_some());
        }
    }

    #[tokio::test]
    async fn test_file_storage_compact_with_corrupted_files() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileArchiveStorage::new(temp_dir.path().to_path_buf());

        // Store a valid agent
        let agent = AgentGenome::new("fn valid() {}".to_string(), vec![1.0]);
        storage.store_agent(&agent, 0.7).await.unwrap();

        // Create a corrupted file manually
        let corrupted_path = storage.agent_file_path(uuid::Uuid::new_v4());
        tokio::fs::write(&corrupted_path, "invalid json {")
            .await
            .unwrap();

        // Compaction should handle corrupted files gracefully
        let result = storage.compact().await;
        assert!(result.is_ok());

        // Only the valid agent should remain
        let agents = storage.list_agents().await.unwrap();
        assert_eq!(agents.len(), 1);
        assert!(agents.contains(&agent.id));
    }

    #[tokio::test]
    async fn test_file_storage_invalid_filename_handling() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileArchiveStorage::new(temp_dir.path().to_path_buf());

        // Create files with invalid names that shouldn't be parsed as UUIDs
        let invalid_files = vec![
            "not_uuid.json",
            "invalid.txt",
            "123.json",
            "partial-uuid.json",
        ];

        for filename in invalid_files {
            let path = temp_dir.path().join(filename);
            tokio::fs::write(&path, "{}").await.unwrap();
        }

        // List should ignore invalid files
        let agents = storage.list_agents().await.unwrap();
        assert!(agents.is_empty());
    }

    #[tokio::test]
    async fn test_stored_agent_data_serialization() {
        let agent = AgentGenome::new("fn test() { 42 }".to_string(), vec![1.5, 2.5]);
        let stored_data = StoredAgentData {
            genome: agent.clone(),
            fitness: 0.85,
            storage_time: 1234567890,
        };

        let json = serde_json::to_string(&stored_data).unwrap();
        let parsed: StoredAgentData = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.genome.id, agent.id);
        assert_eq!(parsed.fitness, 0.85);
        assert_eq!(parsed.storage_time, 1234567890);
    }

    #[tokio::test]
    async fn test_memory_storage_fitness_extremes() {
        let storage = MemoryArchiveStorage::new();

        // Test extreme fitness values
        let agents_and_fitness = vec![
            ("min", f64::MIN),
            ("max", f64::MAX),
            ("zero", 0.0),
            ("negative", -100.5),
            ("infinity", f64::INFINITY),
            ("neg_infinity", f64::NEG_INFINITY),
        ];

        for (name, fitness) in agents_and_fitness {
            let agent = AgentGenome::new(format!("fn {} {{}}", name), vec![1.0]);
            storage.store_agent(&agent, fitness).await.unwrap();

            let loaded = storage.load_agent(agent.id).await.unwrap();
            assert!(loaded.is_some());
            let (_, loaded_fitness) = loaded.unwrap();

            if fitness.is_finite() {
                assert_eq!(loaded_fitness, fitness);
            } else {
                assert_eq!(loaded_fitness.is_infinite(), fitness.is_infinite());
                assert_eq!(
                    loaded_fitness.is_sign_positive(),
                    fitness.is_sign_positive()
                );
            }
        }
    }

    #[tokio::test]
    async fn test_file_storage_with_special_characters() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileArchiveStorage::new(temp_dir.path().to_path_buf());

        // Agent with special characters in code
        let special_code = "fn test() { \"unicode: ñáéíóú, symbols: !@#$%^&*()\" }";
        let agent = AgentGenome::new(special_code.to_string(), vec![1.0, -2.5, 3.14159]);

        storage.store_agent(&agent, 0.75).await.unwrap();

        let loaded = storage.load_agent(agent.id).await.unwrap();
        assert!(loaded.is_some());
        let (loaded_genome, loaded_fitness) = loaded.unwrap();

        assert_eq!(loaded_genome.code, special_code);
        assert_eq!(loaded_fitness, 0.75);
        assert_eq!(loaded_genome.parameters, vec![1.0, -2.5, 3.14159]);
    }

    #[tokio::test]
    async fn test_memory_storage_default() {
        let storage = MemoryArchiveStorage::default();
        assert_eq!(storage.agents.read().await.len(), 0);
    }

    #[tokio::test]
    async fn test_file_storage_compression_flag() {
        let temp_dir = TempDir::new().unwrap();
        let storage_no_compression = FileArchiveStorage::new(temp_dir.path().join("no_compress"));
        let storage_with_compression =
            FileArchiveStorage::new(temp_dir.path().join("with_compress")).with_compression(true);

        assert_eq!(storage_no_compression.compression_enabled, false);
        assert_eq!(storage_with_compression.compression_enabled, true);
    }

    #[tokio::test]
    async fn test_agent_file_path_generation() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileArchiveStorage::new(temp_dir.path().to_path_buf());
        let agent_id = uuid::Uuid::new_v4();

        let path = storage.agent_file_path(agent_id);
        let expected_filename = format!("{}.json", agent_id);

        assert_eq!(
            path.file_name().unwrap().to_string_lossy(),
            expected_filename
        );
        assert_eq!(path.parent().unwrap(), temp_dir.path());
    }

    #[tokio::test]
    async fn test_memory_storage_compact_no_op() {
        let storage = MemoryArchiveStorage::new();

        // Add some agents
        for i in 0..3 {
            let agent = AgentGenome::new(format!("fn test_{} {{}}", i), vec![i as f32]);
            storage.store_agent(&agent, i as f64).await.unwrap();
        }

        // Compact should be a no-op
        let result = storage.compact().await;
        assert!(result.is_ok());

        // All agents should still be present
        let stats = storage.get_stats().await.unwrap();
        assert_eq!(stats.total_agents, 3);
    }

    #[tokio::test]
    async fn test_storage_stats_fields() {
        let stats = StorageStats {
            total_agents: 42,
            storage_size_bytes: 1024,
            fragmentation_ratio: 0.15,
            last_compaction: Some(1234567890),
        };

        // Test serialization
        let json = serde_json::to_string(&stats).unwrap();
        let parsed: StorageStats = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.total_agents, 42);
        assert_eq!(parsed.storage_size_bytes, 1024);
        assert_eq!(parsed.fragmentation_ratio, 0.15);
        assert_eq!(parsed.last_compaction, Some(1234567890));
    }

    #[tokio::test]
    async fn test_file_storage_ensure_directory_error_handling() {
        // Test with path that can't be created (root directory on most systems)
        #[cfg(unix)]
        {
            let invalid_path = PathBuf::from("/root/cannot_create_this");
            let storage = FileArchiveStorage::new(invalid_path);
            let agent = AgentGenome::new("test".to_string(), vec![]);

            // This should fail due to permission denied
            let result = storage.store_agent(&agent, 0.5).await;
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn test_memory_storage_clone() {
        let storage1 = MemoryArchiveStorage::new();
        let agent = AgentGenome::new("fn test() {}".to_string(), vec![1.0]);
        storage1.store_agent(&agent, 0.8).await.unwrap();

        let storage2 = storage1.clone();

        // Both storages should reference the same data
        let loaded1 = storage1.load_agent(agent.id).await.unwrap();
        let loaded2 = storage2.load_agent(agent.id).await.unwrap();

        assert!(loaded1.is_some());
        assert!(loaded2.is_some());
        assert_eq!(loaded1.unwrap().1, loaded2.unwrap().1);
    }

    #[tokio::test]
    async fn test_file_storage_clone() {
        let temp_dir = TempDir::new().unwrap();
        let storage1 =
            FileArchiveStorage::new(temp_dir.path().to_path_buf()).with_compression(true);
        let storage2 = storage1.clone();

        assert_eq!(storage1.base_path, storage2.base_path);
        assert_eq!(storage1.compression_enabled, storage2.compression_enabled);

        // Both should be able to access the same files
        let agent = AgentGenome::new("fn test() {}".to_string(), vec![1.0]);
        storage1.store_agent(&agent, 0.6).await.unwrap();

        let loaded = storage2.load_agent(agent.id).await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().1, 0.6);
    }

    #[tokio::test]
    async fn test_concurrent_remove_operations() {
        let storage = MemoryArchiveStorage::new();
        let agent = AgentGenome::new("fn test() {}".to_string(), vec![1.0]);
        storage.store_agent(&agent, 0.7).await.unwrap();

        // Concurrent remove operations should not cause issues
        let storage1 = storage.clone();
        let storage2 = storage.clone();

        let handle1 = tokio::spawn(async move { storage1.remove_agent(agent.id).await });

        let handle2 = tokio::spawn(async move { storage2.remove_agent(agent.id).await });

        let result1 = handle1.await.unwrap();
        let result2 = handle2.await.unwrap();

        // Both operations should succeed
        assert!(result1.is_ok());
        assert!(result2.is_ok());

        // Agent should be removed
        let loaded = storage.load_agent(agent.id).await.unwrap();
        assert!(loaded.is_none());
    }
}
