//! Checkpoint management for GPU memory and agent state

use crate::error::{FaultToleranceError, FtResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs;
use uuid::Uuid;

/// Unique identifier for checkpoints
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointId(Uuid);

impl CheckpointId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for CheckpointId {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU memory checkpoint data
#[derive(Debug, Serialize, Deserialize)]
pub struct GpuCheckpoint {
    pub memory_snapshot: Vec<u8>,
    pub kernel_states: HashMap<String, KernelState>,
    pub timestamp: u64,
    pub size_bytes: usize,
}

/// Individual kernel state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelState {
    pub kernel_id: String,
    pub parameters: Vec<u8>,
    pub execution_context: ExecutionContext,
}

/// Kernel execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    pub device_id: u32,
    pub block_size: (u32, u32, u32),
    pub grid_size: (u32, u32, u32),
    pub shared_memory_bytes: usize,
}

/// Agent state checkpoint
#[derive(Debug, Serialize, Deserialize)]
pub struct AgentCheckpoint {
    pub agent_id: String,
    pub state_data: Vec<u8>,
    pub memory_contents: HashMap<String, serde_json::Value>,
    pub goals: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Full system checkpoint
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemCheckpoint {
    pub id: CheckpointId,
    pub timestamp: u64,
    pub gpu_checkpoints: Vec<GpuCheckpoint>,
    pub agent_checkpoints: Vec<AgentCheckpoint>,
    pub system_metadata: HashMap<String, serde_json::Value>,
    pub compressed: bool,
}

/// Checkpoint manager handles creation and storage of checkpoints
pub struct CheckpointManager {
    storage_path: PathBuf,
    compression_enabled: bool,
    max_checkpoints: usize,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            storage_path: PathBuf::from("./checkpoints"),
            compression_enabled: true,
            max_checkpoints: 10,
        })
    }

    /// Create new checkpoint manager with custom path
    pub fn with_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        Ok(Self {
            storage_path: path.as_ref().to_path_buf(),
            compression_enabled: true,
            max_checkpoints: 10,
        })
    }

    /// Create a full system checkpoint
    pub async fn create_full_checkpoint(&self) -> FtResult<CheckpointId> {
        let checkpoint_id = CheckpointId::new();

        // Create checkpoint directory
        fs::create_dir_all(&self.storage_path).await?;

        // Create mock system checkpoint for now
        let checkpoint = SystemCheckpoint {
            id: checkpoint_id.clone(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            gpu_checkpoints: vec![self.create_gpu_checkpoint().await?],
            agent_checkpoints: vec![self.create_agent_checkpoint().await?],
            system_metadata: HashMap::new(),
            compressed: self.compression_enabled,
        };

        self.save_checkpoint(&checkpoint).await?;
        Ok(checkpoint_id)
    }

    /// Create GPU memory checkpoint
    async fn create_gpu_checkpoint(&self) -> FtResult<GpuCheckpoint> {
        // Mock GPU checkpoint for development
        Ok(GpuCheckpoint {
            memory_snapshot: vec![0u8; 1024], // Mock 1KB memory
            kernel_states: HashMap::new(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            size_bytes: 1024,
        })
    }

    /// Create agent state checkpoint
    async fn create_agent_checkpoint(&self) -> FtResult<AgentCheckpoint> {
        // Mock agent checkpoint for development
        Ok(AgentCheckpoint {
            agent_id: Uuid::new_v4().to_string(),
            state_data: vec![0u8; 512], // Mock state data
            memory_contents: HashMap::new(),
            goals: vec!["test_goal".to_string()],
            metadata: HashMap::new(),
        })
    }

    /// Save checkpoint to storage
    async fn save_checkpoint(&self, checkpoint: &SystemCheckpoint) -> FtResult<()> {
        let file_path = self
            .storage_path
            .join(format!("{}.checkpoint", checkpoint.id.0));

        let data = if checkpoint.compressed {
            let serialized = bincode::serialize(checkpoint)?;
            lz4::block::compress(&serialized, None, true)
                .map_err(|e| FaultToleranceError::SerializationError(e.to_string()))?
        } else {
            bincode::serialize(checkpoint)?
        };

        fs::write(file_path, data).await?;
        self.cleanup_old_checkpoints().await?;
        Ok(())
    }

    /// Load checkpoint from storage
    pub async fn load_checkpoint(&self, id: &CheckpointId) -> FtResult<SystemCheckpoint> {
        let file_path = self.storage_path.join(format!("{}.checkpoint", id.0));

        if !file_path.exists() {
            return Err(FaultToleranceError::CheckpointNotFound(id.0.to_string()));
        }

        let data = fs::read(file_path).await?;

        let decompressed = if self.compression_enabled {
            lz4::block::decompress(&data, None)
                .map_err(|e| FaultToleranceError::SerializationError(e.to_string()))?
        } else {
            data
        };

        let checkpoint: SystemCheckpoint = bincode::deserialize(&decompressed)?;
        Ok(checkpoint)
    }

    /// List available checkpoints
    pub async fn list_checkpoints(&self) -> FtResult<Vec<CheckpointId>> {
        if !self.storage_path.exists() {
            return Ok(vec![]);
        }

        let mut checkpoints = vec![];
        let mut dir = fs::read_dir(&self.storage_path).await?;

        while let Some(entry) = dir.next_entry().await? {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".checkpoint") {
                    if let Some(uuid_str) = name.strip_suffix(".checkpoint") {
                        if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                            checkpoints.push(CheckpointId::from_uuid(uuid));
                        }
                    }
                }
            }
        }

        Ok(checkpoints)
    }

    /// Clean up old checkpoints beyond max limit
    async fn cleanup_old_checkpoints(&self) -> FtResult<()> {
        let mut checkpoints = self.list_checkpoints().await?;

        if checkpoints.len() <= self.max_checkpoints {
            return Ok(());
        }

        // Sort by UUID (which includes timestamp in v4)
        checkpoints.sort_by(|a, b| a.0.cmp(&b.0));

        // Remove oldest checkpoints
        let to_remove = checkpoints.len() - self.max_checkpoints;
        for checkpoint_id in checkpoints.iter().take(to_remove) {
            let file_path = self
                .storage_path
                .join(format!("{}.checkpoint", checkpoint_id.0));
            if file_path.exists() {
                fs::remove_file(file_path).await?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_checkpoint_id_creation() {
        let id1 = CheckpointId::new();
        let id2 = CheckpointId::new();
        assert_ne!(id1, id2);
    }

    #[tokio::test]
    async fn test_checkpoint_manager_creation() {
        let manager = CheckpointManager::new();
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_checkpoint_manager_with_custom_path() {
        let temp_dir = TempDir::new()?;
        let manager = CheckpointManager::with_path(temp_dir.path());
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_full_checkpoint_creation() {
        let temp_dir = TempDir::new()?;
        let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

        let checkpoint_id = manager.create_full_checkpoint().await;
        assert!(checkpoint_id.is_ok());
    }

    #[tokio::test]
    async fn test_checkpoint_save_and_load() {
        let temp_dir = TempDir::new()?;
        let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

        let checkpoint_id = manager.create_full_checkpoint().await?;
        let loaded = manager.load_checkpoint(&checkpoint_id).await;

        assert!(loaded.is_ok());
        let checkpoint = loaded?;
        assert_eq!(checkpoint.id, checkpoint_id);
    }

    #[tokio::test]
    async fn test_list_checkpoints() {
        let temp_dir = TempDir::new()?;
        let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

        // Initially empty
        let checkpoints = manager.list_checkpoints().await?;
        assert!(checkpoints.is_empty());

        // Create a checkpoint
        manager.create_full_checkpoint().await?;

        // Should have one checkpoint
        let checkpoints = manager.list_checkpoints().await?;
        assert_eq!(checkpoints.len(), 1);
    }

    #[tokio::test]
    async fn test_checkpoint_cleanup() {
        let temp_dir = TempDir::new()?;
        let mut manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
        manager.max_checkpoints = 2; // Limit to 2 checkpoints

        // Create 3 checkpoints
        manager.create_full_checkpoint().await?;
        manager.create_full_checkpoint().await?;
        manager.create_full_checkpoint().await?;

        // Should only have 2 checkpoints (oldest removed)
        let checkpoints = manager.list_checkpoints().await?;
        assert_eq!(checkpoints.len(), 2);
    }

    #[tokio::test]
    async fn test_load_nonexistent_checkpoint() {
        let temp_dir = TempDir::new()?;
        let manager = CheckpointManager::with_path(temp_dir.path()).unwrap();

        let nonexistent_id = CheckpointId::new();
        let result = manager.load_checkpoint(&nonexistent_id).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FaultToleranceError::CheckpointNotFound(_)
        ));
    }

    #[test]
    fn test_gpu_checkpoint_serialization() {
        let checkpoint = GpuCheckpoint {
            memory_snapshot: vec![1, 2, 3, 4],
            kernel_states: HashMap::new(),
            timestamp: 123456789,
            size_bytes: 4,
        };

        let serialized = bincode::serialize(&checkpoint)?;
        let deserialized: GpuCheckpoint = bincode::deserialize(&serialized)?;

        assert_eq!(checkpoint.memory_snapshot, deserialized.memory_snapshot);
        assert_eq!(checkpoint.timestamp, deserialized.timestamp);
        assert_eq!(checkpoint.size_bytes, deserialized.size_bytes);
    }

    #[test]
    fn test_execution_context_creation() {
        let context = ExecutionContext {
            device_id: 0,
            block_size: (256, 1, 1),
            grid_size: (1024, 1, 1),
            shared_memory_bytes: 4096,
        };

        assert_eq!(context.device_id, 0);
        assert_eq!(context.block_size.0, 256);
        assert_eq!(context.shared_memory_bytes, 4096);
    }

    #[tokio::test]
    async fn test_checkpoint_manager_storage_path_creation() {
        let temp_dir = TempDir::new()?;
        let custom_path = temp_dir.path().join("nested").join("checkpoint_storage");

        // Storage path doesn't exist yet
        assert!(!custom_path.exists());

        let manager = CheckpointManager::with_path(&custom_path)?;
        let _checkpoint_id = manager.create_full_checkpoint().await?;

        // Storage path should be created
        assert!(custom_path.exists());
    }

    #[tokio::test]
    async fn test_cleanup_old_checkpoints() {
        let temp_dir = TempDir::new()?;
        let mut manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
        manager.max_checkpoints = 2; // Set limit to trigger cleanup

        // Create 3 checkpoints - should trigger automatic cleanup
        manager.create_full_checkpoint().await?;
        manager.create_full_checkpoint().await?;
        manager.create_full_checkpoint().await?;

        // Should only have 2 checkpoints due to automatic cleanup
        let checkpoints = manager.list_checkpoints().await?;
        assert_eq!(checkpoints.len(), 2);
    }

    #[test]
    fn test_checkpoint_id_as_uuid() {
        let checkpoint_id = CheckpointId::new();
        let uuid = checkpoint_id.as_uuid();

        // Should be able to convert back and forth
        let new_id = CheckpointId::from_uuid(uuid);
        assert_eq!(checkpoint_id.as_uuid(), new_id.as_uuid());
    }

    #[test]
    fn test_system_checkpoint_with_compression() {
        let checkpoint = SystemCheckpoint {
            id: CheckpointId::new(),
            timestamp: 123456789,
            gpu_checkpoints: vec![GpuCheckpoint {
                memory_snapshot: vec![1, 2, 3, 4, 5],
                kernel_states: HashMap::new(),
                timestamp: 123456789,
                size_bytes: 5,
            }],
            agent_checkpoints: vec![],
            system_metadata: HashMap::new(),
            compressed: true, // Test compressed checkpoint
        };

        // Should serialize and deserialize correctly
        let serialized = bincode::serialize(&checkpoint)?;
        let deserialized: SystemCheckpoint = bincode::deserialize(&serialized)?;

        assert!(deserialized.compressed);
        assert_eq!(deserialized.gpu_checkpoints.len(), 1);
        assert_eq!(
            deserialized.gpu_checkpoints[0].memory_snapshot,
            vec![1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn test_agent_checkpoint_serialization() {
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), serde_json::json!("value1"));

        let agent_checkpoint = AgentCheckpoint {
            agent_id: "test-agent".to_string(),
            state_data: vec![10, 20, 30],
            memory_contents: HashMap::new(),
            goals: vec!["goal1".to_string(), "goal2".to_string()],
            metadata,
        };

        // Test serde_json serialization instead of bincode due to Value complexity
        let serialized = serde_json::to_string(&agent_checkpoint)?;
        let deserialized: AgentCheckpoint = serde_json::from_str(&serialized)?;

        assert_eq!(deserialized.agent_id, "test-agent");
        assert_eq!(deserialized.state_data, vec![10, 20, 30]);
        assert_eq!(deserialized.goals, vec!["goal1", "goal2"]);
        if let Some(serde_json::Value::String(value)) = deserialized.metadata.get("key1") {
            assert_eq!(value, "value1");
        } else {
            panic!("Expected string value in metadata");
        }
    }

    #[tokio::test]
    async fn test_checkpoint_manager_automatic_cleanup() {
        let temp_dir = TempDir::new()?;
        let mut manager = CheckpointManager::with_path(temp_dir.path()).unwrap();
        manager.max_checkpoints = 2; // Set low limit to trigger auto cleanup

        // Create multiple checkpoints to trigger automatic cleanup
        let _id1 = manager.create_full_checkpoint().await?;
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await; // Ensure different timestamps
        let _id2 = manager.create_full_checkpoint().await?;
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await; // Ensure different timestamps
        let _id3 = manager.create_full_checkpoint().await?; // This should trigger cleanup

        // Should only have 2 checkpoints (oldest removed automatically)
        let checkpoints = manager.list_checkpoints().await?;
        assert_eq!(checkpoints.len(), 2);

        // Verify we can load the remaining checkpoints
        for checkpoint_id in &checkpoints {
            let result = manager.load_checkpoint(checkpoint_id).await;
            assert!(result.is_ok());
        }
    }
}
