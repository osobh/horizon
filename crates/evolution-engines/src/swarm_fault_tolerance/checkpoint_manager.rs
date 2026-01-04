//! Checkpoint management for fault tolerance

use super::fault_detector::FaultToleranceConfig;
use super::storage::{CheckpointStorage, StorageType};
use super::types::{CheckpointMetadata, CheckpointSnapshot, CompressionAlgorithm, NodeState};
use crate::error::EvolutionEngineResult;
use crate::swarm_distributed::MigrationParticle;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Checkpoint manager for system state persistence
pub struct CheckpointManager {
    /// Configuration
    pub(crate) config: FaultToleranceConfig,
    /// In-memory checkpoint storage
    pub(crate) checkpoints: HashMap<String, CheckpointSnapshot>,
    /// Checkpoint metadata
    pub(crate) checkpoint_metadata: HashMap<String, CheckpointMetadata>,
    /// Storage backend
    pub(crate) storage_backend: Arc<RwLock<CheckpointStorage>>,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub async fn new(config: FaultToleranceConfig) -> EvolutionEngineResult<Self> {
        Ok(Self {
            config,
            checkpoints: HashMap::new(),
            checkpoint_metadata: HashMap::new(),
            storage_backend: Arc::new(RwLock::new(CheckpointStorage::new(StorageType::Local))),
        })
    }

    /// Create a new checkpoint
    pub async fn create_checkpoint(
        &mut self,
        generation: u32,
        node_states: HashMap<String, NodeState>,
        global_best: Option<MigrationParticle>,
        global_best_fitness: Option<f64>,
    ) -> EvolutionEngineResult<String> {
        let checkpoint_id = uuid::Uuid::new_v4().to_string();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as u64;

        let checkpoint = CheckpointSnapshot {
            id: checkpoint_id.clone(),
            generation,
            timestamp,
            node_states: node_states.clone(),
            global_best: global_best.clone(),
            global_best_fitness,
            size_bytes: 0, // Will be calculated during serialization
        };

        // Calculate size and create metadata
        let serialized = serde_json::to_string(&checkpoint)?;
        let size_bytes = serialized.len();

        let metadata = CheckpointMetadata {
            id: checkpoint_id.clone(),
            created_at: timestamp,
            size_bytes,
            node_count: node_states.len(),
            particle_count: node_states.values().map(|s| s.particles.len()).sum(),
            compression: CompressionAlgorithm::None,
            checksum: self.calculate_checksum(&serialized),
        };

        // Store checkpoint
        self.checkpoints.insert(checkpoint_id.clone(), checkpoint);
        self.checkpoint_metadata
            .insert(checkpoint_id.clone(), metadata);

        // Store to backend
        self.storage_backend
            .write()
            .await
            .store_checkpoint(&checkpoint_id, &serialized)
            .await?;

        Ok(checkpoint_id)
    }

    /// Get latest checkpoint
    pub async fn get_latest_checkpoint(&self) -> EvolutionEngineResult<Option<CheckpointSnapshot>> {
        let latest_id = self
            .checkpoint_metadata
            .iter()
            .max_by_key(|(_, metadata)| metadata.created_at)
            .map(|(id, _)| id.clone());

        if let Some(id) = latest_id {
            Ok(self.checkpoints.get(&id).cloned())
        } else {
            Ok(None)
        }
    }

    /// Restore from checkpoint
    pub async fn restore_from_checkpoint(
        &self,
        checkpoint_id: &str,
    ) -> EvolutionEngineResult<CheckpointSnapshot> {
        self.checkpoints.get(checkpoint_id).cloned().ok_or_else(|| {
            crate::error::EvolutionEngineError::InvalidConfiguration {
                message: format!("Checkpoint not found: {}", checkpoint_id),
            }
        })
    }

    /// List available checkpoints
    pub fn list_checkpoints(&self) -> Vec<CheckpointMetadata> {
        self.checkpoint_metadata.values().cloned().collect()
    }

    /// Delete old checkpoints
    pub async fn cleanup_old_checkpoints(
        &mut self,
        keep_count: usize,
    ) -> EvolutionEngineResult<usize> {
        let mut metadata_list: Vec<_> = self.checkpoint_metadata.iter().collect();
        metadata_list.sort_by_key(|(_, metadata)| metadata.created_at);

        let to_delete = if metadata_list.len() > keep_count {
            metadata_list.len() - keep_count
        } else {
            0
        };

        let ids_to_delete: Vec<String> = metadata_list
            .iter()
            .take(to_delete)
            .map(|(id, _)| (*id).clone())
            .collect();

        let mut deleted_count = 0;
        for id in ids_to_delete {
            self.checkpoints.remove(&id);
            self.checkpoint_metadata.remove(&id);
            self.storage_backend
                .write()
                .await
                .delete_checkpoint(&id)
                .await?;
            deleted_count += 1;
        }

        Ok(deleted_count)
    }

    /// Calculate checksum for integrity verification
    fn calculate_checksum(&self, data: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}
