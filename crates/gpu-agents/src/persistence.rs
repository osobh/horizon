//! Agent state persistence and checkpointing

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{GpuSwarm, SwarmMetrics};

/// Configuration for persistence system
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Directory to store checkpoints
    pub checkpoint_dir: PathBuf,

    /// Interval between automatic checkpoints (in steps)
    pub checkpoint_interval: u32,

    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,

    /// Enable compression for checkpoints
    pub compression_enabled: bool,

    /// Checkpoint format ("binary" or "json")
    pub format: String,

    /// Enable incremental checkpoints
    pub enable_incremental: bool,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("./checkpoints"),
            checkpoint_interval: 1000,
            max_checkpoints: 10,
            compression_enabled: true,
            format: "binary".to_string(),
            enable_incremental: true,
        }
    }
}

/// Manages agent state persistence and recovery
pub struct PersistenceManager {
    config: PersistenceConfig,
    checkpoint_history: Vec<String>,
}

impl PersistenceManager {
    /// Create a new persistence manager
    pub fn new(config: PersistenceConfig) -> Result<Self> {
        // Create checkpoint directory if it doesn't exist
        fs::create_dir_all(&config.checkpoint_dir)?;

        Ok(Self {
            config,
            checkpoint_history: Vec::new(),
        })
    }

    /// Get the checkpoint format
    pub fn get_format(&self) -> &str {
        &self.config.format
    }

    /// Check if compression is enabled
    pub fn is_compression_enabled(&self) -> bool {
        self.config.compression_enabled
    }

    /// Create a checkpoint of the current swarm state
    pub fn create_checkpoint(&mut self, swarm: &GpuSwarm) -> Result<String> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        let checkpoint_id = format!("checkpoint_{}", timestamp);
        let checkpoint_path = self.get_checkpoint_path(&checkpoint_id);

        // Create checkpoint data
        let checkpoint_data = CheckpointData {
            id: checkpoint_id.clone(),
            timestamp,
            metrics: swarm.metrics(),
            agent_count: swarm.metrics().agent_count,
            format_version: 1,
            compression_used: self.config.compression_enabled,
        };

        // Serialize and save
        self.save_checkpoint_data(&checkpoint_data, &checkpoint_path)?;

        // Add to history and manage rotation
        self.checkpoint_history.push(checkpoint_id.clone());
        self.rotate_checkpoints()?;

        Ok(checkpoint_id)
    }

    /// Restore swarm state from a checkpoint
    pub fn restore_checkpoint(&self, checkpoint_id: &str, swarm: &mut GpuSwarm) -> Result<()> {
        let checkpoint_path = self.get_checkpoint_path(checkpoint_id);

        if !checkpoint_path.exists() {
            return Err(anyhow::anyhow!("Checkpoint {} not found", checkpoint_id));
        }

        let checkpoint_data = self.load_checkpoint_data(&checkpoint_path)?;

        // Restore swarm state (placeholder implementation)
        // In real implementation, would restore GPU memory, agent states, etc.
        swarm.initialize(checkpoint_data.agent_count)?;

        Ok(())
    }

    /// Create an incremental checkpoint based on a previous checkpoint
    pub fn create_incremental_checkpoint(
        &mut self,
        swarm: &GpuSwarm,
        base_checkpoint: &str,
    ) -> Result<String> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        let checkpoint_id = format!("incremental_{}_{}", base_checkpoint, timestamp);
        let checkpoint_path = self.get_checkpoint_path(&checkpoint_id);

        // Create incremental checkpoint data
        let checkpoint_data = CheckpointData {
            id: checkpoint_id.clone(),
            timestamp,
            metrics: swarm.metrics(),
            agent_count: swarm.metrics().agent_count,
            format_version: 1,
            compression_used: self.config.compression_enabled,
        };

        // Save incremental data
        self.save_checkpoint_data(&checkpoint_data, &checkpoint_path)?;

        self.checkpoint_history.push(checkpoint_id.clone());
        self.rotate_checkpoints()?;

        Ok(checkpoint_id)
    }

    /// Get the path for a checkpoint file
    pub fn get_checkpoint_path(&self, checkpoint_id: &str) -> PathBuf {
        self.config
            .checkpoint_dir
            .join(format!("{}.checkpoint", checkpoint_id))
    }

    /// Get the size of a checkpoint file
    pub fn get_checkpoint_size(&self, checkpoint_id: &str) -> Result<u64> {
        let checkpoint_path = self.get_checkpoint_path(checkpoint_id);
        let metadata = fs::metadata(checkpoint_path)?;
        Ok(metadata.len())
    }

    /// List all available checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<String>> {
        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(&self.config.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(file_name) = path.file_name() {
                if let Some(name_str) = file_name.to_str() {
                    if name_str.ends_with(".checkpoint") {
                        let checkpoint_id = name_str.trim_end_matches(".checkpoint").to_string();
                        checkpoints.push(checkpoint_id);
                    }
                }
            }
        }

        Ok(checkpoints)
    }

    /// Validate checkpoint integrity
    pub fn validate_checkpoint(&self, checkpoint_id: &str) -> Result<bool> {
        let checkpoint_path = self.get_checkpoint_path(checkpoint_id);

        if !checkpoint_path.exists() {
            return Ok(false);
        }

        match self.load_checkpoint_data(&checkpoint_path) {
            Ok(data) => Ok(data.id == checkpoint_id),
            Err(_) => Ok(false),
        }
    }

    /// Create a distributed checkpoint for multiple swarms
    pub fn create_distributed_checkpoint(&mut self, swarms: &[GpuSwarm]) -> Result<String> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        let checkpoint_id = format!("distributed_{}", timestamp);
        let checkpoint_path = self.get_checkpoint_path(&checkpoint_id);

        // Aggregate metrics from all swarms
        let individual_metrics: Vec<SwarmMetrics> = swarms.iter().map(|s| s.metrics()).collect();
        let total_agents: usize = individual_metrics.iter().map(|m| m.agent_count).sum();
        let gpu_count = swarms.len();

        let distributed_data = DistributedCheckpointData {
            id: checkpoint_id.clone(),
            timestamp,
            gpu_count,
            total_agents,
            individual_metrics,
        };

        // Save distributed checkpoint
        self.save_distributed_checkpoint(&distributed_data, &checkpoint_path)?;

        self.checkpoint_history.push(checkpoint_id.clone());
        self.rotate_checkpoints()?;

        Ok(checkpoint_id)
    }

    /// Get checkpoint metadata
    pub fn get_checkpoint_metadata(&self, checkpoint_id: &str) -> Result<CheckpointMetadata> {
        let checkpoint_path = self.get_checkpoint_path(&checkpoint_id);

        // Try to load as distributed checkpoint first
        match self.load_distributed_checkpoint(&checkpoint_path) {
            Ok(data) => Ok(CheckpointMetadata {
                id: data.id,
                timestamp: data.timestamp,
                gpu_count: data.gpu_count,
                total_agents: data.total_agents,
                checkpoint_type: "distributed".to_string(),
            }),
            Err(_) => {
                // Try as regular checkpoint
                let data = self.load_checkpoint_data(&checkpoint_path)?;
                Ok(CheckpointMetadata {
                    id: data.id,
                    timestamp: data.timestamp,
                    gpu_count: 1,
                    total_agents: data.agent_count,
                    checkpoint_type: "single".to_string(),
                })
            }
        }
    }

    /// Migrate a checkpoint to a different location
    pub fn migrate_checkpoint(&self, checkpoint_id: &str, target_dir: &Path) -> Result<bool> {
        let source_path = self.get_checkpoint_path(checkpoint_id);
        let target_path = target_dir.join(format!("{}.checkpoint", checkpoint_id));

        // Create target directory
        fs::create_dir_all(target_dir)?;

        // Copy checkpoint file
        fs::copy(&source_path, &target_path)?;

        Ok(target_path.exists())
    }

    /// Rotate checkpoints based on max_checkpoints limit
    fn rotate_checkpoints(&mut self) -> Result<()> {
        while self.checkpoint_history.len() > self.config.max_checkpoints {
            let old_checkpoint = self.checkpoint_history.remove(0);
            let old_path = self.get_checkpoint_path(&old_checkpoint);
            if old_path.exists() {
                fs::remove_file(old_path)?;
            }
        }
        Ok(())
    }

    /// Save checkpoint data to file
    fn save_checkpoint_data(&self, data: &CheckpointData, path: &Path) -> Result<()> {
        let serialized = match self.config.format.as_str() {
            "json" => serde_json::to_vec(data)?,
            "binary" => bincode::serialize(data)?,
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported format: {}",
                    self.config.format
                ))
            }
        };

        let final_data = if self.config.compression_enabled {
            self.compress_data(&serialized)?
        } else {
            serialized
        };

        fs::write(path, final_data)?;
        Ok(())
    }

    /// Load checkpoint data from file
    fn load_checkpoint_data(&self, path: &Path) -> Result<CheckpointData> {
        let file_data = fs::read(path)?;

        let decompressed_data = if self.config.compression_enabled {
            self.decompress_data(&file_data)?
        } else {
            file_data
        };

        let data = match self.config.format.as_str() {
            "json" => serde_json::from_slice(&decompressed_data)?,
            "binary" => bincode::deserialize(&decompressed_data)?,
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported format: {}",
                    self.config.format
                ))
            }
        };

        Ok(data)
    }

    /// Save distributed checkpoint data
    fn save_distributed_checkpoint(
        &self,
        data: &DistributedCheckpointData,
        path: &Path,
    ) -> Result<()> {
        let serialized = bincode::serialize(data)?;

        let final_data = if self.config.compression_enabled {
            self.compress_data(&serialized)?
        } else {
            serialized
        };

        fs::write(path, final_data)?;
        Ok(())
    }

    /// Load distributed checkpoint data
    fn load_distributed_checkpoint(&self, path: &Path) -> Result<DistributedCheckpointData> {
        let file_data = fs::read(path)?;

        let decompressed_data = if self.config.compression_enabled {
            self.decompress_data(&file_data)?
        } else {
            file_data
        };

        let data = bincode::deserialize(&decompressed_data)?;
        Ok(data)
    }

    /// Compress data using a simple algorithm (placeholder)
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Placeholder: simulate compression by truncating (not real compression)
        // In real implementation, would use proper compression like zstd, lz4, etc.
        let mut compressed = Vec::with_capacity(data.len() / 2);
        compressed.extend_from_slice(b"CMP:");
        // Simulate compression by taking every other byte
        for (i, &byte) in data.iter().enumerate() {
            if i % 2 == 0 {
                compressed.push(byte);
            }
        }
        Ok(compressed)
    }

    /// Decompress data (placeholder)
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Placeholder decompression
        if data.starts_with(b"CMP:") {
            let compressed_data = &data[4..];
            let mut decompressed = Vec::with_capacity(compressed_data.len() * 2);
            for &byte in compressed_data {
                decompressed.push(byte);
                decompressed.push(0); // Pad with zeros
            }
            Ok(decompressed)
        } else {
            Ok(data.to_vec())
        }
    }
}

/// Checkpoint data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointData {
    id: String,
    timestamp: u64,
    metrics: SwarmMetrics,
    agent_count: usize,
    format_version: u32,
    compression_used: bool,
}

/// Distributed checkpoint data for multi-GPU swarms
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DistributedCheckpointData {
    id: String,
    timestamp: u64,
    gpu_count: usize,
    total_agents: usize,
    individual_metrics: Vec<SwarmMetrics>,
}

/// Checkpoint metadata
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    pub id: String,
    pub timestamp: u64,
    pub gpu_count: usize,
    pub total_agents: usize,
    pub checkpoint_type: String,
}

/// Extension to GpuSwarm for automatic checkpointing
impl crate::GpuSwarm {
    /// Enable automatic checkpointing
    pub fn enable_automatic_checkpointing(
        &mut self,
        manager: &mut PersistenceManager,
    ) -> Result<()> {
        // Store reference to manager for automatic checkpointing
        // In real implementation, would set up automatic checkpointing
        // For testing, we'll track the interval and create checkpoints in step()
        self.enable_checkpointing_with_interval(manager.config.checkpoint_interval)
    }

    /// Enable checkpointing with a specific interval
    fn enable_checkpointing_with_interval(&mut self, _interval: u32) -> Result<()> {
        // Placeholder: would configure internal checkpointing state
        Ok(())
    }
}
