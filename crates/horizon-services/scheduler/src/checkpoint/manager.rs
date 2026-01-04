use crate::config::CheckpointConfig;
use crate::error::{HpcError, SchedulerErrorExt};
use crate::models::{Checkpoint, Job};
use crate::Result;
use std::path::PathBuf;
use tokio::fs;

/// Manages job checkpointing for preemption
pub struct CheckpointManager {
    storage_path: PathBuf,
    max_size_gb: u64,
}

impl CheckpointManager {
    pub fn new(config: &CheckpointConfig) -> Self {
        Self {
            storage_path: PathBuf::from(&config.storage_path),
            max_size_gb: config.max_checkpoint_size_gb,
        }
    }

    /// Create a checkpoint for a job
    pub async fn create_checkpoint(&self, job: &Job) -> Result<Checkpoint> {
        // Serialize job state
        let state_data = serde_json::to_value(job)?;

        // Generate storage path
        let checkpoint_path = self.storage_path.join(format!("{}.json", job.id));

        // Write checkpoint to disk
        let checkpoint_json = serde_json::to_string_pretty(&state_data)?;
        fs::write(&checkpoint_path, checkpoint_json.as_bytes()).await?;

        let size_bytes = checkpoint_json.len() as u64;

        // Validate size
        let size_gb = size_bytes / (1024 * 1024 * 1024);
        if size_gb > self.max_size_gb {
            return Err(HpcError::checkpoint_too_large(size_gb, self.max_size_gb));
        }

        Ok(Checkpoint::new(
            job.id,
            state_data,
            checkpoint_path.to_string_lossy().to_string(),
            size_bytes,
        ))
    }

    /// Load a checkpoint from storage
    pub async fn load_checkpoint(&self, checkpoint_id: &str) -> Result<Checkpoint> {
        let checkpoint_path = self.storage_path.join(format!("{}.json", checkpoint_id));

        if !checkpoint_path.exists() {
            return Err(HpcError::checkpoint_not_found(checkpoint_id));
        }

        let contents = fs::read_to_string(&checkpoint_path).await?;
        let state_data: serde_json::Value = serde_json::from_str(&contents)?;

        let metadata = fs::metadata(&checkpoint_path).await?;
        let size_bytes = metadata.len();

        // Parse job_id from state data
        let job_id = state_data["id"]
            .as_str()
            .and_then(|s| uuid::Uuid::parse_str(s).ok())
            .ok_or_else(|| HpcError::checkpoint_not_found(checkpoint_id))?;

        Ok(Checkpoint::new(
            job_id,
            state_data,
            checkpoint_path.to_string_lossy().to_string(),
            size_bytes,
        ))
    }

    /// Delete a checkpoint
    pub async fn delete_checkpoint(&self, checkpoint_id: &str) -> Result<()> {
        let checkpoint_path = self.storage_path.join(format!("{}.json", checkpoint_id));

        if checkpoint_path.exists() {
            fs::remove_file(&checkpoint_path).await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> CheckpointConfig {
        CheckpointConfig {
            storage_path: "/tmp/test_checkpoints".to_string(),
            s3_bucket: None,
            max_checkpoint_size_gb: 100,
            retention_days: 7,
        }
    }

    #[tokio::test]
    async fn test_checkpoint_creation() {
        let config = create_test_config();
        let manager = CheckpointManager::new(&config);

        // Create the test directory
        let _ = std::fs::create_dir_all("/tmp/test_checkpoints");

        let job = Job::builder().user_id("test").gpu_count(4).build().unwrap();

        let result = manager.create_checkpoint(&job).await;
        assert!(result.is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all("/tmp/test_checkpoints");
    }
}
