//! Core types and data structures for memory snapshots

use crate::DebugError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Memory snapshot containing container state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub snapshot_id: Uuid,
    pub container_id: Uuid,
    pub timestamp: u64,
    pub host_memory: Vec<u8>,
    pub device_memory: Vec<u8>,
    pub kernel_parameters: KernelParameters,
    pub execution_context: ExecutionContext,
    pub metadata: SnapshotMetadata,
}

/// Kernel launch parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelParameters {
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
    pub shared_memory_size: u32,
    pub kernel_args: Vec<u8>,
    pub stream_id: Option<u32>,
}

/// Execution context at snapshot time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    pub goal_prompt: String,
    pub agent_id: Option<String>,
    pub generation: u64,
    pub mutation_count: u32,
    pub parent_fitness: Option<f64>,
    pub environment_vars: HashMap<String, String>,
}

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub creation_reason: String,
    pub file_size_bytes: u64,
    pub compression_ratio: f32,
    pub ttl_seconds: u64,
    pub encrypted: bool,
    pub tags: HashMap<String, String>,
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub expired_snapshots: u64,
    pub oldest_snapshot_age_seconds: u64,
}

/// Configuration for snapshot operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotConfig {
    pub auto_snapshot: bool,
    pub default_ttl_seconds: u64,
    pub max_snapshot_size_mb: u64,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
    pub cleanup_interval_seconds: u64,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            auto_snapshot: true,
            default_ttl_seconds: 3600,  // 1 hour
            max_snapshot_size_mb: 1024, // 1 GB
            compression_enabled: true,
            encryption_enabled: false,
            cleanup_interval_seconds: 300, // 5 minutes
        }
    }
}

/// Internal session tracking
#[derive(Debug, Clone)]
pub struct SnapshotSession {
    pub container_id: Uuid,
    pub start_time: u64,
    pub active_snapshots: Vec<Uuid>,
    pub session_config: SnapshotConfig,
}

/// Trait for snapshot storage backends
#[async_trait]
pub trait SnapshotStorage: Send + Sync {
    async fn store_snapshot(&self, snapshot: MemorySnapshot) -> Result<(), DebugError>;
    async fn get_snapshot(&self, snapshot_id: Uuid) -> Result<MemorySnapshot, DebugError>;
    async fn list_snapshots(&self, container_id: Uuid) -> Result<Vec<Uuid>, DebugError>;
    async fn delete_snapshot(&self, snapshot_id: Uuid) -> Result<(), DebugError>;
    async fn get_stats(&self) -> Result<StorageStats, DebugError>;
    async fn cleanup_expired(&self, max_age_seconds: u64) -> Result<u64, DebugError>;
}

impl MemorySnapshot {
    /// Create a new memory snapshot
    pub fn new(
        container_id: Uuid,
        host_memory: Vec<u8>,
        device_memory: Vec<u8>,
        kernel_parameters: KernelParameters,
        execution_context: ExecutionContext,
        creation_reason: String,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let file_size_bytes = (host_memory.len() + device_memory.len()) as u64;

        Self {
            snapshot_id: Uuid::new_v4(),
            container_id,
            timestamp,
            host_memory,
            device_memory,
            kernel_parameters,
            execution_context,
            metadata: SnapshotMetadata {
                creation_reason,
                file_size_bytes,
                compression_ratio: 1.0,
                ttl_seconds: 3600,
                encrypted: false,
                tags: HashMap::new(),
            },
        }
    }

    /// Get the total size of the snapshot in bytes
    pub fn total_size(&self) -> usize {
        self.host_memory.len() + self.device_memory.len()
    }

    /// Check if snapshot has expired
    pub fn is_expired(&self) -> bool {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        current_time > (self.timestamp + self.metadata.ttl_seconds)
    }

    /// Compress the snapshot data (placeholder implementation)
    pub fn compress(&mut self) -> Result<(), DebugError> {
        // Placeholder for compression logic
        self.metadata.compression_ratio = 0.7; // Simulate 30% compression
        Ok(())
    }

    /// Decompress the snapshot data (placeholder implementation)
    pub fn decompress(&mut self) -> Result<(), DebugError> {
        // Placeholder for decompression logic
        self.metadata.compression_ratio = 1.0;
        Ok(())
    }

    /// Add a tag to the snapshot
    pub fn add_tag(&mut self, key: String, value: String) {
        self.metadata.tags.insert(key, value);
    }

    /// Remove a tag from the snapshot
    pub fn remove_tag(&mut self, key: &str) -> Option<String> {
        self.metadata.tags.remove(key)
    }

    /// Check if snapshot has a specific tag
    pub fn has_tag(&self, key: &str, value: &str) -> bool {
        self.metadata.tags.get(key).is_some_and(|v| v == value)
    }
}

impl ExecutionContext {
    /// Create a new execution context
    pub fn new(goal_prompt: String) -> Self {
        Self {
            goal_prompt,
            agent_id: None,
            generation: 0,
            mutation_count: 0,
            parent_fitness: None,
            environment_vars: HashMap::new(),
        }
    }

    /// Add an environment variable
    pub fn add_env_var(&mut self, key: String, value: String) {
        self.environment_vars.insert(key, value);
    }

    /// Get an environment variable
    pub fn get_env_var(&self, key: &str) -> Option<&String> {
        self.environment_vars.get(key)
    }

    /// Set agent information
    pub fn set_agent_info(&mut self, agent_id: String, generation: u64, mutation_count: u32) {
        self.agent_id = Some(agent_id);
        self.generation = generation;
        self.mutation_count = mutation_count;
    }
}

impl Default for KernelParameters {
    fn default() -> Self {
        Self {
            grid_size: (1, 1, 1),
            block_size: (256, 1, 1),
            shared_memory_size: 0,
            kernel_args: Vec::new(),
            stream_id: None,
        }
    }
}

impl KernelParameters {
    /// Create kernel parameters with specific grid and block sizes
    pub fn new(grid_size: (u32, u32, u32), block_size: (u32, u32, u32)) -> Self {
        Self {
            grid_size,
            block_size,
            shared_memory_size: 0,
            kernel_args: Vec::new(),
            stream_id: None,
        }
    }

    /// Set shared memory size
    pub fn with_shared_memory(mut self, size: u32) -> Self {
        self.shared_memory_size = size;
        self
    }

    /// Set kernel arguments
    pub fn with_args(mut self, args: Vec<u8>) -> Self {
        self.kernel_args = args;
        self
    }

    /// Set stream ID
    pub fn with_stream(mut self, stream_id: u32) -> Self {
        self.stream_id = Some(stream_id);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_snapshot_creation() {
        let container_id = Uuid::new_v4();
        let host_memory = vec![1, 2, 3, 4];
        let device_memory = vec![5, 6, 7, 8];
        let kernel_params = KernelParameters::default();
        let context = ExecutionContext::new("Test goal".to_string());

        let snapshot = MemorySnapshot::new(
            container_id,
            host_memory.clone(),
            device_memory.clone(),
            kernel_params,
            context,
            "test creation".to_string(),
        );

        assert_eq!(snapshot.container_id, container_id);
        assert_eq!(snapshot.host_memory, host_memory);
        assert_eq!(snapshot.device_memory, device_memory);
        assert_eq!(snapshot.total_size(), 8);
        assert!(!snapshot.is_expired());
    }

    #[test]
    fn test_execution_context() {
        let mut context = ExecutionContext::new("Test context".to_string());

        context.add_env_var("GPU_DEVICE".to_string(), "0".to_string());
        context.set_agent_info("agent_123".to_string(), 5, 2);

        assert_eq!(context.goal_prompt, "Test context");
        assert_eq!(context.agent_id, Some("agent_123".to_string()));
        assert_eq!(context.generation, 5);
        assert_eq!(context.mutation_count, 2);
        assert_eq!(context.get_env_var("GPU_DEVICE"), Some(&"0".to_string()));
    }

    #[test]
    fn test_kernel_parameters() {
        let params = KernelParameters::new((8, 8, 1), (256, 1, 1))
            .with_shared_memory(1024)
            .with_args(vec![1, 2, 3, 4])
            .with_stream(42);

        assert_eq!(params.grid_size, (8, 8, 1));
        assert_eq!(params.block_size, (256, 1, 1));
        assert_eq!(params.shared_memory_size, 1024);
        assert_eq!(params.kernel_args, vec![1, 2, 3, 4]);
        assert_eq!(params.stream_id, Some(42));
    }

    #[test]
    fn test_snapshot_tags() {
        let mut snapshot = MemorySnapshot::new(
            Uuid::new_v4(),
            vec![],
            vec![],
            KernelParameters::default(),
            ExecutionContext::new("test".to_string()),
            "test tags".to_string(),
        );

        snapshot.add_tag("environment".to_string(), "test".to_string());
        snapshot.add_tag("version".to_string(), "1.0".to_string());

        assert!(snapshot.has_tag("environment", "test"));
        assert!(snapshot.has_tag("version", "1.0"));
        assert!(!snapshot.has_tag("environment", "production"));

        let removed = snapshot.remove_tag("version");
        assert_eq!(removed, Some("1.0".to_string()));
        assert!(!snapshot.has_tag("version", "1.0"));
    }

    #[test]
    fn test_snapshot_config_default() {
        let config = SnapshotConfig::default();

        assert!(config.auto_snapshot);
        assert_eq!(config.default_ttl_seconds, 3600);
        assert_eq!(config.max_snapshot_size_mb, 1024);
        assert!(config.compression_enabled);
        assert!(!config.encryption_enabled);
        assert_eq!(config.cleanup_interval_seconds, 300);
    }
}
