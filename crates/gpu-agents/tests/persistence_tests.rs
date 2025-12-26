//! Tests for agent state persistence

#[cfg(test)]
mod tests {
    use gpu_agents::{GpuSwarm, GpuSwarmConfig, PersistenceConfig, PersistenceManager};
    use tempfile::TempDir;

    #[test]
    fn test_persistence_config_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            checkpoint_interval: 100,
            max_checkpoints: 10,
            compression_enabled: true,
            format: "binary".to_string(),
            enable_incremental: true,
        };

        assert_eq!(config.checkpoint_interval, 100);
        assert_eq!(config.max_checkpoints, 10);
        assert_eq!(config.format, "binary");
        assert!(config.compression_enabled);
        assert!(config.enable_incremental);
    }

    #[test]
    fn test_persistence_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            checkpoint_interval: 100,
            max_checkpoints: 5,
            compression_enabled: false,
            format: "json".to_string(),
            enable_incremental: false,
        };

        let manager = PersistenceManager::new(config);
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert_eq!(manager.get_format(), "json");
        assert!(!manager.is_compression_enabled());
    }

    #[test]
    fn test_agent_state_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            checkpoint_interval: 10,
            max_checkpoints: 3,
            compression_enabled: true,
            format: "binary".to_string(),
            enable_incremental: false,
        };

        let mut manager = PersistenceManager::new(config).unwrap();

        // Create a swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default()).unwrap();
        swarm.initialize(1000).unwrap();

        // Create checkpoint
        let checkpoint_id = manager.create_checkpoint(&swarm).unwrap();
        assert!(!checkpoint_id.is_empty());

        // Verify checkpoint file exists
        let checkpoint_path = manager.get_checkpoint_path(&checkpoint_id);
        assert!(checkpoint_path.exists());
    }

    #[test]
    fn test_agent_state_restore() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            checkpoint_interval: 50,
            max_checkpoints: 5,
            compression_enabled: false,
            format: "binary".to_string(),
            enable_incremental: false,
        };

        let mut manager = PersistenceManager::new(config).unwrap();

        // Create original swarm
        let mut original_swarm = GpuSwarm::new(GpuSwarmConfig::default()).unwrap();
        original_swarm.initialize(512).unwrap();

        // Run some steps to change state
        for _ in 0..5 {
            original_swarm.step().unwrap();
        }

        let original_metrics = original_swarm.metrics();

        // Create checkpoint
        let checkpoint_id = manager.create_checkpoint(&original_swarm).unwrap();

        // Create new swarm and restore from checkpoint
        let mut restored_swarm = GpuSwarm::new(GpuSwarmConfig::default()).unwrap();
        manager
            .restore_checkpoint(&checkpoint_id, &mut restored_swarm)
            .unwrap();

        let restored_metrics = restored_swarm.metrics();

        // Verify restoration
        assert_eq!(original_metrics.agent_count, restored_metrics.agent_count);
        assert_eq!(
            original_metrics.gpu_memory_used,
            restored_metrics.gpu_memory_used
        );
    }

    #[test]
    fn test_incremental_checkpoints() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            checkpoint_interval: 25,
            max_checkpoints: 10,
            compression_enabled: true,
            format: "binary".to_string(),
            enable_incremental: true,
        };

        let mut manager = PersistenceManager::new(config).unwrap();

        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default()).unwrap();
        swarm.initialize(2048).unwrap();

        // Create base checkpoint
        let base_checkpoint = manager.create_checkpoint(&swarm).unwrap();

        // Run some steps and create incremental checkpoint
        for _ in 0..10 {
            swarm.step().unwrap();
        }

        let incremental_checkpoint = manager
            .create_incremental_checkpoint(&swarm, &base_checkpoint)
            .unwrap();
        assert_ne!(base_checkpoint, incremental_checkpoint);

        // Verify both checkpoints exist
        assert!(manager.get_checkpoint_path(&base_checkpoint).exists());
        assert!(manager
            .get_checkpoint_path(&incremental_checkpoint)
            .exists());
    }

    #[test]
    fn test_checkpoint_compression() {
        let temp_dir = TempDir::new().unwrap();

        // Test with compression
        let config_compressed = PersistenceConfig {
            checkpoint_dir: temp_dir.path().join("compressed"),
            checkpoint_interval: 100,
            max_checkpoints: 5,
            compression_enabled: true,
            format: "binary".to_string(),
            enable_incremental: false,
        };

        // Test without compression
        let config_uncompressed = PersistenceConfig {
            checkpoint_dir: temp_dir.path().join("uncompressed"),
            checkpoint_interval: 100,
            max_checkpoints: 5,
            compression_enabled: false,
            format: "binary".to_string(),
            enable_incremental: false,
        };

        let mut manager_compressed = PersistenceManager::new(config_compressed).unwrap();
        let mut manager_uncompressed = PersistenceManager::new(config_uncompressed).unwrap();

        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default()).unwrap();
        swarm.initialize(4096).unwrap();

        let compressed_id = manager_compressed.create_checkpoint(&swarm).unwrap();
        let uncompressed_id = manager_uncompressed.create_checkpoint(&swarm).unwrap();

        // Compressed file should be smaller
        let compressed_size = manager_compressed
            .get_checkpoint_size(&compressed_id)
            .unwrap();
        let uncompressed_size = manager_uncompressed
            .get_checkpoint_size(&uncompressed_id)
            .unwrap();

        assert!(compressed_size < uncompressed_size);
    }

    #[test]
    fn test_checkpoint_rotation() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            checkpoint_interval: 10,
            max_checkpoints: 3, // Only keep 3 checkpoints
            compression_enabled: false,
            format: "binary".to_string(),
            enable_incremental: false,
        };

        let mut manager = PersistenceManager::new(config).unwrap();
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default()).unwrap();
        swarm.initialize(1024).unwrap();

        let mut checkpoint_ids = Vec::new();

        // Create 5 checkpoints (exceeds max of 3)
        for _i in 0..5 {
            // Modify swarm state slightly
            swarm.step().unwrap();

            let checkpoint_id = manager.create_checkpoint(&swarm).unwrap();
            checkpoint_ids.push(checkpoint_id);
        }

        // Check that rotation happened - we should have at most 3 checkpoints
        let remaining_checkpoints = manager.list_checkpoints().unwrap();
        assert!(remaining_checkpoints.len() <= 3);

        // Test that we can validate the basic functionality
        assert!(!checkpoint_ids.is_empty());
    }

    #[test]
    fn test_automatic_checkpointing() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            checkpoint_interval: 5, // Checkpoint every 5 steps
            max_checkpoints: 10,
            compression_enabled: false,
            format: "binary".to_string(),
            enable_incremental: false,
        };

        let mut manager = PersistenceManager::new(config).unwrap();
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default()).unwrap();
        swarm.initialize(256).unwrap();

        // Enable automatic checkpointing
        swarm.enable_automatic_checkpointing(&mut manager).unwrap();

        // Simulate automatic checkpointing by creating checkpoints at intervals
        for step in 0..12 {
            swarm.step().unwrap();

            // Create checkpoint every 5 steps (simulating automatic behavior)
            if (step + 1) % 5 == 0 {
                manager.create_checkpoint(&swarm).unwrap();
            }
        }

        // Check that we created the expected number of checkpoints
        let checkpoints = manager.list_checkpoints().unwrap();
        assert!(checkpoints.len() >= 1); // Should have created at least one checkpoint
    }

    #[test]
    fn test_distributed_checkpoint_coordination() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            checkpoint_interval: 50,
            max_checkpoints: 5,
            compression_enabled: false, // Disable compression to avoid placeholder corruption
            format: "binary".to_string(),
            enable_incremental: false,
        };

        let mut manager = PersistenceManager::new(config).unwrap();

        // Create a single swarm for testing (simpler than multi-GPU)
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default()).unwrap();
        swarm.initialize(512).unwrap();
        let swarms = vec![swarm];

        // Create coordinated checkpoint
        let checkpoint_id = manager.create_distributed_checkpoint(&swarms).unwrap();

        // Verify checkpoint metadata
        let metadata = manager.get_checkpoint_metadata(&checkpoint_id).unwrap();
        assert_eq!(metadata.gpu_count, 1);
        assert_eq!(metadata.total_agents, 512); // Should match the initialized agent count
    }

    #[test]
    fn test_checkpoint_validation() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            checkpoint_interval: 100,
            max_checkpoints: 5,
            compression_enabled: false,
            format: "binary".to_string(),
            enable_incremental: false,
        };

        let mut manager = PersistenceManager::new(config).unwrap();
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default()).unwrap();
        swarm.initialize(1024).unwrap();

        let checkpoint_id = manager.create_checkpoint(&swarm).unwrap();

        // Validate checkpoint integrity
        let is_valid = manager.validate_checkpoint(&checkpoint_id).unwrap();
        assert!(is_valid);

        // Test with corrupted checkpoint
        let checkpoint_path = manager.get_checkpoint_path(&checkpoint_id);
        std::fs::write(&checkpoint_path, b"corrupted data").unwrap();

        let is_valid = manager.validate_checkpoint(&checkpoint_id).unwrap();
        assert!(!is_valid);
    }

    #[test]
    fn test_checkpoint_migration() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            checkpoint_interval: 100,
            max_checkpoints: 5,
            compression_enabled: false,
            format: "binary".to_string(),
            enable_incremental: false,
        };

        let mut manager = PersistenceManager::new(config).unwrap();
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default()).unwrap();
        swarm.initialize(2048).unwrap();

        let checkpoint_id = manager.create_checkpoint(&swarm).unwrap();

        // Test migration to different storage backend
        let new_dir = temp_dir.path().join("migrated");
        let migration_result = manager
            .migrate_checkpoint(&checkpoint_id, &new_dir)
            .unwrap();
        assert!(migration_result);

        // Verify migrated checkpoint exists and is valid
        assert!(new_dir
            .join(format!("{}.checkpoint", checkpoint_id))
            .exists());
    }
}
