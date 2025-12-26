//! TDD tests for GPU agent storage tier
//! Following rust.md TDD principles and cuda.md GPU standards

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // Test configuration for GPU storage path
    const GPU_STORAGE_PATH: &str = "/magikdev/gpu";
    const TEST_GPU_STORAGE_PATH: &str = "/tmp/test_magikdev_gpu";

    /// Helper to create test storage path
    fn get_test_storage_path() -> PathBuf {
        // In tests, use temp directory; in production use /magikdev/gpu
        if cfg!(test) {
            PathBuf::from(TEST_GPU_STORAGE_PATH)
        } else {
            PathBuf::from(GPU_STORAGE_PATH)
        }
    }

    #[test]
    fn test_gpu_storage_config_uses_magikdev_path() {
        // RED: Test that GPU storage configuration uses /magikdev/gpu
        let config = super::GpuStorageConfig::default();
        assert_eq!(config.base_path, PathBuf::from(GPU_STORAGE_PATH));
        assert_eq!(
            config.cache_path,
            PathBuf::from(GPU_STORAGE_PATH).join("cache")
        );
        assert_eq!(config.wal_path, PathBuf::from(GPU_STORAGE_PATH).join("wal"));
    }

    #[test]
    fn test_gpu_storage_tier_initialization() -> Result<()> {
        // RED: Test GPU storage tier can be initialized
        let temp_dir = TempDir::new()?;
        let config = super::GpuStorageConfig {
            base_path: temp_dir.path().to_path_buf(),
            cache_path: temp_dir.path().join("cache"),
            wal_path: temp_dir.path().join("wal"),
            enable_gpu_cache: true,
            cache_size_mb: 1024, // 1GB GPU cache
            enable_compression: true,
            sync_interval_ms: 100,
        };

        let storage = super::GpuAgentStorage::new(config)?;
        assert!(storage.is_initialized());
        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_agent_data_persistence() -> Result<()> {
        // RED: Test that GPU agents can persist their state
        let temp_dir = TempDir::new()?;
        let config = super::GpuStorageConfig::test_config(temp_dir.path());
        let storage = super::GpuAgentStorage::new(config)?;

        // Test agent data structure
        let agent_id = "agent_001";
        let agent_data = super::GpuAgentData {
            id: agent_id.to_string(),
            state: vec![1.0, 2.0, 3.0, 4.0], // Neural state
            memory: vec![0.5; 128],          // Agent memory
            generation: 42,
            fitness: 0.95,
            metadata: Default::default(),
        };

        // Store agent data
        storage.store_agent(agent_id, &agent_data).await?;

        // Retrieve and verify
        let retrieved = storage.retrieve_agent(agent_id).await?;
        assert_eq!(retrieved.id, agent_data.id);
        assert_eq!(retrieved.state, agent_data.state);
        assert_eq!(retrieved.generation, agent_data.generation);

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_cache_integration() -> Result<()> {
        // RED: Test GPU cache for fast agent access
        let temp_dir = TempDir::new()?;
        let config = super::GpuStorageConfig::test_config(temp_dir.path());
        let storage = super::GpuAgentStorage::new(config)?;

        // Pre-warm cache with frequently accessed agents
        let hot_agents = vec!["agent_hot_1", "agent_hot_2", "agent_hot_3"];
        for agent_id in &hot_agents {
            let data = super::GpuAgentData::random(agent_id);
            storage.store_agent(agent_id, &data).await?;
            storage.cache_agent(agent_id).await?;
        }

        // Verify cache hits
        let stats = storage.cache_stats().await?;
        assert!(stats.cached_agents >= hot_agents.len());

        // Test cache performance
        let start = std::time::Instant::now();
        for _ in 0..100 {
            for agent_id in &hot_agents {
                let _ = storage.retrieve_agent_cached(agent_id).await?;
            }
        }
        let elapsed = start.elapsed();

        // Cache access should be very fast (< 1ms per access)
        let avg_access_time = elapsed.as_micros() as f64 / (100.0 * hot_agents.len() as f64);
        assert!(avg_access_time < 1000.0); // Less than 1ms

        Ok(())
    }

    #[tokio::test]
    async fn test_knowledge_graph_storage() -> Result<()> {
        // RED: Test storage of knowledge graphs for GPU agents
        let temp_dir = TempDir::new()?;
        let config = super::GpuStorageConfig::test_config(temp_dir.path());
        let storage = super::GpuAgentStorage::new(config)?;

        // Create test knowledge graph
        let graph = super::GpuKnowledgeGraph {
            nodes: vec![
                super::GraphNode {
                    id: "node_1".to_string(),
                    embedding: vec![0.1; 768],
                    metadata: Default::default(),
                },
                super::GraphNode {
                    id: "node_2".to_string(),
                    embedding: vec![0.2; 768],
                    metadata: Default::default(),
                },
            ],
            edges: vec![super::GraphEdge {
                source: "node_1".to_string(),
                target: "node_2".to_string(),
                weight: 0.8,
                edge_type: "similarity".to_string(),
            }],
        };

        // Store and retrieve graph
        storage.store_knowledge_graph("graph_1", &graph).await?;
        let retrieved = storage.retrieve_knowledge_graph("graph_1").await?;

        assert_eq!(retrieved.nodes.len(), graph.nodes.len());
        assert_eq!(retrieved.edges.len(), graph.edges.len());

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_memory_mapping() -> Result<()> {
        // RED: Test direct GPU memory mapping for zero-copy access
        let temp_dir = TempDir::new()?;
        let config = super::GpuStorageConfig::test_config(temp_dir.path());
        let storage = super::GpuAgentStorage::new(config)?;

        // Store large agent swarm data
        let swarm_size = 10_000;
        let swarm_data = super::SwarmData {
            agent_count: swarm_size,
            state_dimension: 256,
            data: vec![0.0f32; swarm_size * 256],
        };

        // Map to GPU memory
        let gpu_handle = storage.map_to_gpu_memory("swarm_1", &swarm_data).await?;

        // Verify mapping
        assert!(gpu_handle.is_mapped());
        assert_eq!(gpu_handle.size_bytes(), swarm_size * 256 * 4); // f32 = 4 bytes

        Ok(())
    }

    #[test]
    fn test_storage_path_configuration() {
        // RED: Test various storage path configurations

        // Production path
        let prod_config = super::GpuStorageConfig::production();
        assert_eq!(prod_config.base_path, PathBuf::from("/magikdev/gpu"));

        // Development path
        let dev_config = super::GpuStorageConfig::development();
        assert!(dev_config
            .base_path
            .to_str()
            .unwrap()
            .contains("gpu_storage"));

        // Custom path
        let custom_config = super::GpuStorageConfig::with_base_path("/custom/gpu/path");
        assert_eq!(custom_config.base_path, PathBuf::from("/custom/gpu/path"));
    }

    #[tokio::test]
    async fn test_concurrent_agent_access() -> Result<()> {
        // RED: Test concurrent access to agent storage
        use tokio::task;

        let temp_dir = TempDir::new()?;
        let config = super::GpuStorageConfig::test_config(temp_dir.path());
        let storage = std::sync::Arc::new(super::GpuAgentStorage::new(config)?);

        // Spawn multiple tasks accessing storage concurrently
        let mut handles = vec![];

        for i in 0..10 {
            let storage_clone = storage.clone();
            let handle = task::spawn(async move {
                let agent_id = format!("concurrent_agent_{}", i);
                let data = super::GpuAgentData::random(&agent_id);
                storage_clone.store_agent(&agent_id, &data).await
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await??;
        }

        // Verify all agents stored
        for i in 0..10 {
            let agent_id = format!("concurrent_agent_{}", i);
            assert!(storage.agent_exists(&agent_id).await?);
        }

        Ok(())
    }
}
