//! Integration tests for GPU agent storage
//! Following rust.md TDD principles

use anyhow::Result;
use gpu_agents::{
    GpuAgentData, GpuAgentStorage, GpuKnowledgeGraph, GpuStorageConfig, GpuSwarm, GpuSwarmConfig,
    StorageGraphEdge, StorageGraphNode,
};
use std::collections::HashMap;
use tempfile::TempDir;

#[tokio::test]
async fn test_gpu_agent_storage_integration() -> Result<()> {
    // Create temporary directory for testing
    let temp_dir = TempDir::new()?;

    // Configure storage to use temp directory
    let storage_config = GpuStorageConfig::with_base_path(temp_dir.path());
    let storage = GpuAgentStorage::new(storage_config)?;

    // Create test agent data
    let agent_data = GpuAgentData {
        id: "test_agent_001".to_string(),
        state: vec![0.1, 0.2, 0.3, 0.4],
        memory: vec![0.5; 64],
        generation: 10,
        fitness: 0.85,
        metadata: HashMap::new(),
    };

    // Store and retrieve agent
    storage.store_agent(&agent_data.id, &agent_data).await?;
    let retrieved = storage.retrieve_agent(&agent_data.id).await?;

    // Verify data integrity
    assert_eq!(retrieved.id, agent_data.id);
    assert_eq!(retrieved.state, agent_data.state);
    assert_eq!(retrieved.generation, agent_data.generation);
    assert_eq!(retrieved.fitness, agent_data.fitness);

    Ok(())
}

#[tokio::test]
async fn test_gpu_swarm_with_storage() -> Result<()> {
    // Skip if no GPU available
    if cudarc::driver::CudaDevice::new(0).is_err() {
        println!("Skipping GPU test - no CUDA device available");
        return Ok(());
    }

    // Create temporary directory for testing
    let temp_dir = TempDir::new()?;

    // Configure GPU swarm
    let swarm_config = GpuSwarmConfig {
        device_id: 0,
        max_agents: 1000,
        enable_persistence: true,
        enable_knowledge_graph: true,
        ..Default::default()
    };

    // Create GPU swarm
    let mut swarm = GpuSwarm::new(swarm_config)?;
    swarm.initialize(100)?;

    // Configure storage
    let storage_config = GpuStorageConfig::with_base_path(temp_dir.path());
    let storage = GpuAgentStorage::new(storage_config)?;

    // Store swarm state
    for i in 0..10 {
        let agent_data = GpuAgentData {
            id: format!("swarm_agent_{}", i),
            state: vec![i as f32; 256],
            memory: vec![0.0; 128],
            generation: 1,
            fitness: i as f64 / 10.0,
            metadata: HashMap::new(),
        };

        storage.store_agent(&agent_data.id, &agent_data).await?;
    }

    // Verify all agents stored
    for i in 0..10 {
        let agent_id = format!("swarm_agent_{}", i);
        assert!(storage.agent_exists(&agent_id).await?);
    }

    Ok(())
}

#[tokio::test]
async fn test_knowledge_graph_storage_integration() -> Result<()> {
    // Create temporary directory for testing
    let temp_dir = TempDir::new()?;

    // Configure storage
    let storage_config = GpuStorageConfig::with_base_path(temp_dir.path());
    let storage = GpuAgentStorage::new(storage_config)?;

    // Create knowledge graph
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Add nodes
    for i in 0..5 {
        nodes.push(StorageGraphNode {
            id: format!("node_{}", i),
            embedding: vec![i as f32 * 0.1; 768],
            metadata: HashMap::new(),
        });
    }

    // Add edges
    for i in 0..4 {
        edges.push(StorageGraphEdge {
            source: format!("node_{}", i),
            target: format!("node_{}", i + 1),
            weight: 0.5,
            edge_type: "connected".to_string(),
        });
    }

    let graph = GpuKnowledgeGraph { nodes, edges };

    // Store and retrieve graph
    storage.store_knowledge_graph("test_graph", &graph).await?;
    let retrieved = storage.retrieve_knowledge_graph("test_graph").await?;

    // Verify graph integrity
    assert_eq!(retrieved.nodes.len(), 5);
    assert_eq!(retrieved.edges.len(), 4);
    assert_eq!(retrieved.nodes[0].id, "node_0");
    assert_eq!(retrieved.edges[0].source, "node_0");
    assert_eq!(retrieved.edges[0].target, "node_1");

    Ok(())
}

#[tokio::test]
async fn test_production_path_configuration() {
    // Test production configuration uses /magikdev/gpu
    let prod_config = GpuStorageConfig::production();
    assert_eq!(prod_config.base_path.to_str().unwrap(), "/magikdev/gpu");
    assert_eq!(prod_config.cache_path.to_str()?, "/magikdev/gpu/cache");
    assert_eq!(prod_config.wal_path.to_str()?, "/magikdev/gpu/wal");
}

#[tokio::test]
async fn test_storage_performance() -> Result<()> {
    // Create temporary directory for testing
    let temp_dir = TempDir::new()?;

    // Configure storage with performance settings
    let mut storage_config = GpuStorageConfig::with_base_path(temp_dir.path());
    storage_config.enable_gpu_cache = true;
    storage_config.cache_size_mb = 256;

    let storage = GpuAgentStorage::new(storage_config)?;

    // Benchmark storage operations
    let agent_count = 1000;
    let start = std::time::Instant::now();

    // Store agents
    for i in 0..agent_count {
        let agent_data = GpuAgentData {
            id: format!("perf_agent_{}", i),
            state: vec![i as f32; 256],
            memory: vec![0.0; 128],
            generation: 1,
            fitness: 0.5,
            metadata: HashMap::new(),
        };

        storage.store_agent(&agent_data.id, &agent_data).await?;
    }

    let store_duration = start.elapsed();
    let store_rate = agent_count as f64 / store_duration.as_secs_f64();

    println!("Storage rate: {:.0} agents/second", store_rate);

    // Retrieve agents (should hit cache)
    let retrieve_start = std::time::Instant::now();

    for i in 0..100 {
        let agent_id = format!("perf_agent_{}", i);
        let _ = storage.retrieve_agent(&agent_id).await?;
    }

    let retrieve_duration = retrieve_start.elapsed();
    let retrieve_rate = 100.0 / retrieve_duration.as_secs_f64();

    println!("Retrieval rate: {:.0} agents/second", retrieve_rate);

    // Verify performance thresholds
    assert!(
        store_rate > 100.0,
        "Storage should exceed 100 agents/second"
    );
    assert!(
        retrieve_rate > 1000.0,
        "Retrieval should exceed 1000 agents/second"
    );

    Ok(())
}
