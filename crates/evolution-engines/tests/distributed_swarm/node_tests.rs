//! Tests for node creation, network formation, and basic node operations

use super::{create_distributed_config, create_mock_agent};
use exorust_evolution_engines::{
    error::EvolutionEngineResult,
    swarm_distributed::DistributedSwarmEngine,
};
use std::time::Duration;

#[tokio::test]
async fn test_distributed_swarm_single_node_creation() -> EvolutionEngineResult<()> {
    // Test creation of single distributed swarm node
    let config = create_distributed_config("node_1", 8001);
    let distributed_swarm = DistributedSwarmEngine::new(config).await?;

    // Should be able to create without errors
    assert_eq!(distributed_swarm.get_node_id(), "node_1");
    assert_eq!(distributed_swarm.get_local_particle_count().await, 0);

    Ok(())
}

#[tokio::test]
async fn test_multi_node_swarm_network_formation() -> EvolutionEngineResult<()> {
    // Test network formation with multiple nodes
    let mut node_configs = vec![];
    let mut swarm_nodes = vec![];

    // Create 3 distributed swarm nodes
    for i in 1..=3 {
        let mut config = create_distributed_config(&format!("node_{}", i), 8000 + i as u16);

        // Add bootstrap peers (other nodes)
        for j in 1..=3 {
            if i != j {
                config
                    .network_config
                    .bootstrap_peers
                    .push(format!("127.0.0.1:{}", 8000 + j));
            }
        }

        node_configs.push(config.clone());
        let node = DistributedSwarmEngine::new(config).await?;
        swarm_nodes.push(node);
    }

    // Start all nodes
    for node in &mut swarm_nodes {
        node.start().await?;
    }

    // Wait for network formation
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Check network connectivity
    for (i, node) in swarm_nodes.iter().enumerate() {
        let peer_count = node.get_peer_count().await;
        assert!(
            peer_count >= 1,
            "Node {} should have at least 1 peer, got {}",
            i + 1,
            peer_count
        );
    }

    // Cleanup
    for node in &mut swarm_nodes {
        node.stop().await?;
    }

    Ok(())
}