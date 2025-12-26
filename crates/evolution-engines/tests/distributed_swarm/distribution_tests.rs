//! Tests for particle distribution, load balancing, and global best synchronization

use super::{create_distributed_config, create_mock_agent};
use stratoswarm_evolution_engines::{
    error::EvolutionEngineResult,
    swarm_distributed::DistributedSwarmEngine,
};
use std::time::Duration;

#[tokio::test]
async fn test_particle_distribution_across_nodes() -> EvolutionEngineResult<()> {
    // Test particle distribution and load balancing
    let mut swarm_nodes = vec![];

    // Create 2 nodes
    for i in 1..=2 {
        let mut config = create_distributed_config(&format!("node_{}", i), 8010 + i as u16);
        config.base_config.base.population_size = 20; // Total 40 particles across 2 nodes

        // Bootstrap configuration
        if i == 2 {
            config
                .network_config
                .bootstrap_peers
                .push("127.0.0.1:8011".to_string());
        }

        let node = DistributedSwarmEngine::new(config).await?;
        swarm_nodes.push(node);
    }

    // Start nodes
    for node in &mut swarm_nodes {
        node.start().await?;
    }

    // Initialize populations on both nodes
    for node in &mut swarm_nodes {
        let population = node.generate_initial_population(20).await?;
        node.set_local_population(population).await?;
    }

    // Wait for load balancing
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Check particle distribution
    let total_particles: usize = swarm_nodes
        .iter()
        .map(|node| futures::executor::block_on(node.get_local_particle_count()))
        .sum();

    assert_eq!(total_particles, 40, "Total particles should be 40");

    // Each node should have approximately equal particles after load balancing
    for (i, node) in swarm_nodes.iter().enumerate() {
        let count = node.get_local_particle_count().await;
        assert!(
            count >= 15 && count <= 25,
            "Node {} should have 15-25 particles, got {}",
            i + 1,
            count
        );
    }

    // Cleanup
    for node in &mut swarm_nodes {
        node.stop().await?;
    }

    Ok(())
}

#[tokio::test]
async fn test_global_best_synchronization() -> EvolutionEngineResult<()> {
    // Test global best particle synchronization across nodes
    let mut swarm_nodes = vec![];

    // Create 3 nodes
    for i in 1..=3 {
        let mut config = create_distributed_config(&format!("node_{}", i), 8020 + i as u16);

        // Bootstrap configuration
        if i > 1 {
            config
                .network_config
                .bootstrap_peers
                .push("127.0.0.1:8021".to_string());
        }

        let node = DistributedSwarmEngine::new(config).await?;
        swarm_nodes.push(node);
    }

    // Start nodes and form network
    for node in &mut swarm_nodes {
        node.start().await?;
    }

    // Initialize populations
    for node in &mut swarm_nodes {
        let population = node.generate_initial_population(10).await?;
        node.set_local_population(population).await?;
    }

    // Set different best fitness on each node initially
    swarm_nodes[0].set_local_best_fitness(0.8).await?;
    swarm_nodes[1].set_local_best_fitness(0.9).await?; // This should become global best
    swarm_nodes[2].set_local_best_fitness(0.7).await?;

    // Trigger global best synchronization
    for node in &mut swarm_nodes {
        node.synchronize_global_best().await?;
    }

    // Wait for synchronization
    tokio::time::sleep(Duration::from_millis(100)).await;

    // All nodes should have the same global best fitness (0.9)
    for (i, node) in swarm_nodes.iter().enumerate() {
        let global_best = node.get_global_best_fitness().await;
        assert!(
            (global_best - 0.9).abs() < 0.01,
            "Node {} should have global best 0.9, got {}",
            i + 1,
            global_best
        );
    }

    // Cleanup
    for node in &mut swarm_nodes {
        node.stop().await?;
    }

    Ok(())
}