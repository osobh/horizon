//! Tests for fault tolerance, node failure detection, recovery, and checkpointing

use super::{create_distributed_config, create_mock_agent};
use stratoswarm_evolution_engines::{
    error::EvolutionEngineResult,
    swarm_distributed::DistributedSwarmEngine,
};
use std::time::Duration;

#[tokio::test]
async fn test_node_failure_detection_and_recovery() -> EvolutionEngineResult<()> {
    // Test fault tolerance - node failure detection and recovery
    let mut swarm_nodes = vec![];

    // Create 3 nodes
    for i in 1..=3 {
        let mut config = create_distributed_config(&format!("node_{}", i), 8030 + i as u16);
        config.fault_tolerance_config.failure_timeout_ms = 2000; // Short timeout for testing

        // Add bootstrap peers (other nodes)
        for j in 1..=3 {
            if i != j {
                config
                    .network_config
                    .bootstrap_peers
                    .push(format!("127.0.0.1:{}", 8030 + j));
            }
        }

        let node = DistributedSwarmEngine::new(config).await?;
        swarm_nodes.push(node);
    }

    // Start all nodes
    for node in &mut swarm_nodes {
        node.start().await?;
    }

    // Initialize populations
    for node in &mut swarm_nodes {
        let population = node.generate_initial_population(15).await?;
        node.set_local_population(population).await?;
    }

    // Wait for network formation
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify initial connectivity
    let initial_peer_counts: Vec<usize> = swarm_nodes
        .iter()
        .map(|node| futures::executor::block_on(node.get_peer_count()))
        .collect();

    for (i, &count) in initial_peer_counts.iter().enumerate() {
        assert!(count >= 1, "Node {} should have peers initially", i + 1);
    }

    // Simulate node failure by stopping node 2
    swarm_nodes[1].stop().await?;

    // Wait for failure detection
    tokio::time::sleep(Duration::from_millis(3000)).await;

    // Remaining nodes should detect the failure and redistribute particles
    let node1_particles = swarm_nodes[0].get_local_particle_count().await;
    let node3_particles = swarm_nodes[2].get_local_particle_count().await;

    // Total particles should be redistributed between remaining nodes
    let total_remaining = node1_particles + node3_particles;
    assert!(
        total_remaining >= 30,
        "Particles should be redistributed after node failure"
    );

    // Cleanup
    swarm_nodes[0].stop().await?;
    swarm_nodes[2].stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_particle_migration_under_load() -> EvolutionEngineResult<()> {
    // Test particle migration when nodes are under different loads
    let mut swarm_nodes = vec![];

    // Create 2 nodes with different configurations
    let mut config1 = create_distributed_config("heavy_node", 8040);
    let mut config2 = create_distributed_config("light_node", 8041);

    // Configure bidirectional bootstrap peers for proper discovery
    config1
        .network_config
        .bootstrap_peers
        .push("127.0.0.1:8041".to_string());
    config2
        .network_config
        .bootstrap_peers
        .push("127.0.0.1:8040".to_string());

    // Configure for aggressive load balancing
    config1.load_balance_config.rebalance_threshold = 0.1;
    config2.load_balance_config.rebalance_threshold = 0.1;
    config1.load_balance_config.target_particles_per_node = 10;
    config2.load_balance_config.target_particles_per_node = 10;

    let node1 = DistributedSwarmEngine::new(config1).await?;
    let node2 = DistributedSwarmEngine::new(config2).await?;

    swarm_nodes.push(node1);
    swarm_nodes.push(node2);

    // Start nodes
    for node in &mut swarm_nodes {
        node.start().await?;
    }

    // Create imbalanced load - put most particles on node 1
    let heavy_population = swarm_nodes[0].generate_initial_population(25).await?;
    let light_population = swarm_nodes[1].generate_initial_population(5).await?;

    swarm_nodes[0]
        .set_local_population(heavy_population)
        .await?;
    swarm_nodes[1]
        .set_local_population(light_population)
        .await?;

    // Initial particle counts
    let initial_heavy = swarm_nodes[0].get_local_particle_count().await;
    let initial_light = swarm_nodes[1].get_local_particle_count().await;

    assert_eq!(initial_heavy, 25);
    assert_eq!(initial_light, 5);

    // Trigger load balancing on both nodes simultaneously with coordination
    let heavy_task = swarm_nodes[0].trigger_load_balancing();
    let light_task = swarm_nodes[1].trigger_load_balancing();

    // Run both load balancing operations concurrently
    tokio::try_join!(heavy_task, light_task)?;

    // Wait for migration
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Check load is more balanced
    let final_heavy = swarm_nodes[0].get_local_particle_count().await;
    let final_light = swarm_nodes[1].get_local_particle_count().await;

    // Should be more balanced (closer to 10 each)
    let balance_diff = (final_heavy as i32 - final_light as i32).abs();
    assert!(
        balance_diff <= 10,
        "Load should be more balanced after migration"
    );
    assert_eq!(
        final_heavy + final_light,
        30,
        "Total particles should be conserved"
    );

    // Cleanup
    for node in &mut swarm_nodes {
        node.stop().await?;
    }

    Ok(())
}

#[tokio::test]
async fn test_checkpoint_and_recovery_workflow() -> EvolutionEngineResult<()> {
    // Test checkpoint creation and recovery from failure
    let mut config = create_distributed_config("checkpoint_node", 8060);
    config.fault_tolerance_config.checkpoint_interval = 5; // Checkpoint every 5 generations

    let mut node = DistributedSwarmEngine::new(config).await?;
    node.start().await?;

    // Initialize population
    let population = node.generate_initial_population(20).await?;
    node.set_local_population(population).await?;

    // Run evolution for several generations to trigger checkpointing
    for generation in 0..8 {
        node.evolve_step().await?;

        if generation == 5 {
            // Should have created a checkpoint by now
            assert!(
                node.has_checkpoint().await,
                "Should have checkpoint after generation 5"
            );
        }
    }

    // Get state before "failure"
    let pre_failure_fitness = node.get_average_fitness().await;
    let pre_failure_generation = node.get_current_generation().await;

    // Simulate failure and recovery
    node.stop().await?;

    // Create new node instance (simulating restart)
    let recovery_config = create_distributed_config("checkpoint_node", 8060);
    let recovered_node = DistributedSwarmEngine::new(recovery_config).await?;

    // Recover from checkpoint
    recovered_node.recover_from_checkpoint().await?;
    recovered_node.start().await?;

    // Verify recovery
    let post_recovery_generation = recovered_node.get_current_generation().await;
    assert!(
        post_recovery_generation <= pre_failure_generation,
        "Should recover from checkpoint generation"
    );

    // Continue evolution after recovery
    for _ in 0..3 {
        recovered_node.evolve_step().await?;
    }

    let post_recovery_fitness = recovered_node.get_average_fitness().await;

    // Evolution should continue normally after recovery
    assert!(
        post_recovery_fitness >= 0.0,
        "Should have valid fitness after recovery"
    );

    // Cleanup
    recovered_node.stop().await?;

    Ok(())
}