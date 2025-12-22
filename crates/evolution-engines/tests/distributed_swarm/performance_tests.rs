//! Performance and load testing for distributed swarm systems

use super::{create_distributed_config, create_mock_agent};
use exorust_evolution_engines::{
    error::EvolutionEngineResult,
    swarm_distributed::DistributedSwarmEngine,
};
use std::time::{Duration, Instant};

#[tokio::test]
async fn test_distributed_evolution_convergence() -> EvolutionEngineResult<()> {
    // Test distributed evolution convergence across multiple nodes
    let mut swarm_nodes = vec![];

    // Create 2 nodes
    for i in 1..=2 {
        let mut config = create_distributed_config(&format!("evolution_node_{}", i), 8050 + i as u16);

        if i == 2 {
            config
                .network_config
                .bootstrap_peers
                .push("127.0.0.1:8051".to_string());
        }

        let node = DistributedSwarmEngine::new(config).await?;
        swarm_nodes.push(node);
    }

    // Start nodes
    for node in &mut swarm_nodes {
        node.start().await?;
    }

    // Initialize populations
    for node in &mut swarm_nodes {
        let population = node.generate_initial_population(15).await?;
        node.set_local_population(population).await?;
    }

    // Record initial fitness
    let mut initial_fitness = 0.0;
    for node in &swarm_nodes {
        initial_fitness += node.get_average_fitness().await;
    }
    initial_fitness /= swarm_nodes.len() as f64;

    // Run distributed evolution
    for _generation in 0..10 {
        // Evolve each node
        for node in &mut swarm_nodes {
            node.evolve_step().await?;
        }

        // Synchronize global best
        for node in &mut swarm_nodes {
            node.synchronize_global_best().await?;
        }

        // Brief pause for synchronization
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Check final fitness
    let mut final_fitness = 0.0;
    for node in &swarm_nodes {
        final_fitness += node.get_average_fitness().await;
    }
    final_fitness /= swarm_nodes.len() as f64;

    // Fitness should improve or at least maintain (accounting for stochastic nature)
    assert!(
        final_fitness >= initial_fitness - 0.1,
        "Distributed evolution should maintain or improve fitness"
    );

    // All nodes should have similar global best
    let global_bests: Vec<f64> = swarm_nodes
        .iter()
        .map(|node| futures::executor::block_on(node.get_global_best_fitness()))
        .collect();

    let max_diff = global_bests.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        - global_bests.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    assert!(
        max_diff < 0.1,
        "Global best should be synchronized across nodes"
    );

    // Cleanup
    for node in &mut swarm_nodes {
        node.stop().await?;
    }

    Ok(())
}

#[tokio::test]
async fn test_performance_under_high_load() -> EvolutionEngineResult<()> {
    // Test performance with high particle counts and frequent operations
    let mut config = create_distributed_config("perf_node", 8080);
    config.base_config.base.population_size = 100; // Large population
    config.network_config.heartbeat_interval_ms = 500; // Frequent heartbeats

    let mut node = DistributedSwarmEngine::new(config).await?;
    node.start().await?;

    // Initialize large population
    let population = node.generate_initial_population(100).await?;
    node.set_local_population(population).await?;

    // Measure performance under load
    let start_time = Instant::now();

    // Run intensive operations
    for _ in 0..20 {
        node.evolve_step().await?;
        node.synchronize_global_best().await?;
    }

    let elapsed = start_time.elapsed();

    // Should complete within reasonable time (adjust threshold as needed)
    assert!(
        elapsed < Duration::from_secs(30),
        "High load test should complete within 30 seconds, took {:?}",
        elapsed
    );

    // Verify system still functions correctly
    let final_population_size = node.get_local_particle_count().await;
    assert_eq!(
        final_population_size, 100,
        "Population size should be maintained"
    );

    let global_best = node.get_global_best_fitness().await;
    assert!(global_best >= 0.0, "Should have valid global best fitness");

    // Cleanup
    node.stop().await?;

    Ok(())
}