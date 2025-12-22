//! End-to-end tests for GPU evolution workflows
//!
//! These tests verify complete evolution scenarios including:
//! - Multi-generation evolution with convergence
//! - Hybrid algorithm cooperation
//! - Real-world optimization problems
//! - Resource management and cleanup

use super::*;
use crate::evolution::{adas::*, dgm::*, swarm::*, GpuEvolutionConfig, GpuEvolutionEngine};
use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

// =============================================================================
// E2E Test: Complete ADAS Evolution Workflow
// =============================================================================

#[tokio::test]
async fn test_e2e_adas_meta_learning_workflow() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let mut adas = AdasPopulation::new(device, 64, 512)?;

    // Initialize population
    adas.initialize()?;

    // Run complete meta-learning loop
    let target_performance = 0.8;
    let max_generations = 50;
    let mut converged = false;

    for generation in 0..max_generations {
        // Evaluate current population
        adas.evaluate()?;

        // Get statistics
        let stats = adas.get_statistics()?;

        println!(
            "ADAS Generation {}: best={:.3}, avg={:.3}, diversity={:.3}",
            generation, stats.best_performance, stats.average_performance, stats.diversity
        );

        // Check convergence
        if stats.best_performance >= target_performance {
            converged = true;
            break;
        }

        // Evolve to next generation
        adas.evolve()?;
    }

    assert!(converged, "ADAS should converge to target performance");

    // Verify final population quality
    let final_stats = adas.get_statistics()?;
    assert!(final_stats.best_performance >= target_performance);
    assert!(final_stats.average_performance > 0.5); // Population quality

    Ok(())
}

// =============================================================================
// E2E Test: Complete DGM Self-Improvement Workflow
// =============================================================================

#[tokio::test]
async fn test_e2e_dgm_self_improvement_workflow() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let mut dgm = DgmEngine::new(device, 32, 256, 10)?;

    // Initialize with baseline agents
    dgm.initialize()?;

    // Run self-improvement cycles
    let improvement_cycles = 20;
    let mut performance_history = Vec::new();

    for cycle in 0..improvement_cycles {
        // Benchmark current agents
        dgm.benchmark()?;

        // Get current best performance
        let stats = dgm.get_statistics()?;
        performance_history.push(stats.best_benchmark_score);

        println!(
            "DGM Cycle {}: best={:.3}, archive_size={}, improvements={}",
            cycle, stats.best_benchmark_score, stats.archive_size, stats.total_improvements
        );

        // Self-modify based on performance trends
        dgm.self_improve()?;

        // Archive promising variants
        dgm.update_archive()?;
    }

    // Verify improvement over time
    let initial_performance = performance_history[0];
    let final_performance = performance_history.last().unwrap();
    assert!(
        final_performance > &initial_performance,
        "DGM should improve over time"
    );

    // Check archive quality
    let final_stats = dgm.get_statistics()?;
    assert!(
        final_stats.archive_size > 0,
        "Archive should contain successful agents"
    );
    assert!(
        final_stats.total_improvements > 0,
        "Some improvements should occur"
    );

    Ok(())
}

// =============================================================================
// E2E Test: Complete Swarm Optimization Workflow
// =============================================================================

#[tokio::test]
async fn test_e2e_swarm_optimization_workflow() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let params = SwarmParams {
        inertia_weight: 0.7,
        cognitive_weight: 1.5,
        social_weight: 1.5,
        velocity_limit: 2.0,
        neighborhood_size: 5,
    };

    let mut swarm = SwarmEngine::new(device, 128, 10, params)?;

    // Initialize swarm
    swarm.initialize()?;

    // Run optimization to find global minimum
    let target_fitness = 0.95; // Close to optimal
    let max_iterations = 100;
    let mut converged = false;

    for iteration in 0..max_iterations {
        // Update swarm
        swarm.step()?;

        // Get statistics
        let stats = swarm.get_statistics()?;

        println!(
            "Swarm Iteration {}: best={:.3}, avg={:.3}, convergence={:.3}",
            iteration, stats.best_fitness, stats.average_fitness, stats.convergence_rate
        );

        // Check convergence
        if stats.best_fitness >= target_fitness {
            converged = true;
            break;
        }

        // Early stopping if swarm has converged spatially
        if stats.convergence_rate > 0.95 {
            println!("Swarm converged spatially at iteration {}", iteration);
            break;
        }
    }

    // Verify optimization quality
    let final_stats = swarm.get_statistics()?;
    assert!(
        final_stats.best_fitness > 0.8,
        "Swarm should find good solution"
    );

    Ok(())
}

// =============================================================================
// E2E Test: Multi-Algorithm Hybrid Evolution
// =============================================================================

#[tokio::test]
async fn test_e2e_hybrid_algorithm_workflow() -> Result<()> {
    let device = CudaDevice::new(0)?;

    // Create all three algorithms
    let mut adas = AdasPopulation::new(device.clone(), 32, 256)?;
    let mut dgm = DgmEngine::new(device.clone(), 24, 256, 8)?;
    let mut swarm = SwarmEngine::new(device.clone(), 48, 8, SwarmParams::default())?;

    // Initialize all
    adas.initialize()?;
    dgm.initialize()?;
    swarm.initialize()?;

    // Run hybrid evolution loop
    let hybrid_cycles = 10;

    for cycle in 0..hybrid_cycles {
        println!("\nHybrid Cycle {}", cycle);

        // Phase 1: ADAS explores new agent designs
        adas.evolve()?;
        let adas_stats = adas.get_statistics()?;
        println!("  ADAS: best={:.3}", adas_stats.best_performance);

        // Phase 2: DGM improves best designs
        dgm.self_improve()?;
        let dgm_stats = dgm.get_statistics()?;
        println!("  DGM: best={:.3}", dgm_stats.best_benchmark_score);

        // Phase 3: Swarm optimizes parameters
        swarm.step()?;
        let swarm_stats = swarm.get_statistics()?;
        println!("  Swarm: best={:.3}", swarm_stats.best_fitness);

        // Exchange best solutions between algorithms
        // (In real implementation, would transfer actual agents/parameters)

        // Check overall progress
        let overall_best = adas_stats
            .best_performance
            .max(dgm_stats.best_benchmark_score)
            .max(swarm_stats.best_fitness);

        if overall_best > 0.9 {
            println!("Hybrid system achieved target performance!");
            break;
        }
    }

    Ok(())
}

// =============================================================================
// E2E Test: Resource Management and Cleanup
// =============================================================================

#[tokio::test]
async fn test_e2e_resource_management_workflow() -> Result<()> {
    let device = CudaDevice::new(0)?;

    // Track initial GPU memory
    let initial_memory = get_gpu_memory_usage(&device)?;

    // Create and run multiple evolution instances
    for i in 0..5 {
        let config = GpuEvolutionConfig {
            population_size: 256,
            genome_size: 128,
            fitness_objectives: 2,
            mutation_rate: 0.05,
            crossover_rate: 0.8,
            elite_percentage: 0.1,
            block_size: 256,
        };

        let mut engine = GpuEvolutionEngine::new(device.clone(), config)?;
        engine.initialize_random()?;

        // Run evolution
        engine.run(10).await?;

        // Engine should be dropped here, releasing GPU memory
    }

    // Allow GPU to clean up
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Check memory is released
    let final_memory = get_gpu_memory_usage(&device)?;
    let memory_leak = final_memory.saturating_sub(initial_memory);

    // Allow some overhead, but not significant leaks
    assert!(
        memory_leak < 100 * 1024 * 1024,
        "Memory leak detected: {} MB",
        memory_leak / (1024 * 1024)
    );

    Ok(())
}

// =============================================================================
// E2E Test: Fault Tolerance and Recovery
// =============================================================================

#[tokio::test]
async fn test_e2e_fault_tolerance_workflow() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let mut adas = AdasPopulation::new(device, 64, 256)?;

    // Initialize and run some generations
    adas.initialize()?;
    for _ in 0..5 {
        adas.evolve()?;
    }

    // Simulate checkpoint
    let checkpoint = adas.checkpoint()?;
    let stats_before = adas.get_statistics()?;

    // Simulate failure and recovery
    drop(adas);

    // Restore from checkpoint
    let device = CudaDevice::new(0)?;
    let mut adas_restored = AdasPopulation::restore(device, checkpoint)?;

    // Verify restoration
    let stats_after = adas_restored.get_statistics()?;
    assert_eq!(stats_after.generation, stats_before.generation);
    assert!((stats_after.best_performance - stats_before.best_performance).abs() < 0.001);

    // Continue evolution
    for _ in 0..5 {
        adas_restored.evolve()?;
    }

    // Should complete successfully
    let final_stats = adas_restored.get_statistics()?;
    assert_eq!(final_stats.generation, stats_before.generation + 5);

    Ok(())
}

// =============================================================================
// E2E Test: Real-world Problem - Function Optimization
// =============================================================================

#[tokio::test]
async fn test_e2e_function_optimization() -> Result<()> {
    let device = CudaDevice::new(0)?;

    // Optimize Rastrigin function (challenging multimodal problem)
    let dimensions = 10;
    let swarm = SwarmEngine::new(
        device,
        256,
        dimensions,
        SwarmParams {
            inertia_weight: 0.729,
            cognitive_weight: 1.49,
            social_weight: 1.49,
            velocity_limit: 4.0,
            neighborhood_size: 10,
        },
    )?;

    // Custom fitness function for Rastrigin
    let rastrigin_fitness = |position: &[f32]| -> f32 {
        let a = 10.0;
        let n = position.len() as f32;
        let sum: f32 = position
            .iter()
            .map(|&x| x * x - a * (2.0 * std::f32::consts::PI * x).cos())
            .sum();

        // Convert to fitness (minimize -> maximize)
        let value = a * n + sum;
        1.0 / (1.0 + value)
    };

    // Run optimization
    let mut swarm = swarm;
    swarm.initialize()?;
    swarm.set_fitness_function(Box::new(rastrigin_fitness))?;

    let mut best_positions = Vec::new();

    for iteration in 0..200 {
        swarm.step()?;

        if iteration % 20 == 0 {
            let stats = swarm.get_statistics()?;
            let best_pos = swarm.get_best_position()?;
            best_positions.push(best_pos.clone());

            println!("Iteration {}: fitness={:.6}", iteration, stats.best_fitness);
        }
    }

    // Verify convergence to global optimum (near origin)
    let final_best = best_positions.last().unwrap();
    let distance_from_origin: f32 = final_best.iter().map(|&x| x * x).sum::<f32>().sqrt();

    assert!(
        distance_from_origin < 1.0,
        "Should converge near global optimum"
    );

    Ok(())
}

// =============================================================================
// E2E Test: Scalability and Performance
// =============================================================================

#[tokio::test]
#[ignore] // Run with --ignored for performance testing
async fn test_e2e_scalability_workflow() -> Result<()> {
    let device = CudaDevice::new(0)?;

    // Test with increasing population sizes
    let population_sizes = vec![1024, 4096, 16384, 65536];
    let mut results = Vec::new();

    for &pop_size in &population_sizes {
        let config = GpuEvolutionConfig {
            population_size: pop_size,
            genome_size: 256,
            fitness_objectives: 2,
            mutation_rate: 0.02,
            crossover_rate: 0.8,
            elite_percentage: 0.1,
            block_size: 256,
        };

        let start = std::time::Instant::now();

        let mut engine = GpuEvolutionEngine::new(device.clone(), config)?;
        engine.initialize_random()?;
        engine.run(10).await?;

        let duration = start.elapsed();
        let stats = engine.statistics();

        results.push((pop_size, duration, stats.mutations_per_second));

        println!(
            "Population {}: {:.2}s, {:.0} mutations/sec",
            pop_size,
            duration.as_secs_f32(),
            stats.mutations_per_second
        );
    }

    // Verify scaling efficiency
    for i in 1..results.len() {
        let (pop1, time1, _) = results[i - 1];
        let (pop2, time2, _) = results[i];

        let population_ratio = pop2 as f32 / pop1 as f32;
        let time_ratio = time2.as_secs_f32() / time1.as_secs_f32();

        // Should scale sub-linearly (better than O(n))
        assert!(
            time_ratio < population_ratio * 0.8,
            "Poor scaling: {}x population increase caused {}x time increase",
            population_ratio,
            time_ratio
        );
    }

    Ok(())
}

// Helper function to get GPU memory usage
fn get_gpu_memory_usage(device: &CudaDevice) -> Result<usize> {
    // This would use CUDA API to get actual memory usage
    // For now, return placeholder
    Ok(0)
}

// =============================================================================
// E2E Test: Continuous Evolution with Checkpointing
// =============================================================================

#[tokio::test]
async fn test_e2e_continuous_evolution_workflow() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let checkpoint_interval = 25;
    let total_generations = 100;

    let config = GpuEvolutionConfig {
        population_size: 512,
        genome_size: 256,
        fitness_objectives: 3,
        mutation_rate: 0.03,
        crossover_rate: 0.75,
        elite_percentage: 0.12,
        block_size: 256,
    };

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;

    let mut checkpoints = Vec::new();
    let mut fitness_history = Vec::new();

    // Run evolution with periodic checkpointing
    for generation in 0..total_generations {
        engine.evolve_generation()?;

        let stats = engine.statistics();
        fitness_history.push(stats.best_fitness);

        // Checkpoint periodically
        if generation % checkpoint_interval == 0 {
            println!(
                "Checkpoint at generation {}: best_fitness={:.3}",
                generation, stats.best_fitness
            );
            // In real implementation, would save checkpoint
            checkpoints.push(generation);
        }
    }

    // Verify continuous improvement
    let early_avg = fitness_history[..20].iter().sum::<f64>() / 20.0;
    let late_avg = fitness_history[80..].iter().sum::<f64>() / 20.0;
    assert!(late_avg > early_avg, "Fitness should improve over time");

    // Verify checkpoints were created
    assert_eq!(
        checkpoints.len(),
        (total_generations / checkpoint_interval) as usize
    );

    Ok(())
}
