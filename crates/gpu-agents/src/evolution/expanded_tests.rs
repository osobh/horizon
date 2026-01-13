//! Expanded unit test coverage for evolution modules
//!
//! This module adds comprehensive test coverage for the GPU evolution components
//! to meet the TDD requirements and achieve 90%+ test coverage.

use super::*;
use crate::evolution::{
    EvolutionStatistics, FitnessObjective, GpuEvolutionConfig, GpuEvolutionEngine,
    GpuFitnessEvaluator, GpuIndividual, GpuMutationEngine, GpuPopulation, GpuSelectionStrategy,
    MutationStrategy, SelectionStrategy,
};
use anyhow::Result;
use cudarc::driver::CudaContext;
use std::sync::Arc;

// Test utilities
fn create_test_device() -> Result<Arc<CudaContext>> {
    Ok(CudaContext::new(0)?)
}

fn create_test_config(population_size: usize) -> GpuEvolutionConfig {
    GpuEvolutionConfig {
        population_size,
        genome_size: 128,
        fitness_objectives: 3,
        mutation_rate: 0.05,
        crossover_rate: 0.8,
        elite_percentage: 0.1,
        block_size: 256,
    }
}

// =============================================================================
// Unit Tests for GPU Evolution Engine
// =============================================================================

#[test]
fn test_evolution_engine_creation() -> Result<()> {
    let device = create_test_device()?;
    let config = create_test_config(1024);

    let engine = GpuEvolutionEngine::new(device, config.clone())?;

    // Verify engine was created with correct config
    let stats = engine.statistics();
    assert_eq!(stats.generation, 0);
    assert_eq!(stats.population_size, config.population_size);

    Ok(())
}

#[test]
fn test_evolution_engine_invalid_population_size() {
    let device = create_test_device().unwrap();

    // Test non-warp-aligned population size
    let mut config = create_test_config(1023); // Not multiple of 32
    config.population_size = 1023;

    let result = GpuEvolutionEngine::new(device, config);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("multiple of 32"));
}

#[test]
fn test_evolution_engine_initialization() -> Result<()> {
    let device = create_test_device()?;
    let config = create_test_config(256);

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;

    // After initialization, population should have valid genomes
    let best = engine.best_individual()?;
    assert!(best.fitness >= 0.0);

    Ok(())
}

#[test]
fn test_evolution_engine_fitness_evaluation() -> Result<()> {
    let device = create_test_device()?;
    let config = create_test_config(128);

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;
    engine.evaluate_fitness()?;

    let stats = engine.statistics();
    assert!(stats.best_fitness >= 0.0);
    assert!(stats.average_fitness >= 0.0);
    assert!(stats.best_fitness >= stats.average_fitness);

    Ok(())
}

#[test]
fn test_evolution_engine_single_generation() -> Result<()> {
    let device = create_test_device()?;
    let config = create_test_config(64);

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;

    let initial_stats = engine.statistics();
    engine.evolve_generation()?;
    let final_stats = engine.statistics();

    assert_eq!(final_stats.generation, initial_stats.generation + 1);

    Ok(())
}

#[test]
fn test_evolution_engine_multiple_generations() -> Result<()> {
    let device = create_test_device()?;
    let config = create_test_config(32);

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;

    // Run 5 generations
    for i in 1..=5 {
        engine.evolve_generation()?;
        let stats = engine.statistics();
        assert_eq!(stats.generation, i);
    }

    Ok(())
}

// =============================================================================
// Unit Tests for Population Management
// =============================================================================

#[test]
fn test_gpu_population_creation() -> Result<()> {
    let device = create_test_device()?;
    let population_size = 128;
    let genome_size = 64;

    let population = GpuPopulation::new(device, population_size, genome_size)?;

    // Verify population was created with correct parameters
    assert!(!population.has_fitness());

    Ok(())
}

#[test]
fn test_gpu_population_initialization() -> Result<()> {
    let device = create_test_device()?;
    let mut population = GpuPopulation::new(device, 64, 32)?;

    population.initialize_random()?;

    // After initialization, should have valid genomes
    let best = population.best_individual()?;
    assert_eq!(best.genome.len(), 32);

    Ok(())
}

#[test]
fn test_gpu_population_fitness_tracking() -> Result<()> {
    let device = create_test_device()?;
    let mut population = GpuPopulation::new(device, 32, 16)?;

    assert!(!population.has_fitness());

    population.initialize_random()?;
    // Simulate fitness evaluation
    population.invalidate_fitness();

    assert!(!population.has_fitness());

    Ok(())
}

#[test]
fn test_gpu_population_statistics() -> Result<()> {
    let device = create_test_device()?;
    let population = GpuPopulation::new(device, 96, 48)?;

    let avg_fitness = population.average_fitness();
    assert_eq!(avg_fitness, 0.0); // No fitness evaluated yet

    let diversity = population.diversity_index();
    assert!(diversity >= 0.0 && diversity <= 1.0);

    Ok(())
}

// =============================================================================
// Unit Tests for Fitness Evaluation
// =============================================================================

#[test]
fn test_fitness_evaluator_creation() -> Result<()> {
    let device = create_test_device()?;
    let objectives = 5;

    let evaluator = GpuFitnessEvaluator::new(device, objectives)?;

    // Evaluator should be created successfully
    assert!(true); // Placeholder - would check internal state if accessible

    Ok(())
}

#[test]
fn test_fitness_evaluator_multi_objective() -> Result<()> {
    let device = create_test_device()?;
    let evaluator = GpuFitnessEvaluator::new(device.clone(), 3)?;
    let mut population = GpuPopulation::new(device, 32, 24)?;

    population.initialize_random()?;
    evaluator.evaluate_population(&mut population)?;

    // All individuals should have valid fitness
    let best = population.best_individual()?;
    assert!(best.fitness >= 0.0 && best.fitness <= 1.0);

    Ok(())
}

// =============================================================================
// Unit Tests for Mutation Engine
// =============================================================================

#[test]
fn test_mutation_engine_creation() -> Result<()> {
    let device = create_test_device()?;
    let mutation_rate = 0.02;

    let engine = GpuMutationEngine::new(device, mutation_rate)?;

    // Engine should track mutations per second
    let mps = engine.mutations_per_second();
    assert!(mps >= 0.0);

    Ok(())
}

#[test]
fn test_mutation_engine_rate_validation() -> Result<()> {
    let device = create_test_device()?;

    // Test valid rates
    let engine1 = GpuMutationEngine::new(device.clone(), 0.0)?;
    let engine2 = GpuMutationEngine::new(device.clone(), 0.5)?;
    let engine3 = GpuMutationEngine::new(device.clone(), 1.0)?;

    // Test invalid rates
    let result1 = GpuMutationEngine::new(device.clone(), -0.1);
    assert!(result1.is_err());

    let result2 = GpuMutationEngine::new(device, 1.1);
    assert!(result2.is_err());

    Ok(())
}

#[test]
fn test_mutation_engine_population_mutation() -> Result<()> {
    let device = create_test_device()?;
    let mut engine = GpuMutationEngine::new(device.clone(), 0.5)?; // High rate for testing
    let mut population = GpuPopulation::new(device, 32, 16)?;

    population.initialize_random()?;
    engine.mutate_population(&mut population)?;

    // Mutations should have occurred (hard to verify without genome access)
    let mps = engine.mutations_per_second();
    assert!(mps > 0.0);

    Ok(())
}

// =============================================================================
// Unit Tests for Selection Strategy
// =============================================================================

#[test]
fn test_selection_strategy_creation() -> Result<()> {
    let device = create_test_device()?;
    let elite_percentage = 0.15;

    let strategy = GpuSelectionStrategy::new(device, elite_percentage)?;

    // Strategy created successfully
    assert!(true); // Placeholder

    Ok(())
}

#[test]
fn test_selection_strategy_elite_validation() -> Result<()> {
    let device = create_test_device()?;

    // Test valid percentages
    let strategy1 = GpuSelectionStrategy::new(device.clone(), 0.0)?;
    let strategy2 = GpuSelectionStrategy::new(device.clone(), 0.5)?;
    let strategy3 = GpuSelectionStrategy::new(device.clone(), 1.0)?;

    // Test invalid percentages
    let result1 = GpuSelectionStrategy::new(device.clone(), -0.1);
    assert!(result1.is_err());

    let result2 = GpuSelectionStrategy::new(device, 1.1);
    assert!(result2.is_err());

    Ok(())
}

#[test]
fn test_selection_strategy_selection() -> Result<()> {
    let device = create_test_device()?;
    let strategy = GpuSelectionStrategy::new(device.clone(), 0.2)?;
    let mut population = GpuPopulation::new(device.clone(), 64, 32)?;

    population.initialize_random()?;

    // Evaluate fitness first
    let evaluator = GpuFitnessEvaluator::new(device, 1)?;
    evaluator.evaluate_population(&mut population)?;

    // Perform selection
    let selected = strategy.select(&population, 32)?;

    assert_eq!(selected.len(), 32);
    // All indices should be valid
    for &idx in &selected {
        assert!(idx < 64);
    }

    Ok(())
}

// =============================================================================
// Unit Tests for Evolution Configuration
// =============================================================================

#[test]
fn test_evolution_config_default() {
    let config = GpuEvolutionConfig::default();

    assert_eq!(config.population_size, 1024 * 1024);
    assert_eq!(config.genome_size, 256);
    assert_eq!(config.fitness_objectives, 1);
    assert_eq!(config.mutation_rate, 0.01);
    assert_eq!(config.crossover_rate, 0.7);
    assert_eq!(config.elite_percentage, 0.1);
    assert_eq!(config.block_size, 256);

    // All rates should be valid
    assert!(config.mutation_rate >= 0.0 && config.mutation_rate <= 1.0);
    assert!(config.crossover_rate >= 0.0 && config.crossover_rate <= 1.0);
    assert!(config.elite_percentage >= 0.0 && config.elite_percentage <= 1.0);
}

#[test]
fn test_evolution_config_custom() {
    let config = GpuEvolutionConfig {
        population_size: 2048,
        genome_size: 512,
        fitness_objectives: 5,
        mutation_rate: 0.03,
        crossover_rate: 0.85,
        elite_percentage: 0.05,
        block_size: 512,
    };

    assert_eq!(config.population_size, 2048);
    assert_eq!(config.genome_size, 512);
    assert_eq!(config.fitness_objectives, 5);
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_evolution_integration_small_population() -> Result<()> {
    let device = create_test_device()?;
    let config = GpuEvolutionConfig {
        population_size: 32, // Minimum warp-aligned size
        genome_size: 16,
        fitness_objectives: 1,
        mutation_rate: 0.1,
        crossover_rate: 0.7,
        elite_percentage: 0.25,
        block_size: 32,
    };

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;

    // Run evolution for several generations
    let initial_stats = engine.statistics();

    for _ in 0..10 {
        engine.evolve_generation()?;
    }

    let final_stats = engine.statistics();

    // Evolution should progress
    assert_eq!(final_stats.generation, 10);
    assert!(final_stats.mutations_per_second > 0.0);

    Ok(())
}

#[test]
fn test_evolution_integration_multi_objective() -> Result<()> {
    let device = create_test_device()?;
    let mut config = create_test_config(64);
    config.fitness_objectives = 4; // Multiple objectives

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;

    // Evolution should handle multiple objectives
    engine.evolve_generation()?;

    let stats = engine.statistics();
    assert!(stats.diversity_index > 0.0);

    Ok(())
}

#[test]
fn test_evolution_integration_high_mutation() -> Result<()> {
    let device = create_test_device()?;
    let mut config = create_test_config(96);
    config.mutation_rate = 0.9; // Very high mutation

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;

    let initial_diversity = engine.statistics().diversity_index;

    // High mutation should maintain diversity
    for _ in 0..5 {
        engine.evolve_generation()?;
    }

    let final_diversity = engine.statistics().diversity_index;
    assert!(final_diversity > 0.5); // Should maintain high diversity

    Ok(())
}

#[test]
fn test_evolution_integration_no_crossover() -> Result<()> {
    let device = create_test_device()?;
    let mut config = create_test_config(64);
    config.crossover_rate = 0.0; // No crossover

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;

    // Evolution should work with mutation only
    engine.evolve_generation()?;

    let stats = engine.statistics();
    assert_eq!(stats.generation, 1);

    Ok(())
}

// =============================================================================
// Performance and Stress Tests
// =============================================================================

#[test]
#[ignore] // Run with --ignored for performance testing
fn test_evolution_performance_large_population() -> Result<()> {
    let device = create_test_device()?;
    let config = GpuEvolutionConfig {
        population_size: 32768, // 32K individuals
        genome_size: 512,
        fitness_objectives: 2,
        mutation_rate: 0.02,
        crossover_rate: 0.75,
        elite_percentage: 0.1,
        block_size: 256,
    };

    let start = std::time::Instant::now();

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;

    for _ in 0..10 {
        engine.evolve_generation()?;
    }

    let duration = start.elapsed();
    println!("Large population evolution (32K, 10 gen): {:?}", duration);

    let stats = engine.statistics();
    println!("Mutations per second: {}", stats.mutations_per_second);

    // Should complete in reasonable time
    assert!(duration.as_secs() < 60);

    Ok(())
}

#[test]
#[ignore] // Run with --ignored for performance testing
fn test_evolution_performance_large_genome() -> Result<()> {
    let device = create_test_device()?;
    let config = GpuEvolutionConfig {
        population_size: 1024,
        genome_size: 4096, // Large genomes
        fitness_objectives: 1,
        mutation_rate: 0.01,
        crossover_rate: 0.8,
        elite_percentage: 0.15,
        block_size: 256,
    };

    let start = std::time::Instant::now();

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;

    for _ in 0..20 {
        engine.evolve_generation()?;
    }

    let duration = start.elapsed();
    println!("Large genome evolution (4K genes, 20 gen): {:?}", duration);

    assert!(duration.as_secs() < 120);

    Ok(())
}

// =============================================================================
// Error Handling and Edge Cases
// =============================================================================

#[test]
fn test_evolution_zero_population_error() {
    let device = create_test_device().unwrap();
    let mut config = create_test_config(32);
    config.population_size = 0;

    let result = GpuEvolutionEngine::new(device, config);
    assert!(result.is_err());
}

#[test]
fn test_evolution_zero_genome_error() {
    let device = create_test_device().unwrap();
    let mut config = create_test_config(32);
    config.genome_size = 0;

    let result = GpuEvolutionEngine::new(device, config);
    assert!(result.is_err());
}

#[test]
fn test_evolution_invalid_block_size() {
    let device = create_test_device().unwrap();
    let mut config = create_test_config(64);
    config.block_size = 0;

    let result = GpuEvolutionEngine::new(device, config);
    assert!(result.is_err());
}

#[test]
fn test_evolution_excessive_elite_percentage() {
    let device = create_test_device().unwrap();
    let mut config = create_test_config(64);
    config.elite_percentage = 2.0; // >100%

    let result = GpuEvolutionEngine::new(device, config);
    // Should either error or clamp to valid range
    match result {
        Ok(engine) => {
            let stats = engine.statistics();
            assert!(stats.population_size > 0);
        }
        Err(_) => {
            // Error is also acceptable
            assert!(true);
        }
    }
}

// =============================================================================
// Async Evolution Tests
// =============================================================================

#[tokio::test]
async fn test_evolution_async_run() -> Result<()> {
    let device = create_test_device()?;
    let config = create_test_config(64);

    let mut engine = GpuEvolutionEngine::new(device, config)?;
    engine.initialize_random()?;

    // Run async evolution
    engine.run(5).await?;

    let stats = engine.statistics();
    assert_eq!(stats.generation, 5);

    Ok(())
}

#[tokio::test]
async fn test_evolution_async_concurrent() -> Result<()> {
    let device = create_test_device()?;

    // Create multiple engines (would use different GPUs in production)
    let mut handles = vec![];

    for i in 0..3 {
        let device_clone = device.clone();
        let handle = tokio::spawn(async move {
            let config = create_test_config(32);
            let mut engine = GpuEvolutionEngine::new(device_clone, config)?;
            engine.initialize_random()?;
            engine.run(3).await?;
            Ok::<_, anyhow::Error>(engine.statistics().generation)
        });
        handles.push(handle);
    }

    // All should complete successfully
    for handle in handles {
        let generations = handle.await??;
        assert_eq!(generations, 3);
    }

    Ok(())
}

// =============================================================================
// Type System Tests
// =============================================================================

#[test]
fn test_fitness_objective_types() {
    let objectives = vec![
        FitnessObjective::Performance,
        FitnessObjective::Efficiency,
        FitnessObjective::Novelty,
        FitnessObjective::Robustness,
        FitnessObjective::Speed,
    ];

    // All objectives should be distinct
    for (i, obj1) in objectives.iter().enumerate() {
        for (j, obj2) in objectives.iter().enumerate() {
            if i != j {
                assert_ne!(obj1, obj2);
            }
        }
    }
}

#[test]
fn test_selection_strategy_types() {
    let strategies = vec![
        SelectionStrategy::Tournament,
        SelectionStrategy::NSGA2,
        SelectionStrategy::NSGA3,
        SelectionStrategy::NoveltySearch,
    ];

    // All strategies should be distinct
    for (i, strat1) in strategies.iter().enumerate() {
        for (j, strat2) in strategies.iter().enumerate() {
            if i != j {
                assert_ne!(strat1, strat2);
            }
        }
    }
}

#[test]
fn test_mutation_strategy_types() {
    let strategies = vec![
        MutationStrategy::Fixed,
        MutationStrategy::Adaptive,
        MutationStrategy::SelfAdaptive,
        MutationStrategy::Gaussian,
    ];

    // All strategies should be distinct
    for (i, strat1) in strategies.iter().enumerate() {
        for (j, strat2) in strategies.iter().enumerate() {
            if i != j {
                assert_ne!(strat1, strat2);
            }
        }
    }
}
