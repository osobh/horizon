//! Tests for hybrid evolution strategy coordination
//! 
//! Tests hybrid algorithm combinations including:
//! - Strategy selection and switching
//! - Multi-algorithm coordination
//! - Performance-based adaptation
//! - Resource allocation between strategies

use super::hybrid::*;
use super::{adas::*, dgm::*, swarm::*};
use crate::evolution::{GpuEvolutionConfig, FitnessObjective};
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use anyhow::Result;

// Helper to create test device
fn create_test_device() -> Result<Arc<CudaDevice>> {
    Ok(CudaDevice::new(0)?)
}

// Helper to create test config
fn create_test_config() -> HybridConfig {
    HybridConfig {
        population_size: 100,
        genome_size: 64,
        max_strategies: 3,
        adaptation_interval: 10,
        resource_allocation: ResourceAllocation::Dynamic,
        performance_window: 20,
        strategy_weights: vec![0.33, 0.33, 0.34],
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[test]
fn test_hybrid_config_creation() {
    let config = create_test_config();
    
    assert_eq!(config.population_size, 100);
    assert_eq!(config.max_strategies, 3);
    assert_eq!(config.strategy_weights.len(), 3);
    assert!((config.strategy_weights.iter().sum::<f32>() - 1.0).abs() < 0.01);
}

#[test]
fn test_hybrid_coordinator_creation() -> Result<()> {
    let device = create_test_device()?;
    let config = create_test_config();
    
    let coordinator = HybridCoordinator::new(device, config)?;
    
    assert_eq!(coordinator.active_strategies(), 0);
    assert!(!coordinator.is_running());
    
    Ok(())
}

#[test]
fn test_add_evolution_strategies() -> Result<()> {
    let device = create_test_device()?;
    let config = create_test_config();
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    // Add ADAS strategy
    let adas_strategy = EvolutionStrategy::Adas(AdasConfig {
        population_size: 30,
        max_code_size: 512,
        mutation_rate: 0.05,
    });
    coordinator.add_strategy("adas", adas_strategy)?;
    
    // Add DGM strategy
    let dgm_strategy = EvolutionStrategy::Dgm(DgmConfig {
        population_size: 30,
        max_code_size: 512,
        archive_size: 10,
        self_modification_rate: 0.1,
    });
    coordinator.add_strategy("dgm", dgm_strategy)?;
    
    // Add Swarm strategy
    let swarm_strategy = EvolutionStrategy::Swarm(SwarmConfig {
        population_size: 40,
        dimensions: 64,
        params: SwarmParams::default(),
    });
    coordinator.add_strategy("swarm", swarm_strategy)?;
    
    assert_eq!(coordinator.active_strategies(), 3);
    
    Ok(())
}

#[test]
fn test_strategy_initialization() -> Result<()> {
    let device = create_test_device()?;
    let config = create_test_config();
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    // Add strategies
    coordinator.add_strategy("adas", EvolutionStrategy::Adas(AdasConfig::default()))?;
    coordinator.add_strategy("dgm", EvolutionStrategy::Dgm(DgmConfig::default()))?;
    
    // Initialize all strategies
    coordinator.initialize_all()?;
    
    // Check initialization
    let status = coordinator.get_status()?;
    assert!(status.strategies_initialized);
    assert_eq!(status.total_population, 100); // Based on config
    
    Ok(())
}

#[test]
fn test_single_evolution_step() -> Result<()> {
    let device = create_test_device()?;
    let config = HybridConfig {
        population_size: 30,
        ..create_test_config()
    };
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    // Add simple strategy
    coordinator.add_strategy("adas", EvolutionStrategy::Adas(AdasConfig {
        population_size: 30,
        max_code_size: 128,
        mutation_rate: 0.1,
    }))?;
    
    coordinator.initialize_all()?;
    
    // Run single step
    let metrics = coordinator.evolve_step()?;
    
    assert_eq!(metrics.generation, 1);
    assert!(metrics.best_fitness >= 0.0);
    assert!(metrics.strategy_performances.contains_key("adas"));
    
    Ok(())
}

#[test]
fn test_multi_strategy_coordination() -> Result<()> {
    let device = create_test_device()?;
    let config = HybridConfig {
        population_size: 90,
        strategy_weights: vec![0.33, 0.33, 0.34],
        ..create_test_config()
    };
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    // Add multiple strategies
    coordinator.add_strategy("adas", EvolutionStrategy::Adas(AdasConfig {
        population_size: 30,
        ..Default::default()
    }))?;
    
    coordinator.add_strategy("dgm", EvolutionStrategy::Dgm(DgmConfig {
        population_size: 30,
        ..Default::default()
    }))?;
    
    coordinator.add_strategy("swarm", EvolutionStrategy::Swarm(SwarmConfig {
        population_size: 30,
        ..Default::default()
    }))?;
    
    coordinator.initialize_all()?;
    
    // Run multiple steps
    for _ in 0..5 {
        let metrics = coordinator.evolve_step()?;
        
        // All strategies should contribute
        assert_eq!(metrics.strategy_performances.len(), 3);
        assert!(metrics.average_fitness > 0.0);
    }
    
    Ok(())
}

#[test]
fn test_dynamic_resource_allocation() -> Result<()> {
    let device = create_test_device()?;
    let config = HybridConfig {
        population_size: 100,
        resource_allocation: ResourceAllocation::Dynamic,
        adaptation_interval: 3,
        ..create_test_config()
    };
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    // Add strategies with different initial allocations
    coordinator.add_strategy("good", EvolutionStrategy::Adas(AdasConfig {
        population_size: 50,
        ..Default::default()
    }))?;
    
    coordinator.add_strategy("bad", EvolutionStrategy::Dgm(DgmConfig {
        population_size: 50,
        ..Default::default()
    }))?;
    
    coordinator.initialize_all()?;
    
    // Simulate "good" strategy performing better
    coordinator.set_strategy_performance("good", 0.8)?;
    coordinator.set_strategy_performance("bad", 0.2)?;
    
    // Run until adaptation kicks in
    for i in 0..5 {
        coordinator.evolve_step()?;
        
        if i >= 3 {
            // After adaptation interval, resources should shift
            let allocations = coordinator.get_resource_allocations()?;
            assert!(allocations["good"] > allocations["bad"]);
        }
    }
    
    Ok(())
}

#[test]
fn test_strategy_switching() -> Result<()> {
    let device = create_test_device()?;
    let config = HybridConfig {
        population_size: 60,
        resource_allocation: ResourceAllocation::Adaptive,
        ..create_test_config()
    };
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    // Add exploration and exploitation strategies
    coordinator.add_strategy("explore", EvolutionStrategy::Adas(AdasConfig {
        population_size: 30,
        mutation_rate: 0.2, // High mutation for exploration
        ..Default::default()
    }))?;
    
    coordinator.add_strategy("exploit", EvolutionStrategy::Swarm(SwarmConfig {
        population_size: 30,
        params: SwarmParams {
            inertia_weight: 0.5, // Low inertia for exploitation
            ..Default::default()
        },
        ..Default::default()
    }))?;
    
    coordinator.initialize_all()?;
    
    // Set initial mode
    coordinator.set_evolution_phase(EvolutionPhase::Exploration)?;
    
    // Run and switch phases
    for i in 0..10 {
        if i == 5 {
            coordinator.set_evolution_phase(EvolutionPhase::Exploitation)?;
        }
        
        let metrics = coordinator.evolve_step()?;
        
        if i < 5 {
            // Exploration phase - ADAS should dominate
            assert!(metrics.strategy_performances["explore"] > 0.0);
        } else {
            // Exploitation phase - Swarm should be more active
            assert!(metrics.strategy_performances["exploit"] > 0.0);
        }
    }
    
    Ok(())
}

#[test]
fn test_population_migration() -> Result<()> {
    let device = create_test_device()?;
    let config = HybridConfig {
        population_size: 100,
        enable_migration: true,
        migration_rate: 0.1,
        migration_interval: 5,
        ..create_test_config()
    };
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    // Add two strategies
    coordinator.add_strategy("strategy1", EvolutionStrategy::Adas(AdasConfig {
        population_size: 50,
        ..Default::default()
    }))?;
    
    coordinator.add_strategy("strategy2", EvolutionStrategy::Dgm(DgmConfig {
        population_size: 50,
        ..Default::default()
    }))?;
    
    coordinator.initialize_all()?;
    
    // Run until migration occurs
    for i in 0..10 {
        let metrics = coordinator.evolve_step()?;
        
        if i % 5 == 0 && i > 0 {
            // Migration should have occurred
            assert!(metrics.migrations_performed > 0);
        }
    }
    
    Ok(())
}

#[test]
fn test_fitness_objective_handling() -> Result<()> {
    let device = create_test_device()?;
    let config = create_test_config();
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    // Set multiple fitness objectives
    coordinator.set_fitness_objectives(vec![
        FitnessObjective::Performance,
        FitnessObjective::Efficiency,
        FitnessObjective::Novelty,
    ])?;
    
    // Add strategy
    coordinator.add_strategy("multi_obj", EvolutionStrategy::Adas(AdasConfig::default()))?;
    coordinator.initialize_all()?;
    
    // Evolve and check multi-objective handling
    let metrics = coordinator.evolve_step()?;
    
    assert_eq!(metrics.objective_scores.len(), 3);
    assert!(metrics.pareto_front_size > 0);
    
    Ok(())
}

#[test]
fn test_convergence_detection() -> Result<()> {
    let device = create_test_device()?;
    let config = HybridConfig {
        convergence_threshold: 0.001,
        convergence_window: 5,
        ..create_test_config()
    };
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    coordinator.add_strategy("test", EvolutionStrategy::Swarm(SwarmConfig {
        population_size: 20,
        ..Default::default()
    }))?;
    
    coordinator.initialize_all()?;
    
    // Simulate convergence by setting similar fitness values
    for i in 0..10 {
        coordinator.evolve_step()?;
        
        // Force convergence after generation 5
        if i >= 5 {
            coordinator.set_strategy_performance("test", 0.95 + (i as f32) * 0.001)?;
        }
    }
    
    assert!(coordinator.has_converged()?);
    
    Ok(())
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_hybrid_evolution_workflow() -> Result<()> {
    let device = create_test_device()?;
    let config = HybridConfig {
        population_size: 150,
        max_strategies: 3,
        adaptation_interval: 10,
        resource_allocation: ResourceAllocation::Dynamic,
        enable_migration: true,
        migration_rate: 0.05,
        migration_interval: 10,
        ..create_test_config()
    };
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    // Setup complete hybrid system
    coordinator.add_strategy("adas", EvolutionStrategy::Adas(AdasConfig {
        population_size: 50,
        max_code_size: 256,
        mutation_rate: 0.1,
    }))?;
    
    coordinator.add_strategy("dgm", EvolutionStrategy::Dgm(DgmConfig {
        population_size: 50,
        max_code_size: 256,
        archive_size: 20,
        self_modification_rate: 0.05,
    }))?;
    
    coordinator.add_strategy("swarm", EvolutionStrategy::Swarm(SwarmConfig {
        population_size: 50,
        dimensions: 64,
        params: SwarmParams {
            inertia_weight: 0.729,
            cognitive_weight: 1.49,
            social_weight: 1.49,
            ..Default::default()
        },
    }))?;
    
    // Set objectives
    coordinator.set_fitness_objectives(vec![
        FitnessObjective::Performance,
        FitnessObjective::Efficiency,
    ])?;
    
    coordinator.initialize_all()?;
    
    // Run full evolution
    let mut best_fitness_history = Vec::new();
    
    for generation in 0..50 {
        let metrics = coordinator.evolve_step()?;
        best_fitness_history.push(metrics.best_fitness);
        
        // Adapt phase based on progress
        if generation < 20 {
            coordinator.set_evolution_phase(EvolutionPhase::Exploration)?;
        } else if generation < 40 {
            coordinator.set_evolution_phase(EvolutionPhase::Balanced)?;
        } else {
            coordinator.set_evolution_phase(EvolutionPhase::Exploitation)?;
        }
        
        // Log progress
        if generation % 10 == 0 {
            println!("Generation {}: best={:.4}, avg={:.4}, strategies={:?}",
                generation, metrics.best_fitness, metrics.average_fitness,
                metrics.strategy_performances.keys().collect::<Vec<_>>());
        }
    }
    
    // Verify improvement
    let early_avg = best_fitness_history[..10].iter().sum::<f64>() / 10.0;
    let late_avg = best_fitness_history[40..].iter().sum::<f64>() / 10.0;
    assert!(late_avg > early_avg, "Fitness should improve over time");
    
    Ok(())
}

#[test]
fn test_adaptive_strategy_selection() -> Result<()> {
    let device = create_test_device()?;
    let config = HybridConfig {
        population_size: 100,
        resource_allocation: ResourceAllocation::Adaptive,
        performance_window: 10,
        ..create_test_config()
    };
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    // Add strategies with different characteristics
    let strategies = vec![
        ("fast_converge", EvolutionStrategy::Swarm(SwarmConfig {
            population_size: 100,
            params: SwarmParams {
                inertia_weight: 0.4, // Quick convergence
                ..Default::default()
            },
            ..Default::default()
        })),
        ("slow_explore", EvolutionStrategy::Adas(AdasConfig {
            population_size: 100,
            mutation_rate: 0.2, // High exploration
            ..Default::default()
        })),
    ];
    
    for (name, strategy) in strategies {
        coordinator.add_strategy(name, strategy)?;
    }
    
    coordinator.initialize_all()?;
    
    // Track which strategy dominates over time
    let mut strategy_usage = std::collections::HashMap::new();
    
    for i in 0..30 {
        let metrics = coordinator.evolve_step()?;
        
        // Find dominant strategy
        let dominant = metrics.strategy_performances
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(name, _)| name.clone())
            .unwrap();
        
        *strategy_usage.entry(dominant).or_insert(0) += 1;
        
        // Early on, exploration should dominate
        if i < 10 {
            assert!(metrics.strategy_performances.contains_key("slow_explore"));
        }
    }
    
    // Over time, fast convergence should become more prominent
    assert!(strategy_usage.contains_key(&"fast_converge".to_string()));
    
    Ok(())
}

// =============================================================================
// Performance Tests
// =============================================================================

#[test]
#[ignore] // Run with --ignored for performance testing
fn test_hybrid_scalability() -> Result<()> {
    let device = create_test_device()?;
    
    let population_sizes = vec![100, 500, 1000, 5000];
    
    for pop_size in population_sizes {
        let config = HybridConfig {
            population_size: pop_size,
            max_strategies: 3,
            ..create_test_config()
        };
        
        let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
        
        // Add three strategies
        coordinator.add_strategy("adas", EvolutionStrategy::Adas(AdasConfig {
            population_size: pop_size / 3,
            ..Default::default()
        }))?;
        
        coordinator.add_strategy("dgm", EvolutionStrategy::Dgm(DgmConfig {
            population_size: pop_size / 3,
            ..Default::default()
        }))?;
        
        coordinator.add_strategy("swarm", EvolutionStrategy::Swarm(SwarmConfig {
            population_size: pop_size / 3,
            ..Default::default()
        }))?;
        
        coordinator.initialize_all()?;
        
        let start = std::time::Instant::now();
        
        // Run 10 generations
        for _ in 0..10 {
            coordinator.evolve_step()?;
        }
        
        let duration = start.elapsed();
        println!("Hybrid evolution with {} agents: {:?}", pop_size, duration);
        
        // Should scale reasonably
        assert!(duration.as_secs() < 60);
    }
    
    Ok(())
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_invalid_strategy_configuration() {
    let device = create_test_device().unwrap();
    let config = HybridConfig {
        population_size: 100,
        max_strategies: 2, // Limit to 2
        ..create_test_config()
    };
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    // Add two strategies (should work)
    coordinator.add_strategy("s1", EvolutionStrategy::Adas(AdasConfig::default())).unwrap();
    coordinator.add_strategy("s2", EvolutionStrategy::Dgm(DgmConfig::default())).unwrap();
    
    // Try to add third (should fail)
    let result = coordinator.add_strategy("s3", EvolutionStrategy::Swarm(SwarmConfig::default()));
    assert!(result.is_err());
}

#[test]
fn test_resource_allocation_constraints() -> Result<()> {
    let device = create_test_device()?;
    let config = HybridConfig {
        population_size: 100,
        ..create_test_config()
    };
    
    let mut coordinator = HybridCoordinator::new(device.clone(), config)?;
    
    // Try to add strategies exceeding total population
    let result = coordinator.add_strategy("big", EvolutionStrategy::Adas(AdasConfig {
        population_size: 150, // Exceeds total
        ..Default::default()
    }));
    
    assert!(result.is_err());
    
    Ok(())
}