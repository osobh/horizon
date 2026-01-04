//! Tests for population evolution system

use super::*;
use crate::traits::{AgentGenome, ArchitectureGenes, BehaviorGenes, EvolvableAgent};
use std::time::Duration;
use stratoswarm_agent_core::{Agent, AgentConfig, Goal};

type TestResult = Result<(), Box<dyn std::error::Error>>;

// Helper function to create test agents
fn create_test_agents(count: usize) -> Vec<EvolvableAgent> {
    (0..count)
        .map(|i| {
            let config = AgentConfig {
                name: format!("agent_{}", i),
                agent_type: "test".to_string(),
                max_memory: 1024,
                max_gpu_memory: 256,
                priority: 1,
                metadata: serde_json::Value::Null,
            };
            let agent = Agent::new(config).unwrap();
            let genome = AgentGenome {
                goal: Goal::new(
                    format!("goal_{}", i),
                    stratoswarm_agent_core::GoalPriority::Normal,
                ),
                architecture: ArchitectureGenes {
                    memory_capacity: 1024,
                    processing_units: 4,
                    network_topology: vec![64, 32, 16],
                },
                behavior: BehaviorGenes {
                    exploration_rate: i as f64 / count as f64,
                    learning_rate: 0.01,
                    risk_tolerance: 0.5,
                },
            };
            EvolvableAgent { agent, genome }
        })
        .collect()
}

// Helper function to create test config
fn create_test_config() -> PopulationConfig {
    PopulationConfig {
        num_populations: 3,
        population_size: 10,
        max_generations: 50,
        migration_frequency: 5,
        selection_strategy: SelectionStrategy::Tournament { size: 3 },
        crossover_strategy: CrossoverStrategy::Uniform { rate: 0.8 },
        migration_policy: MigrationPolicy::BestAgent { rate: 0.2 },
        diversity_threshold: 0.4,
        convergence_threshold: 0.05,
    }
}

#[test]
fn test_population_evolution_manager_creation() {
    let config = create_test_config();
    let manager = PopulationEvolutionManager::new(config.clone()).unwrap();

    assert_eq!(manager.get_population_count(), config.num_populations);
    assert_eq!(manager.get_config().num_populations, config.num_populations);

    // Verify all populations are initialized
    for i in 0..config.num_populations {
        let pop_id = &format!("population_{}", i);
        assert!(manager.get_population(pop_id).is_some());
    }
}

#[test]
fn test_managed_population_creation() {
    let agents = create_test_agents(10);
    let population = ManagedPopulation::new("test_pop".to_string(), agents.clone()).unwrap();

    assert_eq!(population.id(), "test_pop");
    assert_eq!(population.size(), 10);
    assert_eq!(population.generation(), 0);
    assert!(!population.is_converged());

    // Check diversity calculation
    let diversity = population.calculate_diversity();
    assert!(diversity >= 0.0 && diversity <= 1.0);
}

#[test]
fn test_population_evolution_step() -> TestResult {
    let agents = create_test_agents(10);
    let mut population = ManagedPopulation::new("test_pop".to_string(), agents).unwrap();

    let initial_generation = population.generation();
    population.evolve_generation(
        &SelectionStrategy::Tournament { size: 3 },
        &CrossoverStrategy::Uniform { rate: 0.7 },
    )?;

    assert_eq!(population.generation(), initial_generation + 1);
    assert_eq!(population.size(), 10); // Population size should remain constant
    Ok(())
}

#[test]
fn test_selection_strategies() -> TestResult {
    let agents = create_test_agents(10);
    let population = ManagedPopulation::new("test_pop".to_string(), agents).unwrap();

    // Test tournament selection
    let selected = population.select_parents(&SelectionStrategy::Tournament { size: 3 }, 5)?;
    assert_eq!(selected.len(), 5);

    // Test elite selection
    let selected = population
        .select_parents(&SelectionStrategy::Elite { count: 3 }, 3)
        .unwrap();
    assert_eq!(selected.len(), 3);

    // Elite selection should return best agents
    for agent in &selected {
        assert!(agent.genome.behavior.exploration_rate >= 0.7); // Should be high exploration rate
    }
    Ok(())
}

#[test]
fn test_crossover_strategies() -> TestResult {
    let agents = create_test_agents(4);
    let population = ManagedPopulation::new("test_pop".to_string(), agents).unwrap();
    let parent1 = &population.get_agents()[0];
    let parent2 = &population.get_agents()[1];

    // Test uniform crossover
    let offspring =
        population.crossover(parent1, parent2, &CrossoverStrategy::Uniform { rate: 0.5 })?;
    assert_eq!(offspring.len(), 2);
    assert_ne!(offspring[0].agent.id(), parent1.agent.id());
    assert_ne!(offspring[1].agent.id(), parent2.agent.id());

    // Test single point crossover
    let offspring = population
        .crossover(parent1, parent2, &CrossoverStrategy::SinglePoint)
        .unwrap();
    assert_eq!(offspring.len(), 2);
    Ok(())
}

#[test]
fn test_migration_controller_creation() {
    let config = create_test_config();
    let controller = MigrationController::new(config.migration_policy.clone()).unwrap();

    assert_eq!(controller.get_migration_count(), 0);
}

#[test]
fn test_migration_between_populations() -> TestResult {
    let config = create_test_config();
    let controller = MigrationController::new(config.migration_policy.clone()).unwrap();

    let agents1 = create_test_agents(10);
    let agents2 = create_test_agents(10);
    let pop1 = ManagedPopulation::new("pop1".to_string(), agents1)?;
    let pop2 = ManagedPopulation::new("pop2".to_string(), agents2)?;

    let migration = controller.plan_migration(&pop1, &pop2, 5)?;

    assert_eq!(migration.source_population, "pop1");
    assert_eq!(migration.target_population, "pop2");
    assert!(migration.agents.len() <= 2); // Based on 0.2 rate and size 10
    assert_eq!(migration.generation, 5);
    Ok(())
}

#[test]
fn test_population_metrics_calculation() {
    let config = create_test_config();
    let manager = PopulationEvolutionManager::new(config).unwrap();

    let metrics = manager.calculate_metrics().unwrap();

    assert_eq!(metrics.total_populations, 3);
    assert!(metrics.active_populations <= 3);
    assert!(metrics.global_best_fitness >= 0.0);
    assert!(metrics.average_fitness >= 0.0);
    assert_eq!(metrics.population_diversities.len(), 3);
    assert_eq!(metrics.convergence_status.len(), 3);
}

#[test]
fn test_diversity_calculation() {
    let agents = create_test_agents(10);
    let population = ManagedPopulation::new("test_pop".to_string(), agents).unwrap();

    let diversity = population.calculate_diversity();
    assert!(diversity >= 0.0 && diversity <= 1.0);

    // Test with identical agents (should have low diversity)
    let identical_agents: Vec<EvolvableAgent> = (0..5)
        .map(|i| {
            let config = AgentConfig {
                name: format!("identical_{}", i),
                agent_type: "test".to_string(),
                max_memory: 1024,
                max_gpu_memory: 256,
                priority: 1,
                metadata: serde_json::Value::Null,
            };
            let agent = Agent::new(config).unwrap();
            let genome = AgentGenome {
                goal: Goal::new(
                    "identical_goal".to_string(),
                    stratoswarm_agent_core::GoalPriority::Normal,
                ),
                architecture: ArchitectureGenes {
                    memory_capacity: 1024,
                    processing_units: 4,
                    network_topology: vec![64, 32, 16],
                },
                behavior: BehaviorGenes {
                    exploration_rate: 0.5,
                    learning_rate: 0.01,
                    risk_tolerance: 0.5,
                },
            };
            EvolvableAgent { agent, genome }
        })
        .collect();

    let identical_pop = ManagedPopulation::new("identical".to_string(), identical_agents).unwrap();
    let identical_diversity = identical_pop.calculate_diversity();
    assert!(identical_diversity < diversity); // Should be less diverse
}

#[test]
fn test_convergence_detection() -> TestResult {
    let mut agents = create_test_agents(5);
    // Make all agents have similar behavior (converged)
    for agent in &mut agents {
        agent.genome.behavior.exploration_rate = 0.85;
    }

    let population = ManagedPopulation::new("converged".to_string(), agents)?;
    let converged = population.check_convergence(0.05); // 5% threshold

    assert!(converged);
    Ok(())
}

#[test]
fn test_parallel_evolution() -> TestResult {
    let config = create_test_config();
    let mut manager = PopulationEvolutionManager::new(config).unwrap();

    // Initialize populations with test agents
    for i in 0..3 {
        let agents = create_test_agents(10);
        let pop_id = format!("population_{}", i);
        manager.initialize_population(&pop_id, agents)?;
    }

    // Run one generation
    manager.evolve_generation().unwrap();

    // Check that all populations evolved
    let metrics = manager.calculate_metrics().unwrap();
    assert!(metrics.total_populations > 0);
    Ok(())
}

#[test]
fn test_migration_policy_best_agent() -> TestResult {
    let policy = MigrationPolicy::BestAgent { rate: 0.2 };
    let controller = MigrationController::new(policy).unwrap();

    let agents1 = create_test_agents(10);
    let agents2 = create_test_agents(10);
    let pop1 = ManagedPopulation::new("source".to_string(), agents1)?;
    let pop2 = ManagedPopulation::new("target".to_string(), agents2)?;

    let migration = controller.plan_migration(&pop1, &pop2, 1)?;

    // Should migrate best agent(s) based on rate
    assert!(!migration.agents.is_empty());
    // Best agents should have high exploration rate (fitness proxy)
    for agent in &migration.agents {
        assert!(agent.genome.behavior.exploration_rate >= 0.8);
    }
    Ok(())
}

#[test]
fn test_resource_utilization_tracking() {
    let config = create_test_config();
    let manager = PopulationEvolutionManager::new(config).unwrap();

    let metrics = manager.calculate_metrics().unwrap();
    let resource_util = &metrics.resource_utilization;

    assert!(resource_util.cpu_usage >= 0.0 && resource_util.cpu_usage <= 100.0);
    assert!(resource_util.memory_usage > 0);
    assert!(resource_util.evaluation_time >= Duration::from_nanos(0));
    assert!(resource_util.parallel_efficiency >= 0.0 && resource_util.parallel_efficiency <= 1.0);
}

#[test]
fn test_end_to_end_population_evolution() {
    let config = PopulationConfig {
        num_populations: 2,
        population_size: 5,
        max_generations: 10,
        migration_frequency: 3,
        selection_strategy: SelectionStrategy::Tournament { size: 2 },
        crossover_strategy: CrossoverStrategy::Uniform { rate: 0.6 },
        migration_policy: MigrationPolicy::BestAgent { rate: 0.4 },
        diversity_threshold: 0.3,
        convergence_threshold: 0.1,
    };

    let mut manager = PopulationEvolutionManager::new(config).unwrap();

    // Initialize populations
    for i in 0..2 {
        let agents = create_test_agents(5);
        let pop_id = format!("population_{}", i);
        manager.initialize_population(&pop_id, agents).unwrap();
    }

    let initial_metrics = manager.calculate_metrics().unwrap();

    // Run multiple generations
    for generation in 1..=5 {
        manager.evolve_generation().unwrap();

        // Perform migration if it's time
        if generation % 3 == 0 {
            manager.perform_migrations(generation).unwrap();
        }

        let metrics = manager.calculate_metrics().unwrap();

        // Verify populations are evolving
        for i in 0..2 {
            let pop_id = format!("population_{}", i);
            let population = manager.get_population(&pop_id).unwrap();
            assert!(population.generation() <= generation); // Allow for some variance
        }

        // Global best should not decrease
        assert!(metrics.global_best_fitness >= initial_metrics.global_best_fitness);
    }

    let final_metrics = manager.calculate_metrics().unwrap();

    // Check final state
    assert_eq!(final_metrics.total_populations, 2);
    assert!(final_metrics.migration_stats.total_migrations > 0);
}

#[test]
fn test_population_health_monitoring() {
    let config = create_test_config();
    let mut manager = PopulationEvolutionManager::new(config).unwrap();

    // Initialize with low diversity population
    let mut low_diversity_agents = create_test_agents(5);
    for agent in &mut low_diversity_agents {
        agent.genome.behavior.exploration_rate = 0.5;
        agent.genome.behavior.learning_rate = 0.01;
        agent.genome.behavior.risk_tolerance = 0.5;
    }

    let pop_id = "low_diversity_pop";
    manager
        .initialize_population(pop_id, low_diversity_agents)
        .unwrap();

    let health_report = manager.assess_population_health(pop_id).unwrap();

    assert!(!health_report.is_healthy);
    assert!(!health_report.recommendations.is_empty());
    assert!(health_report.diversity_score < 0.4); // Below threshold
}

#[test]
fn test_migration_history_tracking() -> TestResult {
    let config = create_test_config();
    let mut controller = MigrationController::new(config.migration_policy).unwrap();

    let agents1 = create_test_agents(10);
    let agents2 = create_test_agents(10);
    let pop1 = ManagedPopulation::new("pop1".to_string(), agents1)?;
    let pop2 = ManagedPopulation::new("pop2".to_string(), agents2)?;

    // Perform multiple migrations
    for gen in 1..=3 {
        let migration = controller.plan_migration(&pop1, &pop2, gen).unwrap();
        controller.record_migration(migration).unwrap();
    }

    assert_eq!(controller.get_migration_count(), 3);
    let history = controller.get_migration_history();
    assert_eq!(history.len(), 3);

    // Check migration generations are recorded
    for (i, migration) in history.iter().enumerate() {
        assert_eq!(migration.generation, (i + 1) as u32);
    }
    Ok(())
}
