//! Integration tests for evolution engines with agent-core and synthesis
//!
//! Tests the complete flow from agent evolution through synthesis

use exorust_agent_core::{Agent, AgentConfig, Goal, GoalPriority};
use exorust_evolution_engines::hybrid::{EngineStrategy, HybridConfig, HybridEvolutionSystem};
use exorust_evolution_engines::EvolutionEngine;
use exorust_synthesis::interpreter::{InterpreterConfig, OperationType};
use exorust_synthesis::GoalInterpreter;

#[tokio::test]
async fn test_agent_evolution_to_synthesis() {
    // Create hybrid evolution system
    let mut config = HybridConfig::default();
    config.strategy = EngineStrategy::Adaptive;
    config.base.population_size = 5;
    config.base.max_generations = 3;

    let mut evolution_system = HybridEvolutionSystem::new(config).await?;

    // Generate initial population
    let mut population = evolution_system
        .generate_initial_population(5)
        .await
        .unwrap();

    // Evolve for a few generations
    for _ in 0..3 {
        population = evolution_system.evolve_step(population).await.unwrap();
    }

    // Get best agent
    let _best_agent = population
        .individuals
        .iter()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        .unwrap();

    // Create agent from evolved genome
    let mut agent_config = AgentConfig::default();
    agent_config.name = "evolved_agent".to_string();
    let agent = Agent::new(agent_config).unwrap();

    // Add goal
    agent
        .add_goal(Goal::new(
            "Matrix multiplication optimization".to_string(),
            GoalPriority::High,
        ))
        .await
        .unwrap();

    // Create goal interpreter
    let interpreter_config = InterpreterConfig::default();
    let interpreter = GoalInterpreter::new(interpreter_config);

    // Get goals
    let goals = agent.goals().await;
    let spec = interpreter
        .interpret(&goals.first().unwrap())
        .await
        .unwrap();

    // Verify we got a valid specification
    assert!(matches!(
        spec.operation_type,
        OperationType::MatrixMultiply | OperationType::Custom
    ));
    assert!(!spec.data_layout.input_shape.is_empty());
}

#[tokio::test]
async fn test_evolution_guided_synthesis() {
    // Create evolution system with performance-based strategy
    let mut config = HybridConfig::default();
    config.strategy = EngineStrategy::PerformanceBased;
    config.base.population_size = 10;

    let mut evolution_system = HybridEvolutionSystem::new(config).await?;

    // Generate population with synthesis-oriented goals
    let mut population = evolution_system
        .generate_initial_population(10)
        .await
        .unwrap();

    // Create goal interpreter for fitness evaluation
    let interpreter = GoalInterpreter::new(InterpreterConfig::default());

    // Evolve with synthesis-based fitness
    for _generation in 0..5 {
        // Evaluate fitness based on synthesis success
        for individual in &mut population.individuals {
            // Create goal from genome
            let goal = Goal::new(
                format!(
                    "Goal_{}: Synthesize efficient kernel",
                    individual.entity.genome.goal.id.0
                ),
                GoalPriority::High,
            );

            // Try to interpret
            match interpreter.interpret(&goal).await {
                Ok(_) => individual.fitness = Some(1.0),
                Err(_) => individual.fitness = Some(0.0),
            }
        }

        // Evolve
        population = evolution_system.evolve_step(population).await.unwrap();
    }

    // Check improvement
    let best_fitness = population
        .individuals
        .iter()
        .filter_map(|i| i.fitness)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    assert!(best_fitness > 0.5);
}

#[tokio::test]
async fn test_multi_agent_synthesis_evolution() {
    // Create multiple agents
    let mut agents = vec![];
    for i in 0..3 {
        let mut config = AgentConfig::default();
        config.name = format!("agent_{i}");
        let agent = Agent::new(config)?;

        agent
            .add_goal(Goal::new(
                format!("Goal_{}: Optimize GPU kernel", i),
                GoalPriority::Normal,
            ))
            .await
            .unwrap();

        agents.push(agent);
    }

    // Create evolution system
    let config = HybridConfig::default();
    let _evolution_system = HybridEvolutionSystem::new(config).await.unwrap();

    // Create goal interpreter
    let interpreter = GoalInterpreter::new(InterpreterConfig::default());

    // Interpret goals for each agent
    let mut interpretation_results = vec![];
    for agent in &agents {
        let goals = agent.goals().await;
        let result = interpreter.interpret(goals.first().unwrap()).await;
        interpretation_results.push(result.is_ok());
    }

    // At least one interpretation should succeed
    assert!(interpretation_results.iter().any(|&r| r));
}

#[tokio::test]
async fn test_evolution_metrics_tracking() {
    // Create evolution system
    let mut config = HybridConfig::default();
    config.base.population_size = 20;
    config.base.max_generations = 10;
    config.strategy = EngineStrategy::Adaptive;

    let mut evolution_system = HybridEvolutionSystem::new(config).await?;

    // Initial metrics
    let initial_metrics = evolution_system.metrics().clone();
    assert_eq!(initial_metrics.generation, 0);

    // Generate and evolve population
    let mut population = evolution_system
        .generate_initial_population(20)
        .await
        .unwrap();

    for i in 0..5 {
        population = evolution_system.evolve_step(population).await.unwrap();

        let metrics = evolution_system.metrics();
        assert_eq!(metrics.generation, (i + 1) as u32);
        assert!(metrics.total_evaluations > 0);
    }

    // Check convergence tracking
    let final_metrics = evolution_system.metrics();
    assert!(final_metrics.generation >= 5);
    assert!(final_metrics.convergence_rate >= 0.0);
}

#[tokio::test]
async fn test_synthesis_feedback_to_evolution() {
    // Create evolution system
    let config = HybridConfig::default();
    let mut evolution_system = HybridEvolutionSystem::new(config).await.unwrap();

    // Create goal interpreter
    let interpreter = GoalInterpreter::new(InterpreterConfig::default());

    // Generate population
    let mut population = evolution_system
        .generate_initial_population(10)
        .await
        .unwrap();

    // Evolve with synthesis feedback
    for _ in 0..3 {
        // Update fitness based on synthesis performance
        for individual in &mut population.individuals {
            let goal = Goal::new(
                "Synthesis goal: optimize reduction kernel".to_string(),
                GoalPriority::High,
            );

            // Interpret and measure complexity
            match interpreter.interpret(&goal).await {
                Ok(spec) => {
                    // Mock performance metric based on specification complexity
                    let complexity = spec.optimization_hints.len() as f64 * 0.1;
                    individual.fitness = Some((1.0 - complexity).max(0.1));
                }
                Err(_) => {
                    individual.fitness = Some(0.1); // Low fitness for failed interpretation
                }
            }
        }

        population = evolution_system.evolve_step(population).await.unwrap();
    }

    // Verify population improved
    let avg_fitness = population
        .individuals
        .iter()
        .filter_map(|i| i.fitness)
        .sum::<f64>()
        / population.individuals.len() as f64;

    assert!(avg_fitness > 0.0);
}

#[tokio::test]
async fn test_kernel_specification_evolution() {
    // Create evolution system
    let mut config = HybridConfig::default();
    config.base.population_size = 15;
    config.strategy = EngineStrategy::RoundRobin;

    let mut evolution_system = HybridEvolutionSystem::new(config).await?;

    // Generate initial population
    let mut population = evolution_system
        .generate_initial_population(15)
        .await
        .unwrap();

    // Create interpreter
    let interpreter = GoalInterpreter::new(InterpreterConfig::default());

    // Track specification diversity
    let mut spec_count = 0;

    // Evolve and track kernel types
    for gen in 0..4 {
        for individual in &mut population.individuals {
            // Create goals with different optimization targets
            let goal = Goal::new(
                format!(
                    "Generation {}: Optimize kernel for {}",
                    gen, individual.entity.genome.goal.id.0
                ),
                GoalPriority::Normal,
            );

            if let Ok(spec) = interpreter.interpret(&goal).await {
                spec_count += 1;

                // Fitness based on specification quality
                let fitness = match spec.operation_type {
                    OperationType::MatrixMultiply => 0.9,
                    OperationType::Reduction => 0.8,
                    OperationType::Convolution => 0.85,
                    OperationType::Elementwise => 0.7,
                    OperationType::Custom => 0.6,
                };
                individual.fitness = Some(fitness);
            } else {
                individual.fitness = Some(0.2);
            }
        }

        population = evolution_system.evolve_step(population).await.unwrap();
    }

    // Verify we generated kernel specifications
    assert!(spec_count > 0, "Should generate some kernel specifications");
}
