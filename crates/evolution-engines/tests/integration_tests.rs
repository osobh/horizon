//! Integration tests for evolution engines with agent-core and synthesis
//!
//! Tests the complete flow from agent evolution through synthesis
//!
//! Note: Some tests are ignored until HybridEvolutionSystem implements EvolutionEngine trait

use stratoswarm_agent_core::{Agent, AgentConfig, Goal, GoalPriority};
use stratoswarm_evolution_engines::hybrid::{EngineStrategy, HybridConfig, HybridEvolutionSystem};
use stratoswarm_synthesis::interpreter::{InterpreterConfig, OperationType};
use stratoswarm_synthesis::GoalInterpreter;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[tokio::test]
#[ignore = "HybridEvolutionSystem does not yet implement EvolutionEngine trait"]
async fn test_agent_evolution_to_synthesis() -> TestResult {
    // Create hybrid evolution system
    let mut config = HybridConfig::default();
    config.strategy = EngineStrategy::Adaptive;
    config.base.population_size = 5;
    config.base.max_generations = 3;

    let _evolution_system = HybridEvolutionSystem::new(config).await?;

    // TODO: Enable when HybridEvolutionSystem implements EvolutionEngine trait
    // let mut population = evolution_system.generate_initial_population(5).await?;
    // for _ in 0..3 {
    //     population = evolution_system.evolve_step(population).await?;
    // }

    // Create agent from evolved genome
    let mut agent_config = AgentConfig::default();
    agent_config.name = "evolved_agent".to_string();
    let agent = Agent::new(agent_config)?;

    // Add goal
    agent
        .add_goal(Goal::new(
            "Matrix multiplication optimization".to_string(),
            GoalPriority::High,
        ))
        .await?;

    // Create goal interpreter
    let interpreter_config = InterpreterConfig::default();
    let interpreter = GoalInterpreter::new(interpreter_config);

    // Get goals
    let goals = agent.goals().await;
    let spec = interpreter
        .interpret(&goals.first().unwrap())
        .await?;

    // Verify we got a valid specification
    assert!(matches!(
        spec.operation_type,
        OperationType::MatrixMultiply | OperationType::Custom
    ));
    assert!(!spec.data_layout.input_shape.is_empty());
    Ok(())
}

#[tokio::test]
#[ignore = "HybridEvolutionSystem does not yet implement EvolutionEngine trait"]
async fn test_evolution_guided_synthesis() -> TestResult {
    // Create evolution system with performance-based strategy
    let mut config = HybridConfig::default();
    config.strategy = EngineStrategy::PerformanceBased;
    config.base.population_size = 10;

    let _evolution_system = HybridEvolutionSystem::new(config).await?;

    // TODO: Enable when HybridEvolutionSystem implements EvolutionEngine trait
    // Create goal interpreter for fitness evaluation
    let _interpreter = GoalInterpreter::new(InterpreterConfig::default());

    // Test passes when HybridEvolutionSystem is fully implemented
    Ok(())
}

#[tokio::test]
async fn test_multi_agent_synthesis_evolution() -> TestResult {
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
            .await?;

        agents.push(agent);
    }

    // Create evolution system
    let config = HybridConfig::default();
    let _evolution_system = HybridEvolutionSystem::new(config).await?;

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
    Ok(())
}

#[tokio::test]
#[ignore = "HybridEvolutionSystem does not yet implement EvolutionEngine trait"]
async fn test_evolution_metrics_tracking() -> TestResult {
    // Create evolution system
    let mut config = HybridConfig::default();
    config.base.population_size = 20;
    config.base.max_generations = 10;
    config.strategy = EngineStrategy::Adaptive;

    let _evolution_system = HybridEvolutionSystem::new(config).await?;

    // TODO: Enable when HybridEvolutionSystem implements EvolutionEngine trait
    // let initial_metrics = evolution_system.metrics().clone();
    // assert_eq!(initial_metrics.generation, 0);
    Ok(())
}

#[tokio::test]
#[ignore = "HybridEvolutionSystem does not yet implement EvolutionEngine trait"]
async fn test_synthesis_feedback_to_evolution() -> TestResult {
    // Create evolution system
    let config = HybridConfig::default();
    let _evolution_system = HybridEvolutionSystem::new(config).await?;

    // Create goal interpreter
    let _interpreter = GoalInterpreter::new(InterpreterConfig::default());

    // TODO: Enable when HybridEvolutionSystem implements EvolutionEngine trait
    Ok(())
}

#[tokio::test]
#[ignore = "HybridEvolutionSystem does not yet implement EvolutionEngine trait"]
async fn test_kernel_specification_evolution() -> TestResult {
    // Create evolution system
    let mut config = HybridConfig::default();
    config.base.population_size = 15;
    config.strategy = EngineStrategy::RoundRobin;

    let _evolution_system = HybridEvolutionSystem::new(config).await?;

    // Create interpreter
    let _interpreter = GoalInterpreter::new(InterpreterConfig::default());

    // TODO: Enable when HybridEvolutionSystem implements EvolutionEngine trait
    Ok(())
}
