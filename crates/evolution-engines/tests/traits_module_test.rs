//! Test to verify traits module structure works correctly

use stratoswarm_evolution_engines::traits::{
    AgentGenome, ArchitectureGenes, BehaviorGenes, EngineConfig, EvolutionEngine, Evolvable,
    EvolvableAgent, MockEvolvableAgent,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn test_traits_module_imports() {
    // Test that we can import all types
    let _arch_genes = ArchitectureGenes {
        memory_capacity: 1024,
        processing_units: 4,
        network_topology: vec![10, 5],
    };

    let _behavior_genes = BehaviorGenes {
        exploration_rate: 0.1,
        learning_rate: 0.01,
        risk_tolerance: 0.5,
    };

    // Test MockEvolvableAgent
    let mock = MockEvolvableAgent::new("test".to_string());
    assert_eq!(mock.id, "test");
}

#[test]
fn test_agent_genome() {
    let goal =
        stratoswarm_agent_core::Goal::new("test".to_string(), stratoswarm_agent_core::GoalPriority::Normal);
    let genome = AgentGenome {
        goal,
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 5],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    assert_eq!(genome.architecture.memory_capacity, 1024);
    assert_eq!(genome.behavior.exploration_rate, 0.1);
}

#[tokio::test]
async fn test_mock_evolvable_agent() -> TestResult {
    let mock = MockEvolvableAgent::new("test_agent".to_string());

    // Test Evolvable trait methods
    let fitness = mock.evaluate_fitness().await.unwrap();
    assert!(fitness >= 0.0);

    let mutated = mock.mutate(0.1).await?;
    assert_eq!(mutated.id, "test_agent");

    let other = MockEvolvableAgent::new("other".to_string());
    let (child1, child2) = mock.crossover(&other).await.unwrap();
    assert_ne!(child1.id, child2.id);
    Ok(())
}
