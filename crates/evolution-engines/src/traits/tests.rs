//! Tests for traits module

use super::*;
use crate::traits::{
    AgentGenome, ArchitectureGenes, BehaviorGenes, EngineConfig, EvolutionEngine, Evolvable,
    EvolvableAgent,
};
use stratoswarm_agent_core::GoalPriority;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
struct TestEntity {
    value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestGenome {
    value: f64,
}

#[async_trait::async_trait]
impl Evolvable for TestEntity {
    type Genome = TestGenome;
    type Fitness = f64;

    fn genome(&self) -> &Self::Genome {
        // For test purposes, we need to store a genome reference
        // This is a test limitation - in real implementations, 
        // the genome would be part of the entity structure
        static TEST_GENOME: TestGenome = TestGenome { value: 0.0 };
        &TEST_GENOME
    }

    async fn from_genome(_genome: Self::Genome) -> crate::error::EvolutionEngineResult<Self> {
        Ok(TestEntity { value: 0.0 })
    }

    async fn evaluate_fitness(&self) -> crate::error::EvolutionEngineResult<Self::Fitness> {
        Ok(self.value)
    }

    async fn mutate(&self, mutation_rate: f64) -> crate::error::EvolutionEngineResult<Self> {
        Ok(TestEntity {
            value: self.value + mutation_rate,
        })
    }

    async fn crossover(&self, other: &Self) -> crate::error::EvolutionEngineResult<(Self, Self)> {
        let child1 = TestEntity {
            value: (self.value + other.value) / 2.0,
        };
        let child2 = TestEntity {
            value: (self.value + other.value) / 2.0,
        };
        Ok((child1, child2))
    }
}

#[tokio::test]
async fn test_evolvable_trait() {
    let entity = TestEntity { value: 5.0 };

    // Test fitness evaluation
    let fitness = entity.evaluate_fitness().await.unwrap();
    assert_eq!(fitness, 5.0);

    // Test mutation
    let mutated = entity.mutate(1.0).await.unwrap();
    assert_eq!(mutated.value, 6.0);

    // Test crossover
    let other = TestEntity { value: 3.0 };
    let (child1, child2) = entity.crossover(&other).await.unwrap();
    assert_eq!(child1.value, 4.0);
    assert_eq!(child2.value, 4.0);
}

#[test]
fn test_agent_genome_serialization() {
    let genome = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Test goal".to_string(), GoalPriority::High),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    let json = serde_json::to_string(&genome).unwrap();
    let deserialized: AgentGenome = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.architecture.memory_capacity, 1024);
    assert_eq!(deserialized.behavior.exploration_rate, 0.1);
}

#[tokio::test]
async fn test_evolving_agent_fitness() {
    let genome = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Optimization goal".to_string(), GoalPriority::High),
        architecture: ArchitectureGenes {
            memory_capacity: 2048,
            processing_units: 8,
            network_topology: vec![20, 40, 20],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.2,
            learning_rate: 0.02,
            risk_tolerance: 0.7,
        },
    };

    let agent = EvolvableAgent::from_genome(genome).await.unwrap();
    let fitness = agent.evaluate_fitness().await.unwrap();

    // Fitness should be based on architecture and behavior
    assert!(fitness > 0.0);
    assert!(fitness <= 1.0);
}

#[tokio::test]
async fn test_evolving_agent_mutation() {
    let genome = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Mutation test".to_string(), GoalPriority::Normal),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.5,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    let agent = EvolvableAgent::from_genome(genome).await.unwrap();

    // Test high mutation rate
    let mutated = agent.mutate(1.0).await.unwrap();

    // Architecture should be mutated
    assert_ne!(
        mutated.genome.architecture.memory_capacity,
        agent.genome.architecture.memory_capacity
    );

    // Behavior parameters should be within valid ranges
    assert!(mutated.genome.behavior.exploration_rate >= 0.0);
    assert!(mutated.genome.behavior.exploration_rate <= 1.0);
    assert!(mutated.genome.behavior.learning_rate >= 0.0001);
    assert!(mutated.genome.behavior.learning_rate <= 0.1);
    assert!(mutated.genome.behavior.risk_tolerance >= 0.0);
    assert!(mutated.genome.behavior.risk_tolerance <= 1.0);
}

#[tokio::test]
async fn test_evolving_agent_mutation_low_rate() {
    let genome = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Low mutation test".to_string(), GoalPriority::Low),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.5,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    let agent = EvolvableAgent::from_genome(genome).await.unwrap();

    // Test very low mutation rate (should often not mutate)
    let mutated = agent.mutate(0.01).await.unwrap();

    // With low mutation rate, some properties might remain unchanged
    // but mutation should still be valid
    assert!(mutated.genome.behavior.exploration_rate >= 0.0);
    assert!(mutated.genome.behavior.exploration_rate <= 1.0);
}

#[tokio::test]
async fn test_evolving_agent_crossover() {
    let genome1 = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Parent 1".to_string(), GoalPriority::High),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.3,
        },
    };

    let genome2 = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Parent 2".to_string(), GoalPriority::High),
        architecture: ArchitectureGenes {
            memory_capacity: 2048,
            processing_units: 8,
            network_topology: vec![20, 40, 20],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.9,
            learning_rate: 0.05,
            risk_tolerance: 0.8,
        },
    };

    let agent1 = EvolvableAgent::from_genome(genome1).await.unwrap();
    let agent2 = EvolvableAgent::from_genome(genome2).await.unwrap();

    let (child1, child2) = agent1.crossover(&agent2).await.unwrap();

    // Children should have valid genomes
    assert!(child1.genome.behavior.exploration_rate >= 0.0);
    assert!(child1.genome.behavior.exploration_rate <= 1.0);
    assert!(child2.genome.behavior.exploration_rate >= 0.0);
    assert!(child2.genome.behavior.exploration_rate <= 1.0);

    // Architecture should be inherited from one parent or the other
    assert!(
        child1.genome.architecture.memory_capacity == 1024
            || child1.genome.architecture.memory_capacity == 2048
    );
    assert!(
        child2.genome.architecture.memory_capacity == 1024
            || child2.genome.architecture.memory_capacity == 2048
    );
}

#[test]
fn test_architecture_genes_validation() {
    // Valid architecture
    let valid_arch = ArchitectureGenes {
        memory_capacity: 1024,
        processing_units: 4,
        network_topology: vec![10, 20, 10],
    };
    assert_eq!(valid_arch.memory_capacity, 1024);
    assert_eq!(valid_arch.processing_units, 4);
    assert_eq!(valid_arch.network_topology.len(), 3);

    // Extreme values
    let extreme_arch = ArchitectureGenes {
        memory_capacity: 1048576, // 1MB
        processing_units: 64,
        network_topology: vec![100, 200, 100],
    };
    assert_eq!(extreme_arch.memory_capacity, 1048576);
    assert_eq!(extreme_arch.processing_units, 64);
}

#[test]
fn test_behavior_genes_validation() {
    // Valid behavior
    let valid_behavior = BehaviorGenes {
        exploration_rate: 0.1,
        learning_rate: 0.01,
        risk_tolerance: 0.5,
    };
    assert_eq!(valid_behavior.exploration_rate, 0.1);
    assert_eq!(valid_behavior.learning_rate, 0.01);
    assert_eq!(valid_behavior.risk_tolerance, 0.5);

    // Boundary values
    let boundary_behavior = BehaviorGenes {
        exploration_rate: 0.0,
        learning_rate: 0.0001,
        risk_tolerance: 1.0,
    };
    assert_eq!(boundary_behavior.exploration_rate, 0.0);
    assert_eq!(boundary_behavior.learning_rate, 0.0001);
    assert_eq!(boundary_behavior.risk_tolerance, 1.0);
}

#[test]
fn test_agent_genome_clone() {
    let genome = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Clone test".to_string(), GoalPriority::Normal),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    let cloned = genome.clone();
    assert_eq!(
        cloned.architecture.memory_capacity,
        genome.architecture.memory_capacity
    );
    assert_eq!(
        cloned.behavior.exploration_rate,
        genome.behavior.exploration_rate
    );
}

#[tokio::test]
async fn test_agent_genome_consistency() {
    // Create agent from genome and verify consistency
    let genome = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Consistency test".to_string(), GoalPriority::High),
        architecture: ArchitectureGenes {
            memory_capacity: 2048,
            processing_units: 8,
            network_topology: vec![32, 64, 32],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.3,
            learning_rate: 0.02,
            risk_tolerance: 0.6,
        },
    };

    let agent = EvolvableAgent::from_genome(genome.clone()).await.unwrap();

    // Retrieved genome should match original
    assert_eq!(
        agent.genome().architecture.memory_capacity,
        genome.architecture.memory_capacity
    );
    assert_eq!(
        agent.genome().behavior.exploration_rate,
        genome.behavior.exploration_rate
    );
}

#[test]
fn test_network_topology_variations() {
    // Small network
    let small_net = ArchitectureGenes {
        memory_capacity: 512,
        processing_units: 2,
        network_topology: vec![5, 5],
    };
    assert_eq!(small_net.network_topology.len(), 2);

    // Large network
    let large_net = ArchitectureGenes {
        memory_capacity: 4096,
        processing_units: 16,
        network_topology: vec![50, 100, 200, 100, 50],
    };
    assert_eq!(large_net.network_topology.len(), 5);

    // Deep network
    let deep_net = ArchitectureGenes {
        memory_capacity: 2048,
        processing_units: 8,
        network_topology: vec![10; 10], // 10 layers of 10 nodes each
    };
    assert_eq!(deep_net.network_topology.len(), 10);
    assert!(deep_net.network_topology.iter().all(|&x| x == 10));
}

#[tokio::test]
async fn test_multiple_mutations() {
    let genome = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new(
            "Multi-mutation test".to_string(),
            GoalPriority::Normal,
        ),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.5,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    let mut agent = EvolvableAgent::from_genome(genome).await.unwrap();
    let _original_fitness = agent.evaluate_fitness().await.unwrap();

    // Apply multiple mutations
    for _ in 0..5 {
        agent = agent.mutate(0.3).await.unwrap();
    }

    let final_fitness = agent.evaluate_fitness().await.unwrap();

    // Fitness should still be valid
    assert!(final_fitness >= 0.0);
    assert!(final_fitness <= 1.0);

    // Parameters should still be in valid ranges
    assert!(agent.genome.behavior.exploration_rate >= 0.0);
    assert!(agent.genome.behavior.exploration_rate <= 1.0);
    assert!(agent.genome.behavior.learning_rate >= 0.0001);
    assert!(agent.genome.behavior.learning_rate <= 0.1);
}

#[tokio::test]
async fn test_crossover_multiple_times() {
    let genome1 = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new(
            "Multi-crossover parent 1".to_string(),
            GoalPriority::High,
        ),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.3,
        },
    };

    let genome2 = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new(
            "Multi-crossover parent 2".to_string(),
            GoalPriority::High,
        ),
        architecture: ArchitectureGenes {
            memory_capacity: 2048,
            processing_units: 8,
            network_topology: vec![20, 40, 20],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.9,
            learning_rate: 0.05,
            risk_tolerance: 0.8,
        },
    };

    let agent1 = EvolvableAgent::from_genome(genome1).await.unwrap();
    let agent2 = EvolvableAgent::from_genome(genome2).await.unwrap();

    // Perform multiple crossovers
    let mut children = vec![(agent1, agent2)];

    for _ in 0..3 {
        let (parent1, parent2) = &children.last().unwrap();
        let (child1, child2) = parent1.crossover(parent2).await.unwrap();
        children.push((child1, child2));
    }

    // All children should be valid
    for (child1, child2) in &children {
        assert!(child1.evaluate_fitness().await.unwrap() >= 0.0);
        assert!(child2.evaluate_fitness().await.unwrap() >= 0.0);
    }
}

#[test]
fn test_genome_equality_and_differences() {
    let genome1 = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Equality test".to_string(), GoalPriority::Normal),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    let genome2 = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Equality test".to_string(), GoalPriority::Normal),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    let genome3 = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Different test".to_string(), GoalPriority::High),
        architecture: ArchitectureGenes {
            memory_capacity: 2048,
            processing_units: 8,
            network_topology: vec![20, 40, 20],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.9,
            learning_rate: 0.05,
            risk_tolerance: 0.8,
        },
    };

    // Test structural equality (excluding goal UUID which is different)
    assert_eq!(
        genome1.architecture.memory_capacity,
        genome2.architecture.memory_capacity
    );
    assert_eq!(
        genome1.behavior.exploration_rate,
        genome2.behavior.exploration_rate
    );

    // Test structural differences
    assert_ne!(
        genome1.architecture.memory_capacity,
        genome3.architecture.memory_capacity
    );
    assert_ne!(
        genome1.behavior.exploration_rate,
        genome3.behavior.exploration_rate
    );
}

#[tokio::test]
async fn test_fitness_calculation_consistency() {
    let genome = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new(
            "Fitness consistency test".to_string(),
            GoalPriority::High,
        ),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    let agent = EvolvableAgent::from_genome(genome).await.unwrap();

    // Fitness calculation should be consistent
    let fitness1 = agent.evaluate_fitness().await.unwrap();
    let fitness2 = agent.evaluate_fitness().await.unwrap();
    let fitness3 = agent.evaluate_fitness().await.unwrap();

    assert_eq!(fitness1, fitness2);
    assert_eq!(fitness2, fitness3);
}

#[tokio::test]
async fn test_agent_creation_from_diverse_genomes() {
    let genomes = vec![
        // Minimal genome
        AgentGenome {
            goal: stratoswarm_agent_core::Goal::new("Minimal".to_string(), GoalPriority::Low),
            architecture: ArchitectureGenes {
                memory_capacity: 128,
                processing_units: 1,
                network_topology: vec![5],
            },
            behavior: BehaviorGenes {
                exploration_rate: 0.0,
                learning_rate: 0.0001,
                risk_tolerance: 0.0,
            },
        },
        // Maximal genome
        AgentGenome {
            goal: stratoswarm_agent_core::Goal::new("Maximal".to_string(), GoalPriority::Critical),
            architecture: ArchitectureGenes {
                memory_capacity: 65536,
                processing_units: 32,
                network_topology: vec![100, 200, 400, 200, 100],
            },
            behavior: BehaviorGenes {
                exploration_rate: 1.0,
                learning_rate: 0.1,
                risk_tolerance: 1.0,
            },
        },
        // Balanced genome
        AgentGenome {
            goal: stratoswarm_agent_core::Goal::new("Balanced".to_string(), GoalPriority::Normal),
            architecture: ArchitectureGenes {
                memory_capacity: 2048,
                processing_units: 8,
                network_topology: vec![16, 32, 16],
            },
            behavior: BehaviorGenes {
                exploration_rate: 0.5,
                learning_rate: 0.01,
                risk_tolerance: 0.5,
            },
        },
    ];

    for genome in genomes {
        let agent = EvolvableAgent::from_genome(genome).await.unwrap();
        let fitness = agent.evaluate_fitness().await.unwrap();

        // All agents should have valid fitness
        assert!(fitness >= 0.0);
        assert!(fitness <= 1.0);
    }
}

#[test]
fn test_debug_trait_implementations() {
    let genome = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Debug test".to_string(), GoalPriority::Normal),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    // Should be able to debug print
    let debug_str = format!("{:?}", genome);
    assert!(debug_str.contains("AgentGenome"));
    assert!(debug_str.contains("architecture"));
    assert!(debug_str.contains("behavior"));

    let arch_debug = format!("{:?}", genome.architecture);
    assert!(arch_debug.contains("ArchitectureGenes"));

    let behavior_debug = format!("{:?}", genome.behavior);
    assert!(behavior_debug.contains("BehaviorGenes"));
}

#[tokio::test]
async fn test_edge_case_mutations() {
    // Test mutation with edge case values
    let genome = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Edge case test".to_string(), GoalPriority::Normal),
        architecture: ArchitectureGenes {
            memory_capacity: 1,
            processing_units: 1,
            network_topology: vec![1],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.0,
            learning_rate: 0.0001,
            risk_tolerance: 0.0,
        },
    };

    let agent = EvolvableAgent::from_genome(genome).await.unwrap();

    // Mutation should handle edge cases gracefully
    let mutated = agent.mutate(1.0).await.unwrap();

    // Values should still be valid even after mutation
    assert!(mutated.genome.architecture.memory_capacity >= 1);
    assert!(mutated.genome.architecture.processing_units >= 1);
    assert!(!mutated.genome.architecture.network_topology.is_empty());
    assert!(mutated.genome.behavior.exploration_rate >= 0.0);
    assert!(mutated.genome.behavior.exploration_rate <= 1.0);
    assert!(mutated.genome.behavior.learning_rate >= 0.0001);
}

#[tokio::test]
async fn test_zero_mutation_rate() {
    let genome = AgentGenome {
        goal: stratoswarm_agent_core::Goal::new("Zero mutation test".to_string(), GoalPriority::Normal),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.5,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    let agent = EvolvableAgent::from_genome(genome.clone()).await.unwrap();

    // Zero mutation rate should not change the agent
    let mutated = agent.mutate(0.0).await.unwrap();

    // Agent should be essentially unchanged
    assert_eq!(
        mutated.genome.architecture.memory_capacity,
        genome.architecture.memory_capacity
    );
    assert_eq!(
        mutated.genome.architecture.processing_units,
        genome.architecture.processing_units
    );
    assert_eq!(
        mutated.genome.architecture.network_topology,
        genome.architecture.network_topology
    );

    // Behavior should also be unchanged (or very close due to floating point)
    assert!(
        (mutated.genome.behavior.exploration_rate - genome.behavior.exploration_rate).abs() < 0.01
    );
    assert!((mutated.genome.behavior.learning_rate - genome.behavior.learning_rate).abs() < 0.001);
    assert!((mutated.genome.behavior.risk_tolerance - genome.behavior.risk_tolerance).abs() < 0.01);
}
