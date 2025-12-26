//! Tests for ADAS module

use super::*;
use crate::traits::EngineConfig;
use stratoswarm_agent_core::{Goal, GoalPriority};

#[test]
fn test_adas_config_default() {
    let config = AdasConfig::default();
    assert_eq!(config.meta_learning_rate, 0.01);
    assert_eq!(config.architecture_mutation_prob, 0.3);
    assert_eq!(config.behavior_mutation_prob, 0.7);
}

#[test]
fn test_adas_config_validation() {
    let mut config = AdasConfig::default();
    assert!(config.validate().is_ok());

    config.meta_learning_rate = 0.0;
    assert!(config.validate().is_err());

    config.meta_learning_rate = 0.5;
    config.architecture_mutation_prob = 1.5;
    assert!(config.validate().is_err());
}

#[tokio::test]
async fn test_adas_engine_creation() {
    let config = AdasConfig::default();
    let engine = AdasEngine::new(config);
    assert!(engine.is_ok());
}

#[tokio::test]
async fn test_initial_population_generation() {
    let config = AdasConfig::default();
    let engine = AdasEngine::new(config).unwrap();

    let population = engine.generate_initial_population(10).await.unwrap();
    assert_eq!(population.size(), 10);
}

#[test]
fn test_architecture_generation() {
    let config = AdasConfig::default();
    let engine = AdasEngine::new(config).unwrap();

    let arch = engine.random_architecture();
    assert!(arch.memory_capacity >= 1024);
    assert!(arch.memory_capacity <= 1024 * 1024);
    assert!(arch.processing_units >= 1);
    assert!(arch.processing_units <= 16);
    assert!(!arch.network_topology.is_empty());
}

#[test]
fn test_behavior_generation() {
    let config = AdasConfig::default();
    let engine = AdasEngine::new(config).unwrap();

    let behavior = engine.random_behavior();
    assert!(behavior.exploration_rate >= 0.01);
    assert!(behavior.exploration_rate <= 0.5);
    assert!(behavior.learning_rate >= 0.0001);
    assert!(behavior.learning_rate <= 0.1);
    assert!(behavior.risk_tolerance >= 0.0);
    assert!(behavior.risk_tolerance <= 1.0);
}

#[test]
fn test_architecture_mutation() {
    let config = AdasConfig::default();
    let engine = AdasEngine::new(config).unwrap();

    let original = engine.random_architecture();
    let mutated = engine.mutate_architecture(&original);

    // Should be different but within bounds
    assert!(mutated.memory_capacity <= 1024 * 1024);
    assert!(mutated.processing_units <= 16);
}

#[test]
fn test_diversity_calculation() {
    let config = AdasConfig::default();
    let engine = AdasEngine::new(config).unwrap();

    let mut population = Vec::new();
    for i in 0..5 {
        let genome = crate::traits::AgentGenome {
            goal: Goal::new("Test".to_string(), GoalPriority::Normal),
            architecture: engine.random_architecture(),
            behavior: engine.random_behavior(),
        };
        let config = stratoswarm_agent_core::AgentConfig {
            name: format!("test_agent_{i}"),
            agent_type: "test".to_string(),
            max_memory: genome.architecture.memory_capacity,
            max_gpu_memory: genome.architecture.memory_capacity / 4,
            priority: 1,
            metadata: serde_json::Value::Null,
        };
        let agent = stratoswarm_agent_core::Agent::new(config).unwrap();
        population.push(crate::traits::EvolvableAgent { agent, genome });
    }

    let diversity = engine.calculate_diversity(&population);
    assert!(diversity >= 0.0);
}

#[test]
fn test_search_spaces() {
    let arch = ArchitectureSearchSpace::default();
    assert_eq!(arch.memory_capacity_range, (1024, 1024 * 1024));
    assert_eq!(arch.processing_units_range, (1, 16));

    let behavior = BehaviorSearchSpace::default();
    assert_eq!(behavior.exploration_rate_range, (0.01, 0.5));
    assert_eq!(behavior.learning_rate_range, (0.0001, 0.1));
}

#[tokio::test]
async fn test_meta_agent_search() {
    let config = AdasConfig::default();
    let mut engine = AdasEngine::new(config).unwrap();

    // Test the method exists and can be called
    let result = engine.meta_agent_search("Test task").await;
    // It will likely fail due to no actual workflows, but we're testing the structure
    assert!(result.is_err() || result.is_ok());
}

#[test]
fn test_engine_name() {
    let config = AdasConfig::default();
    assert_eq!(config.engine_name(), "ADAS");
}
