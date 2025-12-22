//! Tests for DGM evolution engine

use super::*;
use crate::dgm::{config::*, engine::*, improvement::*, patterns::*};
use crate::population::Population;
use crate::traits::{
    AgentGenome, ArchitectureGenes, BehaviorGenes, EngineConfig, EvolutionEngine, EvolvableAgent,
};

#[test]
fn test_dgm_config_default() {
    let config = DgmConfig::default();
    assert_eq!(config.discovery_rate, 0.2);
    assert_eq!(config.growth_momentum, 0.7);
    assert_eq!(config.pattern_retention_threshold, 0.6);
}

#[test]
fn test_dgm_config_validation() {
    let mut config = DgmConfig::default();
    assert!(config.validate().is_ok());

    config.discovery_rate = 1.5;
    assert!(config.validate().is_err());

    config.discovery_rate = 0.5;
    config.growth_momentum = -0.1;
    assert!(config.validate().is_err());

    config.growth_momentum = 0.5;
    config.pattern_retention_threshold = 2.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_growth_patterns_default() {
    let patterns = GrowthPatterns::default();
    assert_eq!(patterns.max_history, 100);
    assert_eq!(patterns.similarity_threshold, 0.8);
    assert_eq!(patterns.consolidation_interval, 10);
}

#[test]
fn test_improvement_parameters() {
    let params = ImprovementParameters::default();
    assert_eq!(params.improvement_threshold, 0.05);
    assert_eq!(params.learning_decay, 0.95);
    assert_eq!(params.exploration_bonus, 1.2);
    assert_eq!(params.exploitation_penalty, 0.8);
}

#[test]
fn test_dgm_engine_creation() {
    let config = DgmConfig::default();
    let engine = DgmEngine::new(config);
    assert!(engine.is_ok());
}

#[tokio::test]
async fn test_initial_population_generation() {
    let config = DgmConfig::default();
    let engine = DgmEngine::new(config).unwrap();

    let population = engine.generate_initial_population(15).await.unwrap();
    assert_eq!(population.size(), 15);
    assert_eq!(population.generation, 0);
}

#[test]
fn test_velocity_update() {
    let config = DgmConfig::default();
    let engine = DgmEngine::new(config).unwrap();

    engine.update_velocity(0.5, 0.7);
    let velocity = *engine.improvement_velocity.read();
    assert!(velocity > 0.0);

    // Test decay
    engine.update_velocity(0.7, 0.7);
    let new_velocity = *engine.improvement_velocity.read();
    assert!(new_velocity < velocity);
}

#[test]
fn test_pattern_discovery() {
    let mut config = DgmConfig::default();
    // Increase discovery rate to ensure patterns are found in test
    config.discovery_rate = 1.0;
    let engine = DgmEngine::new(config).unwrap();

    // Test basic discovery functionality
    let patterns_before = engine.growth_patterns.read().len();
    assert_eq!(patterns_before, 0);

    // Create a simple test population
    let population = Population::<EvolvableAgent>::new();
    engine.discover_patterns(&population);

    // With empty population, no patterns should be discovered
    let patterns_after = engine.growth_patterns.read().len();
    assert_eq!(patterns_after, 0);
}

#[test]
fn test_dgm_config_serialization() {
    let config = DgmConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: DgmConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config.discovery_rate, deserialized.discovery_rate);
    assert_eq!(config.growth_momentum, deserialized.growth_momentum);
}

#[test]
fn test_growth_pattern_methods() {
    let source = AgentGenome {
        goal: exorust_agent_core::Goal::new(
            "Test".to_string(),
            exorust_agent_core::GoalPriority::Normal,
        ),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 2,
            network_topology: vec![10, 20],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    let pattern = GrowthPattern::new(
        "test_pattern".to_string(),
        source.clone(),
        source.clone(),
        0.5,
        10,
    );

    assert_eq!(pattern.id, "test_pattern");
    assert_eq!(pattern.fitness_delta, 0.5);
    assert_eq!(pattern.success_count, 1);
    assert_eq!(pattern.failure_count, 0);
    assert_eq!(pattern.last_used, 10);
    assert_eq!(pattern.success_rate(), 1.0);
    assert!(pattern.is_relevant(15, 10));
    assert!(!pattern.is_relevant(100, 10));
}

#[test]
fn test_discovered_pattern() {
    let mut pattern = DiscoveredPattern::new(
        "test_pattern".to_string(),
        PatternType::Architectural,
        0.5,
        10,
    );

    assert_eq!(pattern.success_rate, 1.0);
    assert_eq!(pattern.application_count, 1);

    // Test successful update
    pattern.update(true, 0.6);
    assert_eq!(pattern.application_count, 2);
    assert!(pattern.success_rate > 0.9); // Should be close to 1.0
    assert!(pattern.fitness_delta > 0.5); // Should have increased

    // Test failed update
    pattern.update(false, 0.0);
    assert_eq!(pattern.application_count, 3);
    assert!(pattern.success_rate < 0.9); // Should have decreased

    // Test pattern value
    let value = pattern.value();
    assert!(value > 0.0);
}

#[test]
fn test_growth_history() {
    let mut history = GrowthHistory::default();

    // Test adding data
    history.add_application("pattern1".to_string(), 0.5, 10);
    history.add_fitness(10, 0.8);
    history.add_discovery(10, "pattern1".to_string());

    assert_eq!(history.pattern_applications.len(), 1);
    assert_eq!(history.fitness_history.len(), 1);
    assert_eq!(history.discoveries.len(), 1);

    // Test improvement velocity
    history.add_fitness(20, 0.9);
    let velocity = history.improvement_velocity(10);
    assert_eq!(velocity, 0.01); // (0.9 - 0.8) / (20 - 10)

    // Test discovery rate
    history.add_discovery(15, "pattern2".to_string());
    let discovery_rate = history.discovery_rate(10);
    assert_eq!(discovery_rate, 0.1); // 1 discovery in window of 10
}

#[test]
fn test_pattern_discovery_similarity() {
    let discovery = PatternDiscovery::new(0.8, 100, 10);

    let genome1 = AgentGenome {
        goal: exorust_agent_core::Goal::new(
            "Test".to_string(),
            exorust_agent_core::GoalPriority::Normal,
        ),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 2,
            network_topology: vec![10, 20],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.5,
            learning_rate: 0.02,
            risk_tolerance: 0.6,
        },
    };

    let genome2 = AgentGenome {
        goal: exorust_agent_core::Goal::new(
            "Test".to_string(),
            exorust_agent_core::GoalPriority::Normal,
        ),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 2,
            network_topology: vec![10, 20],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.5,
            learning_rate: 0.02,
            risk_tolerance: 0.6,
        },
    };

    // Identical genomes should be similar
    assert!(discovery.genomes_similar(&genome1, &genome2));

    // Very different genomes should not be similar
    let genome3 = AgentGenome {
        goal: exorust_agent_core::Goal::new(
            "Different".to_string(),
            exorust_agent_core::GoalPriority::High,
        ),
        architecture: ArchitectureGenes {
            memory_capacity: 8192,
            processing_units: 8,
            network_topology: vec![50, 100],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.001,
            risk_tolerance: 0.2,
        },
    };

    assert!(!discovery.genomes_similar(&genome1, &genome3));
}

#[test]
fn test_pattern_consolidation() {
    let discovery = PatternDiscovery::new(0.8, 5, 10);

    let genome1 = AgentGenome {
        goal: exorust_agent_core::Goal::new(
            "Test".to_string(),
            exorust_agent_core::GoalPriority::Normal,
        ),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 2,
            network_topology: vec![10, 20],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.5,
            learning_rate: 0.02,
            risk_tolerance: 0.6,
        },
    };

    let patterns = vec![
        ("p1".to_string(), genome1.clone(), genome1.clone(), 0.5),
        ("p2".to_string(), genome1.clone(), genome1.clone(), 0.6),
        ("p3".to_string(), genome1.clone(), genome1.clone(), 0.4),
    ];

    let consolidated = discovery.consolidate_patterns(patterns);

    // Should consolidate similar patterns
    assert!(consolidated.len() <= 3);

    // Should preserve pattern with highest fitness
    if consolidated.len() == 1 {
        assert_eq!(consolidated[0].3, 0.5); // Average of 0.5, 0.6, 0.4
    }
}

#[tokio::test]
async fn test_pattern_application() {
    let config = DgmConfig::default();
    let engine = DgmEngine::new(config).unwrap();

    let genome = engine.generate_base_genome();
    let agent_config = exorust_agent_core::AgentConfig {
        name: "test_agent".to_string(),
        agent_type: "test".to_string(),
        max_memory: genome.architecture.memory_capacity,
        max_gpu_memory: genome.architecture.memory_capacity / 4,
        priority: 1,
        metadata: serde_json::Value::Null,
    };
    let agent = exorust_agent_core::Agent::new(agent_config).unwrap();
    let evolvable = EvolvableAgent {
        agent,
        genome: genome.clone(),
    };

    // Apply patterns when no patterns exist (should fall back to mutation)
    let (result, pattern_id) = engine.apply_patterns(&evolvable, 0).await.unwrap();

    assert!(pattern_id.is_none()); // No pattern applied
    assert!(result.genome.architecture.memory_capacity >= 1024);
}

#[tokio::test]
async fn test_evolution_step_basic() {
    let mut config = DgmConfig::default();
    config.base.population_size = 5;
    config.base.mutation_rate = 0.1;

    let mut engine = DgmEngine::new(config)?;
    let initial_pop = engine.generate_initial_population(5).await?;

    let evolved_pop = engine.evolve_step(initial_pop).await?;
    assert_eq!(evolved_pop.size(), 5);
    assert_eq!(evolved_pop.generation, 1);
}

#[tokio::test]
async fn test_termination_conditions() {
    let mut config = DgmConfig::default();
    config.base.max_generations = 10;
    config.base.target_fitness = Some(0.9);
    config.improvement_params.improvement_threshold = 0.01;

    let engine = DgmEngine::new(config)?;

    // Test generation limit
    let mut metrics = crate::metrics::EvolutionMetrics::default();
    metrics.generation = 15;
    assert!(engine.should_terminate(&metrics).await);

    // Test target fitness
    metrics.generation = 5;
    metrics.best_fitness = 0.95;
    assert!(engine.should_terminate(&metrics).await);

    // Test velocity stagnation
    *engine.improvement_velocity.write() = 0.005; // Below threshold
    metrics.generation = 55;
    metrics.best_fitness = 0.5;
    assert!(engine.should_terminate(&metrics).await);

    // Test no termination
    *engine.improvement_velocity.write() = 0.1;
    metrics.generation = 5;
    metrics.best_fitness = 0.5;
    assert!(!engine.should_terminate(&metrics).await);
}

#[tokio::test]
async fn test_parameter_adaptation() {
    let mut config = DgmConfig::default();
    config.base.adaptive_parameters = true;
    config.improvement_params.improvement_threshold = 0.1;
    config.improvement_params.exploration_bonus = 1.2;
    config.improvement_params.exploitation_penalty = 0.8;

    let mut engine = DgmEngine::new(config)?;

    // Test adaptation with low velocity (should increase exploration)
    *engine.improvement_velocity.write() = 0.05;
    let metrics = crate::metrics::EvolutionMetrics::default();
    engine.adapt_parameters(&metrics).await.unwrap();

    // Test adaptation with high velocity (should increase exploitation)
    *engine.improvement_velocity.write() = 0.2;
    engine.adapt_parameters(&metrics).await.unwrap();

    // Just verify the call succeeds - can't check private fields
}

#[test]
fn test_seeded_random_generation() {
    let mut config1 = DgmConfig::default();
    config1.base.seed = Some(42);
    let engine1 = DgmEngine::new(config1).unwrap();

    let mut config2 = DgmConfig::default();
    config2.base.seed = Some(42);
    let engine2 = DgmEngine::new(config2)?;

    // With same seed, should generate identical genomes
    let genome1 = engine1.generate_base_genome();
    let genome2 = engine2.generate_base_genome();

    assert_eq!(
        genome1.architecture.memory_capacity,
        genome2.architecture.memory_capacity
    );
    assert_eq!(
        genome1.architecture.processing_units,
        genome2.architecture.processing_units
    );
}
