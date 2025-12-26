use gpu_agents::scenarios::{
    KnowledgeConfig, MemoryPattern, PromptComplexity, ReasoningConfig, ScenarioConfig,
    ScenarioRunner, ScenarioType, SimpleBehavior,
};
use std::time::Duration;

#[test]
fn test_simple_scenario_execution() {
    let config = ScenarioConfig {
        id: "test-simple".to_string(),
        name: "Test Simple Scenario".to_string(),
        description: "Integration test for simple agents".to_string(),
        scenario_type: ScenarioType::Simple {
            behavior: SimpleBehavior::RandomWalk,
            interaction_radius: 10.0,
            update_frequency: 60.0,
        },
        agent_count: 100,
        duration: Duration::from_secs(2),
        seed: Some(42),
        objectives: vec![],
    };

    let runner = ScenarioRunner::new(0).expect("Failed to create runner");
    let result = runner
        .run_scenario(&config)
        .expect("Failed to run scenario");

    assert_eq!(result.scenario_id, "test-simple");
    assert_eq!(result.agent_count, 100);
    assert!(result.duration >= Duration::from_secs(2));
    assert!(result.errors.is_empty());
}

#[test]
fn test_reasoning_scenario_execution() {
    let config = ScenarioConfig {
        id: "test-reasoning".to_string(),
        name: "Test Reasoning Scenario".to_string(),
        description: "Integration test for reasoning agents".to_string(),
        scenario_type: ScenarioType::Reasoning {
            config: ReasoningConfig {
                model: "test-llm".to_string(),
                max_tokens: 128,
                temperature: 0.7,
                prompt_complexity: PromptComplexity::Simple,
                batch_size: 10,
                decision_frequency: 2.0,
            },
        },
        agent_count: 10,
        duration: Duration::from_secs(1),
        seed: None,
        objectives: vec![],
    };

    let runner = ScenarioRunner::new(0).expect("Failed to create runner");
    let result = runner
        .run_scenario(&config)
        .expect("Failed to run scenario");

    assert_eq!(result.scenario_id, "test-reasoning");
    assert_eq!(result.agent_count, 10);
}

#[test]
fn test_knowledge_scenario_execution() {
    let config = ScenarioConfig {
        id: "test-knowledge".to_string(),
        name: "Test Knowledge Scenario".to_string(),
        description: "Integration test for knowledge agents".to_string(),
        scenario_type: ScenarioType::Knowledge {
            config: KnowledgeConfig {
                initial_nodes: 10,
                max_nodes: 100,
                memory_pattern: MemoryPattern::Temporal,
                update_frequency: 5.0,
                sharing_ratio: 0.3,
            },
        },
        agent_count: 20,
        duration: Duration::from_secs(1),
        seed: Some(999),
        objectives: vec![],
    };

    let runner = ScenarioRunner::new(0).expect("Failed to create runner");
    let result = runner
        .run_scenario(&config)
        .expect("Failed to run scenario");

    assert_eq!(result.scenario_id, "test-knowledge");
    assert_eq!(result.agent_count, 20);
}

#[test]
fn test_scenario_with_objectives() {
    use gpu_agents::scenarios::config::PerformanceObjective;

    let config = ScenarioConfig {
        id: "test-objectives".to_string(),
        name: "Test Objectives".to_string(),
        description: "Test scenario with performance objectives".to_string(),
        scenario_type: ScenarioType::Simple {
            behavior: SimpleBehavior::Flocking,
            interaction_radius: 15.0,
            update_frequency: 30.0,
        },
        agent_count: 1000,
        duration: Duration::from_secs(5),
        seed: Some(123),
        objectives: vec![PerformanceObjective {
            metric: "agents_per_second".to_string(),
            target: 10_000.0,
            maximize: true,
        }],
    };

    let runner = ScenarioRunner::new(0).expect("Failed to create runner");
    let result = runner
        .run_scenario(&config)
        .expect("Failed to run scenario");

    // Check that metrics were collected
    assert!(!result.metrics.is_empty());
    assert!(result.metrics.iter().any(|m| m.name == "agents_per_second"));
}

#[test]
fn test_scenario_configuration_loading() {
    // Test YAML loading
    let yaml_content = r#"
id: yaml-test
name: YAML Test Scenario
description: Test loading from YAML
scenario_type:
  type: simple
  behavior: seeking
  interaction_radius: 20.0
  update_frequency: 45.0
agent_count: 500
duration:
  secs: 10
  nanos: 0
seed: 54321
objectives: []
"#;

    std::fs::write("/tmp/test_scenario.yaml", yaml_content).unwrap();
    let config = ScenarioConfig::from_yaml("/tmp/test_scenario.yaml").unwrap();

    assert_eq!(config.id, "yaml-test");
    assert_eq!(config.agent_count, 500);
    if let ScenarioType::Simple { behavior, .. } = config.scenario_type {
        assert_eq!(behavior, SimpleBehavior::Seeking);
    } else {
        panic!("Expected simple scenario type");
    }

    std::fs::remove_file("/tmp/test_scenario.yaml").unwrap();
}

#[test]
fn test_scenario_batch_execution() {
    // Test running multiple scenarios
    let configs = vec![
        ScenarioConfig {
            id: "batch-1".to_string(),
            name: "Batch Test 1".to_string(),
            description: "First batch test".to_string(),
            scenario_type: ScenarioType::Simple {
                behavior: SimpleBehavior::RandomWalk,
                interaction_radius: 5.0,
                update_frequency: 60.0,
            },
            agent_count: 50,
            duration: Duration::from_secs(1),
            seed: Some(1),
            objectives: vec![],
        },
        ScenarioConfig {
            id: "batch-2".to_string(),
            name: "Batch Test 2".to_string(),
            description: "Second batch test".to_string(),
            scenario_type: ScenarioType::Simple {
                behavior: SimpleBehavior::Avoidance,
                interaction_radius: 8.0,
                update_frequency: 30.0,
            },
            agent_count: 75,
            duration: Duration::from_secs(1),
            seed: Some(2),
            objectives: vec![],
        },
    ];

    let runner = ScenarioRunner::new(0).expect("Failed to create runner");

    for config in configs {
        let result = runner
            .run_scenario(&config)
            .expect("Failed to run scenario");
        assert!(result.errors.is_empty());
        assert_eq!(result.scenario_id, config.id);
    }
}
