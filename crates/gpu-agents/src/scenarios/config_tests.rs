#[cfg(test)]
mod tests {
    use crate::scenarios::config::PerformanceObjective;
    use crate::scenarios::{
        KnowledgeConfig, MemoryPattern, PromptComplexity, ReasoningConfig, ScenarioConfig,
        ScenarioType, SimpleBehavior,
    };
    use std::time::Duration;

    #[test]
    fn test_simple_scenario_config() {
        let config = ScenarioConfig {
            id: "simple-flocking".to_string(),
            name: "Simple Flocking Scenario".to_string(),
            description: "Test flocking behavior with simple agents".to_string(),
            scenario_type: ScenarioType::Simple {
                behavior: SimpleBehavior::Flocking,
                interaction_radius: 10.0,
                update_frequency: 60.0,
            },
            agent_count: 1000,
            duration: Duration::from_secs(60),
            seed: Some(42),
            objectives: vec![
                PerformanceObjective {
                    metric: "agents_per_second".to_string(),
                    target: 1_000_000.0,
                    maximize: true,
                },
                PerformanceObjective {
                    metric: "memory_usage_mb".to_string(),
                    target: 1024.0,
                    maximize: false,
                },
            ],
        };

        assert_eq!(config.id, "simple-flocking");
        assert_eq!(config.agent_count, 1000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_reasoning_scenario_config() {
        let config = ScenarioConfig {
            id: "llm-decision-making".to_string(),
            name: "LLM Decision Making".to_string(),
            description: "Test LLM-based agent decision making".to_string(),
            scenario_type: ScenarioType::Reasoning {
                config: ReasoningConfig {
                    model: "llama-7b".to_string(),
                    max_tokens: 256,
                    temperature: 0.7,
                    prompt_complexity: PromptComplexity::Moderate,
                    batch_size: 32,
                    decision_frequency: 2.0,
                },
            },
            agent_count: 100,
            duration: Duration::from_secs(300),
            seed: None,
            objectives: vec![PerformanceObjective {
                metric: "decisions_per_second".to_string(),
                target: 200.0,
                maximize: true,
            }],
        };

        assert!(config.validate().is_ok());
        if let ScenarioType::Reasoning { config: reasoning } = &config.scenario_type {
            assert_eq!(reasoning.model, "llama-7b");
            assert_eq!(reasoning.batch_size, 32);
        } else {
            panic!("Expected reasoning scenario type");
        }
    }

    #[test]
    fn test_knowledge_scenario_config() {
        let config = ScenarioConfig {
            id: "knowledge-sharing".to_string(),
            name: "Knowledge Graph Sharing".to_string(),
            description: "Test knowledge sharing between agents".to_string(),
            scenario_type: ScenarioType::Knowledge {
                config: KnowledgeConfig {
                    initial_nodes: 100,
                    max_nodes: 1000,
                    memory_pattern: MemoryPattern::Associative,
                    update_frequency: 10.0,
                    sharing_ratio: 0.3,
                },
            },
            agent_count: 500,
            duration: Duration::from_secs(120),
            seed: Some(12345),
            objectives: vec![PerformanceObjective {
                metric: "knowledge_operations_per_second".to_string(),
                target: 10_000.0,
                maximize: true,
            }],
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_errors() {
        // Test zero agent count
        let mut config = create_simple_config();
        config.agent_count = 0;
        assert!(config.validate().is_err());

        // Test zero duration
        config = create_simple_config();
        config.duration = Duration::from_secs(0);
        assert!(config.validate().is_err());

        // Test negative interaction radius
        config = create_simple_config();
        if let ScenarioType::Simple {
            ref mut interaction_radius,
            ..
        } = config.scenario_type
        {
            *interaction_radius = -1.0;
        }
        assert!(config.validate().is_err());

        // Test invalid temperature
        let mut reasoning_config = create_reasoning_config();
        if let ScenarioType::Reasoning { ref mut config } = reasoning_config.scenario_type {
            config.temperature = 3.0;
        }
        assert!(reasoning_config.validate().is_err());

        // Test invalid sharing ratio
        let mut knowledge_config = create_knowledge_config();
        if let ScenarioType::Knowledge { ref mut config } = knowledge_config.scenario_type {
            config.sharing_ratio = 1.5;
        }
        assert!(knowledge_config.validate().is_err());
    }

    #[test]
    fn test_yaml_serialization() {
        let config = create_simple_config();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let deserialized: ScenarioConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_toml_serialization() {
        let config = create_reasoning_config();
        let toml = toml::to_string(&config).unwrap();
        let deserialized: ScenarioConfig = toml::from_str(&toml).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_from_yaml_file() {
        // Create a temporary YAML file
        let yaml_content = r#"
id: test-scenario
name: Test Scenario
description: A test scenario
scenario_type:
  type: simple
  behavior: random_walk
  interaction_radius: 5.0
  update_frequency: 30.0
agent_count: 100
duration:
  secs: 60
  nanos: 0
seed: 42
objectives:
  - metric: throughput
    target: 1000.0
    maximize: true
"#;

        // Write to temporary file
        std::fs::write("/tmp/test_scenario.yaml", yaml_content).unwrap();

        // Load and validate
        let config = ScenarioConfig::from_yaml("/tmp/test_scenario.yaml").unwrap();
        assert_eq!(config.id, "test-scenario");
        assert_eq!(config.agent_count, 100);

        // Clean up
        std::fs::remove_file("/tmp/test_scenario.yaml").unwrap();
    }

    // Helper functions
    fn create_simple_config() -> ScenarioConfig {
        ScenarioConfig {
            id: "test".to_string(),
            name: "Test".to_string(),
            description: "Test scenario".to_string(),
            scenario_type: ScenarioType::Simple {
                behavior: SimpleBehavior::RandomWalk,
                interaction_radius: 5.0,
                update_frequency: 30.0,
            },
            agent_count: 100,
            duration: Duration::from_secs(60),
            seed: None,
            objectives: vec![],
        }
    }

    fn create_reasoning_config() -> ScenarioConfig {
        ScenarioConfig {
            id: "test-reasoning".to_string(),
            name: "Test Reasoning".to_string(),
            description: "Test reasoning scenario".to_string(),
            scenario_type: ScenarioType::Reasoning {
                config: ReasoningConfig {
                    model: "test-model".to_string(),
                    max_tokens: 128,
                    temperature: 0.5,
                    prompt_complexity: PromptComplexity::Simple,
                    batch_size: 16,
                    decision_frequency: 1.0,
                },
            },
            agent_count: 50,
            duration: Duration::from_secs(30),
            seed: Some(123),
            objectives: vec![],
        }
    }

    fn create_knowledge_config() -> ScenarioConfig {
        ScenarioConfig {
            id: "test-knowledge".to_string(),
            name: "Test Knowledge".to_string(),
            description: "Test knowledge scenario".to_string(),
            scenario_type: ScenarioType::Knowledge {
                config: KnowledgeConfig {
                    initial_nodes: 50,
                    max_nodes: 500,
                    memory_pattern: MemoryPattern::Random,
                    update_frequency: 5.0,
                    sharing_ratio: 0.2,
                },
            },
            agent_count: 200,
            duration: Duration::from_secs(90),
            seed: None,
            objectives: vec![],
        }
    }
}
