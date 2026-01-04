//! Comprehensive test suite for ExoRust Bootstrap crate
//!
//! This test suite follows TDD principles and provides comprehensive coverage
//! for all bootstrap functionality including configuration, DNA management,
//! genesis loading, population control, safeguards, and monitoring.

use crate::{
    agents::{EvolutionAgent, PrimeAgent, ReplicatorAgent, TemplateAgent},
    config::{
        BootstrapConfig, BootstrapPhase, EvolutionConfig, MonitoringConfig, PopulationConfig,
        ResourceConfig, SafeguardConfig,
    },
    dna::{AgentDNA, AgentType, PerformanceMetrics, VariableTraits},
    genesis::GenesisLoader,
    monitoring::{AlertCategory, AlertSeverity, BootstrapMonitor},
    population::{AgentStatus, PopulationController},
    safeguards::{BootstrapSafeguards, SafeguardIntervention},
    BootstrapResult,
};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn test_default_config_creation() {
        let config = BootstrapConfig::default();

        // Test population defaults
        assert_eq!(config.population.initial_size, 15);
        assert_eq!(config.population.max_size, 1000);
        assert_eq!(config.population.min_size, 5);
        assert_eq!(config.population.max_growth_rate, 0.1);

        // Test evolution defaults
        assert_eq!(config.evolution.initial_mutation_rate, 0.25);
        assert_eq!(config.evolution.target_mutation_rate, 0.05);
        assert_eq!(config.evolution.min_diversity, 0.3);
        assert_eq!(config.evolution.selection_pressure, 0.7);

        // Test resource defaults
        assert_eq!(config.resources.gpu_memory_per_agent, 64);
        assert_eq!(config.resources.cpu_time_slice, 100);
        assert_eq!(config.resources.max_kernel_time, 200);

        // Test monitoring defaults
        assert_eq!(config.monitoring.metrics_interval, Duration::from_secs(10));
        assert_eq!(
            config.monitoring.health_check_interval,
            Duration::from_secs(30)
        );
        assert!(config.monitoring.detailed_logging);

        // Test safeguards defaults
        assert!(config.safeguards.prevent_explosion);
        assert!(config.safeguards.diversity_protection);
        assert!(config.safeguards.resource_protection);
        assert!(config.safeguards.emergency_reset);
        assert_eq!(config.safeguards.max_failures, 3);
    }

    #[test]
    fn test_config_validation_success() {
        let config = BootstrapConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_zero_initial_size() {
        let mut config = BootstrapConfig::default();
        config.population.initial_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_size_limits() {
        let mut config = BootstrapConfig::default();
        config.population.initial_size = 100;
        config.population.max_size = 50;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_mutation_rate() {
        let mut config = BootstrapConfig::default();
        config.evolution.initial_mutation_rate = 1.5; // > 1.0
        assert!(config.validate().is_err());

        config.evolution.initial_mutation_rate = -0.1; // < 0.0
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_zero_gpu_memory() {
        let mut config = BootstrapConfig::default();
        config.resources.gpu_memory_per_agent = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_current_mutation_rate_calculation() {
        let config = BootstrapConfig::default();

        // Genesis phase should use initial rate
        let rate = config.current_mutation_rate(BootstrapPhase::Genesis, 0.0);
        assert_eq!(rate, config.evolution.initial_mutation_rate);

        // Template creation should use initial rate
        let rate = config.current_mutation_rate(BootstrapPhase::TemplateCreation, 0.0);
        assert_eq!(rate, config.evolution.initial_mutation_rate);

        // Specialization should reduce over time
        let rate_start = config.current_mutation_rate(BootstrapPhase::Specialization, 0.0);
        let rate_mid = config.current_mutation_rate(BootstrapPhase::Specialization, 0.5);
        let rate_end = config.current_mutation_rate(BootstrapPhase::Specialization, 1.0);
        assert!(rate_start > rate_mid);
        assert!(rate_mid > rate_end);

        // Self-sustaining should use target rate
        let rate = config.current_mutation_rate(BootstrapPhase::SelfSustaining, 1.0);
        assert_eq!(rate, config.evolution.target_mutation_rate);
    }

    #[tokio::test]
    async fn test_config_file_serialization() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.json");
        let config_path_str = config_path.to_str().unwrap();

        let original_config = BootstrapConfig::default();

        // Save to file
        original_config.save_to_file(config_path_str)?;

        // Load from file
        let loaded_config = BootstrapConfig::load_from_file(config_path_str).unwrap();

        // Verify they match
        assert_eq!(
            original_config.population.initial_size,
            loaded_config.population.initial_size
        );
        assert_eq!(
            original_config.evolution.initial_mutation_rate,
            loaded_config.evolution.initial_mutation_rate
        );
        assert_eq!(
            original_config.resources.gpu_memory_per_agent,
            loaded_config.resources.gpu_memory_per_agent
        );
    }
}

#[cfg(test)]
mod dna_tests {
    use super::*;

    #[test]
    fn test_agent_dna_creation() {
        let dna = AgentDNA::create_template(AgentType::Prime);

        assert_eq!(dna.core_traits.agent_type, AgentType::Prime);
        assert!(dna.core_traits.can_replicate);
        assert!(dna.core_traits.can_synthesize);
        assert!(!dna.core_traits.can_evaluate);
        assert!(dna.core_traits.can_evolve);
        assert_eq!(dna.core_traits.base_memory, 32);
        assert_eq!(dna.core_traits.base_processing, 100);
        assert_eq!(dna.generation, 0);
        assert!(dna.fitness_history.is_empty());
    }

    #[test]
    fn test_different_agent_types() {
        let prime_dna = AgentDNA::create_template(AgentType::Prime);
        let replicator_dna = AgentDNA::create_template(AgentType::Replicator);
        let evolution_dna = AgentDNA::create_template(AgentType::Evolution);

        // Prime agent traits
        assert!(prime_dna.core_traits.can_replicate);
        assert!(prime_dna.core_traits.can_synthesize);
        assert!(!prime_dna.core_traits.can_evaluate);
        assert!(prime_dna.core_traits.can_evolve);

        // Replicator agent traits
        assert!(replicator_dna.core_traits.can_replicate);
        assert!(replicator_dna.core_traits.can_synthesize);
        assert!(replicator_dna.core_traits.can_evaluate);
        assert!(!replicator_dna.core_traits.can_evolve);

        // Evolution agent traits
        assert!(!evolution_dna.core_traits.can_replicate);
        assert!(!evolution_dna.core_traits.can_synthesize);
        assert!(evolution_dna.core_traits.can_evaluate);
        assert!(evolution_dna.core_traits.can_evolve);
    }

    #[test]
    fn test_agent_reproduction_without_partner() {
        let parent_dna = AgentDNA::create_template(AgentType::Prime);
        let offspring_dna = parent_dna.reproduce(None, 0.5).unwrap();

        assert_ne!(parent_dna.id, offspring_dna.id);
        assert_eq!(offspring_dna.generation, parent_dna.generation + 1);
        assert_eq!(offspring_dna.lineage.parent_id, Some(parent_dna.id));
        assert!(offspring_dna.fitness_history.is_empty());
    }

    #[test]
    fn test_agent_reproduction_with_partner() {
        let parent1_dna = AgentDNA::create_template(AgentType::Prime);
        let parent2_dna = AgentDNA::create_template(AgentType::Replicator);
        let offspring_dna = parent1_dna.reproduce(Some(&parent2_dna), 0.5).unwrap();

        assert_ne!(parent1_dna.id, offspring_dna.id);
        assert_ne!(parent2_dna.id, offspring_dna.id);
        assert_eq!(offspring_dna.generation, parent1_dna.generation + 1);
        assert_eq!(offspring_dna.lineage.parent_id, Some(parent1_dna.id));
    }

    #[test]
    fn test_fitness_calculation() {
        let dna = AgentDNA::create_template(AgentType::Prime);
        let metrics = PerformanceMetrics {
            survival_time: 100,
            goals_completed: 5,
            goals_failed: 1,
            offspring_created: 2,
            resources_used: 800,
            resources_allocated: 1000,
            kernels_synthesized: 3,
            successful_interactions: 4,
        };

        let fitness = dna.calculate_fitness(&metrics);
        assert!(fitness > 0.0);

        // Test with different metrics
        let poor_metrics = PerformanceMetrics {
            survival_time: 10,
            goals_completed: 0,
            goals_failed: 5,
            offspring_created: 0,
            resources_used: 1200,
            resources_allocated: 1000,
            kernels_synthesized: 0,
            successful_interactions: 0,
        };

        let poor_fitness = dna.calculate_fitness(&poor_metrics);
        assert!(poor_fitness < fitness);
    }

    #[test]
    fn test_experience_update() {
        let mut dna = AgentDNA::create_template(AgentType::Prime);

        // Test successful experience
        dna.update_experience("test_goal", true, 0.5);
        assert!(dna
            .experience_memory
            .successful_goals
            .contains_key("test_goal"));
        assert_eq!(dna.experience_memory.successful_goals["test_goal"], 1.0);

        // Test failed experience
        dna.update_experience("bad_goal", false, 0.8);
        assert!(dna.experience_memory.failed_goals.contains_key("bad_goal"));
        assert_eq!(dna.experience_memory.failed_goals["bad_goal"], 0.5);
    }
}

#[cfg(test)]
mod genesis_loader_tests {
    use super::*;

    #[tokio::test]
    async fn test_genesis_loader_creation() {
        let genesis = GenesisLoader::new().unwrap();
        // Basic creation test - if it doesn't panic, it works
    }

    #[tokio::test]
    async fn test_genesis_loader_with_config() {
        let config = BootstrapConfig::default();
        let genesis = GenesisLoader::with_config(config).unwrap();
        // Basic creation test with config
    }

    #[tokio::test]
    async fn test_genesis_loader_default() {
        let genesis = GenesisLoader::default();
        // Test default creation
    }

    #[tokio::test]
    async fn test_invalid_config_rejection() {
        let mut config = BootstrapConfig::default();
        config.population.initial_size = 0; // Invalid

        let result = GenesisLoader::with_config(config);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod population_controller_tests {
    use super::*;

    #[tokio::test]
    async fn test_population_controller_creation() {
        let config = BootstrapConfig::default();
        let controller = PopulationController::new(config).unwrap();

        let stats = controller.get_stats();
        assert_eq!(stats.total_agents, 0);
        assert_eq!(stats.active_agents, 0);
    }

    #[tokio::test]
    async fn test_agent_registration() {
        let config = BootstrapConfig::default();
        let mut controller = PopulationController::new(config).unwrap();

        let dna = AgentDNA::create_template(AgentType::Prime);
        let agent_id = dna.id;

        controller.register_agent(dna).await?;

        let stats = controller.get_stats();
        assert_eq!(stats.total_agents, 1);
        assert_eq!(stats.active_agents, 1);
    }

    #[tokio::test]
    async fn test_agent_metrics_update() {
        let config = BootstrapConfig::default();
        let mut controller = PopulationController::new(config).unwrap();

        let dna = AgentDNA::create_template(AgentType::Prime);
        let agent_id = dna.id;

        controller.register_agent(dna).await?;

        let metrics = PerformanceMetrics {
            survival_time: 100,
            goals_completed: 5,
            goals_failed: 1,
            offspring_created: 2,
            resources_used: 800,
            resources_allocated: 1000,
            kernels_synthesized: 3,
            successful_interactions: 4,
        };

        controller
            .update_agent_metrics(agent_id, metrics)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_agent_removal() {
        let config = BootstrapConfig::default();
        let mut controller = PopulationController::new(config).unwrap();

        let dna = AgentDNA::create_template(AgentType::Prime);
        let agent_id = dna.id;

        controller.register_agent(dna).await?;
        assert_eq!(controller.get_stats().total_agents, 1);

        controller.remove_agent(agent_id).await?;
        assert_eq!(controller.get_stats().total_agents, 0);
    }

    #[tokio::test]
    async fn test_health_score_calculation() {
        let config = BootstrapConfig::default();
        let controller = PopulationController::new(config).unwrap();

        // Empty population should have health score 0
        let health = controller.health_score().await?;
        assert_eq!(health, 0.0);
    }

    #[tokio::test]
    async fn test_diversity_score_calculation() {
        let config = BootstrapConfig::default();
        let controller = PopulationController::new(config).unwrap();

        // Empty population should have diversity score 0
        let diversity = controller.diversity_score().await?;
        assert_eq!(diversity, 0.0);
    }

    #[tokio::test]
    async fn test_mutation_rate_setting() {
        let config = BootstrapConfig::default();
        let mut controller = PopulationController::new(config).unwrap();

        controller.set_mutation_rate(0.15).await.unwrap();
        // Test doesn't fail - rate is set internally

        // Test bounds clamping
        controller.set_mutation_rate(-0.5).await?; // Should clamp to 0.0
        controller.set_mutation_rate(1.5).await?; // Should clamp to 1.0
    }

    #[tokio::test]
    async fn test_autonomous_mode() {
        let config = BootstrapConfig::default();
        let mut controller = PopulationController::new(config).unwrap();

        controller.enable_autonomous_mode().await.unwrap();
        // Test autonomous mode can be enabled without error
    }

    #[tokio::test]
    async fn test_evolution_cycle() {
        let config = BootstrapConfig::default();
        let controller = PopulationController::new(config).unwrap();

        // Test evolution cycle on empty population doesn't crash
        controller.run_evolution_cycle().await?;
    }
}

#[cfg(test)]
mod safeguards_tests {
    use super::*;

    #[tokio::test]
    async fn test_safeguards_creation() {
        let config = BootstrapConfig::default();
        let safeguards = BootstrapSafeguards::new(config).unwrap();

        assert!(!safeguards.is_emergency_mode());
        assert_eq!(safeguards.failure_count(), 0);
        assert!(safeguards.time_since_last_intervention().is_none());
    }

    #[tokio::test]
    async fn test_safeguards_intervention_check() {
        let config = BootstrapConfig::default();
        let mut safeguards = BootstrapSafeguards::new(config).unwrap();

        let intervention = safeguards.check_and_intervene().await.unwrap();
        // Should return None for initial state
        assert!(matches!(intervention, SafeguardIntervention::None));
    }

    #[tokio::test]
    async fn test_safeguards_reset() {
        let config = BootstrapConfig::default();
        let mut safeguards = BootstrapSafeguards::new(config).unwrap();

        safeguards.reset();
        assert!(!safeguards.is_emergency_mode());
        assert_eq!(safeguards.failure_count(), 0);
    }

    #[tokio::test]
    async fn test_intervention_execution() {
        let config = BootstrapConfig::default();
        let mut safeguards = BootstrapSafeguards::new(config.clone()).unwrap();
        let population = Arc::new(RwLock::new(PopulationController::new(config).unwrap()));

        // Test various interventions
        let interventions = vec![
            SafeguardIntervention::ReduceMutationRate { new_rate: 0.1 },
            SafeguardIntervention::IncreaseMutationRate { new_rate: 0.3 },
            SafeguardIntervention::InjectTemplateAgents { count: 5 },
            SafeguardIntervention::CullFailingAgents { threshold: 10.0 },
            SafeguardIntervention::EnforceResourceLimits,
            SafeguardIntervention::None,
        ];

        for intervention in interventions {
            safeguards
                .execute_intervention(intervention, population.clone())
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn test_emergency_reset() {
        let config = BootstrapConfig::default();
        let mut safeguards = BootstrapSafeguards::new(config.clone()).unwrap();
        let population = Arc::new(RwLock::new(PopulationController::new(config).unwrap()));

        let intervention = SafeguardIntervention::EmergencyReset;
        safeguards
            .execute_intervention(intervention, population)
            .await?;

        assert!(safeguards.is_emergency_mode());
    }
}

#[cfg(test)]
mod monitor_tests {
    use super::*;

    #[tokio::test]
    async fn test_monitor_creation() {
        let config = BootstrapConfig::default();
        let monitor = BootstrapMonitor::new(config).unwrap();

        assert!(monitor.get_latest_metrics().is_none());
        assert_eq!(monitor.get_phase_transitions().len(), 0);
    }

    #[tokio::test]
    async fn test_phase_transition_recording() {
        let config = BootstrapConfig::default();
        let mut monitor = BootstrapMonitor::new(config).unwrap();

        monitor
            .record_phase_transition(
                BootstrapPhase::Genesis,
                BootstrapPhase::TemplateCreation,
                "Test transition".to_string(),
                true,
            )
            .unwrap();

        assert_eq!(monitor.get_phase_transitions().len(), 1);
        let transition = &monitor.get_phase_transitions()[0];
        assert_eq!(transition.from_phase, BootstrapPhase::Genesis);
        assert_eq!(transition.to_phase, BootstrapPhase::TemplateCreation);
        assert!(transition.success);
    }

    #[tokio::test]
    async fn test_alert_adding() {
        let config = BootstrapConfig::default();
        let mut monitor = BootstrapMonitor::new(config).unwrap();

        let mut context = HashMap::new();
        context.insert("test_key".to_string(), "test_value".to_string());

        monitor
            .add_alert(
                AlertSeverity::Warning,
                AlertCategory::PopulationHealth,
                "Test alert message".to_string(),
                context,
            )
            .unwrap();

        let recent_alerts = monitor.get_recent_alerts(Duration::from_secs(60));
        assert_eq!(recent_alerts.len(), 1);
        assert_eq!(recent_alerts[0].message, "Test alert message");
    }

    #[tokio::test]
    async fn test_metrics_export() {
        let config = BootstrapConfig::default();
        let monitor = BootstrapMonitor::new(config).unwrap();

        let temp_dir = TempDir::new().unwrap();
        let export_path = temp_dir.path().join("test_export.json");
        let export_path_str = export_path.to_str()?;

        monitor.export_metrics(export_path_str).await?;

        // Verify file exists and contains JSON
        let content = tokio::fs::read_to_string(export_path_str).await.unwrap();
        assert!(content.contains("start_time"));
        assert!(content.contains("metrics_history"));
    }
}

#[cfg(test)]
mod agent_tests {
    use super::*;

    #[tokio::test]
    async fn test_prime_agent_creation() {
        let agent = PrimeAgent::new().await.unwrap();
        assert_eq!(agent.dna().core_traits.agent_type, AgentType::Prime);
        assert!(agent.dna().core_traits.can_replicate);
        assert!(agent.dna().core_traits.can_synthesize);
    }

    #[tokio::test]
    async fn test_replicator_agent_creation() {
        let agent = ReplicatorAgent::new().await.unwrap();
        assert_eq!(agent.dna().core_traits.agent_type, AgentType::Replicator);
        assert!(agent.dna().core_traits.can_replicate);
        assert!(agent.dna().core_traits.can_evaluate);
    }

    #[tokio::test]
    async fn test_evolution_agent_creation() {
        let agent = EvolutionAgent::new().await.unwrap();
        assert_eq!(agent.dna().core_traits.agent_type, AgentType::Evolution);
        assert!(!agent.dna().core_traits.can_replicate);
        assert!(agent.dna().core_traits.can_evaluate);
        assert!(agent.dna().core_traits.can_evolve);
    }

    #[tokio::test]
    async fn test_agent_execution() {
        let mut agent = PrimeAgent::new().await.unwrap();
        let metrics = agent.execute_primary_function().await.unwrap();

        assert!(metrics.survival_time > 0);
        assert!(metrics.resources_allocated > 0);
    }

    #[tokio::test]
    async fn test_agent_learning() {
        let mut agent = PrimeAgent::new().await.unwrap();
        let initial_fitness_len = agent.dna().fitness_history.len();

        let metrics = PerformanceMetrics {
            survival_time: 100,
            goals_completed: 5,
            goals_failed: 1,
            offspring_created: 2,
            resources_used: 800,
            resources_allocated: 1000,
            kernels_synthesized: 3,
            successful_interactions: 4,
        };

        agent.learn_from_experience(&metrics).await.unwrap();
        assert_eq!(agent.dna().fitness_history.len(), initial_fitness_len + 1);
    }

    #[tokio::test]
    async fn test_agent_replication() {
        let agent1 = PrimeAgent::new().await.unwrap();
        let agent2 = ReplicatorAgent::new().await.unwrap();

        let offspring = agent1
            .replicate(Some(&agent2 as &dyn TemplateAgent), 0.2)
            .await?;

        assert_ne!(offspring.dna().id, agent1.dna().id);
        assert_ne!(offspring.dna().id, agent2.dna().id);
        assert_eq!(offspring.dna().generation, agent1.dna().generation + 1);
    }

    #[tokio::test]
    async fn test_agent_termination_decision() {
        let agent = PrimeAgent::new().await.unwrap();

        // New agent should not need termination
        assert!(!agent.should_terminate());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_bootstrap_workflow() {
        let config = BootstrapConfig::default();
        let mut genesis = GenesisLoader::with_config(config.clone()).unwrap();

        // Test basic workflow components exist
        let _monitor = BootstrapMonitor::new(config.clone())?;
        let _safeguards = BootstrapSafeguards::new(config.clone())?;
        let _population = PopulationController::new(config)?;

        // Basic workflow test passes if no panics occur
    }

    #[tokio::test]
    async fn test_agent_population_interaction() {
        let config = BootstrapConfig::default();
        let mut population = PopulationController::new(config).unwrap();

        // Create and register multiple agents
        let prime_agent = PrimeAgent::new().await?;
        let replicator_agent = ReplicatorAgent::new().await?;
        let evolution_agent = EvolutionAgent::new().await?;

        population
            .register_agent(prime_agent.dna().clone())
            .await
            .unwrap();
        population
            .register_agent(replicator_agent.dna().clone())
            .await
            .unwrap();
        population
            .register_agent(evolution_agent.dna().clone())
            .await
            .unwrap();

        let stats = population.get_stats();
        assert_eq!(stats.total_agents, 3);
        assert_eq!(stats.active_agents, 3);

        // Test health and diversity with multiple agents
        let health = population.health_score().await.unwrap();
        let diversity = population.diversity_score().await.unwrap();

        assert!(health >= 0.0 && health <= 1.0);
        assert!(diversity >= 0.0 && diversity <= 1.0);
    }

    #[tokio::test]
    async fn test_monitoring_and_safeguards_integration() {
        let config = BootstrapConfig::default();
        let mut monitor = BootstrapMonitor::new(config.clone()).unwrap();
        let mut safeguards = BootstrapSafeguards::new(config.clone()).unwrap();
        let population = Arc::new(RwLock::new(PopulationController::new(config).unwrap()));

        // Test safeguard check
        let intervention = safeguards.check_and_intervene().await?;

        // Test phase recording
        monitor
            .record_phase_transition(
                BootstrapPhase::Genesis,
                BootstrapPhase::TemplateCreation,
                "Integration test".to_string(),
                true,
            )
            .unwrap();

        // Test alert generation
        let mut context = HashMap::new();
        context.insert("source".to_string(), "integration_test".to_string());

        monitor
            .add_alert(
                AlertSeverity::Info,
                AlertCategory::PopulationHealth,
                "Integration test alert".to_string(),
                context,
            )
            .unwrap();

        assert_eq!(monitor.get_phase_transitions().len(), 1);
        assert_eq!(monitor.get_recent_alerts(Duration::from_secs(60)).len(), 1);
    }

    #[tokio::test]
    async fn test_evolution_cycle_with_agents() {
        let config = BootstrapConfig::default();
        let mut population = PopulationController::new(config).unwrap();

        // Add some agents with different fitness levels
        let mut prime_dna = AgentDNA::create_template(AgentType::Prime);
        prime_dna.fitness_history.push(25.0); // High fitness

        let mut replicator_dna = AgentDNA::create_template(AgentType::Replicator);
        replicator_dna.fitness_history.push(15.0); // Medium fitness

        let mut evolution_dna = AgentDNA::create_template(AgentType::Evolution);
        evolution_dna.fitness_history.push(5.0); // Low fitness

        population.register_agent(prime_dna).await.unwrap();
        population.register_agent(replicator_dna).await.unwrap();
        population.register_agent(evolution_dna).await.unwrap();

        // Run evolution cycle
        population.run_evolution_cycle().await.unwrap();

        // Evolution cycle should complete without error
        let stats = population.get_stats();
        assert_eq!(stats.total_agents, 3);
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[tokio::test]
    async fn test_large_population_handling() {
        let mut config = BootstrapConfig::default();
        config.population.max_size = 100;

        let mut population = PopulationController::new(config).unwrap();

        // Add many agents
        for i in 0..50 {
            let agent_type = match i % 3 {
                0 => AgentType::Prime,
                1 => AgentType::Replicator,
                _ => AgentType::Evolution,
            };

            let dna = AgentDNA::create_template(agent_type);
            population.register_agent(dna).await.unwrap();
        }

        let stats = population.get_stats();
        assert_eq!(stats.total_agents, 50);

        // Test health and diversity calculations with large population
        let health = population.health_score().await.unwrap();
        let diversity = population.diversity_score().await.unwrap();

        assert!(health >= 0.0 && health <= 1.0);
        assert!(diversity >= 0.0 && diversity <= 1.0);
    }

    #[tokio::test]
    async fn test_rapid_agent_creation_and_removal() {
        let config = BootstrapConfig::default();
        let mut population = PopulationController::new(config).unwrap();

        let mut agent_ids = Vec::new();

        // Rapidly create agents
        for _ in 0..20 {
            let dna = AgentDNA::create_template(AgentType::Prime);
            agent_ids.push(dna.id);
            population.register_agent(dna).await?;
        }

        assert_eq!(population.get_stats().total_agents, 20);

        // Rapidly remove agents
        for agent_id in agent_ids {
            population.remove_agent(agent_id).await.unwrap();
        }

        assert_eq!(population.get_stats().total_agents, 0);
    }

    #[tokio::test]
    async fn test_concurrent_agent_operations() {
        let config = BootstrapConfig::default();
        let population = Arc::new(RwLock::new(PopulationController::new(config).unwrap()));

        let mut handles = Vec::new();

        // Spawn concurrent tasks for agent registration
        for i in 0..10 {
            let pop = population.clone();
            let handle = tokio::spawn(async move {
                let dna = AgentDNA::create_template(AgentType::Prime);
                let mut pop = pop.write().await;
                pop.register_agent(dna).await.unwrap();
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let pop = population.read().await;
        assert_eq!(pop.get_stats().total_agents, 10);
    }

    #[tokio::test]
    async fn test_memory_efficiency_with_large_history() {
        let config = BootstrapConfig::default();
        let mut safeguards = BootstrapSafeguards::new(config).unwrap();

        // Simulate many safeguard checks to build up history
        for _ in 0..200 {
            safeguards.check_and_intervene().await?;
        }

        // Should not crash with large history (internal VecDeque should cap at 100)
    }
}
