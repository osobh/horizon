//! Integration tests for complete XP-evolution flow
//!
//! This module contains comprehensive tests that verify the entire XP system
//! integrates properly with evolution engines and produces expected behavior.

#[cfg(test)]
mod tests {
    use crate::{
        AgentEvolutionEngine, AgentPerformanceFitnessFunction, AgentXPSharingCoordinator,
        EvolutionXPRewardCalculator, KnowledgeType, LevelBasedFitnessFunction,
        PerformanceMetricType, PerformanceXPConverter, XPSharingConfig,
    };
    use std::collections::HashSet;
    use std::time::Duration;
    use stratoswarm_agent_core::agent::{Agent, AgentConfig};

    /// Helper function to create a test agent with specific configuration
    async fn create_configured_test_agent(name: &str, level: u32, xp: u64) -> Agent {
        let config = AgentConfig {
            name: name.to_string(),
            max_memory: 1024 * 1024 * 1024,    // 1GB
            max_gpu_memory: 512 * 1024 * 1024, // 512MB
            ..Default::default()
        };

        let agent = Agent::new(config).unwrap();
        agent.initialize().await.unwrap();

        if xp > 0 {
            agent
                .award_xp(
                    xp,
                    format!("{} initial XP", name),
                    "initialization".to_string(),
                )
                .await
                .unwrap();
        }

        // Simulate some activity to reach desired level
        for _ in 0..level.saturating_sub(1) * 3 {
            agent.update_goal_stats(true, Duration::from_secs(30)).await;
        }

        agent
    }

    #[tokio::test]
    async fn test_complete_xp_evolution_pipeline() {
        // Test the complete flow: Agent XP gain -> Evolution trigger -> Performance improvement -> More XP

        // Create test agent
        let agent = create_configured_test_agent("pipeline_test", 1, 0).await;
        let fitness_fn = AgentPerformanceFitnessFunction::default();
        let mut evolution_engine = AgentEvolutionEngine::with_defaults(fitness_fn);

        // Initial state verification
        let initial_stats = agent.stats().await;
        assert_eq!(initial_stats.level, 1);
        assert!(initial_stats.current_xp < 100); // Below evolution threshold

        // Simulate agent gaining XP through activities
        agent.update_goal_stats(true, Duration::from_secs(20)).await; // Fast completion
        agent.update_goal_stats(true, Duration::from_secs(25)).await;
        agent.update_goal_stats(true, Duration::from_secs(30)).await;
        agent
            .award_xp(
                75,
                "Performance bonus".to_string(),
                "performance".to_string(),
            )
            .await
            .unwrap();

        // Check that agent is now ready for evolution
        let pre_evolution_stats = agent.stats().await;
        assert!(pre_evolution_stats.current_xp >= 100); // Should be at evolution threshold
        assert!(agent.check_evolution_readiness().await);

        // Trigger evolution through the evolution engine
        let mut agents = vec![agent.clone()];
        let evolution_results = evolution_engine
            .evolve_agent_population(&mut agents)
            .await
            .unwrap();

        // Verify evolution occurred
        assert!(!evolution_results.is_empty());
        let evolution_result = &evolution_results[0];
        assert!(evolution_result.new_level >= evolution_result.previous_level);

        // Check post-evolution state
        let post_evolution_stats = agent.stats().await;
        assert!(post_evolution_stats.level >= pre_evolution_stats.level);
        assert!(post_evolution_stats.total_xp > pre_evolution_stats.total_xp); // Should have gained XP from evolution

        // Verify evolution XP was awarded
        let evolution_xp_gained = post_evolution_stats.total_xp - pre_evolution_stats.total_xp;
        assert!(evolution_xp_gained >= 100); // Should get at least the base evolution reward
    }

    #[tokio::test]
    async fn test_performance_metrics_to_xp_integration() {
        // Test the complete performance metrics to XP conversion pipeline

        let agent = create_configured_test_agent("performance_test", 2, 150).await;
        let mut performance_converter = PerformanceXPConverter::default();

        // Simulate various performance achievements
        for i in 0..5 {
            let execution_time = 20 + i * 5; // Gradually slower execution times
            agent
                .update_goal_stats(i < 4, Duration::from_secs(execution_time))
                .await;
        }

        // Award some optimization XP to simulate performance improvements
        agent
            .award_xp(
                40,
                "Memory optimization".to_string(),
                "optimization".to_string(),
            )
            .await
            .unwrap();
        agent
            .award_xp(
                35,
                "Speed improvement".to_string(),
                "performance".to_string(),
            )
            .await
            .unwrap();

        let initial_xp = agent.stats().await.total_xp;

        // Convert performance to XP
        let performance_breakdown = performance_converter
            .convert_agent_performance_to_xp(&agent)
            .await
            .unwrap();

        // Verify performance XP was calculated
        assert!(performance_breakdown.total_xp > 0);
        assert!(!performance_breakdown.metric_rewards.is_empty());

        // Award the calculated performance XP
        if performance_breakdown.total_xp > 0 {
            agent
                .award_xp(
                    performance_breakdown.total_xp,
                    performance_breakdown.brief_summary(),
                    "performance_conversion".to_string(),
                )
                .await
                .unwrap();
        }

        let final_xp = agent.stats().await.total_xp;
        assert!(final_xp > initial_xp);

        // Verify performance metrics were recorded
        assert!(performance_breakdown
            .metric_rewards
            .contains_key(&PerformanceMetricType::SuccessRate));
        assert!(performance_breakdown
            .metric_rewards
            .contains_key(&PerformanceMetricType::CompletionSpeed));
    }

    #[tokio::test]
    async fn test_cross_agent_xp_sharing_integration() {
        // Test the complete cross-agent learning and XP sharing system

        let mentor_agent = create_configured_test_agent("mentor", 4, 800).await;
        let mentee_agent = create_configured_test_agent("mentee", 2, 150).await;
        let learner_agent = create_configured_test_agent("learner", 1, 50).await;

        let coordinator = AgentXPSharingCoordinator::default();

        // Register agents in the learning network
        coordinator.register_agent(&mentor_agent).await.unwrap();
        coordinator.register_agent(&mentee_agent).await.unwrap();
        coordinator.register_agent(&learner_agent).await.unwrap();

        // Establish mentorship
        coordinator
            .create_mentorship(&mentor_agent, &mentee_agent)
            .await
            .unwrap();

        // Create and share knowledge
        let knowledge_metrics = crate::agent_xp_sharing::KnowledgeMetrics {
            success_rate: 0.85,
            performance_improvement: 0.2,
            resource_efficiency: 0.9,
            applicability_score: 0.8,
            learning_curve_steepness: 0.6,
        };

        let mut context_tags = HashSet::new();
        context_tags.insert("performance".to_string());
        context_tags.insert("optimization".to_string());

        let package_id = coordinator
            .create_knowledge_package(
                &mentor_agent,
                KnowledgeType::Optimization,
                "Advanced resource optimization technique discovered through evolution".to_string(),
                knowledge_metrics,
                context_tags.clone(),
            )
            .await
            .unwrap();

        // Mentee learns from the knowledge package
        let learning_result = coordinator
            .apply_knowledge(&mentee_agent, &package_id)
            .await
            .unwrap();
        assert!(learning_result.xp_gained > 0);

        // Test XP sharing when mentee gains XP
        let initial_mentee_xp = mentee_agent.stats().await.total_xp;
        let sharing_result = coordinator
            .process_xp_gain(&mentee_agent, 100, "successful_goal_completion")
            .await
            .unwrap();

        // Verify XP was shared appropriately
        assert!(sharing_result.shared_to_collective > 0);
        assert!(sharing_result.shared_to_mentors > 0);
        assert!(sharing_result.net_xp_retained < sharing_result.original_xp);

        // Check that collective pool received XP
        let collective_pool = *coordinator.collective_xp_pool.read().await;
        assert!(collective_pool > 0);

        // Verify learning statistics
        let mentee_stats = coordinator
            .get_learning_stats(&mentee_agent.id())
            .await
            .unwrap();
        assert!(mentee_stats.mentor_count > 0);
        assert!(mentee_stats.knowledge_packages_learned > 0);

        let mentor_stats = coordinator
            .get_learning_stats(&mentor_agent.id())
            .await
            .unwrap();
        assert!(mentor_stats.mentee_count > 0);
        assert!(mentor_stats.knowledge_packages_contributed > 0);
    }

    #[tokio::test]
    async fn test_multi_agent_evolution_with_collaboration() {
        // Test multiple agents evolving together with knowledge sharing and collaboration

        let mut agents = Vec::new();
        for i in 0..5 {
            let agent = create_configured_test_agent(
                &format!("collaborative_agent_{}", i),
                2 + i % 3,    // Varied levels
                100 + i * 50, // Varied XP
            )
            .await;
            agents.push(agent);
        }

        // Setup evolution engine and sharing coordinator
        let fitness_fn = AgentPerformanceFitnessFunction::default();
        let mut evolution_engine = AgentEvolutionEngine::with_defaults(fitness_fn);
        let coordinator = AgentXPSharingCoordinator::default();

        // Register all agents
        for agent in &agents {
            coordinator.register_agent(agent).await.unwrap();
        }

        // Create some mentorship relationships
        coordinator
            .create_mentorship(&agents[4], &agents[0])
            .await
            .unwrap(); // Highest level mentors lowest
        coordinator
            .create_mentorship(&agents[3], &agents[1])
            .await
            .unwrap();

        // Simulate collaborative activities and knowledge creation
        for (i, agent) in agents.iter().enumerate() {
            // Each agent gains XP through different activities
            for j in 0..(3 + i) {
                agent
                    .update_goal_stats(j < 2, Duration::from_secs(25 + j as u64 * 5))
                    .await;
            }

            // Higher level agents create knowledge packages
            if i >= 2 {
                let knowledge_metrics = crate::agent_xp_sharing::KnowledgeMetrics {
                    success_rate: 0.7 + (i as f64 * 0.05),
                    performance_improvement: 0.1 + (i as f64 * 0.05),
                    resource_efficiency: 0.8,
                    applicability_score: 0.75,
                    learning_curve_steepness: 0.6,
                };

                let mut tags = HashSet::new();
                tags.insert(format!("technique_{}", i));
                tags.insert("collaboration".to_string());

                coordinator
                    .create_knowledge_package(
                        agent,
                        KnowledgeType::Collaboration,
                        format!("Collaborative technique {} discovered", i),
                        knowledge_metrics,
                        tags,
                    )
                    .await
                    .unwrap();
            }
        }

        // Collect initial stats
        let mut initial_levels: Vec<u32> = Vec::new();
        let mut initial_xp: Vec<u64> = Vec::new();
        for agent in &agents {
            let stats = agent.stats().await;
            initial_levels.push(stats.level);
            initial_xp.push(stats.total_xp);
        }

        // Trigger evolution for all agents
        let evolution_results = evolution_engine
            .evolve_agent_population(&mut agents)
            .await
            .unwrap();

        // Verify collaborative improvements
        let mut evolved_count = 0;
        let mut total_xp_gained = 0u64;

        for (i, agent) in agents.iter().enumerate() {
            let final_stats = agent.stats().await;

            if final_stats.level > initial_levels[i] {
                evolved_count += 1;
            }

            let xp_gained = final_stats.total_xp - initial_xp[i];
            total_xp_gained += xp_gained;

            // Process XP sharing for the gained XP
            if xp_gained > 0 {
                coordinator
                    .process_xp_gain(agent, xp_gained, "evolution_outcome")
                    .await
                    .unwrap();
            }
        }

        // Verify collaborative outcomes
        assert!(
            evolved_count > 0,
            "At least some agents should have evolved"
        );
        assert!(total_xp_gained > 500, "Total XP gain should be substantial");

        // Check that knowledge sharing increased overall performance
        let collective_pool = *coordinator.collective_xp_pool.read().await;
        assert!(
            collective_pool > 0,
            "Collective XP pool should have contributions"
        );

        // Verify learning network effects
        let mut total_knowledge_packages = 0;
        let mut total_mentorships = 0;

        for agent in &agents {
            if let Some(stats) = coordinator.get_learning_stats(&agent.id()).await {
                total_knowledge_packages += stats.knowledge_packages_contributed;
                total_mentorships += stats.mentor_count + stats.mentee_count;
            }
        }

        assert!(
            total_knowledge_packages >= 3,
            "Should have created knowledge packages"
        );
        assert!(
            total_mentorships >= 4,
            "Should have established mentorship relationships"
        );
    }

    #[tokio::test]
    async fn test_reward_calculation_consistency() {
        // Test that XP reward calculations are consistent across different systems

        let agent = create_configured_test_agent("consistency_test", 3, 500).await;

        // Set up all reward calculation systems
        let reward_calculator = EvolutionXPRewardCalculator::default();
        let fitness_fn = LevelBasedFitnessFunction::default();
        let performance_converter = PerformanceXPConverter::default();

        // Simulate evolution result
        let evolution_result = stratoswarm_agent_core::agent::EvolutionResult {
            previous_level: 3,
            new_level: 4,
            xp_at_evolution: 500,
            evolution_timestamp: chrono::Utc::now(),
            capabilities_gained: vec!["advanced_optimization".to_string()],
            previous_metrics: stratoswarm_agent_core::agent::EvolutionMetrics {
                avg_completion_time: Duration::from_secs(45),
                success_rate: 0.7,
                memory_efficiency: 0.75,
                processing_speed: 1.0,
            },
            new_metrics: stratoswarm_agent_core::agent::EvolutionMetrics {
                avg_completion_time: Duration::from_secs(35),
                success_rate: 0.85,
                memory_efficiency: 0.85,
                processing_speed: 1.3,
            },
        };

        // Calculate XP rewards using different methods
        let evolution_breakdown = reward_calculator
            .calculate_evolution_reward(&agent, &evolution_result)
            .await
            .unwrap();

        let fitness_improvement = 0.15; // Simulated fitness improvement
        let fitness_reward =
            fitness_fn.calculate_xp_reward(&evolution_result.new_metrics, fitness_improvement);

        let evolution_performance_xp =
            performance_converter.calculate_evolution_performance_xp(&evolution_result);

        // Verify all systems produce reasonable and consistent rewards
        assert!(evolution_breakdown.total_reward > 0);
        assert!(fitness_reward > 0);
        assert!(evolution_performance_xp > 0);

        // All systems should award significant XP for good improvements
        assert!(evolution_breakdown.total_reward >= 100);
        assert!(fitness_reward >= 50);
        assert!(evolution_performance_xp >= 50);

        // Verify breakdown components
        assert!(evolution_breakdown.level_up_reward > 0);
        assert!(evolution_breakdown.capability_reward > 0);
        assert!(!evolution_breakdown.performance_rewards.is_empty());

        // Test consistency: similar improvements should yield similar rewards
        let similar_evolution = stratoswarm_agent_core::agent::EvolutionResult {
            new_metrics: stratoswarm_agent_core::agent::EvolutionMetrics {
                success_rate: 0.86, // Very similar improvement
                processing_speed: 1.31,
                ..evolution_result.new_metrics
            },
            ..evolution_result
        };

        let similar_breakdown = reward_calculator
            .calculate_evolution_reward(&agent, &similar_evolution)
            .await
            .unwrap();

        // Rewards should be within 20% of each other for similar improvements
        let reward_diff =
            (evolution_breakdown.total_reward as i64 - similar_breakdown.total_reward as i64).abs();
        let avg_reward = (evolution_breakdown.total_reward + similar_breakdown.total_reward) / 2;
        let diff_percentage = (reward_diff as f64 / avg_reward as f64) * 100.0;

        assert!(
            diff_percentage < 20.0,
            "Similar improvements should yield similar rewards"
        );
    }

    #[tokio::test]
    async fn test_edge_cases_and_error_handling() {
        // Test various edge cases and error conditions in the XP-evolution system

        let agent = create_configured_test_agent("edge_case_test", 1, 0).await;
        let fitness_fn = AgentPerformanceFitnessFunction::default();
        let mut evolution_engine = AgentEvolutionEngine::with_defaults(fitness_fn);
        let coordinator = AgentXPSharingCoordinator::default();

        // Test evolution with insufficient XP
        let mut agents = vec![agent.clone()];
        let evolution_results = evolution_engine
            .evolve_agent_population(&mut agents)
            .await
            .unwrap();
        assert!(
            evolution_results.is_empty(),
            "Should not evolve agents without sufficient XP"
        );

        // Test registering same agent multiple times
        coordinator.register_agent(&agent).await.unwrap();
        coordinator.register_agent(&agent).await.unwrap(); // Should not fail

        // Test learning from non-existent knowledge package
        let learning_result = coordinator.apply_knowledge(&agent, "non_existent_id").await;
        assert!(
            learning_result.is_err(),
            "Should fail when learning from non-existent package"
        );

        // Test mentorship with insufficient level
        let low_level_mentor = create_configured_test_agent("low_level", 1, 50).await;
        let mentee = create_configured_test_agent("mentee", 1, 25).await;

        coordinator.register_agent(&low_level_mentor).await.unwrap();
        coordinator.register_agent(&mentee).await.unwrap();

        let mentorship_result = coordinator
            .create_mentorship(&low_level_mentor, &mentee)
            .await;
        assert!(
            mentorship_result.is_err(),
            "Should fail when mentor level is too low"
        );

        // Test XP sharing with zero XP
        let zero_sharing = coordinator
            .process_xp_gain(&agent, 0, "no_gain")
            .await
            .unwrap();
        assert_eq!(zero_sharing.original_xp, 0);
        assert_eq!(zero_sharing.shared_to_collective, 0);
        assert_eq!(zero_sharing.shared_to_mentors, 0);

        // Test collective distribution with empty pool
        let empty_distribution = coordinator.distribute_collective_xp().await.unwrap();
        assert_eq!(empty_distribution.total_distributed, 0);
        assert!(empty_distribution.recipients.is_empty());

        // Test performance conversion with minimal agent activity
        let mut performance_converter = PerformanceXPConverter::default();
        let minimal_breakdown = performance_converter
            .convert_agent_performance_to_xp(&agent)
            .await
            .unwrap();

        // Should work but produce minimal rewards
        assert!(minimal_breakdown.total_xp >= 0);

        // Test finding relevant knowledge with no packages
        let empty_packages = coordinator
            .find_relevant_knowledge(&agent, &[KnowledgeType::Optimization], &HashSet::new())
            .await;
        assert!(empty_packages.is_empty());
    }

    #[tokio::test]
    async fn test_xp_system_scalability() {
        // Test the system with larger numbers of agents to verify scalability

        const AGENT_COUNT: usize = 20;

        // Create many agents with varied characteristics
        let mut agents = Vec::new();
        for i in 0..AGENT_COUNT {
            let agent = create_configured_test_agent(
                &format!("scale_test_agent_{}", i),
                1 + (i % 5) as u32, // Levels 1-5
                (i * 25) as u64,    // Varied XP
            )
            .await;
            agents.push(agent);
        }

        let fitness_fn = AgentPerformanceFitnessFunction::default();
        let mut evolution_engine = AgentEvolutionEngine::with_defaults(fitness_fn);
        let coordinator = AgentXPSharingCoordinator::default();

        // Register all agents
        let start_time = std::time::Instant::now();
        for agent in &agents {
            coordinator.register_agent(agent).await.unwrap();
        }
        let registration_time = start_time.elapsed();
        assert!(
            registration_time < Duration::from_secs(5),
            "Registration should be fast"
        );

        // Create random mentorship networks
        for i in 0..AGENT_COUNT / 4 {
            let mentor_idx = (i * 2) % AGENT_COUNT;
            let mentee_idx = (i * 2 + 1) % AGENT_COUNT;

            // Only create mentorship if mentor has sufficient level
            let mentor_stats = agents[mentor_idx].stats().await;
            if mentor_stats.level >= 3 {
                let _ = coordinator
                    .create_mentorship(&agents[mentor_idx], &agents[mentee_idx])
                    .await;
            }
        }

        // Simulate agent activities
        for agent in &agents {
            for j in 0..5 {
                agent
                    .update_goal_stats(j < 4, Duration::from_secs(30 + j * 10))
                    .await;
            }
        }

        // Trigger mass evolution
        let evolution_start = std::time::Instant::now();
        let evolution_results = evolution_engine
            .evolve_agent_population(&mut agents)
            .await
            .unwrap();
        let evolution_time = evolution_start.elapsed();

        assert!(
            evolution_time < Duration::from_secs(30),
            "Mass evolution should complete reasonably quickly"
        );

        // Process XP sharing for all agents
        let sharing_start = std::time::Instant::now();
        for agent in &agents {
            let _ = coordinator
                .process_xp_gain(agent, 50, "mass_activity")
                .await;
        }
        let sharing_time = sharing_start.elapsed();

        assert!(
            sharing_time < Duration::from_secs(10),
            "Mass XP sharing should be efficient"
        );

        // Verify system remained consistent
        let final_collective_pool = *coordinator.collective_xp_pool.read().await;
        assert!(
            final_collective_pool > 0,
            "Collective pool should have accumulated XP"
        );

        // Verify learning network integrity
        let mut total_relationships = 0;
        for agent in &agents {
            if let Some(stats) = coordinator.get_learning_stats(&agent.id()).await {
                total_relationships += stats.mentor_count + stats.mentee_count;
            }
        }

        assert!(
            total_relationships > 0,
            "Should have maintained learning relationships"
        );

        // Test system memory usage doesn't grow unbounded
        let repository_size = coordinator.knowledge_repository.read().await.len();
        let history_size = coordinator.sharing_history.read().await.len();

        // These should be reasonable sizes (not growing without bounds)
        assert!(
            repository_size < 1000,
            "Knowledge repository should not grow unbounded"
        );
        assert!(
            history_size < 10000,
            "Sharing history should not grow unbounded"
        );
    }
}
