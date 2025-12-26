//! Tests for advanced evolution strategies

#[cfg(test)]
mod tests {
    use gpu_agents::{
        EvolutionConfig, EvolutionManager, FitnessObjective, GpuSwarm, GpuSwarmConfig,
        MutationStrategy, SelectionStrategy,
    };

    #[test]
    fn test_evolution_config_creation() {
        let config = EvolutionConfig {
            population_size: 1000,
            selection_strategy: SelectionStrategy::NSGA2,
            mutation_rate: 0.02,
            crossover_rate: 0.8,
            objectives: vec![
                FitnessObjective::Performance,
                FitnessObjective::Efficiency,
                FitnessObjective::Novelty,
            ],
            elitism_percentage: 0.1,
            tournament_size: 4,
            mutation_strategy: MutationStrategy::Adaptive,
            max_generations: 1000,
            convergence_threshold: 0.001,
            enable_archive: true,
            archive_size_limit: 500,
            novelty_k_nearest: 15,
            behavioral_descriptor_size: 10,
        };

        assert_eq!(config.population_size, 1000);
        assert_eq!(config.objectives.len(), 3);
        assert_eq!(config.selection_strategy, SelectionStrategy::NSGA2);
        assert_eq!(config.mutation_strategy, MutationStrategy::Adaptive);
        assert!(config.mutation_rate > 0.0);
        assert!(config.crossover_rate > 0.0);
    }

    #[test]
    fn test_evolution_manager_creation() {
        let config = EvolutionConfig::default();
        let manager = EvolutionManager::new_legacy(config);

        assert!(manager.is_ok());
        let manager = manager?;

        assert_eq!(manager.current_generation(), 0);
        assert!(!manager.has_converged());
        assert_eq!(manager.objective_count(), 3); // Default objectives
    }

    #[test]
    fn test_multi_objective_fitness_evaluation() {
        let config = EvolutionConfig::default();
        let mut manager = EvolutionManager::new_legacy(config).unwrap();

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(100)?;

        // Run swarm to generate some performance data
        for _ in 0..10 {
            swarm.step()?;
        }

        // Evaluate fitness for all objectives
        let fitness_vectors = manager.evaluate_multi_objective_fitness(&swarm).unwrap();

        assert_eq!(fitness_vectors.len(), 100); // One per agent

        // Each agent should have fitness values for all objectives
        for fitness_vector in &fitness_vectors {
            assert_eq!(fitness_vector.len(), 3); // Performance, Efficiency, Novelty

            // All fitness values should be valid (not NaN or infinite)
            for &fitness in fitness_vector {
                assert!(fitness.is_finite());
                assert!(fitness >= 0.0); // Assuming non-negative fitness
            }
        }
    }

    #[test]
    fn test_pareto_dominance_calculation() {
        let config = EvolutionConfig::default();
        let manager = EvolutionManager::new_legacy(config).unwrap();

        // Test cases for Pareto dominance
        let fitness_a = vec![0.8, 0.6, 0.7]; // Performance, Efficiency, Novelty
        let fitness_b = vec![0.9, 0.5, 0.6]; // Better performance, worse efficiency, worse novelty
        let fitness_c = vec![0.7, 0.7, 0.8]; // Worse performance, better efficiency, better novelty
        let fitness_d = vec![0.9, 0.7, 0.8]; // Dominates all others

        // A vs B: Neither dominates (trade-offs)
        assert!(!manager.dominates(&fitness_a, &fitness_b));
        assert!(!manager.dominates(&fitness_b, &fitness_a));

        // A vs C: Neither dominates (trade-offs)
        assert!(!manager.dominates(&fitness_a, &fitness_c));
        assert!(!manager.dominates(&fitness_c, &fitness_a));

        // D dominates all others
        assert!(manager.dominates(&fitness_d, &fitness_a));
        assert!(manager.dominates(&fitness_d, &fitness_b));
        assert!(manager.dominates(&fitness_d, &fitness_c));
    }

    #[test]
    fn test_nsga2_selection() {
        let config = EvolutionConfig {
            selection_strategy: SelectionStrategy::NSGA2,
            population_size: 50,
            ..EvolutionConfig::default()
        };
        let mut manager = EvolutionManager::new_legacy(config)?;

        // Create test swarm with diverse agents
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(50)?;

        // Run several steps to differentiate agent performance
        for _ in 0..20 {
            swarm.step().unwrap();
        }

        // Evaluate fitness
        let fitness_vectors = manager.evaluate_multi_objective_fitness(&swarm).unwrap();

        // Perform NSGA-II selection
        let selected_indices = manager.nsga2_selection(&fitness_vectors).unwrap();

        assert_eq!(selected_indices.len(), 50); // Should maintain population size

        // All selected indices should be valid
        for &index in &selected_indices {
            assert!(index < 50);
        }

        // Selection should prefer non-dominated solutions
        let pareto_front = manager.compute_pareto_fronts(&fitness_vectors).unwrap();
        assert!(!pareto_front.is_empty());
        assert!(pareto_front[0].len() > 0); // First front should not be empty
    }

    #[test]
    fn test_adaptive_mutation() {
        let config = EvolutionConfig {
            mutation_strategy: MutationStrategy::Adaptive,
            mutation_rate: 0.02,
            ..EvolutionConfig::default()
        };
        let mut manager = EvolutionManager::new_legacy(config)?;

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(100)?;

        // Track diversity over generations
        let mut diversity_history = Vec::new();

        for generation in 0..10 {
            // Evaluate fitness
            let fitness_vectors = manager.evaluate_multi_objective_fitness(&swarm).unwrap();

            // Calculate population diversity
            let diversity = manager
                .calculate_population_diversity(&fitness_vectors)
                .unwrap();
            diversity_history.push(diversity);

            // Perform adaptive mutation
            let mutation_rate = manager
                .calculate_adaptive_mutation_rate(generation, diversity)
                .unwrap();

            // Mutation rate should be reasonable
            assert!(mutation_rate >= 0.001);
            assert!(mutation_rate <= 0.1);

            // Apply evolution step
            manager.evolve_generation(&mut swarm).unwrap();
        }

        // Verify we have diversity measurements
        assert_eq!(diversity_history.len(), 10);

        // All diversity values should be non-negative
        for diversity in diversity_history {
            assert!(diversity >= 0.0);
        }
    }

    #[test]
    fn test_novelty_search() {
        let config = EvolutionConfig {
            objectives: vec![FitnessObjective::Novelty],
            ..EvolutionConfig::default()
        };
        let manager = EvolutionManager::new_legacy(config)?;

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(50)?;

        // Run agents to generate behavioral data
        for _ in 0..15 {
            swarm.step().unwrap();
        }

        // Extract behavioral descriptors
        let behavioral_descriptors = manager.extract_behavioral_descriptors(&swarm).unwrap();

        assert_eq!(behavioral_descriptors.len(), 50);

        // Each descriptor should have consistent dimensionality
        let descriptor_dim = behavioral_descriptors[0].len();
        assert!(descriptor_dim > 0);

        for descriptor in &behavioral_descriptors {
            assert_eq!(descriptor.len(), descriptor_dim);

            // All values should be finite
            for &value in descriptor {
                assert!(value.is_finite());
            }
        }

        // Calculate novelty scores
        let novelty_scores = manager
            .calculate_novelty_scores(&behavioral_descriptors)
            .unwrap();

        assert_eq!(novelty_scores.len(), 50);

        // All novelty scores should be non-negative
        for score in novelty_scores {
            assert!(score >= 0.0);
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_evolution_convergence() {
        let config = EvolutionConfig {
            max_generations: 100,
            convergence_threshold: 0.01,
            ..EvolutionConfig::default()
        };
        let mut manager = EvolutionManager::new_legacy(config)?;

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(50)?;

        let mut best_fitness_history = Vec::new();

        // Run evolution for several generations
        for generation in 0..20 {
            // Evolve one generation
            manager.evolve_generation(&mut swarm).unwrap();

            // Track best fitness
            let fitness_vectors = manager.evaluate_multi_objective_fitness(&swarm).unwrap();
            let best_fitness = manager.get_best_fitness(&fitness_vectors).unwrap();
            best_fitness_history.push(best_fitness);

            // Check convergence
            if manager.check_convergence(&best_fitness_history).unwrap() {
                println!("Converged at generation {}", generation);
                break;
            }
        }

        assert!(manager.current_generation() > 0);
        assert!(!best_fitness_history.is_empty());

        // Best fitness should generally improve or stabilize
        let final_fitness = best_fitness_history.last().unwrap();
        let initial_fitness = best_fitness_history.first().unwrap();

        // At least one objective should show improvement
        let mut improved = false;
        for i in 0..final_fitness.len() {
            if final_fitness[i] >= initial_fitness[i] {
                improved = true;
                break;
            }
        }
        assert!(
            improved,
            "Evolution should show improvement in at least one objective"
        );
    }

    #[test]
    fn test_gpu_memory_with_evolution() {
        let config = EvolutionConfig::default();
        let mut manager = EvolutionManager::new_legacy(config).unwrap();

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(200)?;

        // Get initial metrics
        let initial_metrics = swarm.metrics();
        let initial_memory = initial_metrics.gpu_memory_used;

        // Run evolution for several generations
        for _ in 0..5 {
            manager.evolve_generation(&mut swarm).unwrap();
        }

        // Check final metrics
        let final_metrics = swarm.metrics();
        let final_memory = final_metrics.gpu_memory_used;

        // Memory usage should remain stable (no leaks)
        let memory_diff = if final_memory > initial_memory {
            final_memory - initial_memory
        } else {
            initial_memory - final_memory
        };

        // Allow for some evolution metadata overhead
        let max_allowed_increase = initial_memory / 10; // 10% increase allowed
        assert!(
            memory_diff <= max_allowed_increase,
            "Memory usage increased too much: {} -> {} bytes",
            initial_memory,
            final_memory
        );
    }

    #[test]
    fn test_evolution_performance_metrics() {
        let config = EvolutionConfig {
            population_size: 100,
            max_generations: 10,
            ..EvolutionConfig::default()
        };
        let mut manager = EvolutionManager::new_legacy(config)?;

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(100)?;

        let start_time = std::time::Instant::now();

        // Run evolution
        for _ in 0..10 {
            manager.evolve_generation(&mut swarm).unwrap();
        }

        let elapsed = start_time.elapsed();
        let generations_per_second = 10.0 / elapsed.as_secs_f64();

        // Get evolution performance metrics
        let metrics = manager.get_performance_metrics().unwrap();

        assert!(metrics.total_generations >= 10);
        assert!(metrics.total_evolution_time_ms > 0.0);
        assert!(metrics.average_generation_time_ms > 0.0);
        assert!(metrics.fitness_evaluations > 0);
        assert!(generations_per_second > 0.0);

        // Performance should be reasonable (at least 1 generation per second)
        assert!(
            generations_per_second >= 1.0,
            "Evolution too slow: {} generations/second",
            generations_per_second
        );

        println!(
            "Evolution performance: {:.2} generations/second",
            generations_per_second
        );
        println!(
            "Average generation time: {:.2} ms",
            metrics.average_generation_time_ms
        );
    }

    #[test]
    fn test_evolution_archive_management() {
        let config = EvolutionConfig {
            population_size: 50,
            enable_archive: true,
            archive_size_limit: 200,
            ..EvolutionConfig::default()
        };
        let mut manager = EvolutionManager::new_legacy(config)?;

        // Create test swarm
        let mut swarm = GpuSwarm::new(GpuSwarmConfig::default())?;
        swarm.initialize(50).unwrap();

        // Run evolution to populate archive
        for _ in 0..8 {
            manager.evolve_generation(&mut swarm).unwrap();
        }

        // Check archive contents
        let archive = manager.get_archive().unwrap();
        assert!(!archive.is_empty());
        assert!(archive.len() <= 200); // Should not exceed limit

        // All archived agents should have valid fitness
        for archived_agent in archive {
            assert!(!archived_agent.fitness_vector.is_empty());
            assert!(archived_agent.generation > 0);
            assert!(archived_agent.novelty_score >= 0.0);

            for &fitness in &archived_agent.fitness_vector {
                assert!(fitness.is_finite());
            }
        }

        // Archive should contain diverse solutions
        let mut unique_fitness_vectors = std::collections::HashSet::new();
        for agent in archive {
            let fitness_key = format!(
                "{:.3}_{:.3}_{:.3}",
                agent.fitness_vector[0], agent.fitness_vector[1], agent.fitness_vector[2]
            );
            unique_fitness_vectors.insert(fitness_key);
        }

        // Should have at least some diversity
        assert!(unique_fitness_vectors.len() > 1, "Archive lacks diversity");
    }
}
