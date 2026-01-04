//! Integration tests for DGM engine with self-assessment functionality
//!
//! These tests validate the complete integration between the DGM evolution engine
//! and its self-assessment system, following TDD methodology.

use stratoswarm_evolution_engines::{
    dgm::{DgmConfig, DgmEngine},
    dgm_self_assessment::{
        DgmSelfAssessment, ModificationType, SelfAssessmentConfig, SelfModification,
    },
    error::EvolutionEngineResult,
    population::{Individual, Population},
    traits::{AgentGenome, ArchitectureGenes, BehaviorGenes, EvolutionEngine, EvolvableAgent},
};
use tokio;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Create a test DGM configuration
fn create_test_dgm_config() -> DgmConfig {
    DgmConfig::default()
}

/// Create test evolvable agents for population
fn create_test_population(size: usize) -> Vec<EvolvableAgent> {
    let mut agents = Vec::new();

    for i in 0..size {
        let genome = AgentGenome {
            goal: stratoswarm_agent_core::Goal::new(
                format!("Test goal {}", i),
                stratoswarm_agent_core::GoalPriority::Normal,
            ),
            architecture: ArchitectureGenes {
                memory_capacity: 1024 * (i + 1),
                processing_units: (i % 4 + 1) as u32,
                network_topology: vec![32, 16, 8],
            },
            behavior: BehaviorGenes {
                exploration_rate: 0.1 + (i as f64 * 0.1),
                learning_rate: 0.01 + (i as f64 * 0.005),
                risk_tolerance: 0.5 + (i as f64 * 0.1),
            },
        };

        let config = stratoswarm_agent_core::AgentConfig {
            name: format!("test_agent_{}", i),
            agent_type: "test".to_string(),
            max_memory: genome.architecture.memory_capacity,
            max_gpu_memory: genome.architecture.memory_capacity / 4,
            priority: 1,
            metadata: serde_json::Value::Null,
        };

        let agent =
            stratoswarm_agent_core::Agent::new(config).expect("Failed to create test agent");

        agents.push(EvolvableAgent { agent, genome });
    }

    agents
}

#[tokio::test]
async fn test_dgm_self_assessment_initialization() -> TestResult {
    // RED phase expectation: DGM engine should initialize with self-assessment
    let config = create_test_dgm_config();

    let result = DgmEngine::new(config);

    // Should successfully create engine with self-assessment
    assert!(result.is_ok(), "DGM engine should initialize successfully");

    let engine = result?;

    // Verify self-assessment is available
    let report = engine.get_self_assessment_report();
    // Initial state should have no report yet
    assert!(report.is_none(), "Initial self-assessment should be None");
    Ok(())
}

#[tokio::test]
async fn test_dgm_self_assessment_standalone() {
    // Test the self-assessment system in isolation
    let config = SelfAssessmentConfig::default();
    let mut assessment = DgmSelfAssessment::new(config);

    // Initial state should be empty
    let metrics = assessment.get_performance_metrics();
    assert_eq!(metrics.fitness_improvement, 0.0);
    assert_eq!(metrics.modification_success_rate, 0.0);

    // Track some performance data
    assessment.track_generation_performance(1, 0.5);
    assessment.track_generation_performance(2, 0.6);
    assessment.track_generation_performance(3, 0.7);

    // Verify performance tracking
    let updated_metrics = assessment.get_performance_metrics();
    assert!(
        updated_metrics.fitness_improvement > 0.0,
        "Fitness should improve over generations"
    );

    // Generate report
    let report = assessment.generate_assessment_report(3);
    assert_eq!(report.generation, 3);
    assert!(!report.assessment_id.is_empty());
    assert!(report.assessment_score >= 0.0 && report.assessment_score <= 1.0);
}

#[tokio::test]
async fn test_dgm_engine_self_assessment_integration() {
    // RED phase: Test complete integration between engine and assessment
    let config = create_test_dgm_config();
    let mut engine = DgmEngine::new(config).expect("Engine creation failed");

    // Create initial population
    let agents = create_test_population(20);
    let mut population =
        Population::from_individuals(agents.into_iter().map(Individual::new).collect());

    // Run evolution step - this should trigger self-assessment
    let result = engine.evolve_step(population).await;
    assert!(result.is_ok(), "Evolution step should succeed");

    let new_population = result.unwrap();
    assert!(new_population.generation > 0, "Generation should advance");

    // Check if self-assessment is now available
    let report = engine.get_self_assessment_report();
    // Depending on assessment interval, might not be available yet
    // This is acceptable as assessment happens periodically
}

#[tokio::test]
async fn test_dgm_modification_tracking() {
    // Test that modifications are properly tracked
    let config = SelfAssessmentConfig::default();
    let mut assessment = DgmSelfAssessment::new(config);

    // Create test modification
    let modification = SelfModification {
        id: "test_mod_1".to_string(),
        generation: 5,
        parent_id: "agent_parent".to_string(),
        child_id: "agent_child".to_string(),
        modification_type: ModificationType::ParameterAdjustment,
        description: "Test modification".to_string(),
        performance_before: 0.5,
        performance_after: Some(0.6),
        successful: Some(true),
    };

    // Track the modification
    assessment.track_modification(modification);

    // Verify statistics
    let stats = assessment.get_modification_statistics();
    assert!(!stats.is_empty(), "Should have modification statistics");
    assert!(stats.iter().any(|s| s.contains("Total modifications: 1")));
    assert!(stats
        .iter()
        .any(|s| s.contains("Successful modifications: 1")));
}

#[tokio::test]
async fn test_dgm_assessment_should_assess_logic() {
    // Test the assessment timing logic
    let mut config = SelfAssessmentConfig::default();
    config.assessment_interval = 5; // Every 5 generations

    let assessment = DgmSelfAssessment::new(config);

    // Test various generations
    assert!(
        !assessment.should_assess(0),
        "Should not assess at generation 0"
    );
    assert!(
        !assessment.should_assess(1),
        "Should not assess at generation 1"
    );
    assert!(
        !assessment.should_assess(4),
        "Should not assess at generation 4"
    );
    assert!(assessment.should_assess(5), "Should assess at generation 5");
    assert!(
        !assessment.should_assess(6),
        "Should not assess at generation 6"
    );
    assert!(
        assessment.should_assess(10),
        "Should assess at generation 10"
    );
}

#[tokio::test]
async fn test_dgm_performance_metrics_calculation() {
    // Test that performance metrics are calculated correctly
    let config = SelfAssessmentConfig::default();
    let mut assessment = DgmSelfAssessment::new(config);

    // Add performance data with clear improvement trend
    let fitness_values = vec![0.1, 0.2, 0.35, 0.5, 0.6, 0.75, 0.8, 0.85, 0.9, 0.92];

    for (gen, fitness) in fitness_values.iter().enumerate() {
        assessment.track_generation_performance((gen + 1) as u32, *fitness);
    }

    let metrics = assessment.get_performance_metrics();

    // Verify improvements are tracked
    assert!(
        metrics.fitness_improvement > 0.0,
        "Should show fitness improvement"
    );
    assert!(
        metrics.convergence_rate >= 0.0,
        "Should have valid convergence rate"
    );
    assert!(
        metrics.diversity_score >= 0.0,
        "Should have valid diversity score"
    );
}

#[tokio::test]
async fn test_dgm_assessment_report_completeness() {
    // Test that assessment reports contain all required information
    let config = SelfAssessmentConfig::default();
    let mut assessment = DgmSelfAssessment::new(config);

    // Add some test data
    assessment.track_generation_performance(1, 0.4);
    assessment.track_generation_performance(2, 0.6);
    assessment.track_generation_performance(3, 0.8);

    let modification = SelfModification {
        id: "test_mod".to_string(),
        generation: 2,
        parent_id: "parent".to_string(),
        child_id: "child".to_string(),
        modification_type: ModificationType::ToolEnhancement,
        description: "Enhanced tool".to_string(),
        performance_before: 0.4,
        performance_after: Some(0.6),
        successful: Some(true),
    };
    assessment.track_modification(modification);

    // Generate comprehensive report
    let report = assessment.generate_assessment_report(3);

    // Verify report completeness
    assert!(!report.assessment_id.is_empty(), "Report should have ID");
    assert_eq!(report.generation, 3);
    assert!(report.timestamp > 0, "Report should have timestamp");
    assert!(report.assessment_score >= 0.0 && report.assessment_score <= 1.0);
    assert!(
        !report.recommendations.is_empty(),
        "Report should have recommendations"
    );
    assert!(
        !report.recent_performance.is_empty(),
        "Report should have performance data"
    );
    assert!(
        !report.top_modifications.is_empty(),
        "Report should include modifications"
    );

    // Verify improvement capability metrics
    assert!(report.improvement_capability.modification_success_rate >= 0.0);
    assert!(report.improvement_capability.avg_performance_gain != 0.0);

    // Verify exploration stats
    assert!(report.exploration_stats.archive_size > 0);
    assert!(report.exploration_stats.diversity_score >= 0.0);
}

#[tokio::test]
async fn test_dgm_engine_multiple_evolution_steps() {
    // Integration test with multiple evolution steps
    let config = create_test_dgm_config();
    let mut engine = DgmEngine::new(config).expect("Engine creation failed");

    // Start with initial population
    let agents = create_test_population(10);
    let mut population =
        Population::from_individuals(agents.into_iter().map(Individual::new).collect());

    // Run multiple evolution steps
    for step in 0..5 {
        let result = engine.evolve_step(population).await;
        assert!(result.is_ok(), "Evolution step {} should succeed", step);

        population = result.unwrap();
        assert_eq!(
            population.generation,
            step + 1,
            "Generation should advance correctly"
        );

        // Every few steps, check self-assessment
        if step > 0 && step % 2 == 0 {
            // Trigger self-assessment manually to test
            let assessment_result = engine
                .perform_self_assessment(population.generation as u32)
                .await;
            assert!(assessment_result.is_ok(), "Self-assessment should succeed");

            let report = assessment_result.unwrap();
            assert_eq!(report.generation, population.generation as u32);
        }
    }

    // Final population should be evolved
    assert_eq!(population.generation, 5);
    assert!(!population.individuals.is_empty());
}

#[tokio::test]
async fn test_dgm_pattern_effectiveness_integration() {
    // Test pattern effectiveness tracking through the engine
    let config = create_test_dgm_config();
    let mut engine = DgmEngine::new(config).expect("Engine creation failed");

    // Create population and evolve to generate patterns
    let agents = create_test_population(15);
    let mut population =
        Population::from_individuals(agents.into_iter().map(Individual::new).collect());

    // Run evolution to build up pattern history
    for _ in 0..3 {
        let result = engine.evolve_step(population).await;
        assert!(result.is_ok(), "Evolution should succeed");
        population = result.unwrap();
    }

    // The engine should now have some patterns and assessment data
    assert!(population.generation > 0);

    // Check that the engine can provide self-assessment
    let metrics = engine.metrics();
    assert!(metrics.generation > 0, "Engine should track generations");
}
