//! Integration test to verify DGM self assessment module structure works correctly

use exorust_evolution_engines::dgm_self_assessment::{
    AssessmentCriteria, AssessmentReport, LineageTracker, ModificationType, PerformanceMetrics,
    PerformanceTracker, SelfAssessmentConfig, SelfAssessmentEngine,
};

#[test]
fn test_dgm_self_assessment_module_imports() {
    // Test that we can import all types from the module
    let config = SelfAssessmentConfig::default();
    assert!(config.assessment_interval > 0);
    assert!(config.min_improvement_threshold > 0.0);

    // Test assessment engine creation
    let engine = SelfAssessmentEngine::new(config);
    assert!(!engine.id.is_empty());
    assert_eq!(engine.current_generation, 0);

    // Test performance metrics
    let metrics = PerformanceMetrics::default();
    assert!(metrics.fitness_improvement >= 0.0);
    assert!(metrics.convergence_rate >= 0.0);
}

#[test]
fn test_assessment_criteria_types() {
    // Test all assessment criteria are accessible
    let criteria = vec![
        AssessmentCriteria::FitnessImprovement,
        AssessmentCriteria::DiversityMaintenance,
        AssessmentCriteria::ConvergenceSpeed,
        AssessmentCriteria::ResourceEfficiency,
        AssessmentCriteria::AdaptabilityScore,
    ];
    assert_eq!(criteria.len(), 5);
}

#[tokio::test]
async fn test_self_assessment_operations() {
    let mut engine = SelfAssessmentEngine::new(SelfAssessmentConfig::default());

    // Test performance tracking
    let tracker = &engine.performance_tracker;
    assert_eq!(tracker.generation_count, 0);

    // Test lineage tracking
    let lineage = &engine.lineage_tracker;
    assert!(lineage.lineage_graph.is_empty());

    // Test assessment generation
    let report = engine.generate_assessment_report().await.unwrap();
    assert!(!report.assessment_id.is_empty());
    assert_eq!(report.generation, 0);
}
