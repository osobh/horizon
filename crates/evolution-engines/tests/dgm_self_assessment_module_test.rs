//! Integration test to verify DGM self assessment module structure works correctly

use stratoswarm_evolution_engines::dgm_self_assessment::{
    AssessmentCriteria, DgmSelfAssessment, ModificationType, PerformanceMetrics,
    SelfAssessmentConfig,
};

#[test]
fn test_dgm_self_assessment_module_imports() {
    // Test that we can import all types from the module
    let config = SelfAssessmentConfig::default();
    assert!(config.assessment_interval > 0);
    assert!(config.min_improvement_threshold > 0.0);

    // Test DgmSelfAssessment creation
    let _assessment = DgmSelfAssessment::new(config);

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

#[test]
fn test_modification_types() {
    // Test ModificationType enum
    let mod_types = vec![
        ModificationType::ParameterAdjustment,
        ModificationType::ArchitectureChange,
        ModificationType::ToolEnhancement,
        ModificationType::WorkflowImprovement,
        ModificationType::RandomMutation,
    ];
    assert_eq!(mod_types.len(), 5);
}
