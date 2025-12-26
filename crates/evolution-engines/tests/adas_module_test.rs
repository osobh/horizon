//! Test to verify ADAS module structure works correctly

use stratoswarm_evolution_engines::adas::{
    AdasConfig, AdasEngine, ArchitectureSearchSpace, BehaviorSearchSpace,
};
use stratoswarm_evolution_engines::traits::{EngineConfig, EvolutionEngine};

#[test]
fn test_adas_module_imports() {
    // Test that we can create config types
    let _config = AdasConfig::default();
    let _arch_space = ArchitectureSearchSpace::default();
    let _behavior_space = BehaviorSearchSpace::default();

    // Test that engine can be created
    let engine_result = AdasEngine::new(AdasConfig::default());
    assert!(engine_result.is_ok());
}

#[test]
fn test_config_validation() {
    let config = AdasConfig::default();
    assert!(config.validate().is_ok());
}

#[tokio::test]
async fn test_engine_creation() {
    let config = AdasConfig::default();
    let engine = AdasEngine::new(config).unwrap();

    // Test basic functionality
    let population = engine.generate_initial_population(5).await?;
    assert_eq!(population.size(), 5);
}
